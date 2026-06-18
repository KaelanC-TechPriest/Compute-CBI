"""Standalone one-shot CBI for a single MTBS fire perimeter.

Self-contained consolidation of the modular pipeline. Given one MTBS perimeter:

  1. ensure the RF model (load the repo cache, else train from the CBI CSV + cache it);
  2. ensure the static `def` raster (load the repo cache, else download the TerraClimate
     1981-2010 normal, sum to annual, cache it);
  3. fetch Landsat C2 L2 from AWS Earth Search for the fire bbox (ephemeral, no cache);
  4. composite -> predictors -> RF predict -> CBI/CBI_bc;
  5. write CBI/CBI_bc GeoTIFFs on the EPSG:5070 NLCD-snapped grid, clipped to the perimeter.

Caches (model, def) live at the wider-repo locations so they're shared with the modular code.
Faithful to Parks et al. (2019); see the repo modules this consolidates for provenance.

Landsat streams from AWS Earth Search, which reads the requester-pays `usgs-landsat`
bucket (us-west-2). This needs AWS credentials (`aws configure`, AWS_ACCESS_KEY_ID/
AWS_SECRET_ACCESS_KEY env vars, an instance profile, etc.) and costs a few cents/fire in
egress (cheapest run from EC2 in us-west-2).

Run:
    uv run python scripts/cbi_oneshot.py --gpkg data/fire_perims/test.gpkg \
        --event-id NV4071111641720150629 --out-dir data/cbi/annie
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pystac_client
import rasterio
import rioxarray  # noqa: F401  -- registers the `.rio` accessor
import sklearn
import xarray as xr
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------------------------------- paths
REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / "data/model/cbi_rf.joblib"
DEF_TIF = REPO / "data/terraclimate/def_19812010_annual.tif"
DEF_NC = REPO / "data/terraclimate/TerraClimate_19812010_def.nc"
TRAIN_CSV = REPO / "parks_2019/data/data_for_ee_model.csv"

# --------------------------------------------------------------------------- model
PREDICTORS = ["def", "lat", "rbr", "dmirbi", "dndvi", "post_mirbi"]
TARGET = "CBI"
N_TREES = 500
SEED = 123

# ----------------------------------------------------------------------- terraclimate
DEF_URL = (
    "http://thredds.northwestknowledge.net:8080/thredds/fileServer"
    "/TERRACLIMATE_ALL/climatology/TerraClimate_19812010_def.nc"
)
NODATA_DEF = -9999

# ---------------------------------------------------------------------------- landsat
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "landsat-c2-l2"
OPTICAL = ["blue", "green", "red", "nir08", "swir16", "swir22"]
QA = "qa_pixel"
BANDS = OPTICAL + [QA]
SR_SCALE, SR_OFFSET = 0.0000275, -0.2
QA_MASK_BITS = (0, 1, 3, 4, 5, 7)  # fill, dilated cloud, cloud, shadow, snow, water
INDICES = ["nbr", "ndvi", "ndmi", "evi", "mirbi"]
RES = 30
DEFAULT_MAX_CLOUD = 80.0
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # Earth Search hrefs are uppercase `..._SR_B4.TIF`; the allow-list match is
    # case-sensitive, so `.TIF` must be present or GDAL refuses to open them.
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    # The `usgs-landsat` COG bucket is requester-pays in us-west-2; GDAL's native
    # S3 driver resolves AWS credentials from the env / ~/.aws (no boto3 needed).
    "AWS_REQUEST_PAYER": "requester",
    "AWS_REGION": "us-west-2",
}

# ----------------------------------------------------------------------------- mtbs
SW, WEST, SE, NORTH = (91, 181), (152, 258), (121, 212), (140, 243)
STATE_WINDOWS: dict[str, tuple[int, int]] = {
    "AZ": SW, "NM": SW, "KS": SW, "OK": SW, "TX": SW,
    "CA": WEST, "CO": WEST, "ID": WEST, "MT": WEST, "ND": WEST, "NE": WEST,
    "NV": WEST, "OR": WEST, "SD": WEST, "UT": WEST, "WA": WEST, "WY": WEST,
    "AL": SE, "AR": SE, "CT": SE, "DE": SE, "FL": SE, "GA": SE, "IA": SE,
    "IL": SE, "IN": SE, "KY": SE, "LA": SE, "MA": SE, "MD": SE, "MO": SE,
    "MS": SE, "NC": SE, "NJ": SE, "NY": SE, "OH": SE, "PA": SE, "RI": SE,
    "SC": SE, "TN": SE, "VA": SE, "WV": SE,
    "ME": NORTH, "MI": NORTH, "MN": NORTH, "NH": NORTH, "VT": NORTH, "WI": NORTH,
}
WILDFIRE_CODE = 1

# ----------------------------------------------------------------------------- 5070
NLCD_ORIGIN = (-2493045.0, 3310005.0)


# ========================================================================== model
def ensure_model(model_path: Path, csv_path: Path) -> dict:
    if model_path.is_file():
        print(f"model: using cache {model_path}", flush=True)
        return joblib.load(model_path)
    print(f"model: training from {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    X, y = df[PREDICTORS], df[TARGET]
    model = RandomForestRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=int(round(len(X) / 75 / len(PREDICTORS))),
        max_features="sqrt", random_state=SEED, n_jobs=-1, oob_score=True,
    ).fit(X, y)
    bundle = {
        "model": model, "predictors": PREDICTORS, "target": TARGET,
        "n_samples": int(len(X)), "oob_r2": float(model.oob_score_),
        "sklearn_version": sklearn.__version__,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"model: trained n={len(X)} OOB R2={model.oob_score_:.4f} -> {model_path}",
          flush=True)
    return bundle


# ============================================================================ def
def ensure_def(def_tif: Path, def_nc: Path) -> Path:
    if def_tif.is_file():
        print(f"def: using cache {def_tif}", flush=True)
        return def_tif
    def_tif.parent.mkdir(parents=True, exist_ok=True)
    if not def_nc.is_file():
        import requests
        print(f"def: downloading {DEF_URL}", flush=True)
        tmp = def_nc.with_suffix(".nc.tmp")
        with requests.get(DEF_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.replace(def_nc)
    print("def: summing 12 monthly normals -> annual", flush=True)
    ds = xr.open_dataset(def_nc)
    annual = np.floor(ds["def"].sum("time", skipna=False)).fillna(NODATA_DEF).astype("int16")
    annual = annual.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    annual = annual.rio.write_crs("EPSG:4326").rio.write_nodata(NODATA_DEF)
    ds.close()
    annual.rio.to_raster(def_tif, tiled=True, compress="ZSTD", zstd_level=1)
    print(f"def: wrote {def_tif}", flush=True)
    return def_tif


# =========================================================================== mtbs
def read_perimeter(gpkg: str, event_id: str | None = None,
                   index: int | None = None, layer: str | None = None):
    gdf = gpd.read_file(gpkg, layer=layer).to_crs(5070)
    if event_id is not None:
        sel = gdf[gdf["Event_ID"] == event_id]
        if sel.empty:
            raise SystemExit(f"event-id {event_id} not found in {gpkg}")
        row = sel.iloc[0]
    elif index is not None:
        if not 0 <= index < len(gdf):
            raise SystemExit(f"index {index} out of range (0..{len(gdf) - 1})")
        row = gdf.iloc[index]
    else:  # default: first wildfire
        wf = gdf[gdf["Incid_Type"] == WILDFIRE_CODE] if "Incid_Type" in gdf else gdf
        if wf.empty:
            raise SystemExit("no wildfire perimeter found")
        row = wf.iloc[0]
    eid, state = row["Event_ID"], str(row["Event_ID"])[:2].upper()
    if state not in STATE_WINDOWS:
        raise SystemExit(f"{eid}: no image-season window for state {state!r}")
    sd, ed = STATE_WINDOWS[state]
    geom = row.geometry
    minx, miny, maxx, maxy = geom.bounds
    b = 1000.0
    bbox = transform_bounds("EPSG:5070", "EPSG:4326",
                            minx - b, miny - b, maxx + b, maxy + b)
    return {
        "fire_id": eid, "state": state, "year": int(row["Ig_Date"].year),
        "start_day": sd, "end_day": ed, "geometry": geom, "bbox": bbox,
    }


# ========================================================================= grid
def utm_epsg(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1


def fire_grid(bbox):
    w, s, e, n = bbox
    epsg = utm_epsg((w + e) / 2, (s + n) / 2)
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", f"EPSG:{epsg}", w, s, e, n)
    minx, miny = math.floor(minx / RES) * RES, math.floor(miny / RES) * RES
    maxx, maxy = math.ceil(maxx / RES) * RES, math.ceil(maxy / RES) * RES
    width, height = round((maxx - minx) / RES), round((maxy - miny) / RES)
    return epsg, Affine(RES, 0, minx, 0, -RES, maxy), width, height


# ====================================================================== landsat
_S3_AUTH_HINT = (
    "Earth Search streams Landsat from the requester-pays `usgs-landsat` bucket "
    "(us-west-2), which needs AWS credentials. Set them with `aws configure` or the "
    "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables (an EC2 instance "
    "profile or SSO session also works)."
)


def _aws_creds_available() -> bool:
    """Best-effort check for a resolvable AWS credential source.

    Recognizes static keys, AWS_PROFILE, and container/web-identity roles in the
    environment, plus the ~/.aws files. It cannot see an EC2 instance-profile
    (IMDS) without a network probe, so a False result is advisory only -- callers
    warn and continue rather than hard-exit on it.
    """
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    if any(os.environ.get(k) for k in (
        "AWS_PROFILE", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    )):
        return True
    aws_dir = Path.home() / ".aws"
    return (aws_dir / "credentials").is_file() or (aws_dir / "config").is_file()


def _is_s3_auth_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in (
        "access denied", "403", "credential", "aws_secret",
        "request payer", "requester pays", "not authorized", "signature",
    ))


def _search(bbox, start, end, max_cloud, start_day, end_day):
    cat = pystac_client.Client.open(STAC_URL)
    items = list(cat.search(
        collections=[COLLECTION], bbox=list(bbox), datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).items())
    if start_day is not None:
        items = [it for it in items
                 if start_day <= it.datetime.timetuple().tm_yday <= end_day]
    items.sort(key=lambda it: it.datetime)
    return items


def _item_window(item, grid):
    """Read the fire-bbox window of an item, scale + QA-mask, reproject to grid."""
    g_epsg, g_tr, g_w, g_h = grid
    epsg = int(item.properties["proj:code"].split(":")[1])
    tr = item.properties["proj:transform"]
    sh = item.properties["proj:shape"]
    H, W = int(sh[0]), int(sh[1])
    A = Affine(*tr[:6])

    left = g_tr.c
    top = g_tr.f
    right = left + g_w * RES
    bottom = top - g_h * RES
    minx, miny, maxx, maxy = transform_bounds(f"EPSG:{g_epsg}", f"EPSG:{epsg}",
                                              left, bottom, right, top)
    c0 = max(0, math.floor((minx - A.c) / A.a))
    c1 = min(W, math.ceil((maxx - A.c) / A.a))
    r0 = max(0, math.floor((maxy - A.f) / A.e))
    r1 = min(H, math.ceil((miny - A.f) / A.e))
    if c1 <= c0 or r1 <= r0:
        return None
    win = Window(c0, r0, c1 - c0, r1 - r0)
    wtr = A * Affine.translation(c0, r0)

    arr = {}
    with rasterio.Env(**_GDAL_ENV):
        for band in BANDS:
            # Earth Search hrefs are `s3://usgs-landsat/...`; read via GDAL /vsis3/.
            path = item.assets[band].href.replace("s3://", "/vsis3/")
            with rasterio.open(path) as ds:
                arr[band] = ds.read(1, window=win)

    h, w = arr[QA].shape
    xs = wtr.c + (np.arange(w) + 0.5) * wtr.a
    ys = wtr.f + (np.arange(h) + 0.5) * wtr.e
    maskbits = 0
    for b in QA_MASK_BITS:
        maskbits |= 1 << b
    invalid = (arr[QA].astype("uint16") & maskbits) != 0
    opt = np.stack([arr[b].astype("float32") * SR_SCALE + SR_OFFSET for b in OPTICAL])
    opt[:, invalid] = np.nan

    da = xr.DataArray(opt, dims=("band", "y", "x"),
                      coords={"band": OPTICAL, "y": ys, "x": xs})
    da = da.rio.write_crs(epsg).rio.write_nodata(np.nan)
    out = da.rio.reproject(f"EPSG:{g_epsg}", transform=g_tr, shape=(g_h, g_w),
                           resampling=Resampling.bilinear, nodata=np.nan)
    return out.drop_vars("spatial_ref", errors="ignore")


def fetch_landsat(bbox, year, start_day, end_day, max_cloud):
    grid = fire_grid(bbox)
    items = _search(bbox, f"{year - 2}-01-01", f"{year + 3}-01-01",
                    max_cloud, start_day, end_day)
    # The composite only uses {Y-2, Y-1} (pre) and {Y+1, Y+2} (post); the fire
    # year is fetched-but-unused, so skip it before reading any bytes (~20% less).
    items = [it for it in items if it.datetime.year != year]
    arrs = []
    try:
        for it in items:
            a = _item_window(it, grid)
            if a is not None:
                arrs.append(a.assign_coords(time=it.datetime).expand_dims("time"))
    except RasterioIOError as e:
        if _is_s3_auth_error(e):
            raise SystemExit(f"error: cannot read Landsat from S3. {_S3_AUTH_HINT}\n"
                             f"  (underlying error: {e})")
        raise
    if not arrs:
        raise SystemExit("no overlapping Landsat scenes found")
    cube = xr.concat(arrs, dim="time", coords="minimal",
                     compat="override").assign_coords(band=OPTICAL)
    cube.rio.write_crs(f"EPSG:{grid[0]}", inplace=True)
    print(f"landsat: {cube.sizes['time']} scenes, grid "
          f"{cube.sizes['y']}x{cube.sizes['x']} EPSG:{grid[0]}", flush=True)
    return cube


# ===================================================================== composite
def _scene_indices(cube):
    nir, red, blue = cube.sel(band="nir08"), cube.sel(band="red"), cube.sel(band="blue")
    sw1, sw2 = cube.sel(band="swir16"), cube.sel(band="swir22")
    nbr = (nir - sw2) / (nir + sw2)
    ndvi = (nir - red) / (nir + red)
    ndmi = (nir - sw1) / (nir + sw1)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)
    mirbi = 10.0 * sw1 - 9.8 * sw2 + 2.0
    out = xr.concat([nbr, ndvi, ndmi, evi, mirbi], dim="index")
    return out.assign_coords(index=INDICES).astype("float32").drop_vars("band", errors="ignore")


def composite(cube, year):
    idx = _scene_indices(cube)
    years = pd.to_datetime(cube.time.values).year.to_numpy()
    nan_slice = xr.full_like(idx.isel(time=0, drop=True), np.nan)

    def wmean(yset):
        m = np.isin(years, list(yset))
        if not m.any():
            return nan_slice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return idx.isel(time=m).mean("time", skipna=True)

    pre = wmean({year - 1}).combine_first(wmean({year - 2, year - 1}))
    post = wmean({year + 1}).combine_first(wmean({year + 1, year + 2}))
    ds = xr.Dataset({"pre": pre, "post": post}).drop_vars("band", errors="ignore")
    ds["pre"].rio.write_crs(cube.rio.crs, inplace=True)
    ds["post"].rio.write_crs(cube.rio.crs, inplace=True)
    return ds


# ===================================================================== predictors
def _trunc(da):
    return np.trunc(da).astype("float32").drop_vars("index", errors="ignore")


def build_stack(comp, def_path):
    pre, post = comp["pre"], comp["post"]
    pre_nbr = pre.sel(index="nbr")
    dnbr = _trunc((pre_nbr - post.sel(index="nbr")) * 1000.0)
    rbr = _trunc(dnbr / (pre_nbr + 1.001))
    dndvi = _trunc((pre.sel(index="ndvi") - post.sel(index="ndvi")) * 1000.0)
    dmirbi = _trunc((pre.sel(index="mirbi") - post.sel(index="mirbi")) * 1000.0)
    post_mirbi = _trunc(post.sel(index="mirbi") * 1000.0)

    grid = rbr
    da = rioxarray.open_rasterio(def_path, masked=True).squeeze("band", drop=True)
    defp = da.rio.reproject_match(grid, resampling=Resampling.nearest).load()
    da.close()

    xs, ys = grid["x"].values, grid["y"].values
    xx, yy = np.meshgrid(xs, ys)
    _, lat = Transformer.from_crs(comp.rio.crs, "EPSG:4326", always_xy=True).transform(xx, yy)
    latp = xr.DataArray(np.round(lat).astype("float32"), dims=("y", "x"),
                        coords={"y": grid["y"], "x": grid["x"]})
    latp = latp.rio.write_crs(comp.rio.crs)

    bands = {"def": defp.astype("float32"), "lat": latp, "rbr": rbr,
             "dmirbi": dmirbi, "dndvi": dndvi, "post_mirbi": post_mirbi}
    stack = xr.concat([bands[n] for n in PREDICTORS], dim="band",
                      coords="minimal", compat="override")
    stack = stack.assign_coords(band=PREDICTORS).astype("float32")
    stack.rio.write_crs(comp.rio.crs, inplace=True)
    return stack


# ======================================================================== predict
def _floor2(a):
    return np.floor(a * 100.0) / 100.0


def predict(stack, bundle):
    X = stack.sel(band=bundle["predictors"])
    arr = X.values
    nb, ny, nx = arr.shape
    flat = arr.reshape(nb, -1).T
    valid = np.isfinite(flat).all(axis=1)
    cbi_flat = np.full(flat.shape[0], np.nan, dtype="float32")
    if valid.any():
        cbi_flat[valid] = bundle["model"].predict(
            pd.DataFrame(flat[valid], columns=bundle["predictors"]))
    cbi = _floor2(cbi_flat.reshape(ny, nx)).astype("float32")
    bc = np.where(cbi <= 1.5, (cbi - 1.5) * 1.3 + 1.5, (cbi - 1.5) * 1.175 + 1.5)
    bc = _floor2(np.clip(bc, 0.0, 3.0)).astype("float32")
    coords = {"y": stack["y"], "x": stack["x"]}
    ds = xr.Dataset({
        "CBI": xr.DataArray(cbi, dims=("y", "x"), coords=coords),
        "CBI_bc": xr.DataArray(bc, dims=("y", "x"), coords=coords),
    })
    ds.rio.write_crs(stack.rio.crs, inplace=True)
    ds["CBI"].rio.write_nodata(np.nan, inplace=True)
    ds["CBI_bc"].rio.write_nodata(np.nan, inplace=True)
    return ds


# ========================================================================== 5070
def to_5070_clip(ds, geometry):
    ox, oy = NLCD_ORIGIN
    minx, miny, maxx, maxy = geometry.bounds
    left = ox + math.floor((minx - ox) / RES) * RES
    right = ox + math.ceil((maxx - ox) / RES) * RES
    top = oy - math.floor((oy - maxy) / RES) * RES
    bottom = oy - math.ceil((oy - miny) / RES) * RES
    tr = Affine(RES, 0, left, 0, -RES, top)
    w, h = int(round((right - left) / RES)), int(round((top - bottom) / RES))
    out = ds.rio.reproject("EPSG:5070", transform=tr, shape=(h, w),
                           resampling=Resampling.nearest, nodata=np.nan)
    return out.rio.clip([geometry], crs="EPSG:5070", drop=True)


# =========================================================================== main
def main() -> int:
    p = argparse.ArgumentParser(description="One-shot CBI for one MTBS perimeter.")
    p.add_argument("--gpkg", required=True)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--event-id", default=None,
                     help="MTBS Event_ID (default: first wildfire).")
    sel.add_argument("--index", type=int, default=None,
                     help="0-based row index in the gpkg layer.")
    p.add_argument("--layer", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--def", dest="def_tif", default=str(DEF_TIF))
    p.add_argument("--csv", default=str(TRAIN_CSV))
    args = p.parse_args()

    bundle = ensure_model(Path(args.model), Path(args.csv))
    def_path = ensure_def(Path(args.def_tif), DEF_NC)
    fire = read_perimeter(args.gpkg, args.event_id, args.index, args.layer)
    print(f"fire={fire['fire_id']} state={fire['state']} year={fire['year']} "
          f"DOY=[{fire['start_day']},{fire['end_day']}]", flush=True)

    if not _aws_creds_available():
        print(f"warning: no AWS credentials detected. {_S3_AUTH_HINT}\n"
              "  Continuing in case an instance profile or SSO session is available...",
              flush=True)

    cube = fetch_landsat(fire["bbox"], fire["year"], fire["start_day"],
                         fire["end_day"], args.max_cloud)
    comp = composite(cube, fire["year"])
    stack = build_stack(comp, def_path)
    ds = predict(stack, bundle)
    ds = to_5070_clip(ds, fire["geometry"])

    cbi = ds["CBI"].values
    print(f"out: grid={ds.sizes['y']}x{ds.sizes['x']} crs={ds.rio.crs} "
          f"valid_px={int(np.isfinite(cbi).sum())} "
          f"CBI med={np.nanmedian(cbi):.2f} max={np.nanmax(cbi):.2f}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("CBI", "CBI_bc"):
        ds[name].rio.to_raster(out / f"{fire['fire_id']}_{name}.tif",
                               tiled=True, compress="ZSTD", zstd_level=1)
    print(f"wrote {fire['fire_id']}_CBI.tif, _CBI_bc.tif to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
