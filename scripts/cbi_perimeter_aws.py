"""CBI for all MTBS fire perimeters in a state (or one perimeter), via AWS Earth Search.

Self-contained consolidation of the modular pipeline. For each MTBS perimeter:

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

Run (whole state):
    uv run python scripts/cbi_perimeter_aws.py --gpkg data/fire_perims/mtbs_perims.gpkg \
        --state MT --out-dir data/cbi/MT

Run (single fire):
    uv run python scripts/cbi_perimeter_aws.py --gpkg data/fire_perims/mtbs_perims.gpkg \
        --event-id NV4071111641720150629 --out-dir data/cbi/annie
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pystac_client
import rasterio
import rioxarray  # noqa: F401  -- registers the `.rio` accessor
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from utils import (
    BANDS, COLLECTION, DEF_NC, DEF_TIF, DEFAULT_MAX_CLOUD,
    MODEL_PATH, OPTICAL, QA, QA_MASK_BITS, RES, SR_OFFSET, SR_SCALE,
    STATE_WINDOWS, TRAIN_CSV, WILDFIRE_CODE,
    build_stack, composite, ensure_def, ensure_model, fire_grid, predict, to_5070_clip,
)

# ---------------------------------------------------------------------------- landsat
STAC_URL = "https://earth-search.aws.element84.com/v1"
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
        raise RuntimeError("no overlapping Landsat scenes found")
    cube = xr.concat(arrs, dim="time", coords="minimal",
                     compat="override").assign_coords(band=OPTICAL)
    cube.rio.write_crs(f"EPSG:{grid[0]}", inplace=True)
    print(f"  landsat: {cube.sizes['time']} scenes, grid "
          f"{cube.sizes['y']}x{cube.sizes['x']} EPSG:{grid[0]}", flush=True)
    return cube


# =========================================================================== main
def main() -> int:
    p = argparse.ArgumentParser(
        description="CBI for all MTBS perimeters in a state (or one perimeter), via AWS."
    )
    p.add_argument("--gpkg", required=True)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--event-id", default=None,
                     help="MTBS Event_ID (default: first wildfire).")
    sel.add_argument("--index", type=int, default=None,
                     help="0-based row index in the gpkg layer.")
    p.add_argument("--layer", default=None,
                   help="Layer name in the gpkg (default: first layer).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--def", dest="def_tif", default=str(DEF_TIF))
    p.add_argument("--csv", default=str(TRAIN_CSV))
    p.add_argument("--state", default=None,
                   help="2-letter state abbreviation (e.g. MT, NV) to filter fires by state.")
    p.add_argument("--start-year", type=int, default=1986,
                   help="First fire ignition year to process (default: 1986).")
    p.add_argument("--end-year", type=int, default=2020,
                   help="Last fire ignition year to process (default: 2020).")
    args = p.parse_args()

    if args.state is not None:
        args.state = args.state.upper()
        if args.state not in STATE_WINDOWS:
            p.error(f"Unknown state '{args.state}'. Known states: {', '.join(sorted(STATE_WINDOWS))}")

    if args.start_year < 1986 or args.start_year > 2020:
        p.error("Start year out of bounds (must be in range 1986-2020)")

    if args.end_year < 1986 or args.end_year > 2020:
        p.error("End year out of bounds (must be in range 1986-2020)")

    if not _aws_creds_available():
        print(f"warning: no AWS credentials detected. {_S3_AUTH_HINT}\n"
              "  Continuing in case an instance profile or SSO session is available...",
              flush=True)

    start_time = time.perf_counter()

    bundle = ensure_model(Path(args.model), Path(args.csv))
    def_path = ensure_def(Path(args.def_tif), DEF_NC)

    gdf = gpd.read_file(args.gpkg, layer=args.layer).to_crs(5070)
    gdf = gdf[gdf["Incid_Type"] == WILDFIRE_CODE]

    if args.state is not None:
        gdf = gdf[gdf["Event_ID"].str[:2].str.upper() == args.state]

    if args.event_id is not None:
        gdf = gdf[gdf["Event_ID"] == args.event_id]
        if gdf.empty:
            raise SystemExit(f"event-id {args.event_id} not found in {args.gpkg}")
    elif args.index is not None:
        if not 0 <= args.index < len(gdf):
            raise SystemExit(f"index {args.index} out of range (0..{len(gdf) - 1})")
        gdf = gdf.iloc[[args.index]]

    if gdf.empty:
        print(f"No wildfires found for state={args.state} event_id={args.event_id} "
              f"index={args.index}", flush=True)
        return 0

    state_str = f" in {args.state}" if args.state else ""
    print(f"Found {len(gdf)} wildfire(s){state_str}.", flush=True)

    out_base = Path(args.out_dir)
    if args.state:
        out_base = out_base / args.state
    out_base.mkdir(parents=True, exist_ok=True)

    n_total = len(gdf)
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, (_, row) in enumerate(gdf.iterrows(), start=1):
        fire_id = "<unknown>"
        try:
            fire_id = row["Event_ID"]
            year = int(row["Ig_Date"].year)

            if year < args.start_year or year > args.end_year:
                print(f"[{i}/{n_total}] Skipping {fire_id}: year {year} outside "
                      f"{args.start_year}-{args.end_year}", flush=True)
                n_skip += 1
                continue

            state = str(fire_id)[:2].upper()
            if state not in STATE_WINDOWS:
                print(f"[{i}/{n_total}] Skipping {fire_id}: no image-season window "
                      f"for state {state!r}", flush=True)
                n_skip += 1
                continue

            out_cbi = out_base / f"{fire_id}_CBI.tif"
            out_cbi_bc = out_base / f"{fire_id}_CBI_bc.tif"
            if out_cbi.exists() and out_cbi_bc.exists():
                print(f"[{i}/{n_total}] Skipping {fire_id}: outputs already exist",
                      flush=True)
                n_skip += 1
                continue

            sd, ed = STATE_WINDOWS[state]
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds
            b = 1000.0
            bbox = transform_bounds("EPSG:5070", "EPSG:4326",
                                    minx - b, miny - b, maxx + b, maxy + b)

            print(f"[{i}/{n_total}] {fire_id} state={state} year={year} "
                  f"DOY=[{sd},{ed}]", flush=True)

            cube = fetch_landsat(bbox, year, sd, ed, args.max_cloud)
            comp = composite(cube, year)
            stack = build_stack(comp, def_path)
            ds = predict(stack, bundle)
            ds = to_5070_clip(ds, geom)

            for name in ("CBI", "CBI_bc"):
                ds[name].rio.to_raster(
                    out_base / f"{fire_id}_{name}.tif",
                    tiled=True, compress="ZSTD", zstd_level=1,
                )

            cbi = ds["CBI"].values
            print(f"  -> Done: grid={ds.sizes['y']}x{ds.sizes['x']} crs={ds.rio.crs} "
                  f"valid_px={int(np.isfinite(cbi).sum())} "
                  f"CBI med={np.nanmedian(cbi):.2f} max={np.nanmax(cbi):.2f}", flush=True)
            n_ok += 1

        except Exception as e:
            print(f"  -> Failed on {fire_id}: {type(e).__name__}: {e}\n"
                  f"{traceback.format_exc()}", flush=True)
            n_fail += 1
            continue

    elapsed = (time.perf_counter() - start_time) / 60
    print(f"\nDone{state_str}: {n_ok} ok, {n_skip} skipped, {n_fail} failed "
          f"({elapsed:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())