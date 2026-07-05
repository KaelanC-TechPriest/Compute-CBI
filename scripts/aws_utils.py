"""AWS Earth Search helpers for Landsat C2 L2 streaming (requester-pays usgs-landsat)."""
from __future__ import annotations

import gc
import math
import os

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
    BANDS, COLLECTION, OPTICAL, QA, QA_MASK_BITS, RES, SR_OFFSET, SR_SCALE,
    composite, fire_grid,
)

# ---------------------------------------------------------------------------- constants
STAC_URL = "https://earth-search.aws.element84.com/v1"
_GDAL_ENV = {
    "GDAL_CACHEMAX": "256",   # or 512 (MB). Prevents GDAL from grabbing too
    # much RAM for HTTP/S3 block caching during reads + warps.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # Earth Search hrefs are uppercase `..._SR_B4.TIF`; the allow-list match is
    # case-sensitive, so `.TIF` must be present or GDAL refuses to open them.
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
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


# --------------------------------------------------------------------------- credentials
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
    from pathlib import Path
    aws_dir = Path.home() / ".aws"
    return (aws_dir / "credentials").is_file() or (aws_dir / "config").is_file()


def _is_s3_auth_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in (
        "access denied", "403", "credential", "aws_secret",
        "request payer", "requester pays", "not authorized", "signature",
    ))


# ------------------------------------------------------------------------------ search
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


# ------------------------------------------------------------------------------ fetch
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


def lazy_fetch_and_composite(bbox, year, start_day, end_day, max_cloud):
    """Fetch Landsat scenes and build pre/post composites, downloading Y±2 only if needed.

    Processes pre and post phases independently so only one half of the raw data
    is live at a time, roughly halving peak memory. Each phase fetches the
    preferred year first (Y-1 for pre, Y+1 for post) and downloads the fallback
    (Y-2 / Y+2) only when NaN pixels remain.
    """
    grid = fire_grid(bbox)
    grid_str: str | None = None  # filled on first cube

    def _fetch_year(y):
        items = _search(bbox, f"{y}-01-01", f"{y + 1}-01-01",
                        max_cloud, start_day, end_day)
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
        return arrs

    def _make_cube(arrs_by_year):
        all_arrs = [a for al in arrs_by_year.values() for a in al]
        if not all_arrs:
            return None
        c = xr.concat(all_arrs, dim="time", coords="minimal",
                      compat="override").assign_coords(band=OPTICAL)
        c.rio.write_crs(f"EPSG:{grid[0]}", inplace=True)
        return c

    def _phase(preferred_y, fallback_y, slot):
        """Fetch one composite slot ("pre" or "post"), return its DataArray."""
        nonlocal grid_str
        arrs: dict[int, list] = {}
        yr_arrs = _fetch_year(preferred_y)
        if yr_arrs:
            arrs[preferred_y] = yr_arrs

        cube = _make_cube(arrs)
        if cube is None:
            raise RuntimeError(
                f"no overlapping Landsat scenes found for Y{preferred_y - year:+d}")

        if grid_str is None:
            grid_str = f"{cube.sizes['y']}x{cube.sizes['x']} EPSG:{grid[0]}"

        comp = composite(cube, year)
        del cube
        gc.collect()

        if bool(np.any(np.isnan(comp[slot].values))):
            fb_arrs = _fetch_year(fallback_y)
            if fb_arrs:
                arrs[fallback_y] = fb_arrs
                print(f"  landsat Y{fallback_y - year:+d}: {len(fb_arrs)} scene(s) "
                      f"(fallback for {slot})", flush=True)
                cube = _make_cube(arrs)
                assert cube is not None
                comp = composite(cube, year)
                del cube
                gc.collect()

        result = comp[slot]
        total = sum(len(al) for al in arrs.values())
        print(f"  landsat {slot}: {total} scene(s)", flush=True)
        return result

    pre_da  = _phase(year - 1, year - 2, "pre")
    post_da = _phase(year + 1, year + 2, "post")

    print(f"  landsat: grid {grid_str}", flush=True)
    return xr.Dataset({"pre": pre_da, "post": post_da})
