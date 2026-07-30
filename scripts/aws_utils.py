"""AWS Earth Search helpers for Landsat C2 L2 streaming (requester-pays usgs-landsat)."""
from __future__ import annotations

import gc
import math
import os
import time

import numpy as np
import pystac
import pystac_client
import rasterio
import rioxarray  # noqa: F401  -- registers the `.rio` accessor
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    BANDS, COLLECTION, OPTICAL, QA, QA_MASK_BITS, RES, SR_OFFSET, SR_SCALE,
    composite, fire_grid,
)

# ---------------------------------------------------------------------------- constants
STAC_URL = "https://earth-search.aws.element84.com/v1"
_GDAL_ENV = {
    "AWS_REGION": "us-west-2",
    "AWS_REQUEST_PAYER": "requester", # The `usgs-landsat` COG bucket is requester-pays in us-west-2; GDAL's native S3 driver resolves AWS credentials from the env / ~/.aws (no boto3 needed).
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff", # Earth Search hrefs are uppercase `..._SR_B4.TIF`; the allow-list match is case-sensitive, so `.TIF` must be present or GDAL refuses to open them.
    "CPL_VSIL_CURL_CACHE_SIZE": "800000000",        # ~200 MB VSI curl cache
    "CPL_VSIL_CURL_CHUNK_SIZE": "2097152",      # 2 MB (default is much smaller)
    "CPL_VSIL_CURL_USE_HEAD": "NO",                 # sometimes helps with S3
    # "GDAL_CACHEMAX": 1073741824,  # 1 GB in bytes; rasterio calls GDALSetCacheMax64() directly so bytes are required.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",   # faster open on S3
    "GDAL_HTTP_MAX_CACHED_CONNECTIONS": "100",     # keep-alive cache (GDAL ≥ 3.11)
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_MAX_TOTAL_CONNECTIONS": "200",      # total simultaneous connections
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MULTIRANGE": "YES",                 # or "SERIAL" / "SINGLE_GET"
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_VERSION": "2",                  # helps multiplexing when supported
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "25000000",              # 25 MB per-file cache
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


def _item_window(
    item: pystac.Item,
    grid: tuple[int, Affine, int, int],
    debug: bool = False,
) -> xr.DataArray | None:
    """Read the fire-bbox window of an item, scale + QA-mask, reproject to grid."""

    def _read_band(band: str, path: str, win: Window):
        with rasterio.Env(**_GDAL_ENV):
            with rasterio.open(path) as ds:
                return band, ds.read(1, window=win)

    t0 = time.perf_counter() if debug else 0.0
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
    t_s3 = time.perf_counter() if debug else 0.0
    with rasterio.Env(**_GDAL_ENV):
        paths = {
            band: item.assets[band].href.replace("s3://", "/vsis3/")
            for band in BANDS
        }

        with ThreadPoolExecutor(max_workers=len(BANDS)) as ex:
            futures = {
                ex.submit(_read_band, band, path, win): band
                for band, path in paths.items()
            }

            for fut in as_completed(futures):
                band, data = fut.result()
                arr[band] = data

    t_s3_done = time.perf_counter() if debug else 0.0

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
    t_repro = time.perf_counter() if debug else 0.0
    out = da.rio.reproject(f"EPSG:{g_epsg}", transform=g_tr, shape=(g_h, g_w),
                           resampling=Resampling.bilinear, nodata=np.nan)
    if debug:
        print(
            f"debug [_item_window]: s3_read={t_s3_done - t_s3:.2f}s "
            f"reproject={time.perf_counter() - t_repro:.2f}s "
            f"total={time.perf_counter() - t0:.2f}s "
            f"win={int(win.width)}x{int(win.height)} id={item.id}",
            flush=True,
        )
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
    # print(f"  landsat: {cube.sizes['time']} scenes, grid "
    #       f"{cube.sizes['y']}x{cube.sizes['x']} EPSG:{grid[0]}", flush=True)
    return cube


def lazy_fetch_and_composite(
    bbox,
    year: int,
    slot,
    start_day,
    end_day,
    max_cloud,
    debug: bool = False,
) -> xr.DataArray:
    """Fetch Landsat for one composite slot, downloading fallback year only if
    needed.

    Call once for pre (preferred=Y-1, fallback=Y-2) and once for post
    (preferred=Y+1, fallback=Y+2). Returns the composite DataArray for that
    slot.
    """

    t_all = time.perf_counter() if debug else 0.0
    preferred_y = year - 1
    fallback_y = year - 2

    if slot == "post":
        preferred_y = year + 1
        fallback_y = year + 2

    grid = fire_grid(bbox)

    def _fetch_year(y: int) -> list[xr.DataArray]:
        t_year = time.perf_counter() if debug else 0.0
        if debug: print(f"debug [_fetch_year]: fetching landsat for year {y}", flush=True)
        t_search = time.perf_counter() if debug else 0.0
        items = _search(bbox, f"{y}-01-01", f"{y + 1}-01-01",
                        max_cloud, start_day, end_day)
        if debug:
            print(f"debug [_fetch_year]: stac_search={time.perf_counter() - t_search:.2f}s "
                  f"got {len(items)} items from year {y}", flush=True)
        arrs: list[xr.DataArray] = []
        t_read = time.perf_counter() if debug else 0.0
        try:
            for it in items:
                a = _item_window(it, grid, debug=debug)
                if a is not None:
                    arrs.append(a.assign_coords(time=it.datetime).expand_dims("time"))
        except RasterioIOError as e:
            if _is_s3_auth_error(e):
                raise SystemExit(f"error: cannot read Landsat from S3. {_S3_AUTH_HINT}\n"
                                 f"  (underlying error: {e})")
            raise
        if debug:
            print(f"debug [_fetch_year]: scene_reads={time.perf_counter() - t_read:.2f}s "
                  f"{len(arrs)}/{len(items)} scenes overlap grid for year {y} "
                  f"(year_total={time.perf_counter() - t_year:.2f}s)", flush=True)
        return arrs

    def _make_cube(arrs_by_year: dict[int, list[xr.DataArray]]) -> xr.DataArray | None:
        all_arrs = [a for al in arrs_by_year.values() for a in al]
        if debug: print(f"debug [_make_cube]: making cube with {len(all_arrs)} arrays", flush=True)
        if not all_arrs:
            return None
        c = xr.concat(all_arrs, dim="time", coords="minimal",
                      compat="override").assign_coords(band=OPTICAL)
        c.rio.write_crs(f"EPSG:{grid[0]}", inplace=True)
        return c

    arrs: dict[int, list[xr.DataArray]] = {}
    yr_arrs = _fetch_year(preferred_y)
    if yr_arrs:
        arrs[preferred_y] = yr_arrs
    else:
        if debug:
            print(f"debug [lazy_fetch_and_composite]: preferred Y{preferred_y} empty, "
                  f"trying fallback Y{fallback_y}", flush=True)
        fb_arrs = _fetch_year(fallback_y)
        if fb_arrs:
            arrs[fallback_y] = fb_arrs

    cube = _make_cube(arrs)
    if cube is None:
        raise RuntimeError(
            f"no overlapping Landsat scenes found for "
            f"Y{preferred_y - year:+d} or Y{fallback_y - year:+d}")

    if debug:
        print(f"debug [lazy_fetch_and_composite]: landsat {slot}: "
              f"grid {cube.sizes['y']}x{cube.sizes['x']} EPSG:{grid[0]}", flush=True)

    t_comp = time.perf_counter() if debug else 0.0
    comp = composite(cube, year, debug=debug)
    if debug:
        print(f"debug [lazy_fetch_and_composite]: composite={time.perf_counter() - t_comp:.2f}s "
              f"slot={slot}", flush=True)

    if bool(np.any(np.isnan(comp[slot].values))) and fallback_y not in arrs:
        if debug:
            print(f"debug [lazy_fetch_and_composite]: landsat Y{preferred_y} has NaNs, "
                  f"falling back to Y{fallback_y}", flush=True)
        fb_arrs = _fetch_year(fallback_y)
        if fb_arrs:
            arrs[fallback_y] = fb_arrs
            cube = _make_cube(arrs)
            assert cube is not None
            t_comp = time.perf_counter() if debug else 0.0
            comp = composite(cube, year, debug=debug)
            if debug:
                print(f"debug [lazy_fetch_and_composite]: composite_fallback="
                      f"{time.perf_counter() - t_comp:.2f}s slot={slot}", flush=True)

    result = comp[slot]
    del cube, comp
    gc.collect()

    if debug:
        total = sum(len(al) for al in arrs.values())
        print(f"debug [lazy_fetch_and_composite]: landsat {slot}: {total} scene(s) "
              f"total={time.perf_counter() - t_all:.2f}s", flush=True)
    return result
