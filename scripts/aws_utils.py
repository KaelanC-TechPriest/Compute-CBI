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
    OPTICAL, composite, fire_grid, _search, _item_window
)

# ---------------------------------------------------------------------------- constants
STAC_URL = "https://earth-search.aws.element84.com/v1"
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
            a = _item_window(it, grid, aws=True)
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

        if len(items) == 0:
            return []

        if debug:
            print(f"debug [_fetch_year]: stac_search={time.perf_counter() - t_search:.2f}s "
                  f"got {len(items)} items from year {y}", flush=True)
        arrs: list[xr.DataArray] = []


        def _fetch_item(it):
            a = _item_window(it, grid, debug=debug, aws=True)
            if a is None:
                return None
            return a.assign_coords(time=it.datetime).expand_dims("time")

        t_read = time.perf_counter() if debug else 0.0

        with ThreadPoolExecutor(max_workers=min(4, len(items))) as ex:
            futures = [ex.submit(_fetch_item, item)
                for item in items]

            for fut in as_completed(futures):
                try:
                    data = fut.result()
                except RasterioIOError as e:
                    if _is_s3_auth_error(e):
                        raise SystemExit(f"error: cannot read Landsat from S3. {_S3_AUTH_HINT}\n"
                                         f"  (underlying error: {e})")
                    raise

                if data is not None:
                    arrs.append(data)

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
