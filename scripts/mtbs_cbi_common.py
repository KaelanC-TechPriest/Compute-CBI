"""Shared constants and per-year logic for the MTBS-vs-CBI comparison scripts.

See mtbs_cbi_histogram.py (compute) and mtbs_cbi_plot.py (aggregate + plot).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401  -- registers the `.rio` accessor
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

DEFAULT_MTBS_DIR = Path("/run/data_raid5/shared_data/fire_analysis/data/mtbs/cog")
DEFAULT_CBI_DIR = Path("/run/host/run/media/kaelan/CBI")

# Padding (pixels) added to each per-fire MTBS read window before rounding to
# whole pixels. MTBS (ESRI:102039) and CBI (EPSG:5070) share projection
# parameters but not a datum, so a tightly-rounded window can clip a thin
# real edge of data that reproject_match would otherwise need -- padding
# costs nothing (the window stays tiny) and avoids starving fire-boundary
# pixels.
_WINDOW_PAD = 2

MTBS_CLASSES = (1, 2, 3, 4, 5, 6)
MTBS_LABELS = {
    1: "Unburned to Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Increased Greenness",
    6: "Non-Mapped/Mask",
}
# dataviz skill categorical palette, slots 1-6, light mode, fixed order
MTBS_COLORS = {
    1: "#2a78d6",  # blue
    2: "#eb6834",  # orange
    3: "#1baf7a",  # aqua
    4: "#eda100",  # yellow
    5: "#e87ba4",  # magenta
    6: "#008300",  # green
}


def discover_years(mtbs_dir: Path, cbi_dir: Path) -> list[int]:
    mtbs_years = {int(p.stem) for p in mtbs_dir.glob("*.tif") if p.stem.isdigit()}
    cbi_years = {
        int(p.name) for p in cbi_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and any(p.glob("*_CBI_bc.tif"))
    }
    return sorted(mtbs_years & cbi_years)


def _read_mtbs_window(mtbs_ds: rasterio.DatasetReader, cbi: xr.DataArray) -> xr.DataArray:
    """Read the small MTBS window matching a CBI fire tile's extent, as a DataArray."""
    bounds = transform_bounds(cbi.rio.crs, mtbs_ds.crs, *cbi.rio.bounds())
    win = from_bounds(*bounds, transform=mtbs_ds.transform)
    col_off = math.floor(win.col_off) - _WINDOW_PAD
    row_off = math.floor(win.row_off) - _WINDOW_PAD
    col_end = math.ceil(win.col_off + win.width) + _WINDOW_PAD
    row_end = math.ceil(win.row_off + win.height) + _WINDOW_PAD
    window = Window(col_off, row_off, col_end - col_off, row_end - row_off)

    arr = mtbs_ds.read(1, window=window, boundless=True, fill_value=0)
    transform = mtbs_ds.window_transform(window)
    h, w = arr.shape
    xs = transform.c + (np.arange(w) + 0.5) * transform.a
    ys = transform.f + (np.arange(h) + 0.5) * transform.e
    mtbs_da = xr.DataArray(arr, dims=("y", "x"), coords={"y": ys, "x": xs})
    mtbs_da = mtbs_da.rio.write_crs(mtbs_ds.crs)
    # value 0 is both the nodata sentinel and the ambiguous "Background" class -- drop both
    return mtbs_da.where(mtbs_da != 0)


def process_year(year: int, mtbs_dir: Path, cbi_dir: Path,
                 bin_edges: np.ndarray) -> np.ndarray | None:
    """Return a (n_bins, 6) joint histogram of (CBI value, MTBS class) for one year.

    Iterates the year's individual fire-perimeter CBI tiles rather than the
    full CONUS-wide mosaic (the mosaic has ~125x more pixels than actually
    belong to any fire that year). For each fire, only the matching small
    window is read from the MTBS COG, relying on its internal 256x256 tiling
    for a cheap read instead of warping the entire file.

    Returns None (rather than a zero array) when there's no source data at
    all, so callers can distinguish "not computed" from "computed and
    genuinely zero".
    """
    mtbs_path = mtbs_dir / f"{year}.tif"
    fire_dir = cbi_dir / str(year)
    fire_paths = sorted(fire_dir.glob("*_CBI_bc.tif")) if fire_dir.is_dir() else []

    if not mtbs_path.is_file() or not fire_paths:
        print(f"  {year}: skipping, missing {mtbs_path if not mtbs_path.is_file() else fire_dir}",
              flush=True)
        return None

    n_bins = len(bin_edges) - 1
    H = np.zeros((n_bins, len(MTBS_CLASSES)))
    n_valid_total = 0
    t0 = time.perf_counter()

    with rasterio.open(mtbs_path) as mtbs_ds:
        for fire_path in fire_paths:
            cbi = rioxarray.open_rasterio(fire_path, masked=True).squeeze("band", drop=True)
            mtbs_da = _read_mtbs_window(mtbs_ds, cbi)
            mtbs_on_cbi = mtbs_da.rio.reproject_match(cbi, resampling=Resampling.nearest)

            cbi_vals = cbi.values.ravel()
            mtbs_vals = mtbs_on_cbi.values.ravel()
            cbi.close()

            valid = np.isfinite(cbi_vals) & np.isfinite(mtbs_vals)
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue

            class_valid = np.rint(mtbs_vals[valid]).astype(np.int16)
            h, _, _ = np.histogram2d(
                cbi_vals[valid], class_valid,
                bins=[bin_edges, np.arange(0.5, len(MTBS_CLASSES) + 1.5)],
            )
            H += h
            n_valid_total += n_valid

    print(f"  {year}: {len(fire_paths)} fires, {n_valid_total:,} valid pixels "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return H
