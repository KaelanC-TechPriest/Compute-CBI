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
DEFAULT_NLCD_DIR = Path("/run/media/fire_analysis/data/nlcd/cleaned")

NLCD_NODATA = 250

LAND_COVER_CATEGORIES = (
    "Water", "Developed", "Barren", "Forest",
    "Shrubland", "Grassland", "Agriculture", "Wetland",
)
_NLCD_CODE_TO_CATEGORY = {
    11: "Water", 12: "Water",
    21: "Developed", 22: "Developed", 23: "Developed", 24: "Developed",
    31: "Barren",
    41: "Forest", 42: "Forest", 43: "Forest",
    52: "Shrubland",
    71: "Grassland",
    81: "Agriculture", 82: "Agriculture",
    90: "Wetland", 95: "Wetland",
}
# Byte pixel value -> land-cover category index (-1 for nodata/unmapped codes).
# A vectorized lut[nlcd_vals] lookup is much cheaper than a per-pixel dict lookup.
_NLCD_LUT = np.full(256, -1, dtype=np.int8)
for _code, _cat in _NLCD_CODE_TO_CATEGORY.items():
    _NLCD_LUT[_code] = LAND_COVER_CATEGORIES.index(_cat)

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


def _cbi_suffix(bias_corrected: bool) -> str:
    return "_CBI_bc.tif" if bias_corrected else "_CBI.tif"


def discover_years(mtbs_dir: Path, cbi_dir: Path, bias_corrected: bool = True) -> list[int]:
    suffix = _cbi_suffix(bias_corrected)
    mtbs_years = {int(p.stem) for p in mtbs_dir.glob("*.tif") if p.stem.isdigit()}
    cbi_years = {
        int(p.name) for p in cbi_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and any(p.glob(f"*{suffix}"))
    }
    return sorted(mtbs_years & cbi_years)


def _read_aligned_window(ds: rasterio.DatasetReader, cbi: xr.DataArray,
                         nodata_value: int) -> xr.DataArray:
    """Read the small window of `ds` matching a CBI fire tile's extent, as a DataArray.

    `nodata_value` pixels (and anything read as fill outside `ds`'s own extent)
    are set to NaN. Shared by MTBS (nodata=0, which also doubles as the
    ambiguous "Background" class -- dropping it is intentional) and NLCD
    (nodata=NLCD_NODATA).
    """
    bounds = transform_bounds(cbi.rio.crs, ds.crs, *cbi.rio.bounds())
    win = from_bounds(*bounds, transform=ds.transform)
    col_off = math.floor(win.col_off) - _WINDOW_PAD
    row_off = math.floor(win.row_off) - _WINDOW_PAD
    col_end = math.ceil(win.col_off + win.width) + _WINDOW_PAD
    row_end = math.ceil(win.row_off + win.height) + _WINDOW_PAD
    window = Window(col_off, row_off, col_end - col_off, row_end - row_off)

    arr = ds.read(1, window=window, boundless=True, fill_value=nodata_value)
    transform = ds.window_transform(window)
    h, w = arr.shape
    xs = transform.c + (np.arange(w) + 0.5) * transform.a
    ys = transform.f + (np.arange(h) + 0.5) * transform.e
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": ys, "x": xs})
    da = da.rio.write_crs(ds.crs)
    return da.where(da != nodata_value)


def process_year(year: int, mtbs_dir: Path, cbi_dir: Path, nlcd_dir: Path,
                 bin_edges: np.ndarray, bias_corrected: bool = True) -> np.ndarray | None:
    """Return a (n_bins, 6, 8) joint histogram of (CBI value, MTBS class,
    NLCD land-cover category) for one year.

    Iterates the year's individual fire-perimeter CBI tiles rather than the
    full CONUS-wide mosaic (the mosaic has ~125x more pixels than actually
    belong to any fire that year). For each fire, only the matching small
    window is read from the MTBS COG and the NLCD COG, relying on their
    internal 256x256 tiling for a cheap read instead of warping the entire
    file. Land cover is read from year - 1 (the year before the fire), so it
    reflects the vegetation that actually burned rather than any
    fire-caused reclassification in the fire's own year.

    `bias_corrected` selects between the `_CBI_bc.tif` (bias-corrected) and
    `_CBI.tif` (raw) per-fire CBI tiles -- see mtbs_cbi_histogram.py's
    --bias-corrected/--no-bias-corrected flag.

    Returns None (rather than a zero array) when there's no source data at
    all, so callers can distinguish "not computed" from "computed and
    genuinely zero".
    """
    mtbs_path = mtbs_dir / f"{year}.tif"
    nlcd_path = nlcd_dir / f"Annual_NLCD_LndCov_{year - 1}_CU_C1V2.tif"
    fire_dir = cbi_dir / str(year)
    fire_paths = sorted(fire_dir.glob(f"*{_cbi_suffix(bias_corrected)}")) if fire_dir.is_dir() else []

    if not mtbs_path.is_file() or not nlcd_path.is_file() or not fire_paths:
        missing = mtbs_path if not mtbs_path.is_file() \
            else nlcd_path if not nlcd_path.is_file() else fire_dir
        print(f"  {year}: skipping, missing {missing}", flush=True)
        return None

    n_bins = len(bin_edges) - 1
    n_land = len(LAND_COVER_CATEGORIES)
    H = np.zeros((n_bins, len(MTBS_CLASSES), n_land))
    n_valid_total = 0
    t0 = time.perf_counter()

    with rasterio.open(mtbs_path) as mtbs_ds, rasterio.open(nlcd_path) as nlcd_ds:
        for fire_path in fire_paths:
            cbi = rioxarray.open_rasterio(fire_path, masked=True).squeeze("band", drop=True)
            mtbs_da = _read_aligned_window(mtbs_ds, cbi, nodata_value=0)
            nlcd_da = _read_aligned_window(nlcd_ds, cbi, nodata_value=NLCD_NODATA)
            mtbs_on_cbi = mtbs_da.rio.reproject_match(cbi, resampling=Resampling.nearest)
            nlcd_on_cbi = nlcd_da.rio.reproject_match(cbi, resampling=Resampling.nearest)

            cbi_vals = cbi.values.ravel()
            mtbs_vals = mtbs_on_cbi.values.ravel()
            nlcd_vals = nlcd_on_cbi.values.ravel()
            cbi.close()

            valid = np.isfinite(cbi_vals) & np.isfinite(mtbs_vals) & np.isfinite(nlcd_vals)
            if not valid.any():
                continue

            class_valid = np.rint(mtbs_vals[valid]).astype(np.int16)
            land_valid = _NLCD_LUT[np.rint(nlcd_vals[valid]).astype(np.int16)]
            keep = land_valid >= 0  # drop any code outside our category map (shouldn't occur)
            n_valid = int(keep.sum())
            if n_valid == 0:
                continue

            h, _ = np.histogramdd(
                (cbi_vals[valid][keep], class_valid[keep], land_valid[keep]),
                bins=[bin_edges, np.arange(0.5, len(MTBS_CLASSES) + 1.5), np.arange(-0.5, n_land + 0.5)],
            )
            H += h
            n_valid_total += n_valid

    print(f"  {year}: {len(fire_paths)} fires, {n_valid_total:,} valid pixels "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return H
