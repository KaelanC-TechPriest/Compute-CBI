"""Compute per-year MTBS-vs-CBI-vs-NLCD joint histograms, one CSV per year.

For every requested year with an MTBS COG (`<mtbs-dir>/<year>.tif`), per-fire
CBI tiles (`<cbi-dir>/<year>/<Event_ID>_CBI[_bc].tif` -- bias-corrected by
default, or raw via --no-bias-corrected), and an NLCD annual land-cover layer
for the year *before* the fire
(`<nlcd-dir>/Annual_NLCD_LndCov_<year - 1>_CU_C1V2.tif`), this warps MTBS and
NLCD onto the CBI grid (nearest-neighbor, since both are categorical -- see
why a real CRS-aware warp is required) and writes the joint histogram of
(CBI value, MTBS class, NLCD land-cover category) to `<out-dir>/<year>.csv`.
NLCD is read from year - 1, not the fire's own year, so it reflects the
vegetation that actually burned rather than any fire-caused reclassification;
the 16 raw NLCD classes are collapsed into 8 general categories (Water,
Developed, Barren, Forest, Shrubland, Grassland, Agriculture, Wetland).

Designed to run incrementally: each year only ever writes its own file, so
recomputing a year overwrites just that file and leaves every other year's
file untouched. A year with no source data is skipped and gets no file at
all (never a spurious all-zero file), so its absence in --out-dir means
"not computed", not "computed and zero".

Run (single year):
    uv run python scripts/joint_histogram.py --years 2020

Run (a range, still one file per year):
    uv run python scripts/joint_histogram.py --years 2018,2019,2020,2021

Run (everything available -- processes years one at a time, still safe to re-run):
    uv run python scripts/joint_histogram.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
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

LAND_COVER_CATEGORIES = (
    "Water", "Developed", "Barren", "Forest",
    "Shrubland", "Grassland", "Agriculture", "Wetland",
)

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
NLCD_NODATA = 250

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


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute per-year MTBS-vs-CBI-vs-NLCD joint histograms as one CSV per year.")
    p.add_argument("--mtbs-dir", type=Path, default=DEFAULT_MTBS_DIR,
                   help="Directory of normalized MTBS COGs, <year>.tif.")
    p.add_argument("--cbi-dir", type=Path, default=DEFAULT_CBI_DIR,
                   help="Directory of per-fire CBI tiles, <year>/<Event_ID>_CBI[_bc].tif.")
    p.add_argument("--bias-corrected", action=argparse.BooleanOptionalAction, default=True,
                   help="Use the bias-corrected CBI tiles (<Event_ID>_CBI_bc.tif). Pass "
                        "--no-bias-corrected to use the raw tiles (<Event_ID>_CBI.tif) instead. "
                        "Default: bias-corrected. Note the 'skip if already exists' behavior "
                        "below is per-year-file, not per-variant -- switching this flag for a "
                        "year already computed under the other variant will silently keep the "
                        "old results unless you also change --out-dir or delete that year's file.")
    p.add_argument("--nlcd-dir", type=Path, default=DEFAULT_NLCD_DIR,
                   help="Directory of NLCD annual land-cover COGs, "
                        "Annual_NLCD_LndCov_<year>_CU_C1V2.tif.")
    p.add_argument("--years", default=None,
                   help="Comma list of years to (re)compute (default: all years present in both dirs).")
    p.add_argument("--bins", type=int, default=60, help="Number of CBI histogram bins.")
    p.add_argument("--cbi-min", type=float, default=0.0, help="Lower edge of the CBI axis.")
    p.add_argument("--cbi-max", type=float, default=3.0, help="Upper edge of the CBI axis.")
    p.add_argument("--out-dir", type=Path, default=Path("out/histograms_nlcd"),
                   help="Directory to write one <year>.csv per computed year into. "
                        "Defaults to a separate tree from out/histograms/ since this "
                        "adds a new required column (land_cover) -- mixing old and new "
                        "schema files in one directory would break anything that "
                        "assumes consistent columns across files.")
    args = p.parse_args()

    if args.years:
        years = sorted(int(y) for y in args.years.split(","))
    else:
        suffix = "_CBI_bc.tif" if args.bias_corrected else "_CBI.tif"
        mtbs_years = {int(p.stem) for p in args.mtbs_dir.glob("*.tif") if p.stem.isdigit()}
        cbi_years = {
            int(p.name) for p in args.cbi_dir.iterdir()
            if p.is_dir() and p.name.isdigit() and any(p.glob(f"*{suffix}"))
        }
        years = sorted(mtbs_years & cbi_years)

    if not years:
        variant = "bias-corrected" if args.bias_corrected else "raw"
        print(f"No years found with both an MTBS COG and {variant} CBI tiles.", file=sys.stderr)
        return 1
    print(f"Processing {len(years)} year(s): {years}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = np.linspace(args.cbi_min, args.cbi_max, args.bins + 1)

    n_written = 0
    for year in years:
        final_path = args.out_dir / f"{year}.csv"
        tmp_path = args.out_dir / f".{year}.csv.tmp"

        if final_path.exists():
            print(f"{year}: skipping, already exists")
            continue

        mtbs_path = args.mtbs_dir / f"{year}.tif"
        nlcd_path = args.nlcd_dir / f"Annual_NLCD_LndCov_{year - 1}_CU_C1V2.tif"
        fire_dir = args.cbi_dir / str(year)
        cbi_suffix = "_CBI_bc.tif" if args.bias_corrected else "_CBI.tif"
        fire_paths = sorted(fire_dir.glob(f"*{cbi_suffix}")) if fire_dir.is_dir() else []

        if not mtbs_path.is_file() or not nlcd_path.is_file() or not fire_paths:
            missing = mtbs_path if not mtbs_path.is_file() \
                else nlcd_path if not nlcd_path.is_file() else fire_dir
            print(f"  {year}: skipping, missing {missing}", flush=True)
            continue

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

        with open(tmp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "cbi_bin_left", "cbi_bin_right", "mtbs_class", "land_cover", "count"])
            for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                for j, c in enumerate(MTBS_CLASSES):
                    for k, lc in enumerate(LAND_COVER_CATEGORIES):
                        w.writerow([year, left, right, c, lc, int(H[i, j, k])])
        os.replace(tmp_path, final_path)

        print(f"  wrote {final_path}", flush=True)
        n_written += 1

    print(f"Done: {n_written}/{len(years)} year(s) written to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
