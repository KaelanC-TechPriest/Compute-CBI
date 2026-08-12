"""Compute per-year MTBS-vs-CBI-vs-NLCD joint histograms, one CSV per year.

For every requested year with an MTBS COG (`<mtbs-dir>/<year>.tif`), per-fire
CBI tiles (`<cbi-dir>/<year>/<Event_ID>_CBI[_bc].tif` -- bias-corrected by
default, or raw via --no-bias-corrected), and an NLCD annual land-cover layer
for the year *before* the fire
(`<nlcd-dir>/Annual_NLCD_LndCov_<year - 1>_CU_C1V2.tif`), this warps MTBS and
NLCD onto the CBI grid (nearest-neighbor, since both are categorical -- see
mtbs_cbi_common.py for why a real CRS-aware warp is required) and writes the
joint histogram of (CBI value, MTBS class, NLCD land-cover category) to
`<out-dir>/<year>.csv`. NLCD is read from year - 1, not the fire's own year,
so it reflects the vegetation that actually burned rather than any
fire-caused reclassification; the 16 raw NLCD classes are collapsed into 8
general categories (Water, Developed, Barren, Forest, Shrubland, Grassland,
Agriculture, Wetland).

Designed to run incrementally: each year only ever writes its own file, so
recomputing a year overwrites just that file and leaves every other year's
file untouched. A year with no source data is skipped and gets no file at
all (never a spurious all-zero file), so its absence in --out-dir means
"not computed", not "computed and zero".

Run (single year):
    uv run python scripts/mtbs_cbi_histogram.py --years 2020

Run (a range, still one file per year):
    uv run python scripts/mtbs_cbi_histogram.py --years 2018,2019,2020,2021

Run (everything available -- processes years one at a time, still safe to re-run):
    uv run python scripts/mtbs_cbi_histogram.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from mtbs_cbi_common import (
    DEFAULT_CBI_DIR,
    DEFAULT_MTBS_DIR,
    DEFAULT_NLCD_DIR,
    LAND_COVER_CATEGORIES,
    MTBS_CLASSES,
    discover_years,
    process_year,
)


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

    years = (sorted(int(y) for y in args.years.split(","))
             if args.years else discover_years(args.mtbs_dir, args.cbi_dir, args.bias_corrected))
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

        H = process_year(year, args.mtbs_dir, args.cbi_dir, args.nlcd_dir, bin_edges,
                         bias_corrected=args.bias_corrected)
        if H is None:
            continue

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
