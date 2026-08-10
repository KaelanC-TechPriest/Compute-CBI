"""Compute per-year MTBS-vs-CBI joint histograms and write one CSV per year.

For every requested year with both an MTBS COG (`<mtbs-dir>/<year>.tif`) and a
bias-corrected CBI mosaic (`<cbi-dir>/<year>_bc.tif`), this warps MTBS onto the
CBI grid (nearest-neighbor, since MTBS classes are categorical -- see
mtbs_cbi_common.py for why a real CRS-aware warp is required) and writes the
joint histogram of (CBI value, MTBS class) to `<out-dir>/<year>.csv`.

Designed to run incrementally: each year only ever writes its own file, so
recomputing a year overwrites just that file and leaves every other year's
file untouched. A year with no source data is skipped and gets no file at
all (never a spurious all-zero file), so its absence in --out-dir means
"not computed", not "computed and zero".

Run (single year):
    uv run python scripts/mtbs_cbi_histogram.py --years 2020

Run (a range, still one file per year):
    uv run python scripts/mtbs_cbi_histogram.py --years 2018,2019,2020,2021

Run (everything available, still safe to re-run):
    uv run python scripts/mtbs_cbi_histogram.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mtbs_cbi_common import (
    DEFAULT_CBI_DIR,
    DEFAULT_MTBS_DIR,
    MTBS_CLASSES,
    discover_years,
    process_year,
)


def _write_year_csv(year: int, H: np.ndarray, bin_edges: np.ndarray, out_dir: Path) -> Path:
    final_path = out_dir / f"{year}.csv"
    tmp_path = out_dir / f".{year}.csv.tmp"
    with open(tmp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "cbi_bin_left", "cbi_bin_right", "mtbs_class", "count"])
        for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            for j, c in enumerate(MTBS_CLASSES):
                w.writerow([year, left, right, c, int(H[i, j])])
    os.replace(tmp_path, final_path)
    return final_path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute per-year MTBS-vs-CBI joint histograms as one CSV per year.")
    p.add_argument("--mtbs-dir", type=Path, default=DEFAULT_MTBS_DIR,
                   help="Directory of normalized MTBS COGs, <year>.tif.")
    p.add_argument("--cbi-dir", type=Path, default=DEFAULT_CBI_DIR,
                   help="Directory of CBI mosaics, <year>_bc.tif (bias-corrected only).")
    p.add_argument("--years", default=None,
                   help="Comma list of years to (re)compute (default: all years present in both dirs).")
    p.add_argument("--bins", type=int, default=60, help="Number of CBI histogram bins.")
    p.add_argument("--cbi-min", type=float, default=0.0, help="Lower edge of the CBI axis.")
    p.add_argument("--cbi-max", type=float, default=3.0, help="Upper edge of the CBI axis.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel year workers. Keep modest -- a full-aggregate run at "
                        "--workers 4 previously crashed with BrokenProcessPool, almost "
                        "certainly OOM: each worker can transiently hold 100GB+ for a "
                        "large-fire-year CBI mosaic. Raise only while watching `free -h`.")
    p.add_argument("--out-dir", type=Path, default=Path("out/histograms"),
                   help="Directory to write one <year>.csv per computed year into.")
    args = p.parse_args()

    years = (sorted(int(y) for y in args.years.split(","))
             if args.years else discover_years(args.mtbs_dir, args.cbi_dir))
    if not years:
        print("No years found with both an MTBS COG and a CBI _bc mosaic.", file=sys.stderr)
        return 1
    print(f"Processing {len(years)} year(s): {years}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = np.linspace(args.cbi_min, args.cbi_max, args.bins + 1)

    n_written = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_year, y, args.mtbs_dir, args.cbi_dir, bin_edges): y
                  for y in years}
        for fut in as_completed(futures):
            year = futures[fut]
            H = fut.result()
            if H is None:
                continue
            path = _write_year_csv(year, H, bin_edges, args.out_dir)
            print(f"  wrote {path}", flush=True)
            n_written += 1

    print(f"Done: {n_written}/{len(years)} year(s) written to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
