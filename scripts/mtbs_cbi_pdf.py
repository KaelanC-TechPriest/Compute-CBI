"""Compare MTBS burn-severity classes against CBI values, pixel by pixel.

For every year with both an MTBS COG (`<mtbs-dir>/<year>.tif`) and a
bias-corrected CBI mosaic (`<cbi-dir>/<year>_bc.tif`), this warps the MTBS
raster onto the CBI raster's own grid (nearest-neighbor, since MTBS classes
are categorical) and builds a joint histogram of (CBI value, MTBS class) over
every pixel where both are valid. MTBS class 0 doubles as "Background" and
the file's nodata sentinel, so it is indistinguishable from no-data and is
dropped -- only classes 1-6 are plotted.

The MTBS grid (ESRI:102039, USGS spherical Albers) and the CBI grid
(EPSG:5070, NAD83 ellipsoidal Albers) share projection parameters but not a
datum, so alignment requires a real CRS-aware warp, not just a shared-origin
crop -- `rio.reproject_match` handles that.

Run (smoke test):
    uv run python scripts/mtbs_cbi_pdf.py --years 2020 --out /tmp/test.png --workers 1

Run (full aggregate, 1986-2024):
    uv run python scripts/mtbs_cbi_pdf.py --out mtbs_cbi_pdf.png --out-csv mtbs_cbi_pdf.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  -- registers the `.rio` accessor
from rasterio.enums import Resampling

DEFAULT_MTBS_DIR = Path("/run/data_raid5/shared_data/fire_analysis/data/mtbs/cog")
DEFAULT_CBI_DIR = Path("/run/host/run/media/kaelan/CBI/mosaics")

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


def _discover_years(mtbs_dir: Path, cbi_dir: Path) -> list[int]:
    mtbs_years = {int(p.stem) for p in mtbs_dir.glob("*.tif") if p.stem.isdigit()}
    cbi_years = {int(p.name[:4]) for p in cbi_dir.glob("*_bc.tif") if p.name[:4].isdigit()}
    return sorted(mtbs_years & cbi_years)


def _process_year(year: int, mtbs_dir: Path, cbi_dir: Path,
                  bin_edges: np.ndarray) -> np.ndarray:
    """Return a (n_bins, 6) joint histogram of (CBI value, MTBS class) for one year."""
    mtbs_path = mtbs_dir / f"{year}.tif"
    cbi_path = cbi_dir / f"{year}_bc.tif"
    n_bins = len(bin_edges) - 1
    empty = np.zeros((n_bins, len(MTBS_CLASSES)))

    if not mtbs_path.is_file() or not cbi_path.is_file():
        print(f"  {year}: skipping, missing {mtbs_path if not mtbs_path.is_file() else cbi_path}",
              flush=True)
        return empty

    t0 = time.perf_counter()
    cbi = rioxarray.open_rasterio(cbi_path, masked=True).squeeze("band", drop=True)
    mtbs = rioxarray.open_rasterio(mtbs_path, masked=True).squeeze("band", drop=True)
    mtbs_on_cbi = mtbs.rio.reproject_match(cbi, resampling=Resampling.nearest)

    cbi_vals = cbi.values.ravel()
    mtbs_vals = mtbs_on_cbi.values.ravel()
    cbi.close()
    mtbs.close()
    del cbi, mtbs, mtbs_on_cbi

    valid = np.isfinite(cbi_vals) & np.isfinite(mtbs_vals)
    n_valid = int(valid.sum())
    if n_valid == 0:
        print(f"  {year}: 0 valid pixels ({time.perf_counter() - t0:.0f}s)", flush=True)
        return empty

    cbi_valid = cbi_vals[valid]
    class_valid = np.rint(mtbs_vals[valid]).astype(np.int16)
    del cbi_vals, mtbs_vals, valid

    H, _, _ = np.histogram2d(
        cbi_valid, class_valid,
        bins=[bin_edges, np.arange(0.5, len(MTBS_CLASSES) + 1.5)],
    )
    print(f"  {year}: {n_valid:,} valid pixels ({time.perf_counter() - t0:.0f}s)", flush=True)
    return H


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot P(CBI value | MTBS class) for classes 1-6, aggregated across years.")
    p.add_argument("--mtbs-dir", type=Path, default=DEFAULT_MTBS_DIR,
                   help="Directory of normalized MTBS COGs, <year>.tif.")
    p.add_argument("--cbi-dir", type=Path, default=DEFAULT_CBI_DIR,
                   help="Directory of CBI mosaics, <year>_bc.tif (bias-corrected only).")
    p.add_argument("--years", default=None,
                   help="Comma list of years to include (default: all years present in both dirs).")
    p.add_argument("--bins", type=int, default=60, help="Number of CBI histogram bins.")
    p.add_argument("--cbi-min", type=float, default=0.0, help="Lower edge of the CBI axis.")
    p.add_argument("--cbi-max", type=float, default=3.0, help="Upper edge of the CBI axis.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel year workers. Keep modest -- each worker can transiently "
                        "hold tens of GB for a large-fire-year CBI mosaic; workers * peak "
                        "year memory must stay under available RAM.")
    p.add_argument("--out", type=Path, default=Path("mtbs_cbi_pdf.png"), help="Output PNG path.")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional path to dump per-bin counts and densities as CSV.")
    args = p.parse_args()

    years = (sorted(int(y) for y in args.years.split(","))
             if args.years else _discover_years(args.mtbs_dir, args.cbi_dir))
    if not years:
        print("No years found with both an MTBS COG and a CBI _bc mosaic.", file=sys.stderr)
        return 1
    print(f"Processing {len(years)} year(s): {years}", flush=True)

    bin_edges = np.linspace(args.cbi_min, args.cbi_max, args.bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    H_total = np.zeros((args.bins, len(MTBS_CLASSES)))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_year, y, args.mtbs_dir, args.cbi_dir, bin_edges): y
                  for y in years}
        for fut in as_completed(futures):
            H_total += fut.result()
    print(f"All years done in {time.perf_counter() - t0:.0f}s.", flush=True)

    col_sums = H_total.sum(axis=0)
    pdf = np.full_like(H_total, np.nan)
    for j, c in enumerate(MTBS_CLASSES):
        if col_sums[j] > 0:
            pdf[:, j] = H_total[:, j] / (col_sums[j] * bin_width)
        else:
            print(f"  warning: MTBS class {c} ({MTBS_LABELS[c]}) has zero valid pixels "
                  "in the selected years -- omitted from the plot.", flush=True)

    if args.out_csv is not None:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cbi_bin_center", *[f"count_class_{c}" for c in MTBS_CLASSES],
                       *[f"density_class_{c}" for c in MTBS_CLASSES]])
            for i, center in enumerate(bin_centers):
                w.writerow([center, *H_total[i].tolist(), *pdf[i].tolist()])
        print(f"Wrote {args.out_csv}", flush=True)

    _plot(bin_centers, pdf, args.out)
    print(f"Wrote {args.out}", flush=True)
    return 0


def _plot(bin_centers: np.ndarray, pdf: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for j, c in enumerate(MTBS_CLASSES):
        if np.isnan(pdf[:, j]).all():
            continue
        ax.plot(bin_centers, pdf[:, j], color=MTBS_COLORS[c], linewidth=2,
               label=f"{c} – {MTBS_LABELS[c]}")

    ax.set_xlabel("CBI value", color="#0b0b0b")
    ax.set_ylabel("Density", color="#0b0b0b")
    ax.set_title("CBI distribution by MTBS burn-severity class", color="#0b0b0b")
    ax.tick_params(colors="#52514e")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.legend(title="MTBS class", frameon=False)
    legend.get_title().set_color("#0b0b0b")
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
