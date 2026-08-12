"""Plot a CBI-vs-MTBS joint density heatmap with a Spearman correlation.

Loads every <hist-dir>/<year>.csv written by aggregate_pixel_categories.py,
sums raw pixel counts across the selected years, restricts to MTBS classes
1-4 (Unburned-to-Low through High -- the genuine ordinal severity scale;
5=Increased Greenness and 6=Non-Mapped/Mask are excluded since they aren't
part of a severity progression and would make the correlation misleading),
then renders a log-scaled heatmap of the joint (CBI value, MTBS class)
distribution plus a Spearman rank correlation between the two.

No new raster computation happens here -- the per-year CSVs are already a
full joint histogram; this just aggregates and plots them differently than
mtbs_cbi_pdf_plot.py does.

Run (everything computed so far):
    uv run python scripts/data_analysis/mtbs_cbi_heatmap.py

Run (forest fires only, a few recent years):
    uv run python scripts/data_analysis/mtbs_cbi_heatmap.py --years 2018,2019,2020,2021 --land-cover Forest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aggregate_pixel_categories import LAND_COVER_CATEGORIES, MTBS_LABELS

HEATMAP_CLASSES = (1, 2, 3, 4)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot a CBI-vs-MTBS joint density heatmap with a Spearman correlation.")
    p.add_argument("--hist-dir", type=Path, default=Path("out/histograms_nlcd"),
                   help="Directory of <year>.csv files written by aggregate_pixel_categories.py.")
    p.add_argument("--years", default=None,
                   help="Comma list of years to include (default: every file present).")
    p.add_argument("--land-cover", choices=LAND_COVER_CATEGORIES, default=None,
                   help="Optional: restrict to one NLCD land-cover category (e.g. Forest). "
                        "Requires --hist-dir files with a land_cover column. Default: no "
                        "filter, pixels are summed across all land-cover categories.")
    p.add_argument("--out", type=Path, default=Path("out/mtbs_cbi_heatmap.png"), help="Output PNG path.")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional path to dump the aggregated (bin, class) count matrix as CSV.")
    args = p.parse_args()

    years = [int(y) for y in args.years.split(",")] if args.years else None
    paths = sorted(args.hist_dir.glob("*.csv"))
    if years is not None:
        wanted = set(years)
        paths = [p for p in paths if p.stem.isdigit() and int(p.stem) in wanted]
    if not paths:
        raise SystemExit(f"No histogram files found in {args.hist_dir} "
                         f"(years filter: {years or 'all'}). Run aggregate_pixel_categories.py first.")
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    loaded_years = sorted(df["year"].unique().tolist())
    print(f"Loaded {len(loaded_years)} year(s): {loaded_years}", flush=True)

    if args.land_cover is not None:
        if "land_cover" not in df.columns:
            raise SystemExit(
                f"--land-cover was given but {args.hist_dir} has no land_cover column -- "
                "recompute with aggregate_pixel_categories.py's --nlcd-dir (e.g. out/histograms_nlcd/) "
                "and point --hist-dir there.")
        df = df[df["land_cover"] == args.land_cover]
        if df.empty:
            raise SystemExit(f"No rows left after filtering to land_cover={args.land_cover!r} "
                             f"in the selected years.")
        print(f"Filtered to land_cover={args.land_cover!r}", flush=True)

    per_year_edges = df.groupby("year").apply(
        lambda g: tuple(sorted(map(tuple, g[["cbi_bin_left", "cbi_bin_right"]].values.tolist()))),
        include_groups=False,
    )
    edge_sets = per_year_edges.unique()
    if len(edge_sets) > 1:
        reference = edge_sets[0]
        mismatched = per_year_edges[per_year_edges != reference].index.tolist()
        raise SystemExit(
            f"Bin edges differ across years -- likely recomputed with different "
            f"--bins/--cbi-min/--cbi-max. Mismatched year(s): {mismatched}. "
            f"Recompute all years with the same bin settings before plotting.")

    df = df[df["mtbs_class"].isin(HEATMAP_CLASSES)]

    agg = (df.groupby(["cbi_bin_left", "cbi_bin_right", "mtbs_class"])["count"]
            .sum().reset_index().sort_values(["cbi_bin_left", "mtbs_class"]))
    bin_lefts = np.sort(agg["cbi_bin_left"].unique())
    bin_width = agg["cbi_bin_right"].iloc[0] - agg["cbi_bin_left"].iloc[0]
    bin_edges = np.append(bin_lefts, bin_lefts[-1] + bin_width)
    bin_centers = bin_lefts + bin_width / 2

    wide = agg.pivot(index="cbi_bin_left", columns="mtbs_class", values="count").reindex(
        columns=HEATMAP_CLASSES, fill_value=0).reindex(index=bin_lefts, fill_value=0)
    matrix = wide.values  # shape (n_bins, 4): rows=cbi bins, cols=classes 1-4

    total_n = int(matrix.sum())
    if total_n == 0:
        raise SystemExit("No pixels in classes 1-4 for the selected years/land-cover filter.")

    # Weighted Spearman via repeating the small (n_bins x 4) count table --
    # simpler and less error-prone than a hand-derived weighted-rank formula.
    # The repeated arrays top out in the hundreds of millions of elements even
    # across the full 39-year dataset, well within this machine's RAM.
    cbi_grid, class_grid = np.meshgrid(bin_centers, HEATMAP_CLASSES, indexing="ij")
    flat_counts = matrix.ravel()
    cbi_repeated = np.repeat(cbi_grid.ravel(), flat_counts)
    class_repeated = np.repeat(class_grid.ravel(), flat_counts)
    rho, pval = spearmanr(cbi_repeated, class_repeated)
    print(f"Spearman rho = {rho:.4f} (p={pval:.2e}, n={total_n:,})", flush=True)

    if args.out_csv is not None:
        out_df = wide.copy()
        out_df.columns = [f"count_class_{c}" for c in out_df.columns]
        out_df.insert(0, "cbi_bin_center", bin_centers)
        out_df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    # dataviz skill sequential ramp (blue, light -> dark; reference-palette steps 100-700)
    cmap = LinearSegmentedColormap.from_list(
        "blue_sequential",
        ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    )
    cmap.set_bad(color="#fcfcfb")  # zero-count cells render as the chart surface, not log(0)

    plot_matrix = matrix.astype(float)
    plot_matrix[plot_matrix == 0] = np.nan

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    mesh = ax.pcolormesh(bin_edges, np.arange(0.5, len(HEATMAP_CLASSES) + 1.5),
                         plot_matrix.T, cmap=cmap, norm=LogNorm())

    ax.set_yticks(HEATMAP_CLASSES)
    ax.set_yticklabels([f"{c} – {MTBS_LABELS[c]}" for c in HEATMAP_CLASSES], color="#0b0b0b")
    ax.set_xlabel("CBI value", color="#0b0b0b")
    ax.set_ylabel("MTBS class", color="#0b0b0b")
    title = "CBI vs MTBS burn-severity class (joint pixel density)"
    if args.land_cover is not None:
        title += f" ({args.land_cover} only)"
    ax.set_title(title, color="#0b0b0b")
    ax.tick_params(colors="#52514e")
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Pixel count (log scale)", color="#0b0b0b")
    cbar.ax.tick_params(colors="#52514e")

    ax.text(0.98, 0.03, f"Spearman ρ = {rho:.3f}  (n = {total_n:,})",
           transform=ax.transAxes, ha="right", va="bottom", color="#0b0b0b",
           fontsize=10, bbox=dict(facecolor="#fcfcfb", edgecolor="#c3c2b7", boxstyle="round,pad=0.4"))

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
