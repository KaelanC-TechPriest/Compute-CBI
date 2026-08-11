"""Aggregate per-year MTBS-vs-CBI histograms (from mtbs_cbi_histogram.py) and plot.

Loads every <hist-dir>/<year>.csv written by mtbs_cbi_histogram.py, sums raw
pixel counts across the selected years (summing counts -- not densities -- is
what makes cross-year aggregation correct), then plots P(CBI value | MTBS
class) for classes 1-6.

Run (everything computed so far):
    uv run python scripts/mtbs_cbi_plot.py --out out/mtbs_cbi_pdf.png

Run (a subset of already-computed years):
    uv run python scripts/mtbs_cbi_plot.py --years 2018,2019,2020,2021 --out out/recent.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mtbs_cbi_common import MTBS_CLASSES, MTBS_COLORS, MTBS_LABELS


def main() -> int:
    p = argparse.ArgumentParser(
        description="Aggregate per-year MTBS-vs-CBI histograms and plot P(CBI | MTBS class).")
    p.add_argument("--hist-dir", type=Path, default=Path("out/histograms"),
                   help="Directory of <year>.csv files written by mtbs_cbi_histogram.py.")
    p.add_argument("--years", default=None,
                   help="Comma list of years to include (default: every file present).")
    p.add_argument("--out", type=Path, default=Path("out/mtbs_cbi_pdf.png"), help="Output PNG path.")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional path to dump the aggregated (bin, class) counts/densities as CSV.")
    args = p.parse_args()

    years = [int(y) for y in args.years.split(",")] if args.years else None
    paths = sorted(args.hist_dir.glob("*.csv"))
    if years is not None:
        wanted = set(years)
        paths = [p for p in paths if p.stem.isdigit() and int(p.stem) in wanted]
    if not paths:
        raise SystemExit(f"No histogram files found in {args.hist_dir} "
                         f"(years filter: {years or 'all'}). Run mtbs_cbi_histogram.py first.")
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    loaded_years = sorted(df["year"].unique().tolist())
    print(f"Loaded {len(loaded_years)} year(s): {loaded_years}", flush=True)

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

    agg = (df.groupby(["cbi_bin_left", "cbi_bin_right", "mtbs_class"])["count"]
            .sum().reset_index().sort_values(["cbi_bin_left", "mtbs_class"]))
    bin_centers = np.sort(agg["cbi_bin_left"].unique())
    bin_width = agg["cbi_bin_right"].iloc[0] - agg["cbi_bin_left"].iloc[0]
    bin_centers = (bin_centers + bin_width / 2)

    wide_counts = agg.pivot(index="cbi_bin_left", columns="mtbs_class", values="count").reindex(
        columns=MTBS_CLASSES, fill_value=0)
    col_sums = wide_counts.sum(axis=0)
    pdf = np.full(wide_counts.shape, np.nan)
    for j, c in enumerate(MTBS_CLASSES):
        if col_sums[c] > 0:
            pdf[:, j] = wide_counts[c].values / (col_sums[c] * bin_width)
        else:
            print(f"  warning: MTBS class {c} ({MTBS_LABELS[c]}) has zero valid pixels "
                  "in the selected years -- omitted from the plot.", flush=True)

    if args.out_csv is not None:
        out_df = wide_counts.copy()
        out_df.columns = [f"count_class_{c}" for c in out_df.columns]
        for j, c in enumerate(MTBS_CLASSES):
            out_df[f"density_class_{c}"] = pdf[:, j]
        out_df.insert(0, "cbi_bin_center", bin_centers)
        out_df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for j, c in enumerate((1,2,3,4)):
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
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
