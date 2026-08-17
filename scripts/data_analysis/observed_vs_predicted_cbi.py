"""Plot observed (field-measured) vs predicted (model) CBI, one point per field plot.

For each plot in data/cbi_field_plots/cbi_field_plots.csv (see
prepare_cbi_field_plots.py), finds the per-fire CBI raster tile for that
plot's fire year whose extent contains the plot's coordinates, and reads the
single predicted-CBI pixel at that exact point. X-axis = predicted CBI
(from the raster), Y-axis = observed CBI (the field-measured ground truth).
A 1:1 line marks perfect agreement; RMSE, mean bias, and Pearson r
quantify how far the model's predictions actually are from the field data.

Plots are matched to fires spatially (point-in-raster-bounds, using the
plot's own fire year to narrow the search to that year's directory) rather
than by fire name, since the field database's FireName strings don't
correspond to the Event_ID naming used by the per-fire CBI tiles.

Run:
    uv run python scripts/data_analysis/observed_vs_predicted_cbi.py

Run (against the raw, non-bias-corrected CBI predictions instead):
    uv run python scripts/data_analysis/observed_vs_predicted_cbi.py --no-bias-corrected
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

DEFAULT_FIELD_CSV = Path("data/cbi_field_plots/cbi_field_plots.csv")
DEFAULT_CBI_DIR = Path("/run/host/run/media/kaelan/CBI")


def _year_tiles(cbi_dir: Path, year: int, suffix: str) -> list[tuple[Path, rasterio.coords.BoundingBox]]:
    fire_dir = cbi_dir / str(year)
    if not fire_dir.is_dir():
        return []
    tiles = []
    for tif in sorted(fire_dir.glob(f"*{suffix}")):
        with rasterio.open(tif) as ds:
            tiles.append((tif, ds.bounds))
    return tiles


def _sample_point(path: Path, x: float, y: float) -> float | None:
    with rasterio.open(path) as ds:
        row, col = ds.index(x, y)
        if not (0 <= row < ds.height and 0 <= col < ds.width):
            return None
        val = ds.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
    return float(val) if np.isfinite(val) else None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot observed (field) vs predicted (model) CBI per plot, with fit stats.")
    p.add_argument("--field-csv", type=Path, default=DEFAULT_FIELD_CSV,
                   help="Cleaned field-plot CSV written by prepare_cbi_field_plots.py.")
    p.add_argument("--cbi-dir", type=Path, default=DEFAULT_CBI_DIR,
                   help="Directory of per-fire CBI tiles, <year>/<Event_ID>_CBI[_bc].tif.")
    p.add_argument("--bias-corrected", action=argparse.BooleanOptionalAction, default=True,
                   help="Sample the bias-corrected CBI tiles (_CBI_bc.tif, default) or the "
                        "raw tiles (_CBI.tif) via --no-bias-corrected.")
    p.add_argument("--out", type=Path, default=Path("out/observed_vs_predicted_cbi.png"),
                   help="Output PNG path.")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional path to dump the matched (observed, predicted) pairs as CSV.")
    args = p.parse_args()

    if not args.field_csv.is_file():
        print(f"{args.field_csv} not found -- run prepare_cbi_field_plots.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(args.field_csv)
    df = df.dropna(subset=["fire_year", "x", "y", "cbi_observed"])
    print(f"Loaded {len(df):,} field plots with a valid fire year/coordinates/observed CBI", flush=True)

    suffix = "_CBI_bc.tif" if args.bias_corrected else "_CBI.tif"
    tile_cache: dict[int, list[tuple[Path, rasterio.coords.BoundingBox]]] = {}

    observed, predicted = [], []
    n_no_year_dir, n_no_match, n_nodata = 0, 0, 0
    for row in df.itertuples():
        year = int(row.fire_year)
        if year not in tile_cache:
            tile_cache[year] = _year_tiles(args.cbi_dir, year, suffix)
        tiles = tile_cache[year]
        if not tiles:
            n_no_year_dir += 1
            continue

        match = None
        for tif, b in tiles:
            if b.left <= row.x <= b.right and b.bottom <= row.y <= b.top:
                val = _sample_point(tif, row.x, row.y)
                if val is not None:
                    match = val
                    break
        if match is None:
            if any(b.left <= row.x <= b.right and b.bottom <= row.y <= b.top for _, b in tiles):
                n_nodata += 1
            else:
                n_no_match += 1
            continue

        observed.append(row.cbi_observed)
        predicted.append(match)

    n_matched = len(observed)
    print(f"Matched {n_matched:,}/{len(df):,} plots to a predicted-CBI pixel "
          f"(no fires computed that year: {n_no_year_dir}, no covering tile: {n_no_match}, "
          f"covered but nodata: {n_nodata})", flush=True)
    if n_matched == 0:
        print("No plots matched -- nothing to plot.", file=sys.stderr)
        return 1

    observed = np.array(observed)
    predicted = np.array(predicted)

    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    r = float(np.corrcoef(observed, predicted)[0, 1])
    r_squared = r ** 2
    fit_slope, fit_intercept = np.polyfit(predicted, observed, 1)
    print(f"RMSE={rmse:.3f}  Pearson r={r:.3f}  "
          f"R²={r_squared:.3f}  best fit: observed = {fit_slope:.3f}*predicted + {fit_intercept:+.3f}",
          flush=True)

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"observed_cbi": observed, "predicted_cbi": predicted}).to_csv(
            args.out_csv, index=False)
        print(f"Wrote {args.out_csv}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    zero_obs = observed == 0
    ax.scatter(predicted[~zero_obs], observed[~zero_obs], s=56, color="#2a78d6",
              alpha=0.35, linewidths=0, label="Observed CBI > 0")
    ax.scatter(predicted[zero_obs], observed[zero_obs], s=56, color="#e34948",
              alpha=0.35, linewidths=0, label="Observed CBI = 0", zorder=3)

    lo, hi = 0.0, 3.0
    ax.plot([lo, hi], [fit_slope * lo + fit_intercept, fit_slope * hi + fit_intercept],
           color="black", linestyle="-", linewidth=1.5, label="Best fit")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted CBI", color="#0b0b0b")
    ax.set_ylabel("Observed CBI (field-measured)", color="#0b0b0b")
    variant = "bias-corrected" if args.bias_corrected else "raw"
    ax.set_title(f"Observed vs predicted CBI ({variant} model, n={n_matched:,})", color="#0b0b0b")
    ax.tick_params(colors="#52514e")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.legend(frameon=False, loc="upper left")
    for text in legend.get_texts():
        text.set_color("#0b0b0b")
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    ax.text(0.98, 0.03,
           f"RMSE = {rmse:.3f}\n\nPearson r = {r:.3f}\nR² = {r_squared:.3f}",
           transform=ax.transAxes, ha="right", va="bottom", color="#0b0b0b",
           fontsize=10, bbox=dict(facecolor="#fcfcfb", edgecolor="#c3c2b7", boxstyle="round,pad=0.4"))

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
