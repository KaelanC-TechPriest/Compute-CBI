"""Shared constants and per-year logic for the MTBS-vs-CBI comparison scripts.

See mtbs_cbi_histogram.py (compute) and mtbs_cbi_plot.py (aggregate + plot).
"""

from __future__ import annotations

import time
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


def discover_years(mtbs_dir: Path, cbi_dir: Path) -> list[int]:
    mtbs_years = {int(p.stem) for p in mtbs_dir.glob("*.tif") if p.stem.isdigit()}
    cbi_years = {int(p.name[:4]) for p in cbi_dir.glob("*_bc.tif") if p.name[:4].isdigit()}
    return sorted(mtbs_years & cbi_years)


def process_year(year: int, mtbs_dir: Path, cbi_dir: Path,
                 bin_edges: np.ndarray) -> np.ndarray | None:
    """Return a (n_bins, 6) joint histogram of (CBI value, MTBS class) for one year.

    Returns None (rather than a zero array) when the source files are missing,
    so callers can distinguish "not computed" from "computed and genuinely zero".
    """
    mtbs_path = mtbs_dir / f"{year}.tif"
    cbi_path = cbi_dir / f"{year}_bc.tif"

    if not mtbs_path.is_file() or not cbi_path.is_file():
        print(f"  {year}: skipping, missing {mtbs_path if not mtbs_path.is_file() else cbi_path}",
              flush=True)
        return None

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
    n_bins = len(bin_edges) - 1
    if n_valid == 0:
        print(f"  {year}: 0 valid pixels ({time.perf_counter() - t0:.0f}s)", flush=True)
        return np.zeros((n_bins, len(MTBS_CLASSES)))

    cbi_valid = cbi_vals[valid]
    class_valid = np.rint(mtbs_vals[valid]).astype(np.int16)
    del cbi_vals, mtbs_vals, valid

    H, _, _ = np.histogram2d(
        cbi_valid, class_valid,
        bins=[bin_edges, np.arange(0.5, len(MTBS_CLASSES) + 1.5)],
    )
    print(f"  {year}: {n_valid:,} valid pixels ({time.perf_counter() - t0:.0f}s)", flush=True)
    return H


def plot_pdf(bin_centers: np.ndarray, pdf: np.ndarray, out_path: Path) -> None:
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
