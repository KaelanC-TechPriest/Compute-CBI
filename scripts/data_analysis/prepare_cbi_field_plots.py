"""Reformat the raw conus_cbi_v4 field-CBI shapefile into a clean, typed CSV.

The source is a ground-truth field-measured Composite Burn Index plot
database (point locations where field crews scored real burn severity
post-fire), downloaded as an ESRI Shapefile bundle (.shp/.shx/.dbf/.prj) and
copied as-is into data/cbi_field_plots/raw/. That format is awkward to work
with directly (10-char-truncated field names, numeric values stored as
shapefile String fields, no data dictionary) -- this script reads it once
with geopandas and writes a small, typed, human-readable CSV with renamed
columns, coordinates in EPSG:5070 (same CRS as the per-fire CBI rasters,
confirmed via `gdf.crs`), and rows with an unparseable observed CBI value
dropped (reported, not silently ignored).

Run:
    uv run python scripts/data_analysis/prepare_cbi_field_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

DEFAULT_SHP = Path("data/cbi_field_plots/raw/conus_cbi_v4.shp")
DEFAULT_OUT = Path("data/cbi_field_plots/cbi_field_plots.csv")

_COLUMN_RENAME = {
    "Id": "plot_id",
    "X": "x",
    "Y": "y",
    "FireName": "fire_name",
    "FireDate": "fire_date",
    "FieldDate": "field_date",
    "Examiners": "examiners",
    "Frg": "fire_regime_group",
    "Om1": "om1",
    "Om2": "om2",
    "Om3": "om3",
    "Nvc": "nvc",
    "Bps": "bps",
    "PreImage": "pre_image",
    "PostImage": "post_image",
    "PreNBR_val": "pre_nbr",
    "PostNBR_va": "post_nbr",
    "dNBR_off": "dnbr_offset",
    "dNBR_sd": "dnbr_sd",
    "dNBR_val": "dnbr",
    "Cbi": "cbi_observed",
}


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shp", type=Path, default=DEFAULT_SHP,
                   help="Path to the raw conus_cbi_v4.shp shapefile.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Path to write the cleaned CSV to.")
    args = p.parse_args()

    if not args.shp.is_file():
        print(f"Shapefile not found: {args.shp}", file=sys.stderr)
        return 1

    gdf = gpd.read_file(args.shp)
    print(f"Read {len(gdf):,} plots, CRS={gdf.crs}", flush=True)

    df = gdf.rename(columns=_COLUMN_RENAME).drop(columns=["geometry"])

    df["cbi_observed"] = pd.to_numeric(df["cbi_observed"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["cbi_observed"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped} plot(s) with an unparseable/missing observed CBI value "
              f"({n_dropped}/{n_before})", flush=True)

    df["fire_date"] = pd.to_datetime(df["fire_date"], errors="coerce")
    df["field_date"] = pd.to_datetime(df["field_date"], errors="coerce")
    df["fire_year"] = df["fire_date"].dt.year

    for col in ("pre_nbr", "post_nbr", "dnbr_offset", "dnbr_sd", "dnbr"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} plots to {args.out}", flush=True)
    print(f"  cbi_observed range: {df['cbi_observed'].min():.2f} - {df['cbi_observed'].max():.2f}, "
          f"mean {df['cbi_observed'].mean():.2f}", flush=True)
    print(f"  fire_year range: {int(df['fire_year'].min())} - {int(df['fire_year'].max())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
