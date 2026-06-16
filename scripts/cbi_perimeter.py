"""Standalone one-shot CBI for a single MTBS fire perimeter.

Self-contained consolidation of the modular pipeline. Given one MTBS perimeter:

  1. ensure the RF model (load the repo cache, else train from the CBI CSV + cache it);
  2. ensure the static `def` raster (load the repo cache, else download the TerraClimate
     1981-2010 normal, sum to annual, cache it);
  3. fetch Landsat C2 L2 from the Planetary Computer for the fire bbox (ephemeral, no cache);
  4. composite -> predictors -> RF predict -> CBI/CBI_bc;
  5. write CBI/CBI_bc GeoTIFFs on the EPSG:5070 NLCD-snapped grid, clipped to the perimeter.

Caches (model, def) live at the wider-repo locations so they're shared with the modular code.
Faithful to Parks et al. (2019); see the repo modules this consolidates for provenance.

Run:
    uv run python scripts/cbi_oneshot.py --gpkg data/fire_perims/test.gpkg \
        --event-id NV4071111641720150629 --out-dir data/cbi/annie
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.warp import transform_bounds

from utils import (
    DEF_NC, DEF_TIF, DEFAULT_MAX_CLOUD, MODEL_PATH, TRAIN_CSV,
    STATE_WINDOWS, WILDFIRE_CODE,
    _scene_cache_stats,
    build_stack, composite, ensure_def, ensure_model,
    fetch_landsat, predict, to_5070_clip,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="One-shot CBI for one MTBS perimeter OR all fires in a given state."
    )
    p.add_argument("--gpkg", required=True)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--event-id", default=None,
                     help="MTBS Event_ID (default: first wildfire).")
    sel.add_argument("--index", type=int, default=None,
                     help="0-based row index in the gpkg layer.")
    p.add_argument("--layer", default=None,
                   help="Layer name in the gpkg (default: first layer).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--def", dest="def_tif", default=str(DEF_TIF))
    p.add_argument("--csv", default=str(TRAIN_CSV))
    p.add_argument("--state", default=None,
                   help="2-letter state abbreviation (e.g. MT, AK) to filter fires by state.")
    p.add_argument("--landsat-cache", default=None,
                   help="Directory for cached Landsat scenes (default: data/landsat_cache).")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete all cached Landsat scenes before running.")
    p.add_argument("--start-year", type=int, default=1986,
                   help="First fire ignition year to process (default: 1986).")
    p.add_argument("--end-year", type=int, default=2020,
                   help="Last fire ignition year to process (default: 2020).")

    args = p.parse_args()

    if args.state is not None:
        args.state = args.state.upper()
        if args.state not in STATE_WINDOWS:
            p.error(f"Unknown state '{args.state}'. Known states: {', '.join(sorted(STATE_WINDOWS))}")

    if args.start_year < 1986 or args.start_year > 2020:
        p.error(f"Start year out of bounds (must be in range 1986-2020)")

    if args.end_year < 1986 or args.end_year > 2020:
        p.error(f"End year out of bounds (must be in range 1986-2020)")

    start_time = time.perf_counter()

    landsat_cache = None
    if args.landsat_cache is not None:
        landsat_cache = Path(args.landsat_cache)
        if args.clear_cache and landsat_cache.exists():
            import shutil
            shutil.rmtree(landsat_cache)
            print(f"cache: cleared {landsat_cache}", flush=True)
        landsat_cache.mkdir(parents=True, exist_ok=True)

    # Always ensure model and def raster once (outside any loop)
    bundle = ensure_model(Path(args.model), Path(args.csv))
    def_path = ensure_def(Path(args.def_tif), DEF_NC)

    out_base = Path(args.out_dir)

    # Load the layer once
    gdf = gpd.read_file(args.gpkg, layer=args.layer).to_crs(5070)

    sub_parts = [args.state] if args.state else []
    out_base = out_base.joinpath(*sub_parts)
    out_base.mkdir(parents=True, exist_ok=True)

    gdf = gdf[gdf["Incid_Type"] == WILDFIRE_CODE]

    if args.state is not None:
        gdf = gdf[gdf["Event_ID"].str[:2].str.upper() == args.state]

    if args.event_id is not None:
        gdf = gdf[gdf["Event_ID"] == args.event_id]
        if gdf.empty:  # type: ignore[union-attr]
            raise SystemExit(f"event-id {args.event_id} not found in {args.gpkg}")

    elif args.index is not None:
        if not 0 <= args.index < len(gdf):
            raise SystemExit(f"index {args.index} out of range (0..{len(gdf) - 1})")
        gdf = gdf.iloc[args.index]  # type: ignore[union-attr]

    if gdf.empty:  # type: ignore[union-attr]
        print(f"No wildfires found for state {args.state}, id {args.event_id}, index {args.index}", flush=True)
        return 0

    state_str = f" in {args.state}" if args.state else ""
    print(f"Found {len(gdf)} wildfires{state_str}.", flush=True)

    for i, (_, row) in enumerate(gdf.iterrows(), start=1):  # type: ignore[union-attr]
        fire_id = "<unknown>"
        try:
            fire_id = row["Event_ID"]

            year = int(row["Ig_Date"].year)
            if year < args.start_year or year > args.end_year:
                print(f"[{i}/{len(gdf)}] Skipping {fire_id}: year {year} outside range "
                      f"{args.start_year}–{args.end_year}")
                continue

            state = str(fire_id)[:2].upper()

            if state not in STATE_WINDOWS:
                print(f"[{i}/{len(gdf)}] Skipping {fire_id}: no image-season window for state {state}")
                continue

            sd, ed = STATE_WINDOWS[state]
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds
            b = 1000.0
            bbox = transform_bounds(
                "EPSG:5070", "EPSG:4326",
                minx - b, miny - b, maxx + b, maxy + b
            )

            fire = {
                "fire_id": fire_id,
                "state": state,
                "year": year,
                "start_day": sd,
                "end_day": ed,
                "geometry": geom,
                "bbox": bbox,
            }

            print(f"[{i}/{len(gdf)}] Processing {fire_id} ...", flush=True)

            cube = fetch_landsat(fire["bbox"], fire["year"], fire["start_day"],
                                 fire["end_day"], args.max_cloud,
                                 landsat_cache=landsat_cache)
            comp = composite(cube, fire["year"])
            stack = build_stack(comp, def_path)
            ds = predict(stack, bundle)
            ds = to_5070_clip(ds, fire["geometry"])

            for name in ("CBI", "CBI_bc"):
                ds[name].rio.to_raster(
                    out_base / f"{fire['fire_id']}_{name}.tif",
                    tiled=True, compress="ZSTD", zstd_level=1
                )

            cbi = ds["CBI"].values
            print(f"  → Done: {ds.sizes['y']}x{ds.sizes['x']} grid, "
                  f"valid pixels = {int(np.isfinite(cbi).sum())}, "
                  f"grid={ds.sizes['y']}x{ds.sizes['x']} crs={ds.rio.crs} "
                  f"CBI med={np.nanmedian(cbi):.2f} max={np.nanmax(cbi):.2f}", flush=True)

            if landsat_cache is not None:
                print(f"  → Cache: {_scene_cache_stats['hits']} hits, "
                      f"{_scene_cache_stats['misses']} misses", flush=True)

            _scene_cache_stats["hits"] = 0
            _scene_cache_stats["misses"] = 0

        except Exception as e:
            print(f"  → Failed on {fire_id}: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            continue

    suffix_parts = [args.state] if args.state else []
    print(f"Processing{' for ' + ' '.join(suffix_parts) if suffix_parts else ''} completed.", flush=True)

    print(f"Finished in {(time.perf_counter() - start_time) / 60:.2f} minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())