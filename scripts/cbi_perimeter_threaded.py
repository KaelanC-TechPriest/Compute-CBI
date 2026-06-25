"""Multithreaded CBI for one or more MTBS fire perimeters.

Single-queue design: each worker thread handles one fire end-to-end
(STAC search, cube build, full CBI pipeline). Workers drain fire_q
and terminate when it is empty.

Run:
    uv run python scripts/cbi_perimeter_threaded.py \
        --gpkg data/fire_perims/test.gpkg \
        --event-id NV4071111641720150629 \
        --out-dir data/cbi/test_threaded \
        --workers 4
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.warp import transform_bounds

from utils import (
    DEF_NC, DEF_TIF, DEFAULT_MAX_CLOUD, MODEL_PATH, TRAIN_CSV,
    STATE_WINDOWS, WILDFIRE_CODE,
    build_stack, composite, ensure_def, ensure_model,
    fetch_landsat, predict, to_5070_clip,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Multithreaded CBI for one or more MTBS perimeters."
    )
    p.add_argument("--gpkg", required=True)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--event-id", default=None,
                     help="MTBS Event_ID (default: all wildfires).")
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
                   help="2-letter state abbreviation to filter fires.")
    p.add_argument("--workers", type=int, default=4,
                   help="Worker threads (search + download + process per fire, default: 4).")
    p.add_argument("--start-year", type=int, default=1986,
                   help="First fire ignition year to process (default: 1986).")
    p.add_argument("--end-year", type=int, default=2020,
                   help="Last fire ignition year to process (default: 2020).")

    args = p.parse_args()

    if args.state is not None:
        args.state = args.state.upper()
        if args.state not in STATE_WINDOWS:
            p.error(f"Unknown state '{args.state}'. Known: {', '.join(sorted(STATE_WINDOWS))}")

    if args.start_year < 1986 or args.start_year > 2020:
        p.error(f"Start year out of bounds (must be in range 1986-2020)")

    if args.end_year < 1986 or args.end_year > 2020:
        p.error(f"End year out of bounds (must be in range 1986-2020)")

    start_time = time.perf_counter()

    bundle = ensure_model(Path(args.model), Path(args.csv))
    bundle["model"].n_jobs = 1  # no joblib sub-threads inside worker threads
    def_path = ensure_def(Path(args.def_tif), DEF_NC)

    out_base = Path(args.out_dir)
    if args.state:
        out_base = out_base / args.state
    out_base.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(args.gpkg, layer=args.layer).to_crs(5070)
    gdf = gdf[gdf["Incid_Type"] == WILDFIRE_CODE]

    if args.state is not None:
        gdf = gdf[gdf["Event_ID"].str[:2].str.upper() == args.state]

    if args.event_id is not None:
        gdf = gdf[gdf["Event_ID"] == args.event_id]
        if gdf.empty:
            raise SystemExit(f"event-id {args.event_id} not found in {args.gpkg}")
    elif args.index is not None:
        if not 0 <= args.index < len(gdf):
            raise SystemExit(f"index {args.index} out of range (0..{len(gdf) - 1})")
        gdf = gdf.iloc[[args.index]]

    if gdf.empty:
        print("No wildfires found for the given filters.", flush=True)
        return 0

    state_str = f" in {args.state}" if args.state else ""
    print(f"Found {len(gdf)} wildfires{state_str}.", flush=True)

    # -------------------------------------------------------------------------
    # Build fire list (no STAC search yet — workers do that per fire)
    # -------------------------------------------------------------------------
    fires: list[dict] = []

    for _, row in gdf.iterrows():
        fire_id = row["Event_ID"]

        year = int(row["Ig_Date"].year)
        if year < args.start_year or year > args.end_year:
            print(f"  Skipping {fire_id}: year {year} outside range "
                  f"{args.start_year}–{args.end_year}", flush=True)
            continue

        state = str(fire_id)[:2].upper()
        if state not in STATE_WINDOWS:
            print(f"  Skipping {fire_id}: no image-season window for state {state}", flush=True)
            continue

        sd, ed = STATE_WINDOWS[state]
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        b = 1000.0
        bbox = transform_bounds("EPSG:5070", "EPSG:4326",
                                minx - b, miny - b, maxx + b, maxy + b)

        fires.append({
            "fire_id": fire_id, "state": state, "year": year,
            "start_day": sd, "end_day": ed,
            "geometry": geom, "bbox": bbox,
        })

    if not fires:
        print("No eligible fires after filtering.", flush=True)
        return 0

    total = len(fires)

    # -------------------------------------------------------------------------
    # Unified worker pool: each thread does search + download + process per fire
    # -------------------------------------------------------------------------
    fire_q: queue.Queue = queue.Queue()
    for i, fire in enumerate(fires, start=1):
        fire_q.put((i, fire))

    print(f"Processing {total} fires with {args.workers} workers ...", flush=True)

    counts = {"ok": 0, "err": 0}
    counts_lock = threading.Lock()

    def _worker():
        while True:
            try:
                i, fire = fire_q.get_nowait()
            except queue.Empty:
                return
            fire_id = fire["fire_id"]
            try:
                cbi_path = out_base / f"{fire_id}_CBI.tif"
                if cbi_path.is_file():
                    print(f"[{i}/{total}] Skipping {fire_id} (already exists)", flush=True)
                    with counts_lock:
                        counts["ok"] += 1
                    continue

                print(f"[{i}/{total}] Fetching {fire_id} ...", flush=True)
                cube = fetch_landsat(
                    fire["bbox"], fire["year"],
                    fire["start_day"], fire["end_day"],
                    args.max_cloud,
                )
                print(f"[{i}/{total}] Processing {fire_id} ...", flush=True)
                comp = composite(cube, fire["year"])
                stack = build_stack(comp, def_path)
                ds = predict(stack, bundle)
                ds = to_5070_clip(ds, fire["geometry"])

                for name in ("CBI", "CBI_bc"):
                    ds[name].rio.to_raster(
                        out_base / f"{fire_id}_{name}.tif",
                        tiled=True, compress="ZSTD", zstd_level=1,
                    )

                cbi = ds["CBI"].values
                print(f"  → Done {fire_id}: grid={ds.sizes['y']}x{ds.sizes['x']} "
                      f"valid={int(np.isfinite(cbi).sum())} "
                      f"CBI med={np.nanmedian(cbi):.2f} max={np.nanmax(cbi):.2f}",
                      flush=True)
                with counts_lock:
                    counts["ok"] += 1
            except (Exception, SystemExit) as e:
                print(f"  → Failed {fire_id}: {type(e).__name__}: {e}\n"
                      f"{traceback.format_exc()}", flush=True)
                with counts_lock:
                    counts["err"] += 1

    fire_threads = [threading.Thread(target=_worker, daemon=True)
                    for _ in range(args.workers)]
    for t in fire_threads:
        t.start()
    for t in fire_threads:
        t.join()

    print(f"Finished: {counts['ok']} ok, {counts['err']} failed. "
          f"({(time.perf_counter() - start_time) / 60:.2f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
