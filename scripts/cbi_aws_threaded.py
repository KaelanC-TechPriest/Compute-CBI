"""Multithreaded CBI for all MTBS fire perimeters in a state (or one perimeter), via AWS.

Combines the lazy-fetch AWS strategy from cbi_perimeter_aws.py with the
queue-based worker pool from cbi_perimeter_threaded.py. Each worker thread
handles one fire end-to-end: STAC search, lazy fetch (Y±1 first, Y±2 fallback),
composite, predict, write.

Run (whole state):
    uv run python scripts/cbi_aws_threaded.py --gpkg data/fire_perims/mtbs_perims.gpkg \
        --state MT --out-dir data/cbi/MT --workers 4

Run (single fire):
    uv run python scripts/cbi_aws_threaded.py --gpkg data/fire_perims/mtbs_perims.gpkg \
        --event-id NV4071111641720150629 --out-dir data/cbi/annie --workers 1
"""

from __future__ import annotations

import argparse
import queue
import sys
import gc
import threading
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  -- registers the `.rio` accessor
from rasterio.warp import transform_bounds

from utils import (
    DEF_NC, DEF_TIF, DEFAULT_MAX_CLOUD, MODEL_PATH, STATE_WINDOWS, TRAIN_CSV, WILDFIRE_CODE,
    build_stack, ensure_def, ensure_model, predict, to_5070_clip,
)
from aws_utils import _aws_creds_available, _S3_AUTH_HINT, lazy_fetch_and_composite


def main() -> int:
    p = argparse.ArgumentParser(
        description="Multithreaded CBI for all MTBS perimeters in a state (or one), via AWS."
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
                   help="2-letter state abbreviation (e.g. MT, NV) to filter fires by state.")
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
            p.error(f"Unknown state '{args.state}'. Known states: {', '.join(sorted(STATE_WINDOWS))}")

    if args.start_year < 1986 or args.start_year > 2020:
        p.error("Start year out of bounds (must be in range 1986-2020)")

    if args.end_year < 1986 or args.end_year > 2020:
        p.error("End year out of bounds (must be in range 1986-2020)")

    if not _aws_creds_available():
        print(f"warning: no AWS credentials detected. {_S3_AUTH_HINT}\n"
              "  Continuing in case an instance profile or SSO session is available...",
              flush=True)

    start_time = time.perf_counter()

    bundle = ensure_model(Path(args.model), Path(args.csv))
    bundle["model"].n_jobs = 1  # no joblib sub-threads inside worker threads
    def_path = ensure_def(Path(args.def_tif), DEF_NC)

    gdf = gpd.read_file(args.gpkg, layer=args.layer).to_crs(5070)
    gdf = gdf[gdf["Incid_Type"] == WILDFIRE_CODE]

    if args.state is not None:
        gdf = gdf[gdf["Event_ID"].str[:2].str.upper() == args.state]

    gdf = gdf[(gdf["Ig_Date"].dt.year >= args.start_year) &
              (gdf["Ig_Date"].dt.year <= args.end_year)]

    if args.event_id is not None:
        gdf = gdf[gdf["Event_ID"] == args.event_id]
        if gdf.empty:
            raise SystemExit(f"event-id {args.event_id} not found in {args.gpkg}")
    elif args.index is not None:
        if not 0 <= args.index < len(gdf):
            raise SystemExit(f"index {args.index} out of range (0..{len(gdf) - 1})")
        gdf = gdf.iloc[[args.index]]

    if gdf.empty:
        print(f"No wildfires found for state={args.state} event_id={args.event_id} "
              f"index={args.index}", flush=True)
        return 0

    state_str = f" in {args.state}" if args.state else ""
    print(f"Found {len(gdf)} wildfire(s){state_str}.", flush=True)

    out_base = Path(args.out_dir)
    if args.state:
        out_base = out_base / args.state
    out_base.mkdir(parents=True, exist_ok=True)

    fires: list[dict] = []
    for _, row in gdf.iterrows():
        fire_id = row["Event_ID"]
        state = str(fire_id)[:2].upper()
        if state not in STATE_WINDOWS:
            print(f"  Skipping {fire_id}: no image-season window for state {state!r}", flush=True)
            continue

        sd, ed = STATE_WINDOWS[state]
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        b = 1000.0
        bbox = transform_bounds("EPSG:5070", "EPSG:4326",
                                minx - b, miny - b, maxx + b, maxy + b)
        fires.append({
            "fire_id": fire_id,
            "state": state,
            "year": int(row["Ig_Date"].year),
            "start_day": sd,
            "end_day": ed,
            "geometry": geom,
            "bbox": bbox,
            "area_m2": geom.area,
        })

    if not fires:
        print("No eligible fires after filtering.", flush=True)
        return 0

    fires.sort(key=lambda f: f["area_m2"])

    total = len(fires)
    fire_q: queue.Queue = queue.Queue()
    for i, fire in enumerate(fires, start=1):
        fire_q.put((i, fire))

    print(f"Processing {total} fire(s){state_str} with {args.workers} worker(s) ...", flush=True)

    counts = {"ok": 0, "skip": 0, "err": 0}
    counts_lock = threading.Lock()

    def _worker():
        tid = threading.current_thread().name
        while True:
            try:
                i, fire = fire_q.get_nowait()
            except queue.Empty:
                return

            fire_id = fire["fire_id"]
            try:
                out_cbi = out_base / f"{fire_id}_CBI.tif"
                out_cbi_bc = out_base / f"{fire_id}_CBI_bc.tif"
                if out_cbi.exists() and out_cbi_bc.exists():
                    print(f"[{tid}:{i}/{total}] Skipping {fire_id}: outputs already exist", flush=True)
                    with counts_lock:
                        counts["skip"] += 1
                    continue

                print(f"[{tid}:{i}/{total}] {fire_id} state={fire['state']} year={fire['year']} "
                      f"DOY=[{fire['start_day']},{fire['end_day']}]", flush=True)

                comp = lazy_fetch_and_composite(
                    fire["bbox"], fire["year"],
                    fire["start_day"], fire["end_day"],
                    args.max_cloud,
                )
                stack = build_stack(comp, def_path)
                del comp
                gc.collect()
                ds = predict(stack, bundle)
                del stack
                gc.collect()
                ds = to_5070_clip(ds, fire["geometry"])

                for name in ("CBI", "CBI_bc"):
                    ds[name].rio.to_raster(
                        out_base / f"{fire_id}_{name}.tif",
                        tiled=True, compress="ZSTD", zstd_level=1,
                    )

                cbi = ds["CBI"].values
                del ds
                gc.collect()
                print(f"  [{tid}] Done {fire_id}: grid={cbi.shape[0]}x{cbi.shape[1]} "
                      f"valid={int(np.isfinite(cbi).sum())} "
                      f"CBI med={np.nanmedian(cbi):.2f} max={np.nanmax(cbi):.2f}",
                      flush=True)
                del cbi
                gc.collect()
                with counts_lock:
                    counts["ok"] += 1

            except (Exception, SystemExit) as e:
                print(f"  [{tid}] Failed {fire_id}: {type(e).__name__}: {e}\n"
                      f"{traceback.format_exc()}", flush=True)
                with counts_lock:
                    counts["err"] += 1

    threads = [threading.Thread(target=_worker, daemon=True, name=f"W{i+1}")
               for i in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = (time.perf_counter() - start_time) / 60
    print(f"\nDone{state_str}: {counts['ok']} ok, {counts['skip']} skipped, "
          f"{counts['err']} failed ({elapsed:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
