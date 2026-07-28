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
import xarray as xr
from shapely import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
import rioxarray  # noqa: F401  -- registers the `.rio` accessor
from rasterio.warp import transform_bounds
from shapely.geometry import box as shapely_box
from rioxarray.merge import merge_arrays

from utils import (
    DEF_NC, DEF_TIF, DEFAULT_MAX_CLOUD, MODEL_PATH, STATE_WINDOWS, TRAIN_CSV, 
    WILDFIRE_CODES, PADDING,
    build_stack, ensure_def, ensure_model, predict, to_5070_clip,
)
from aws_utils import _aws_creds_available, _S3_AUTH_HINT, lazy_fetch_and_composite

_M2_PER_ACRE: float = 4_046.8564224
_SPLIT_THRESHOLD_M2: float = 150_000 * _M2_PER_ACRE # based on 8GB RAM capacity

def split_polygon(geom: BaseGeometry, threshold_m2:float) -> list[BaseGeometry]:
    minx, miny, maxx, maxy = geom.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    if bbox_area <= threshold_m2: return [geom]

    if (maxx - minx >= maxy - miny):
        mid = (maxx + minx) / 2.0
        halves = [shapely_box(minx, miny, mid, maxy), shapely_box(mid, miny, maxx, maxy)]
    else:
        mid = (maxy + miny) / 2.0
        halves = [shapely_box(minx, miny, maxx, mid), shapely_box(minx, mid, maxx, maxy)]

    pieces = []
    for half in halves:
        piece = geom.intersection(half)
        if (piece.is_empty or piece.area == 0):
            continue

        if (isinstance(piece, (MultiPolygon, GeometryCollection))):
            sub_geoms = [g for g in piece.geoms if g.geom_type == "Polygon"
                and not (g.is_empty or g.area == 0)]
        else:
            sub_geoms = [piece]

        for sub_geom in sub_geoms:
            pieces.extend(split_polygon(sub_geom, threshold_m2))
    return pieces

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
    p.add_argument("--state", type=str, default=None,
                   help="List of comma-separated 2-letter state abbreviations (e.g. MT,NV).")
    p.add_argument("--workers", type=int, default=4,
                   help="Worker threads (search + download + process per fire, default: 4).")
    p.add_argument("--debug", action="store_true", default=False,
                   help="Enable verbose debug output (scene counts, grid info, fallback fetches).")
    p.add_argument("--exclude-list", default=None,
                   help="Path to a text file with one Event_ID per line to skip.")
    p.add_argument("--start-year", type=int, default=1986,
                   help="First fire ignition year to process (default: 1986).")
    p.add_argument("--end-year", type=int, default=2024,
                   help="Last fire ignition year to process (default: 2024).")
    args = p.parse_args()

    states: list[str] = []
    if args.state is not None:
        states = [s.strip() for s in args.state.upper().split(',') if s.strip()]
        unknown = [s for s in states if s not in STATE_WINDOWS]
        if unknown:
            p.error(f"Unknown state(s) '{", ".join(unknown)}'. Known states: {', '.join(sorted(STATE_WINDOWS))}")

    if args.start_year < 1986 or args.start_year > 2024:
        p.error("Start year out of bounds (must be in range 1986-2024)")

    if args.end_year < 1986 or args.end_year > 2024:
        p.error("End year out of bounds (must be in range 1986-2024)")

    if not _aws_creds_available():
        print(f"warning: no AWS credentials detected. {_S3_AUTH_HINT}\n"
              "  Continuing in case an instance profile or SSO session is available...",
              flush=True)

    start_time = time.perf_counter()

    bundle = ensure_model(Path(args.model), Path(args.csv))
    bundle["model"].n_jobs = 1  # no joblib sub-threads inside worker threads
    def_path = ensure_def(Path(args.def_tif), DEF_NC)

    exclude_ids: set[str] = set()
    if args.exclude_list is not None:
        with open(args.exclude_list) as f:
            exclude_ids = {line.strip() for line in f if line.strip()}

    # ===================== FILTERING ========================================
    gdf = gpd.read_file(args.gpkg, layer=args.layer).to_crs(5070)
    gdf = gdf[gdf["Incid_Type"].isin(WILDFIRE_CODES)]

    if states:
        gdf = gdf[gdf["Event_ID"].str[:2].str.upper().isin(states)]

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
        print(f"No wildfires found for state(s)=({", ".join(states)})"
            f" | event_id={args.event_id} "
            f" | index={args.index}", flush=True)
        return 0

    if exclude_ids:
        gdf = gdf[~gdf["Event_ID"].isin(exclude_ids)]

    state_str = f" in {", ".join(states)}" if states else ""
    print(f"Found {len(gdf)} wildfire(s){state_str}.", flush=True)

    out_base = Path(args.out_dir)

    for year in range(args.start_year, args.end_year + 1):
        (out_base / str(year)).mkdir(parents=True, exist_ok=True)

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
        bbox = transform_bounds("EPSG:5070", "EPSG:4326",
                                minx - PADDING, miny - PADDING,
                                maxx + PADDING, maxy + PADDING)
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

    worker_split_threshold = _SPLIT_THRESHOLD_M2 / args.workers

    total = len(fires)
    fire_q: queue.Queue = queue.Queue()
    for i, fire in enumerate(fires, start=1):
        fire_q.put((i, fire))

    print(f"Processing {total} fire(s){state_str} with {args.workers} worker(s) ...", flush=True)
    print(f"Size threshold: {worker_split_threshold} m^2")

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
                out_cbi = out_base / str(fire["year"]) / f"{fire_id}_CBI.tif"
                out_cbi_bc = out_base / str(fire["year"]) / f"{fire_id}_CBI_bc.tif"
                if out_cbi.exists() and out_cbi_bc.exists():
                    print(f"[{tid}:{i}/{total}] Skipping {fire_id}: outputs already exist", flush=True)
                    with counts_lock:
                        counts["skip"] += 1
                    continue

                print(f"[{tid}:{i}/{total}] {fire_id} state={fire['state']} year={fire['year']} "
                          f"DOY=[{fire['start_day']},{fire['end_day']}]", flush=True)

                pieces = split_polygon(fire["geometry"], worker_split_threshold)
                if args.debug and len(pieces) > 1:
                    n_acres = fire["area_m2"] / _M2_PER_ACRE
                    print(f"debug [_worker]: {fire_id}: {n_acres:.0f} acres -> split into {len(pieces)} piece(s)", flush=True)

                raw_pieces = []
                for pi, piece_geom in enumerate(pieces, start=1):
                    if args.debug: print(f"debug [_worker]: processing piece {pi}")
                    minx, miny, maxx, maxy = piece_geom.bounds
                    piece_bbox = transform_bounds("EPSG:5070", "EPSG:4326",
                                                  minx - PADDING, miny - PADDING,
                                                  maxx + PADDING, maxy + PADDING)
                    try:
                        pre_da  = lazy_fetch_and_composite(piece_bbox,
                                                           fire["year"],
                                                           "pre",
                                                           fire["start_day"],
                                                           fire["end_day"],
                                                           args.max_cloud,
                                                           debug=args.debug)
                        post_da  = lazy_fetch_and_composite(piece_bbox,
                                                            fire["year"],
                                                            "post",
                                                            fire["start_day"],
                                                            fire["end_day"],
                                                            args.max_cloud,
                                                            debug=args.debug)
                        comp = xr.Dataset({"pre": pre_da, "post": post_da})
                        stack = build_stack(comp, def_path)
                        del comp, pre_da, post_da; gc.collect()
                        ds  = predict(stack, bundle);       del stack; gc.collect()
                        raw_pieces.append(to_5070_clip(ds, piece_geom))
                        del ds; gc.collect()
                    except Exception as e:
                        print(f"  [{tid}] piece failed {piece_geom.bounds}: {type(e).__name__}: {e}",
                              flush=True)
                        raw_pieces.append(None)

                piece_datasets = [p for p in raw_pieces if p is not None]
                del raw_pieces; gc.collect()

                if not piece_datasets:
                    raise RuntimeError(f"all {len(pieces)} piece(s) failed for {fire_id}")

                if len(piece_datasets) == 1:
                    ds = piece_datasets.pop()
                    del piece_datasets
                else:
                    cbi_merged    = merge_arrays([p["CBI"]    for p in piece_datasets], nodata=np.nan)
                    cbi_bc_merged = merge_arrays([p["CBI_bc"] for p in piece_datasets], nodata=np.nan)
                    ds = xr.Dataset({"CBI": cbi_merged, "CBI_bc": cbi_bc_merged})
                    ds.rio.write_crs("EPSG:5070", inplace=True)
                    ds["CBI"].rio.write_nodata(np.nan, inplace=True)
                    ds["CBI_bc"].rio.write_nodata(np.nan, inplace=True)
                    del piece_datasets, cbi_merged, cbi_bc_merged; gc.collect()

                out_cbi.parent.mkdir(parents=True, exist_ok=True)
                out_cbi_bc.parent.mkdir(parents=True, exist_ok=True)
                ds["CBI"].rio.to_raster(out_cbi, tiled=True, compress="ZSTD", zstd_level=1)
                ds["CBI_bc"].rio.to_raster(out_cbi_bc, tiled=True, compress="ZSTD", zstd_level=1)
                cbi = ds["CBI"].values
                del ds; gc.collect()
                print(f"[{tid}] Done {fire_id}: grid={cbi.shape[0]}x{cbi.shape[1]} "
                      f"valid={int(np.isfinite(cbi).sum())} "
                      f"CBI med={np.nanmedian(cbi):.2f} max={np.nanmax(cbi):.2f}",
                      flush=True)
                del cbi; gc.collect()
                with counts_lock:
                    counts["ok"] += 1

            except (Exception, SystemExit) as e:
                print(f"[{tid}] Failed {fire_id}: {type(e).__name__}: {e}\n"
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
