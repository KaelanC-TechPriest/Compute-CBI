"""Statewide CBI with threaded prefetch and threaded fire processing.

Two parallelism phases:
  1. Prefetch: main thread collects all raw scenes to download across all years,
     populates a queue once (no duplicates), N download workers drain it.
  2. CBI: fire dicts are built in the main thread, queued, N compute workers
     drain the queue running the full pipeline per fire.

Run:
    uv run python scripts/cbi_statewide_threaded.py \
        --gpkg data/fire_perims/mtbs_perims_DD.gpkg \
        --state MT --out-dir data/cbi/MT \
        --landsat-cache data/landsat_cache \
        --download-workers 8 --fire-workers 4
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from utils import (
    BANDS, DEF_NC, DEF_TIF, DEFAULT_MAX_CLOUD, MODEL_PATH, TRAIN_CSV,
    STATE_WINDOWS, WILDFIRE_CODE,
    _GDAL_ENV, _search,
    build_stack, composite, ensure_def, ensure_model,
    fetch_landsat, predict, raw_scene_path, to_5070_clip,
)


def compute_state_bbox(gdf_state: gpd.GeoDataFrame, buffer_m: float = 50_000.0):
    """Return (w, s, e, n) in WGS84 for the union of state fire perimeters plus buffer.

    gdf_state must be in EPSG:5070 so the buffer is in metres.
    """
    buffered = gdf_state.union_all().buffer(buffer_m)
    minx, miny, maxx, maxy = buffered.bounds
    return transform_bounds("EPSG:5070", "EPSG:4326", minx, miny, maxx, maxy)


def _download_raw_scene(item, bbox_4326, out_path: Path) -> bool:
    """Download the bbox window of a scene in its native projection as a multi-band GeoTIFF.

    Returns True if the file was written, False if the scene doesn't overlap bbox.
    """
    epsg = int(item.properties["proj:code"].split(":")[1])
    tr = item.properties["proj:transform"]
    sh = item.properties["proj:shape"]
    H, W = int(sh[0]), int(sh[1])
    A = Affine(*tr[:6])

    w, s, e, n = bbox_4326
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", f"EPSG:{epsg}", w, s, e, n)
    c0 = max(0, math.floor((minx - A.c) / A.a))
    c1 = min(W, math.ceil((maxx - A.c) / A.a))
    r0 = max(0, math.floor((maxy - A.f) / A.e))
    r1 = min(H, math.ceil((miny - A.f) / A.e))
    if c1 <= c0 or r1 <= r0:
        return False

    win = Window(c0, r0, c1 - c0, r1 - r0)  # type: ignore[call-arg]
    wtr: Affine = A * Affine.translation(c0, r0)  # type: ignore[assignment]

    arrays = []
    with rasterio.Env(**_GDAL_ENV):  # type: ignore[arg-type]
        for band in BANDS:
            with rasterio.open(item.assets[band].href) as ds:
                arrays.append(ds.read(1, window=win))

    h, w_px = arrays[0].shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.tif")
    with rasterio.open(
        tmp, "w", driver="GTiff",
        count=len(BANDS), dtype=arrays[0].dtype,
        crs=f"EPSG:{epsg}", transform=wtr,
        width=w_px, height=h,
        tiled=True, compress="ZSTD", zstd_level=1,
    ) as ds:
        for i, arr in enumerate(arrays, start=1):
            ds.write(arr, i)
    tmp.replace(out_path)
    return True


def prefetch_state(state: str, state_bbox, years, cache_dir: Path,
                   max_cloud: float, start_day: int, end_day: int,
                   n_workers: int) -> None:
    """Cache raw Landsat scene windows covering the full state bbox.

    Phase 1 (main thread): collect all items across all years, filter already-cached
    paths, enqueue the remainder. Each item appears at most once, so no two workers
    ever race to download the same file.

    Phase 2 (n_workers threads): drain the queue with _download_raw_scene.
    """
    to_download = []
    already_cached = 0
    for year in years:
        items = _search(state_bbox, f"{year}-01-01", f"{year + 1}-01-01",
                        max_cloud, start_day, end_day)
        for item in items:
            path = raw_scene_path(cache_dir, item, state)
            if path.is_file():
                already_cached += 1
            else:
                to_download.append((item, path))

        print(f"  prefetch year={year}: queued")

    num_to_download = len(to_download)
    print(f"  prefetch: {already_cached} already cached, "
          f"{num_to_download} queued for download ({n_workers} workers)", flush=True)

    dl_q: queue.Queue = queue.Queue()
    for entry in to_download:
        dl_q.put(entry)

    downloaded = errors = 0
    lock = threading.Lock()

    def worker():
        nonlocal downloaded, errors
        while True:
            try:
                item, path = dl_q.get_nowait()
            except queue.Empty:
                return
            try:
                if _download_raw_scene(item, state_bbox, path):
                    with lock:
                        downloaded += 1
                        print(f"  prefetch {downloaded}/{num_to_download} downloaded", flush=True)
            except Exception as e:
                print(f"  prefetch: skipping {item.id}: {e}", flush=True)
                with lock:
                    errors += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"prefetch done: {downloaded} downloaded, {errors} errors", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Statewide CBI with parallel prefetch and parallel fire processing."
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
    p.add_argument("--state", default=None, required=True,
                   help="2-letter state abbreviation (e.g. MT, AK).")
    p.add_argument("--landsat-cache", default=None, required=True,
                   help="Directory for cached Landsat scenes.")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete all cached Landsat scenes before running.")
    p.add_argument("--download-workers", type=int, default=4,
                   help="Threads for prefetch downloads (default: 4).")
    p.add_argument("--fire-workers", type=int, default=4,
                   help="Threads for CBI computation (default: 4).")

    args = p.parse_args()

    args.state = args.state.upper()
    if args.state not in STATE_WINDOWS:
        p.error(f"Unknown state '{args.state}'. Known states: {', '.join(sorted(STATE_WINDOWS))}")

    start_time = time.perf_counter()

    landsat_cache = Path(args.landsat_cache)
    if args.clear_cache and landsat_cache.exists():
        import shutil
        shutil.rmtree(landsat_cache)
        print(f"cache: cleared {landsat_cache}", flush=True)
    landsat_cache.mkdir(parents=True, exist_ok=True)

    bundle = ensure_model(Path(args.model), Path(args.csv))
    # Disable internal joblib parallelism so predict() doesn't spawn threads
    # inside each fire worker thread.
    bundle["model"].n_jobs = 1
    def_path = ensure_def(Path(args.def_tif), DEF_NC)

    out_base = Path(args.out_dir) / args.state
    out_base.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(args.gpkg, layer=args.layer).to_crs(5070)
    gdf = gdf[gdf["Incid_Type"] == WILDFIRE_CODE]
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
        print(f"No wildfires found for state {args.state}.", flush=True)
        return 0

    print(f"Found {len(gdf)} wildfires in {args.state}.", flush=True)

    # --- prefetch ---
    sd_pre, ed_pre = STATE_WINDOWS[args.state]
    state_bbox = compute_state_bbox(gdf)
    print(f"prefetch: {args.state} bbox={tuple(round(x, 4) for x in state_bbox)}, "
          f"years 1984-2022", flush=True)
    prefetch_state(args.state, state_bbox, range(1984, 2023),
                   landsat_cache, args.max_cloud, sd_pre, ed_pre,
                   n_workers=args.download_workers)

    # --- build fire list in main thread ---
    fires = []
    for _, row in gdf.iterrows():
        year = int(row["Ig_Date"].year)
        if year < 1986 or year > 2020:
            print(f"  Skipping {row['Event_ID']}: year {year} out of range", flush=True)
            continue
        state = str(row["Event_ID"])[:2].upper()
        if state not in STATE_WINDOWS:
            print(f"  Skipping {row['Event_ID']}: no window for state {state}", flush=True)
            continue
        sd, ed = STATE_WINDOWS[state]
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        b = 1000.0
        bbox = transform_bounds("EPSG:5070", "EPSG:4326",
                                minx - b, miny - b, maxx + b, maxy + b)
        fires.append({"fire_id": row["Event_ID"], "state": state, "year": year,
                      "start_day": sd, "end_day": ed, "geometry": geom, "bbox": bbox})

    total = len(fires)
    print(f"Processing {total} fires with {args.fire_workers} workers ...", flush=True)

    fire_q: queue.Queue = queue.Queue()
    for i, fire in enumerate(fires, start=1):
        fire_q.put((i, fire))

    counts = {"ok": 0, "err": 0}
    counts_lock = threading.Lock()

    def fire_worker():
        while True:
            try:
                i, fire = fire_q.get_nowait()
            except queue.Empty:
                return
            fire_id = fire["fire_id"]
            try:
                print(f"[{i}/{total}] Processing {fire_id} ...", flush=True)
                cube = fetch_landsat(fire["bbox"], fire["year"], fire["start_day"],
                                     fire["end_day"], args.max_cloud,
                                     landsat_cache=landsat_cache, state=fire["state"],
                                     cache_processed=False)
                comp = composite(cube, fire["year"])
                stack = build_stack(comp, def_path)
                ds = predict(stack, bundle)
                ds = to_5070_clip(ds, fire["geometry"])
                for name in ("CBI", "CBI_bc"):
                    ds[name].rio.to_raster(
                        out_base / f"{fire_id}_{name}.tif",
                        tiled=True, compress="ZSTD", zstd_level=1)
                cbi = ds["CBI"].values
                print(f"  → Done {fire_id}: grid={ds.sizes['y']}x{ds.sizes['x']} "
                      f"valid={int(np.isfinite(cbi).sum())} "
                      f"CBI med={np.nanmedian(cbi):.2f} max={np.nanmax(cbi):.2f}",
                      flush=True)
                with counts_lock:
                    counts["ok"] += 1
            except Exception as e:
                print(f"  → Failed {fire_id}: {type(e).__name__}: {e}\n"
                      f"{traceback.format_exc()}", flush=True)
                with counts_lock:
                    counts["err"] += 1

    fire_threads = [threading.Thread(target=fire_worker, daemon=True)
                    for _ in range(args.fire_workers)]
    for t in fire_threads:
        t.start()
    for t in fire_threads:
        t.join()

    print(f"Finished: {counts['ok']} ok, {counts['err']} failed. "
          f"({(time.perf_counter() - start_time) / 60:.2f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
