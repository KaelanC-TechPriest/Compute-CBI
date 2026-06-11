"""Multithreaded CBI for one or more MTBS fire perimeters.

Two-queue design that prevents redundant work:
  Queue 1 (scene_q): unique Landsat scenes to download — main thread deduplicates
    by (scene_id, state) so each raw file is written at most once.
  Queue 2 (fire_q): fires to compute CBI for — main thread skips fires whose
    output TIFs already exist, enabling resumable runs.

Phase 0  main thread
  ├─ build fire list and STAC-search per fire (no downloads yet)
  ├─ aggregate unique scenes → compute per-scene union bbox → populate scene_q
  └─ join Phase 1, then populate fire_q (skipping existing outputs)

Phase 1  --download-workers threads
  └─ drain scene_q: write landsat_cache/raw/{state}/{scene_id}.tif

Phase 2  --fire-workers threads
  └─ drain fire_q: build cube from stored items + raw cache → CBI/CBI_bc TIFs

Run:
    uv run python scripts/cbi_perimeter_threaded.py \
        --gpkg data/fire_perims/test.gpkg \
        --event-id NV4071111641720150629 \
        --out-dir data/cbi/test_threaded \
        --landsat-cache data/landsat_cache \
        --download-workers 4 --fire-workers 2
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
import xarray as xr
from affine import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from utils import (
    BANDS, DEF_NC, DEF_TIF, DEFAULT_MAX_CLOUD, MODEL_PATH, OPTICAL, TRAIN_CSV,
    STATE_WINDOWS, WILDFIRE_CODE,
    _GDAL_ENV, _item_window, _search,
    build_stack, composite, ensure_def, ensure_model,
    fire_grid, predict, raw_scene_path, to_5070_clip,
)


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


def _build_cube(items, fire: dict, landsat_cache: Path | None) -> xr.Dataset:
    """Build a (time, band, y, x) Landsat cube from pre-searched STAC items.

    Uses the raw scene cache when available so no STAC re-search is needed.
    Mirrors the inner loop of fetch_landsat without the redundant _search call.
    """
    grid = fire_grid(fire["bbox"])
    arrs = []
    for it in items:
        rp = raw_scene_path(landsat_cache, it, fire["state"]) if landsat_cache else None
        a = _item_window(it, grid, raw_path=rp)
        if a is not None:
            arrs.append(a.assign_coords(time=it.datetime).expand_dims("time"))
    if not arrs:
        raise RuntimeError("no overlapping Landsat scenes found")
    cube = xr.concat(arrs, dim="time", coords="minimal",
                     compat="override").assign_coords(band=OPTICAL)
    cube.rio.write_crs(f"EPSG:{grid[0]}", inplace=True)
    return cube


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
    p.add_argument("--landsat-cache", default=None,
                   help="Directory for raw Landsat scene cache.")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete all cached Landsat scenes before running.")
    p.add_argument("--download-workers", type=int, default=4,
                   help="Threads for Phase 1 scene downloads (default: 4).")
    p.add_argument("--fire-workers", type=int, default=4,
                   help="Threads for Phase 2 CBI computation (default: 4).")

    args = p.parse_args()

    if args.state is not None:
        args.state = args.state.upper()
        if args.state not in STATE_WINDOWS:
            p.error(f"Unknown state '{args.state}'. Known: {', '.join(sorted(STATE_WINDOWS))}")

    if args.landsat_cache is None:
        print("warning: --landsat-cache not set; scenes will not be cached between fires", flush=True)

    start_time = time.perf_counter()

    landsat_cache: Path | None = None
    if args.landsat_cache is not None:
        landsat_cache = Path(args.landsat_cache)
        if args.clear_cache and landsat_cache.exists():
            import shutil
            shutil.rmtree(landsat_cache)
            print(f"cache: cleared {landsat_cache}", flush=True)
        landsat_cache.mkdir(parents=True, exist_ok=True)

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
    # Phase 0: build fire list + STAC search (no downloads)
    # -------------------------------------------------------------------------
    fires: list[dict] = []
    # (scene_id, state) → (item, [w, s, e, n])  — union bbox across all fires needing it
    scene_index: dict[tuple[str, str], tuple] = {}

    for _, row in gdf.iterrows():
        fire_id = row["Event_ID"]
        year = int(row["Ig_Date"].year)
        if year < 1986 or year > 2020:
            print(f"  Skipping {fire_id}: year {year} out of range", flush=True)
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

        items = _search(bbox, f"{year - 2}-01-01", f"{year + 3}-01-01",
                        args.max_cloud, sd, ed)

        fire = {
            "fire_id": fire_id, "state": state, "year": year,
            "start_day": sd, "end_day": ed,
            "geometry": geom, "bbox": bbox,
            "items": items,
        }
        fires.append(fire)

        # Accumulate unique scenes; expand union bbox
        fw, fs, fe, fn = bbox
        for it in items:
            scene_id = it.properties.get("landsat:scene_id") or it.id
            key = (scene_id, state)
            if key in scene_index:
                prev_item, (pw, ps, pe, pn) = scene_index[key]
                scene_index[key] = (prev_item,
                                    (min(pw, fw), min(ps, fs),
                                     max(pe, fe), max(pn, fn)))
            else:
                scene_index[key] = (it, (fw, fs, fe, fn))

    if not fires:
        print("No eligible fires after filtering.", flush=True)
        return 0

    total = len(fires)

    # -------------------------------------------------------------------------
    # Phase 1: parallel scene downloads (Queue 1)
    # -------------------------------------------------------------------------
    if landsat_cache is not None:
        scene_q: queue.Queue = queue.Queue()
        already_cached = 0
        for (scene_id, state), (it, union_bbox) in scene_index.items():
            rp = raw_scene_path(landsat_cache, it, state)
            if rp.is_file():
                already_cached += 1
            else:
                scene_q.put((it, union_bbox, rp))

        to_download = scene_q.qsize()
        print(f"prefetch: {already_cached} already cached, "
              f"{to_download} queued for download ({args.download_workers} workers)",
              flush=True)

        dl_ok = dl_err = 0
        dl_lock = threading.Lock()

        def _dl_worker():
            nonlocal dl_ok, dl_err
            while True:
                try:
                    it, union_bbox, rp = scene_q.get_nowait()
                except queue.Empty:
                    return
                try:
                    if _download_raw_scene(it, union_bbox, rp):
                        with dl_lock:
                            dl_ok += 1
                except Exception as e:
                    scene_id = it.properties.get("landsat:scene_id") or it.id
                    print(f"  prefetch: skipping {scene_id}: {e}", flush=True)
                    with dl_lock:
                        dl_err += 1

        dl_threads = [threading.Thread(target=_dl_worker, daemon=True)
                      for _ in range(args.download_workers)]
        for t in dl_threads:
            t.start()
        for t in dl_threads:
            t.join()

        print(f"prefetch done: {dl_ok} downloaded, {dl_err} errors", flush=True)

    # -------------------------------------------------------------------------
    # Phase 2: parallel CBI computation (Queue 2)
    # -------------------------------------------------------------------------
    fire_q: queue.Queue = queue.Queue()
    skipped = 0
    for i, fire in enumerate(fires, start=1):
        fire_q.put((i, fire))

    print(f"Processing {fire_q.qsize()} fires ({skipped} skipped) "
          f"with {args.fire_workers} workers ...", flush=True)

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
                cbi_path = out_base / f"{fire_id}_CBI.tif"
                if cbi_path.is_file():
                    with counts_lock:
                        counts["ok"] += 1
                    return

                print(f"[{i}/{total}] Processing {fire_id} ...", flush=True)
                cube = _build_cube(fire["items"], fire, landsat_cache)
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
