"""CBI for all MTBS fire perimeters in a state (or one perimeter), via AWS Earth Search.

Self-contained consolidation of the modular pipeline. For each MTBS perimeter:

  1. ensure the RF model (load the repo cache, else train from the CBI CSV + cache it);
  2. ensure the static `def` raster (load the repo cache, else download the TerraClimate
     1981-2010 normal, sum to annual, cache it);
  3. fetch Landsat C2 L2 from AWS Earth Search for the fire bbox (ephemeral, no cache);
  4. composite -> predictors -> RF predict -> CBI/CBI_bc;
  5. write CBI/CBI_bc GeoTIFFs on the EPSG:5070 NLCD-snapped grid, clipped to the perimeter.

Caches (model, def) live at the wider-repo locations so they're shared with the modular code.
Faithful to Parks et al. (2019); see the repo modules this consolidates for provenance.

Landsat streams from AWS Earth Search, which reads the requester-pays `usgs-landsat`
bucket (us-west-2). This needs AWS credentials (`aws configure`, AWS_ACCESS_KEY_ID/
AWS_SECRET_ACCESS_KEY env vars, an instance profile, etc.) and costs a few cents/fire in
egress (cheapest run from EC2 in us-west-2).

Run (whole state):
    uv run python scripts/cbi_perimeter_aws.py --gpkg data/fire_perims/mtbs_perims.gpkg \
        --state MT --out-dir data/cbi/MT

Run (single fire):
    uv run python scripts/cbi_perimeter_aws.py --gpkg data/fire_perims/mtbs_perims.gpkg \
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
import rioxarray  # noqa: F401  -- registers the `.rio` accessor
from rasterio.warp import transform_bounds

from utils import (
    DEF_NC, DEF_TIF, DEFAULT_MAX_CLOUD, MODEL_PATH, STATE_WINDOWS, TRAIN_CSV, WILDFIRE_CODE,
    build_stack, ensure_def, ensure_model, predict, to_5070_clip,
)
from aws_utils import _aws_creds_available, _S3_AUTH_HINT, fetch_landsat, lazy_fetch_and_composite


# =========================================================================== main
def main() -> int:
    p = argparse.ArgumentParser(
        description="CBI for all MTBS perimeters in a state (or one perimeter), via AWS."
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
                   help="2-letter state abbreviation (e.g. MT, NV) to filter fires by state.")
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
    def_path = ensure_def(Path(args.def_tif), DEF_NC)

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
        print(f"No wildfires found for state={args.state} event_id={args.event_id} "
              f"index={args.index}", flush=True)
        return 0

    state_str = f" in {args.state}" if args.state else ""
    print(f"Found {len(gdf)} wildfire(s){state_str}.", flush=True)

    out_base = Path(args.out_dir)
    if args.state:
        out_base = out_base / args.state
    out_base.mkdir(parents=True, exist_ok=True)

    n_total = len(gdf)
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, (_, row) in enumerate(gdf.iterrows(), start=1):
        fire_id = "<unknown>"
        try:
            fire_id = row["Event_ID"]
            year = int(row["Ig_Date"].year)

            if year < args.start_year or year > args.end_year:
                print(f"[{i}/{n_total}] Skipping {fire_id}: year {year} outside "
                      f"{args.start_year}-{args.end_year}", flush=True)
                n_skip += 1
                continue

            state = str(fire_id)[:2].upper()
            if state not in STATE_WINDOWS:
                print(f"[{i}/{n_total}] Skipping {fire_id}: no image-season window "
                      f"for state {state!r}", flush=True)
                n_skip += 1
                continue

            out_cbi = out_base / f"{fire_id}_CBI.tif"
            out_cbi_bc = out_base / f"{fire_id}_CBI_bc.tif"
            if out_cbi.exists() and out_cbi_bc.exists():
                print(f"[{i}/{n_total}] Skipping {fire_id}: outputs already exist",
                      flush=True)
                n_skip += 1
                continue

            sd, ed = STATE_WINDOWS[state]
            geom = row.geometry
            minx, miny, maxx, maxy = geom.bounds
            b = 1000.0
            bbox = transform_bounds("EPSG:5070", "EPSG:4326",
                                    minx - b, miny - b, maxx + b, maxy + b)

            print(f"[{i}/{n_total}] {fire_id} state={state} year={year} "
                  f"DOY=[{sd},{ed}]", flush=True)

            comp = lazy_fetch_and_composite(bbox, year, sd, ed, args.max_cloud)
            stack = build_stack(comp, def_path)
            ds = predict(stack, bundle)
            ds = to_5070_clip(ds, geom)

            for name in ("CBI", "CBI_bc"):
                ds[name].rio.to_raster(
                    out_base / f"{fire_id}_{name}.tif",
                    tiled=True, compress="ZSTD", zstd_level=1,
                )

            cbi = ds["CBI"].values
            print(f"  -> Done: grid={ds.sizes['y']}x{ds.sizes['x']} crs={ds.rio.crs} "
                  f"valid_px={int(np.isfinite(cbi).sum())} "
                  f"CBI med={np.nanmedian(cbi):.2f} max={np.nanmax(cbi):.2f}", flush=True)
            n_ok += 1

        except Exception as e:
            print(f"  -> Failed on {fire_id}: {type(e).__name__}: {e}\n"
                  f"{traceback.format_exc()}", flush=True)
            n_fail += 1
            continue

    elapsed = (time.perf_counter() - start_time) / 60
    print(f"\nDone{state_str}: {n_ok} ok, {n_skip} skipped, {n_fail} failed "
          f"({elapsed:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
