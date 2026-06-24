"""Standalone one-shot CBI for a single MTBS fire perimeter.

Self-contained consolidation of the modular pipeline. Given one MTBS perimeter:

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

Run:
    uv run python scripts/cbi_oneshot.py --gpkg data/fire_perims/test.gpkg \
        --event-id NV4071111641720150629 --out-dir data/cbi/annie
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  -- registers the `.rio` accessor

from utils import (
    DEF_NC, DEF_TIF, DEFAULT_MAX_CLOUD, MODEL_PATH, TRAIN_CSV,
    build_stack, composite, ensure_def, ensure_model, predict, read_perimeter, to_5070_clip,
)
from aws_utils import _aws_creds_available, _S3_AUTH_HINT, fetch_landsat


# =========================================================================== main
def main() -> int:
    p = argparse.ArgumentParser(description="One-shot CBI for one MTBS perimeter.")
    p.add_argument("--gpkg", required=True)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--event-id", default=None,
                     help="MTBS Event_ID (default: first wildfire).")
    sel.add_argument("--index", type=int, default=None,
                     help="0-based row index in the gpkg layer.")
    p.add_argument("--layer", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--def", dest="def_tif", default=str(DEF_TIF))
    p.add_argument("--csv", default=str(TRAIN_CSV))
    args = p.parse_args()

    bundle = ensure_model(Path(args.model), Path(args.csv))
    def_path = ensure_def(Path(args.def_tif), DEF_NC)
    fire = read_perimeter(args.gpkg, args.event_id, args.index, args.layer)
    print(f"fire={fire['fire_id']} state={fire['state']} year={fire['year']} "
          f"DOY=[{fire['start_day']},{fire['end_day']}]", flush=True)

    if not _aws_creds_available():
        print(f"warning: no AWS credentials detected. {_S3_AUTH_HINT}\n"
              "  Continuing in case an instance profile or SSO session is available...",
              flush=True)

    cube = fetch_landsat(fire["bbox"], fire["year"], fire["start_day"],
                         fire["end_day"], args.max_cloud)
    comp = composite(cube, fire["year"])
    stack = build_stack(comp, def_path)
    ds = predict(stack, bundle)
    ds = to_5070_clip(ds, fire["geometry"])

    cbi = ds["CBI"].values
    print(f"out: grid={ds.sizes['y']}x{ds.sizes['x']} crs={ds.rio.crs} "
          f"valid_px={int(np.isfinite(cbi).sum())} "
          f"CBI med={np.nanmedian(cbi):.2f} max={np.nanmax(cbi):.2f}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("CBI", "CBI_bc"):
        ds[name].rio.to_raster(out / f"{fire['fire_id']}_{name}.tif",
                               tiled=True, compress="ZSTD", zstd_level=1)
    print(f"wrote {fire['fire_id']}_CBI.tif, _CBI_bc.tif to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
