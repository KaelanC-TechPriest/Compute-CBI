"""Verify that every pixel inside each MTBS fire perimeter has a valid CBI value.

For every <Event_ID>_CBI.tif / <Event_ID>_CBI_bc.tif produced by aws_threaded.py,
this rasterizes that fire's MTBS polygon (from the same gpkg used to run the
pipeline) onto the output file's own grid, then checks every pixel that falls
inside the polygon against the expected CBI range. A pixel is flagged as
invalid if, inside the polygon, it is:

  - nodata   : NaN / non-finite (a gap in the prediction where a number is
               expected -- e.g. a failed split-piece, a bad merge seam)
  - oor      : finite but outside [--min, --max] (default 0.0-3.0, the CBI
               scale; CBI_bc is supposed to be hard-clipped to this range by
               predict() in utils.py, so any CBI_bc violation is a bug)

Files whose Event_ID has no matching polygon in the gpkg are reported
separately (status=no-polygon-match) since there's nothing to check them
against.

Run (whole dataset):
    uv run python scripts/verify_cbi.py \
        --gpkg data/fire_perims/mtbs/mtbs_perims_trimmed.gpkg \
        --cbi-dir /run/host/run/media/kaelan/CBI \
        --out-dir ./cbi_verify_report --workers 8

Run (quick smoke test, one year, raw CBI only):
    uv run python scripts/verify_cbi.py \
        --gpkg data/fire_perims/mtbs/mtbs_perims_trimmed.gpkg \
        --cbi-dir /run/host/run/media/kaelan/CBI \
        --out-dir ./cbi_verify_report --years 2020 --bands CBI --limit 50
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

_SUFFIX_BAND = {
    "_CBI_bc.tif": "CBI_bc",
    "_CBI.tif": "CBI",
}


def _fire_id_and_band(name: str) -> tuple[str, str] | None:
    for suf, band in _SUFFIX_BAND.items():
        if name.endswith(suf):
            return name[: -len(suf)], band
    return None


def _iter_targets(cbi_dir: Path, years: set[int] | None, bands: set[str],
                  states: set[str] | None):
    for year_dir in sorted(p for p in cbi_dir.iterdir() if p.is_dir()):
        if not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if years is not None and year not in years:
            continue
        for tif in sorted(year_dir.glob("*.tif")):
            parsed = _fire_id_and_band(tif.name)
            if parsed is None:
                continue
            fire_id, band = parsed
            if band not in bands:
                continue
            if states is not None and fire_id[:2].upper() not in states:
                continue
            yield year, fire_id, band, tif


def _check_one(task: tuple) -> dict:
    year, fire_id, band, tif_path, geometry, vmin, vmax, max_detail = task
    result = {
        "year": year, "fire_id": fire_id, "band": band, "file": str(tif_path),
        "status": "ok", "n_polygon_px": 0, "n_nodata": 0, "n_oor": 0,
        "n_invalid": 0, "min_val": "", "max_val": "", "error": "",
        "detail": [],
    }
    if geometry is None:
        result["status"] = "no-polygon-match"
        return result

    try:
        with rasterio.open(tif_path) as ds:
            arr = ds.read(1)
            inside = geometry_mask([geometry], out_shape=ds.shape,
                                   transform=ds.transform, invert=True)
            n_inside = int(inside.sum())
            result["n_polygon_px"] = n_inside
            if n_inside == 0:
                result["status"] = "empty-polygon-mask"
                return result

            vals = arr[inside]
            finite = np.isfinite(vals)
            if finite.any():
                result["min_val"] = float(vals[finite].min())
                result["max_val"] = float(vals[finite].max())

            nodata_full = inside & ~np.isfinite(arr)
            oor_full = inside & np.isfinite(arr) & ((arr < vmin) | (arr > vmax))
            invalid_full = nodata_full | oor_full

            n_nodata = int(nodata_full.sum())
            n_oor = int(oor_full.sum())
            result["n_nodata"] = n_nodata
            result["n_oor"] = n_oor
            result["n_invalid"] = n_nodata + n_oor
            if result["n_invalid"] > 0:
                result["status"] = "invalid"
                rows, cols = np.where(invalid_full)
                for i, (r, c) in enumerate(zip(rows, cols)):
                    if i >= max_detail:
                        break
                    x, y = ds.xy(r, c)
                    v = arr[r, c]
                    reason = "nodata" if nodata_full[r, c] else "out-of-range"
                    result["detail"].append({
                        "row": int(r), "col": int(c),
                        "x": float(x), "y": float(y),
                        "value": "" if not np.isfinite(v) else float(v),
                        "reason": reason,
                    })
                if len(result["detail"]) < result["n_invalid"]:
                    result["error"] = (
                        f"detail truncated at {max_detail} of {result['n_invalid']} "
                        "invalid pixels"
                    )
    except Exception as e:  # noqa: BLE001
        result["status"] = "read-error"
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def main() -> int:
    p = argparse.ArgumentParser(
        description="Verify all in-polygon CBI output pixels are valid numbers."
    )
    p.add_argument("--gpkg", required=True,
                   help="Path to the MTBS perimeters gpkg used by aws_threaded.py.")
    p.add_argument("--cbi-dir", required=True,
                   help="Path to the CBI output directory (contains <year>/ subdirs).")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write summary.csv / details.csv into.")
    p.add_argument("--min", type=float, default=0.0, help="Valid CBI lower bound.")
    p.add_argument("--max", type=float, default=3.0, help="Valid CBI upper bound.")
    p.add_argument("--bands", default="CBI,CBI_bc",
                   help="Comma list of bands to check: CBI, CBI_bc, or both.")
    p.add_argument("--years", default=None,
                   help="Comma list of years to check (default: all).")
    p.add_argument("--states", default=None,
                   help="Comma list of 2-letter state prefixes to check (default: all).")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--max-detail-per-file", type=int, default=500,
                   help="Cap on invalid-pixel rows recorded per file (counts are exact).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only check the first N matched files (smoke testing).")
    args = p.parse_args()

    bands = {b.strip() for b in args.bands.split(",") if b.strip()}
    years = ({int(y) for y in args.years.split(",")} if args.years else None)
    states = ({s.strip().upper() for s in args.states.split(",")} if args.states else None)

    cbi_dir = Path(args.cbi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading perimeters from {args.gpkg} ...", flush=True)
    gdf = gpd.read_file(args.gpkg).to_crs(5070)
    geom_by_id = dict(zip(gdf["Event_ID"], gdf.geometry))
    print(f"Loaded {len(geom_by_id)} perimeter(s).", flush=True)

    print(f"Scanning {cbi_dir} ...", flush=True)
    targets = list(_iter_targets(cbi_dir, years, bands, states))
    if args.limit is not None:
        targets = targets[: args.limit]
    total = len(targets)
    print(f"Found {total} output file(s) to check "
          f"(bands={sorted(bands)}, years={sorted(years) if years else 'all'}, "
          f"states={sorted(states) if states else 'all'}).", flush=True)
    if total == 0:
        return 0

    tasks = [
        (year, fire_id, band, tif_path, geom_by_id.get(fire_id),
         args.min, args.max, args.max_detail_per_file)
        for year, fire_id, band, tif_path in targets
    ]

    summary_path = out_dir / "summary.csv"
    detail_path = out_dir / "details.csv"
    summary_fields = ["year", "fire_id", "band", "file", "status",
                      "n_polygon_px", "n_nodata", "n_oor", "n_invalid",
                      "min_val", "max_val", "error"]
    detail_fields = ["year", "fire_id", "band", "file", "row", "col", "x", "y",
                     "value", "reason"]

    t0 = time.perf_counter()
    n_done = 0
    n_invalid_files = 0
    n_no_match = 0
    n_read_err = 0
    total_invalid_px = 0

    with open(summary_path, "w", newline="") as sf, \
         open(detail_path, "w", newline="") as df:
        sw = csv.DictWriter(sf, fieldnames=summary_fields)
        sw.writeheader()
        dw = csv.DictWriter(df, fieldnames=detail_fields)
        dw.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_check_one, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                n_done += 1

                sw.writerow({k: res[k] for k in summary_fields})

                for d in res["detail"]:
                    dw.writerow({
                        "year": res["year"], "fire_id": res["fire_id"],
                        "band": res["band"], "file": res["file"], **d,
                    })

                if res["status"] == "invalid":
                    n_invalid_files += 1
                    total_invalid_px += res["n_invalid"]
                    print(f"  INVALID {res['fire_id']} [{res['band']}] "
                          f"{res['n_invalid']} bad px "
                          f"(nodata={res['n_nodata']} oor={res['n_oor']}) "
                          f"-> {res['file']}", flush=True)
                elif res["status"] == "no-polygon-match":
                    n_no_match += 1
                elif res["status"] == "read-error":
                    n_read_err += 1
                    print(f"  READ-ERROR {res['fire_id']} [{res['band']}]: "
                          f"{res['error']} -> {res['file']}", flush=True)

                if n_done % 500 == 0 or n_done == total:
                    elapsed = time.perf_counter() - t0
                    print(f"  progress: {n_done}/{total} checked "
                          f"({elapsed:.0f}s elapsed)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s.", flush=True)
    print(f"  files checked        : {total}", flush=True)
    print(f"  files with bad pixels: {n_invalid_files}", flush=True)
    print(f"  total invalid pixels : {total_invalid_px}", flush=True)
    print(f"  no matching polygon  : {n_no_match}", flush=True)
    print(f"  read errors          : {n_read_err}", flush=True)
    print(f"  summary -> {summary_path}", flush=True)
    print(f"  details -> {detail_path}", flush=True)

    return 1 if (n_invalid_files or n_read_err) else 0


if __name__ == "__main__":
    sys.exit(main())
