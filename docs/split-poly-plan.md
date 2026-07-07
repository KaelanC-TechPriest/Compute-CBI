# Plan: Large-Fire Polygon Splitting in `cbi_aws_threaded.py`

## Context

Large fires (>25k acres) push Landsat fetching and compositing into very high memory use,
and the resulting raster grids can be unwieldy. The fix is to split big fire polygons into
sub-25k-acre pieces via recursive bbox bisection, run the full pipeline on each piece
independently, then mosaic the pieces back together before writing output GeoTIFFs. Fires
under the threshold are untouched.

---

## Files Modified

- `scripts/cbi_aws_threaded.py` — only file changed

No changes to `utils.py`, `aws_utils.py`, or `main()` outside `_worker`.

---

## Implementation

### 1. New imports (top of file, with existing imports)

```python
from shapely.geometry import box as shapely_box   # polygon splitting
from rioxarray.merge import merge_arrays           # piece mosaicing
```

Both are already project dependencies (see `pyproject.toml`).

### 2. New module-level constants (between imports and `main()`)

```python
_M2_PER_ACRE: float = 4_046.8564224
_SPLIT_THRESHOLD_M2: float = 25_000 * _M2_PER_ACRE   # ~101 171 411 m²
```

### 3. `_split_polygon(geom, threshold_m2)` — module-level helper above `main()`

Recursively bisects the polygon along the longer bbox axis until all pieces are under
`threshold_m2`. Uses shapely `intersection` with the half-bbox.

```
if geom.area <= threshold_m2: return [geom]

bisect along longer axis (dx vs dy) at midpoint
  → left_box / right_box  (or bottom_box / top_box)

for each half:
    piece = geom.intersection(half_box)
    skip if empty or zero-area
    if MultiPolygon/GeometryCollection: flatten to Polygon members
    recurse each sub-polygon
```

Edge cases handled inside `_split_polygon`:
- Empty/zero-area intersections → skipped
- GeometryCollection (concave perimeters) → exploded to Polygon members only
- MultiPolygon input → bisection still works; area/bounds operate on the union

### 4. `_process_piece(piece_geom, fire, def_path, bundle, max_cloud)` — module-level helper

Runs the full pipeline for one piece geometry. Returns `xr.Dataset` on success, `None` on
any exception (logs the failure).

```
recompute piece bbox (with 1000 m buffer) via transform_bounds("EPSG:5070", "EPSG:4326", ...)
comp  = lazy_fetch_and_composite(piece_bbox, ...)
stack = build_stack(comp, def_path);  del comp; gc.collect()
ds    = predict(stack, bundle);       del stack; gc.collect()
ds    = to_5070_clip(ds, piece_geom)
return ds
```

The per-piece bbox is computed from `piece_geom.bounds`, NOT from the precomputed
`fire["bbox"]` — each piece needs its own STAC query sized to the piece.

### 5. Worker thread changes (inside `_worker`, inside the existing `try` block)

Replace the existing single pipeline block with a branch:

```
if fire["area_m2"] > _SPLIT_THRESHOLD_M2:
    pieces = _split_polygon(fire["geometry"], _SPLIT_THRESHOLD_M2)
    log: "N acres -> splitting into K piece(s)"

    raw_pieces = []
    for pi, piece_geom in enumerate(pieces, start=1):
        log: "piece pi/K (X acres)"
        raw_pieces.append(_process_piece(piece_geom, fire, def_path, bundle, args.max_cloud))

    piece_datasets = [p for p in raw_pieces if p is not None]
    del raw_pieces; gc.collect()

    if not piece_datasets:
        raise RuntimeError(f"all {len(pieces)} piece(s) failed for {fire_id}")

    if len(piece_datasets) == 1:
        ds = piece_datasets[0]
    else:
        cbi_merged    = merge_arrays([p["CBI"]    for p in piece_datasets], nodata=np.nan)
        cbi_bc_merged = merge_arrays([p["CBI_bc"] for p in piece_datasets], nodata=np.nan)
        ds = xr.Dataset({"CBI": cbi_merged, "CBI_bc": cbi_bc_merged})
        ds.rio.write_crs("EPSG:5070", inplace=True)
        ds["CBI"].rio.write_nodata(np.nan, inplace=True)
        ds["CBI_bc"].rio.write_nodata(np.nan, inplace=True)
        del piece_datasets, cbi_merged, cbi_bc_merged; gc.collect()

else:
    # --- existing pipeline, unchanged ---
    comp  = lazy_fetch_and_composite(fire["bbox"], ...)
    stack = build_stack(comp, def_path);  del comp; gc.collect()
    ds    = predict(stack, bundle);       del stack; gc.collect()
    ds    = to_5070_clip(ds, fire["geometry"])

# --- write block shared by both paths (unchanged) ---
for name in ("CBI", "CBI_bc"):
    ds[name].rio.to_raster(out_base / f"{fire_id}_{name}.tif", ...)
cbi = ds["CBI"].values
del ds; gc.collect()
print(f"  [{tid}] Done {fire_id}: ...")
del cbi; gc.collect()
counts["ok"] += 1
```

The skip-if-exists check and the error handler are unchanged.

### Memory management

- `_process_piece` frees `comp` and `stack` internally before returning `ds`
- After the piece loop, `raw_pieces` is deleted before mosaic
- After mosaic, `piece_datasets`, `cbi_merged`, `cbi_bc_merged` are deleted before write
- At most: list of N piece datasets + 2 merged arrays in memory simultaneously during `merge_arrays`

---

## Reused Functions

| Function | File | Used for |
|---|---|---|
| `lazy_fetch_and_composite` | `aws_utils.py` | Per-piece STAC fetch + composite |
| `build_stack` | `utils.py` | Per-piece predictor stack |
| `predict` | `utils.py` | Per-piece CBI prediction |
| `to_5070_clip` | `utils.py` | Per-piece clip to sub-polygon |
| `transform_bounds` | `rasterio.warp` | Per-piece bbox (EPSG:5070 → 4326) |

---

## Verification

1. **Small fire (unchanged path):** Run with `--event-id` on a fire under 25k acres. Outputs
   should be identical to pre-change behavior. Confirm no `_split_polygon` log appears.

2. **Large fire (split path):** Run with `--event-id` on a known large fire (e.g. a Montana
   fire >25k acres). Confirm the "splitting into K piece(s)" log appears, piece logs appear,
   and the final `_CBI.tif`/`_CBI_bc.tif` are written with no NaN holes at seams.

3. **State-wide run:** `--state MT --workers 4` — confirm mixed large/small fires all
   complete, counts reported correctly at the end.

4. **Partial failure:** Can be simulated by passing a known bad event-id as one piece; confirm
   the fire is counted as `err` but other fires in the run are unaffected.