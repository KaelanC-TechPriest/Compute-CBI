# Issue: composite() hangs on MT4548111089520130816

## Symptom

The pipeline stalls silently after this debug output with no further progress:

```
debug [composite.wmean]: computing mean over 26 scenes...
```

The line `mean done` never prints, meaning `idx.isel(time=m).mean("time", skipna=True)` never returns.

## Confirmed hang location

`utils.py` — `composite()` → `wmean()` → `idx.isel(time=m).mean("time", skipna=True)`

## Memory breakdown

Observed cube shape: `time=26, band=6, y=1557, x=1293, dtype=float32`

| Array | Calculation | Size |
|---|---|---|
| `cube` (raw scenes) | 26 × 6 × 1557 × 1293 × 4 B | ~1.25 GB |
| `idx` from `_scene_indices` | 26 × 5 × 1557 × 1293 × 4 B | ~1.04 GB |
| NaN mask (`skipna=True`) | 26 × 5 × 1557 × 1293 × 1 B | ~260 MB |
| **Peak for one piece** | | **~2.6 GB** |

With multiple workers running simultaneously, total RAM demand is `2.6 GB × n_workers`, which easily exhausts physical RAM and causes swap thrashing on an 8 GB machine.

## Contributing factors

### 1. `skipna=True` doubles intermediate memory

`np.mean` is not used. Instead xarray materializes a full boolean NaN mask of the same shape as `idx`, then does element-wise accumulation to skip NaN values. This is the direct cause of the excess memory at this step.

### 2. `wmean` is called twice unconditionally

In `composite()`:

```python
pre = wmean({year - 1}).combine_first(wmean({year - 2, year - 1}))
```

Both calls execute regardless of whether the first result has any NaN pixels. The second call covers a potentially larger scene set (Y-2 + Y-1), making it even more expensive. The lazy fallback logic already implemented in `lazy_fetch_and_composite._phase` handles this correctly — `composite()` does not.

### 3. 26 scenes for a single year is high

26 scenes passed the DOY window and cloud filter for Y-1=2012. Each scene adds another 1557×1293 slice to the mean computation. The large scene count combined with the large bounding box grid drives peak memory up.

### 4. Grid size reflects bounding box, not fire perimeter

The 1557×1293 grid (≈46 km × 39 km at 30 m resolution) is the padded bounding box of the fire, not the fire polygon itself. A fire with a compact perimeter but elongated bounding box will have many no-data pixels in the grid, making the computation wasteful.

## Proposed fixes

- **Drop `skipna=True`** — NaN pixels from cloud masking are expected; use `skipna=False` and accept that pixels missing in any scene will be NaN in the composite. Alternatively, filter to only scenes with valid coverage over the fire bbox before concatenating.
- **Lazy second `wmean` call** — only call `wmean({year - 2, year - 1})` if the first result has NaN pixels, mirroring the logic already in `_phase`.
- **`.load()` before `_scene_indices`** — ensure the cube is a single contiguous numpy allocation before index arithmetic, avoiding scattered memory access across per-scene arrays.
- **Reduce scene count** — tighten the DOY window or add a maximum-scenes cap before building the cube, preferring the least-cloudy scenes.
