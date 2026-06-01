# CBI Landsat Scene Caching Plan

## Goal
Modify the `cbi_oneshot.py` script to cache individual Landsat scenes globally so that scenes can be reused across multiple fires (especially within the same year). This will significantly reduce redundant downloads when processing many MTBS fires.

---

## Cache Folder Structure (Recommended)

```bash
data/landsat_cache/
├── scenes/                  # Raw processed scenes (global cache)
│   ├── LC08_043027_20150715.tif
│   ├── LC08_043027_20150731.tif
│   └── ...
└── index.json               # Optional: cache metadata (future enhancement)
```

---

## New Command-Line Arguments

Add the following to the argument parser in `main()`:

- `--landsat-cache` (default: `data/landsat_cache`)
- `--force-refresh` (flag to ignore cache and re-download)
- `--clear-cache` (optional flag to delete existing cache before run)

---

## Required Code Modifications

### 1. New Helper Functions (Add These)

- **`get_scene_cache_path(cache_dir: Path, item)`**  
  Returns the full path where a scene should be saved/loaded based on its unique scene ID.

- **`load_or_download_scene(item, cache_dir: Path, grid, force_refresh: bool)`**  
  Main caching logic: checks if scene exists in cache → loads it, otherwise downloads, processes, saves to cache, and returns the xarray DataArray.

---

### 2. Function Changes

#### `fetch_landsat(bbox, year, start_day, end_day, max_cloud)`
- Should accept new parameter: `landsat_cache: Path, force_refresh: bool`
- Pass the cache directory and `force_refresh` flag down to `_search()` or directly to scene loading.
- After searching for items, loop through items and call `load_or_download_scene()` instead of directly calling `_item_window()`.

#### `_item_window(item, grid)`
- Rename to `_process_scene(item, grid)` or keep name but change purpose.
- This function should now **only** handle the processing logic (windowing, scaling, QA masking, reprojection).
- It should **not** handle downloading or caching — that moves to the new `load_or_download_scene()`.

#### `_search(bbox, start, end, max_cloud, start_day, end_day)`
- Minor change: optionally return more metadata or keep as-is.  
  Main logic stays the same (STAC search).

#### `main()`
- Add new arguments to parser (`--landsat-cache`, `--force-refresh`).
- Create the cache directory at the start (`landsat_cache.mkdir(parents=True, exist_ok=True)`).
- Pass `landsat_cache` and `force_refresh` to `fetch_landsat()`.

---

## Implementation Order (Recommended)

1. Add new command-line arguments in `main()`.
2. Create the two new helper functions (`get_scene_cache_path` and `load_or_download_scene`).
3. Modify `_item_window()` to become a pure processing function.
4. Update `fetch_landsat()` to use the caching layer.
5. Update `main()` to pass cache parameters.
6. Test thoroughly with the same fire multiple times (should hit cache on second run).
7. Test with multiple fires in the same year.

---

## Notes

- Use `ZSTD` compression when saving cache files to save disk space.
- Scene filenames should be based on the official `landsat:scene_id` or `item.id`.
- Handle cases where scene ID is missing gracefully.
