# One-shot CBI

Compute Composite Burn Index (`CBI`) and bias-corrected CBI (`CBI_bc`) for a
single MTBS fire perimeter, output as GeoTIFFs on the **EPSG:5070** grid (snapped
to the NLCD grid), clipped to the perimeter.

Method: Parks et al. (2019), *Remote Sensing* 11, 1735 — a Random Forest of
Landsat spectral indices + climatic water deficit + latitude.

## What's included
- `pyproject.toml` + `uv.lock` — the [uv] project definition (deps + pinned lock).
- `scripts/cbi_oneshot.py` — the entire pipeline in one self-contained file.
- `parks_2019/data/data_for_ee_model.csv` — RF training table (the model is
  trained from this on first run and cached to `data/model/cbi_rf.joblib`).
- `data/terraclimate/def_19812010_annual.tif` — the static climatic-water-deficit
  input (so it isn't re-downloaded).
- `data/fire_perims/test.gpkg` — example MTBS perimeters to run against.

## Setup & run (uses [uv])
Install uv if you don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`
(see https://docs.astral.sh/uv/). Then, from this directory:
```bash
uv run python scripts/cbi_oneshot.py --gpkg data/fire_perims/test.gpkg --index 80 --out-dir out
```
uv reads `pyproject.toml`/`uv.lock`, creates a local `.venv`, installs the pinned deps
(Python 3.12), and runs — no manual environment step.

- `--index N` selects the 0-based perimeter in the gpkg; or `--event-id <ID>`;
  default is the first wildfire.
- `--year Y` — batch mode: process every wildfire in year Y (mutually exclusive with
  `--event-id`/`--index`). Outputs go under `<out-dir>/<year>/`.
- Single-fire mode writes `<out-dir>/<Event_ID>_CBI.tif` and `<out-dir>/<Event_ID>_CBI_bc.tif`.
- First run trains + caches the model (~15 s); later runs reuse it.
- `--landsat-cache DIR` — directory for cached Landsat scenes
  (default: `data/landsat_cache`). Scenes are cached on first download and reused
  on subsequent runs for the same or overlapping fire footprints.
- `--force-refresh` — ignore the scene cache and re-download everything.
- `--clear-cache` — delete the cache directory before running.

**Needs internet** — Landsat is streamed from the Microsoft Planetary Computer
(free, no account). Each new scene is downloaded once and cached locally; later
runs skip the download for scenes already in the cache.

### Worked example
A forested Idaho 2016 wildfire (`Event_ID ID4634811469320160718`, ~21 km²) included
in `test.gpkg`:
```bash
uv run python scripts/cbi_oneshot.py \
  --gpkg data/fire_perims/test.gpkg \
  --event-id ID4634811469320160718 \
  --out-dir out
```
Expected console summary (≈80 Landsat scenes fetched; takes a few minutes):
```
fire=ID4634811469320160718 state=ID year=2016 DOY=[152,258]
landsat: 80 scenes, grid 422x394 EPSG:32611
out: grid=286x253 crs=EPSG:5070 valid_px=23089 CBI med=1.82 max=2.78
```
Writes `out/ID4634811469320160718_CBI.tif` and `out/ID4634811469320160718_CBI_bc.tif`
(EPSG:5070, 30 m, clipped to the perimeter). The moderate-to-high CBI (median ~1.8)
is the expected signal for a real forested wildfire.

## Caveats
- **Forest-trained model.** Non-forest fires (e.g. sagebrush, grassland) produce
  low/unreliable CBI — the model was built on forested CBI plots.
- Selecting a perimeter by `--index`/`--event-id` runs it regardless of fire type;
  the default-first-wildfire behavior only applies when neither is given.

# ROADMAP

## 06/01/2026 

- Caching is getting successful hits when running the same file twice.

Testing on all fires in data/fire_perims/test.gpkg in the year 2019.

| Year | Normal | Caching (empty) | Caching (full) |
| ---- | ----- | --------------- | --------------- |
| 2006 | ~89.82 min | ~89.82 min |  |
| 2019 | ~18.07 min | ~17.95 min | ~0.19 min |


## 06/02/2026

- Begin testing multithreading.


| Year | Multithreading |
| ------ | ---- |
| 2006 | ~45.07 min |
| 2019 | ~11.04 min |


## 06/04/2026

| METHOD | RUNTIME NJ |
| ----- | -- |
| Sequential, fire-bounds | 224.43 min |
| Sequential, state-bounds | 1387.40 min |

- Both runs were incomplete due to issues with landsat data.
- The state-bounds run started with the first few years cached.

## 06/09/2026

- A run covering NJ for the entire timespan had 0 cache hits. This suggests
    that caching is most likely useless if the scenes are only as large as the fire
    perimeters.

I intend to transition to having multiple different scripts to test the
different methods we've discussed. These will be the scripts in use:

```
scripts/
├── cbi_oneshot.py - The original oneshot file provided by Fred.
├── cbi_yearly.py - Uses the perimeter method for each state in a given year.
├── cbi_statewide.py - Preemptively caches the entire state.
├── cbi_statewide_threaded.py - Same functionality with multithreading.
├── cbi_perimeter.py - Caches each fire as it appears in the list.
├── cbi_perimeter_threaded.py - Same functionality with multithreading.
└── utils.py - A collection of commonly used functions.
```

- Doing perimeter-wide caching results in almost 0 cache hits, so it may be
    worth creating separate, non-caching scripts as well

# Issues

## Bad URL passing to rasterio

Sometimes, this url is getting passed to rasterio which throws the error below.
This is (probably) because of the little `/vsicurl/` at the beginning.

```
rasterio._err.CPLE_OpenFailedError: '/vsicurl/https://landsateuwest.blob.core.windows.net/landsat-c2/level-2/standard/oli-tirs/2022/014/033/LC09_L2SP_014033_20220704_20220802_02_T1/LC09_L2SP_014033_20220704_20220802_02_T1_QA_PIXEL.TIF?st=2026-06-10T21%3A59%3A31Z&se=2026-06-11T22%3A44%3A31Z&sp=rl&sv=2025-07-05&sr=c&skoid=9c8ff44a-6a2c-4dfb-b298-1c9212f64d9a&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2026-06-11T21%3A13%3A23Z&ske=2026-06-18T21%3A13%3A23Z&sks=b&skv=2025-07-05&sig=8OndyN56m6dr5/bwGyXbXxyPikdTAxWbkOmSUoZqri8%3D' not recognized as being in a supported file format.
```
