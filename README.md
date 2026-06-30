# One-shot CBI

Compute Composite Burn Index (`CBI`) and bias-corrected CBI (`CBI_bc`) for a
single MTBS fire perimeter, output as GeoTIFFs on the **EPSG:5070** grid (snapped
to the NLCD grid), clipped to the perimeter.

Method: Parks et al. (2019), *Remote Sensing* 11, 1735 — a Random Forest of
Landsat spectral indices + climatic water deficit + latitude.

## What's included
```
scripts/
├── cbi_oneshot.py - The original oneshot file provided by Fred.
├── cbi_oneshot_aws.py - The oneshot script modified for use on AWS EC2.
├── cbi_yearly.py - Uses the perimeter method for each state in a given year.
├── cbi_statewide.py - Preemptively caches the entire state.
├── cbi_statewide_threaded.py - Same functionality with multithreading.
├── cbi_perimeter.py - Caches each fire as it appears in the list.
├── cbi_perimeter_threaded.py - Same functionality with multithreading.
├── cbi_perimeter_aws.py - Single-threaded, no caching, AWS-compatible.
├── cbi_aws_threaded.py - Multithreaded, no caching, AWS-compatible.
├── aws_utils.py - A collection of functions for running on AWS.
└── utils.py - A collection of commonly used functions.
```

- `pyproject.toml` + `uv.lock` — the [uv] project definition (deps + pinned lock).
- `parks_2019/data/data_for_ee_model.csv` — RF training table (the model is
  trained from this on first run and cached to `data/model/cbi_rf.joblib`).
- `data/terraclimate/def_19812010_annual.tif` — the static climatic-water-deficit
  input (so it isn't re-downloaded).
- `data/fire_perims/` — Should have fire perimeters. Too large to put on github.

## Setup & run (uses [uv])

Install uv if you don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`
(see https://docs.astral.sh/uv/). Then, from this directory:

```bash
uv run python scripts/cbi_oneshot.py --gpkg data/fire_perims/test.gpkg --index 80 --out-dir out
```

uv reads `pyproject.toml`/`uv.lock`, creates a local `.venv`, installs the pinned deps
(Python 3.12), and runs — no manual environment step.

- First run trains + caches the model (~15 s); later runs reuse it.

- `--index N` selects the 0-based perimeter in the gpkg; or `--event-id <ID>`;
  default is the first wildfire.
- Some scripts can cache. Scenes are cached on first download and reused
  on subsequent runs for the same or overlapping fire footprints.

> [!NOTE]
> Caching was abandoned in later script versions due to extremely low cache hit
> rate.

> [!WARNING] **Needs internet** 
> Some scripts stream Landsat from the Microsoft Planetary Computer (free, no
> account). Each new scene is downloaded once and cached locally; later runs
> skip the download for scenes already in the cache.

### Example

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
- Cache hit rate is extremely low.

# Testing

## Testing in the year 2019. (data/fire_perims/test.gpkg)

| Year | Normal | Caching (empty) | Caching (full) |
| ---- | ----- | --------------- | --------------- |
| 2006 | ~89.82 min | ~89.82 min |  |
| 2019 | ~18.07 min | ~17.95 min | ~0.19 min |


## Testing with Multithreading


| Year | Multithreading |
| ------ | ---- |
| 2006 | ~45.07 min |
| 2019 | ~11.04 min |


## Fire-bounds vs State-bounds

| METHOD | RUNTIME NJ |
| ----- | -- |
| Sequential, fire-bounds | 224.43 min |
| Sequential, state-bounds | 1387.40 min |

- Both runs were incomplete due to issues with landsat data.
- The state-bounds run started with the first few years cached.

# Roadmap

## Global
- [-] Check if we can send a single request to get the whole state or even
    multiple years. (no endpoint exists)
- [ ] Compute CBI for *only* wildfires
- [x] We can check before downloading the second previous or next year

## statewide

- [x] Optional start/end date params

## statewide_threaded

- [x] Optional start/end date params

## perimeter

- [x] Optional start/end date params

## perimeter_threaded

- [ ] Optional start/end date params
