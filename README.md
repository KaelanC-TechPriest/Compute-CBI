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
- Outputs `out/<Event_ID>_CBI.tif` and `out/<Event_ID>_CBI_bc.tif`.
- First run trains + caches the model (~15 s); later runs reuse it.

**Needs internet** — Landsat is streamed from the Microsoft Planetary Computer at
run time (free, no account). Each run downloads only the fire's footprint; nothing
is cached between fires.

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
