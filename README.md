# One-shot CBI

Compute Composite Burn Index (`CBI`) and bias-corrected CBI (`CBI_bc`) for a
single MTBS fire perimeter, output as GeoTIFFs on the **EPSG:5070** grid (snapped
to the NLCD grid), clipped to the perimeter.

Method: Parks et al. (2019), *Remote Sensing* 11, 1735 — a Random Forest of
Landsat spectral indices + climatic water deficit + latitude.

## What's included
```
scripts/
├── cbi_oneshot.py - The original oneshot file provided by Fred Bunt.
├── aws_oneshot.py - The oneshot script modified for use on AWS EC2, also by Fred.
├── aws_threaded.py - Multithreaded, AWS-compatible.
├── cbi_perimeter_threaded.py - Multithreaded, Planetary Computer.
├── aws_utils.py - A collection of functions for running on AWS.
├── utils.py - A collection of commonly used functions.
└── legacy/
    ├── cbi_yearly.py - Uses the perimeter method for each state in a given year.
    ├── cbi_statewide.py - Preemptively caches the entire state.
    ├── cbi_statewide_threaded.py - Same functionality with multithreading.
    ├── cbi_perimeter.py - Caches each fire as it appears in the list.
    └── cbi_perimeter_aws.py - Single-threaded, no caching, AWS-compatible.
```

- `pyproject.toml` + `uv.lock` — the [uv] project definition (deps + pinned lock).
- `parks_2019/data/data_for_ee_model.csv` — RF training table (the model is
  trained from this on first run and cached to `data/model/cbi_rf.joblib`).
- `data/terraclimate/def_19812010_annual.tif` — the static climatic-water-deficit
  input (so it isn't re-downloaded).
- `data/fire_perims/` — Fire perimeters saved as GeoPackage (.gpkg) files. Too
large to put on github.

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

> [!WARNING]
> **Needs internet**
> The scripts that use Planetary Computer need internet access to be able to
> reach the landsat data source.

> [!WARNING]
> **AWS scripts need credentials**
> The scripts that run on AWS require AWS credentials to download landsat data
> from an S3 bucket. You can set credentials with `aws configure`, creating
> access keys, or assigning a IAM role to your EC2 instance.

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
- Lacks a non-processing-area-mask, so bodies of water will be given a CBI number.
- In a small number of cases, the ±2 year buffer is not enough. Like the fire `TX3417209989919910222`, which has 89% of its area as missing data.
    - As of 08/04/2026, we intend to continue with Parks' method, leaving these fires as they are.
- Selecting a perimeter by `--index`/`--event-id` runs it regardless of fire type;
  the default-first-wildfire behavior only applies when neither is given.
- The `aws_threaded.py` script spawns `workers × 4 × 4` threads total.
    - 4 threads per fire or fire piece
    - 4 threads per piece (for parallelized band downloads)

## Optimizations and Discoveries

### Why doesn't it cache landsat data?

Caching showed minimal improvements due to extremely low hit-rate. Examine the
following table to see that a full cache is significantly faster than a
cacheless run. However, an empty cache shows little (or no) improvements.

| Year | Normal | Caching (empty) | Caching (full) |
| ---- | ----- | --------------- | --------------- |
| 2006 | ~89.82 min | ~89.82 min |  |
| 2019 | ~18.07 min | ~17.95 min | ~0.19 min |

After testing the program's cache-hit rate, we observed that it was
consistantly near 0%. We tried preemptively downloading the entire state for
each year to ensure a 100% hit-rate, but the download time far outweighed the
savings from cache hits. See the following table to see the time it took to
compute every fire in New Jersey (from 1986 to 2020).

| METHOD | RUNTIME NJ |
| ----- | -- |
| Sequential, fire-bounds | 224.43 min |
| Sequential, state-bounds | 1387.40 min |

- Both runs were incomplete due to issues with landsat data.
- The state-bounds run started with the first few years cached.

Caching was abandoned in later versions of the scripts (after commit d727762).
Currently, only scripts in `scripts/legacy` have caching.

### Why split on fire polygon bounds?

First, we found that the script was crashing due to lack of memory. At the time
of one crash, the script had four workers running on the fires shown in the
table. Querying the burned area of each fire (using the shown sql query) gives
the area burned.

`sqlite3 .../data/fire_perims/mtbs/mtbs_perims_trimmed.gpkg "SELECT Event_ID, Incid_Name, area_m2, area_acres FROM mtbs_perims_trimmed WHERE Event_ID IN ('MT4643911179319880809','MT4878711426219880906','MT4697011032419901123','MT4803310853419901111');"`

|       Event_ID        |      area_m2         |     area_acres     |
| --------------------- | ------------------   | -------------------|
| MT4878711426219880906 |  136967446.5812026   |  33845.395674426698 |
| MT4643911179319880809 |  145142245.13088387  |  35865.432539965113 |
| MT4803310853419901111 |   95757366.107284427 |  23662.16225488696 |
| MT4697011032419901123 |  103849068.49944615  |  25661.665611183042 |

In total, the script was running on 4.817161e8 square meters or 119034.65608
acres when it crashed. This is problematic because Montana's largest fire is
over a million acres of burned area, which we do not have enough memory for.

The script was then made more efficient by checking if the area burned was
greater than a threshold defined by 100 thousand acres (roughly the total seen
above) divided by the number of workers. If the area burned was greater, the
fire was split in half recursively by the bound's longest axis until the burned
area was less than 25k acres per piece. The script was later updated to check
if the area within the *bounds* of the fire was greater than the threshold. This
decision was made in the light of fire complexes (collections of fires entered
as one fire), which have a much larger bounded area, but relatively small
burned area. This change was the key to accurately controlling the script's
memory usage.

# Roadmap
