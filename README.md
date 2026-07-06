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

> [!WARNING] **AWS needs credentials**
> AWS script versions require AWS credentials because they download from a
> landsat S3 bucket. You can set credentials with `aws configure`, creating
> access keys, or creating/assigning a IAM role.

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

# Memory optimization

We have issues where some fires are too large for the AWS EC2 instances that we
use. To fix this, out multithreaded AWS script orders the fires from smallest
to largest. After performing a single run, we discovered the maximum size of
fires that four workers can simultaneously compute. Those fire ids can be
cross-referenced in the geopackage to get their sizes as shown below.

## Fires that killed my boy

`sqlite3 /home/kix/work/johnson-lab/Compute-CBI/data/fire_perims/mtbs/mtbs_perims_trimmed.gpkg "SELECT Event_ID, Incid_Name, area_m2, area_acres FROM mtbs_perims_trimmed WHERE Event_ID IN ('MT4643911179319880809','MT4878711426219880906','MT4697011032419901123','MT4803310853419901111');"`

|       Event_ID        |      area_m2         |     area_acres     |
| --------------------- | ------------------   | -------------------|
| MT4878711426219880906 |  136967446.5812026   |  33845.395674426698 |
| MT4643911179319880809 |  145142245.13088387  |  35865.432539965113 |
| MT4803310853419901111 |   95757366.107284427 |  23662.16225488696 |
| MT4697011032419901123 |  103849068.49944615  |  25661.665611183042 |

In total, the script was running on a total of 4.817161e8 square meters or
119034.65608 acres when it crashed.

> [!WARNING] This will not be enough
> The largest fires are around a million acres. Even Montana's largest fire is
> over a million.

## Top 10 largest fires in CA

|       Event ID        |       Fire Name       | Size (acres) |
| --------------------- | --------------------- | ------------ |
| CA3966012280920200817 | AUGUST COMPLEX        | 1,068,793    |
| CA3987612137920210714 | DIXIE                 | 979,807      |
| CA3924012311020180727 | RANCH                 | 427,048      |
| CA3742412156820200816 | SCU LIGHTNING COMPLEX | 405,796      |
| CA3720111927220200905 | CREEK                 | 381,450      |
| CA4009112093120200817 | NORTH COMPLEX         | 316,545      |
| CA3850412233720200817 | HENNESSEY             | 314,230      |
| CA4062112015220120812 | RUSH                  | 306,811      |
| CA3442911910020171205 | THOMAS                | 281,983      |
| CA3293911676620031025 | CEDAR                 | 268,362      |

## Top 10 largest fires in montana

|       Event ID        |        Fire Name        | Size (acres) |
| --------------------- | ----------------------- | ------------ |
| MT4566910646920120625 | ASH CREEK               | 253,414      |
| MT4721710790020170719 | BRIDGE COULEE           | 222,572      |
| MT4559210981020060822 | DERBY                   | 200,993      |
| MT4726811348520170724 | RICE RIDGE              | 171,473      |
| MT4580910676420210808 | RICHARD SPRING          | 168,764      |
| MT4724011275119880625 | CANYON CREEK            | 167,875      |
| MT4625810827219840825 | HAWK CREEK              | 157,778      |
| MT4838610920219911016 | BLAINE C                | 138,192      |
| MT4751410764220030719 | MISSOURI BREAKS COMPLEX | 137,947      |
| MT4573810684020120801 | CHALKY                  | 132,681      |

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
