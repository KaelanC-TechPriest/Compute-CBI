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
