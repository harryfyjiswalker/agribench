# YieldSAT evaluation protocol

Status: **agreed 2026-11-09, nothing run yet.** This page sets how every AgriBench model is scored on YieldSAT so results are
comparable with the paper and with each other. Sections marked **[paper]** follow Miranda et al. (2026)
exactly; sections marked **[AgriBench]** fill gaps the paper leaves open or go beyond it. All decisions are
listed in section 4.

Paper: Miranda, Pathak et al., *YieldSAT: A Multimodal Benchmark Dataset for High-Resolution Crop Yield
Prediction*, arXiv:2604.00940 (2026). Website: https://yieldsat.github.io/. The authors publish no fold files and
no benchmark training code (checked 2026-11-09); the only code is a demo notebook that uses `GroupKFold` on
`field_shared_name` and calls `farm_identifier` the "region".

> The data is under a custom licence and contains field locations. This repository is public, so nothing
> field-level (names, coordinates, fold lists, predictions) is committed here. Fold files and run outputs live
> under `/lustre/shared/agriscience-benchmark/`; results JSON in `results/` only carries aggregate metrics and
> hashes.

---

## 1. What is in the dataset

Location on Anunna: `/lustre/shared/agriscience-benchmark/datasets/yieldsat/`

| Folder | Contents |
|---|---|
| `Preprocessed/<Country>/<Country>/merge_s2-soil-dem-weather-coords.nc` | the ML-ready version: one row per labelled 10 m pixel (NetCDF4/HDF5, 7–63 GB per country) |
| `Raw/<Country>/<Country>/<field>/` | one folder per field: Sentinel-2 GeoTIFFs per date, SCL masks, soil, DEM, weather CSV, yield rasters, metadata JSON |
| `Both2.zip` | the original download (24.7 GB), readable only by its owner |

### Fields per country and crop (matches the paper's Table 2: 2,173 fields)

| | Corn | Rapeseed | Soybean | Wheat | Total | Labelled pixels |
|---|---:|---:|---:|---:|---:|---:|
| Argentina | 185 | – | 440 | 126 | 751 | 5,325,807 |
| Brazil | 118 | – | 293 | 140 | 551 | 4,260,262 |
| Germany | – | 111 | – | 188 | 299 | 609,645 |
| Uruguay | – | – | 572 | – | 572 | 2,177,206 |
| **Total** | 303 | 111 | 1,305 | 454 | 2,173 | 12,372,920 |

This gives nine country–crop **subsets** (ARG-C, ARG-S, ARG-W, BRA-C, BRA-S, BRA-W, GER-R, GER-W, URG-S); the
paper reports every result per subset, never pooled.

Years, farms and quality per subset:

| Subset | Years | Fields per year | Farms | Quality good / average / bad |
|---|---|---|---:|---|
| ARG-C | 2017–2024 (8) | 9, 8, 7, 13, 26, 70, 37, 15 | 29 | Argentina overall: 695 / 54 / 2 |
| ARG-S | 2017–2024 (8) | 8, 28, 37, 71, 104, 78, 79, 35 | 44 | |
| ARG-W | 2017–2023 (7) | 6, 12, 6, 32, 42, 13, 15 | 21 | |
| BRA-C | 2017–2020, 2023–2024 (6) | 11, 10, 13, 20, 44, 20 | 7 | Brazil overall: 109 / 308 / 134 |
| BRA-S | 2017–2024 (8) | 31, 48, 48, 28, 42, 43, 24, 29 | 9 | |
| BRA-W | 2017–2023 (7) | 16, 23, 25, 18, 19, 21, 18 | 6 | |
| GER-R | 2016–2022 (7) | 21, 27, 15, 15, 19, 9, 5 | 6 | Germany overall: 150 / 120 / 29 |
| GER-W | 2016–2022 (7) | 27, 29, 30, 49, 16, 20, 17 | 6 | |
| URG-S | 2018–2022 (5) | 236, 163, 12, 75, 86 | 10 | Uruguay overall: 56 / 349 / 167 |

Each country has exactly **one data provider** (Argentina DUP1, Brazil DUP2, Germany DUP3, Uruguay DUP4), so the
paper's "region" can only mean a farm within a provider (see 2.3).

### The preprocessed file (what models read)

| Variable | Shape | Meaning |
|---|---|---|
| `sample` | (pixels, 24, 120) float32 | input features: 24 time steps × 120 bands, NaN where missing |
| `target` | (pixels,) float32 | dry ("scaled") yield in t/ha for that pixel, no missing values |
| `times` | (pixels, 24) float64 | day offset of each time step for that pixel, NaN for empty steps |
| `field_shared_name`, `farm_identifier`, `crop`, `year`, `country`, `seeding_date`, `harvesting_date`, `seeding_date_type` | (pixels,) int | integer codes; the code → label table is in each variable's attributes |
| `row`, `col` | (pixels,) int | pixel position inside its field's raster |
| `index` | (pixels,) str | unique pixel ID (UUID) |
| `stats-mean/std/min/max` | (120,) | per-band statistics over the whole country file |
| global attributes | | `<field>_<>_yield_ground_truth`: the farmer-reported field yield |

The 120 bands are:

- **Sentinel-2 (12):** B01–B09, B11, B12, B8A (L2A reflectance × 10,000). They vary over time and are NaN when
  no image falls in that step (about 72% of S2 slots are empty).
- **Weather, ERA5 (4):** `temp_mean`, `temp_max`, `temp_min` (Kelvin), `total_prec`. They vary over time.
- **Topography, SRTM (5):** `dem`, `slope`, `aspect`, `curvature`, `twi`. Static.
- **Soil, SoilGrids (96):** cec, cfvo, clay, nitrogen, phh2o, sand, silt and soc at six depths, each with an
  `_uncertainty` band. Static.
- **Coordinates (3):** `coord_x`, `coord_y`, `coord_z`. Static.

The paper's "S2" input is the 12 S2 bands. Its "S2 + ADM" input is all bands. The SCL cloud mask exists only in
`Raw/`, not in the preprocessed file.

### Things that differ from the paper or need care

1. **Quality labels.** Every field has a label. The paper says "a total of 332 fields", but 332 is the number
   of *bad* fields. The full count is 1,010 good, 831 average and 332 bad.
2. **Standard moisture for soybean** is 13% in the metadata, but 15% in the paper's Table 6. This doesn't matter
   for us because `target` is already dry yield.
3. **Do not use the metadata `yield_ground_truth` or `area_*` as targets.** They are farmer-reported, sometimes
   stored as strings, and the area can disagree a lot with the mapped area (e.g. 32 ha reported vs 11.2 ha
   mapped). The field-level target is the mean of the field's pixel `target` values (2.4).
4. **Values of 65,535** appear in the S2 band maxima of Argentina, Brazil and Uruguay. They are saturated or
   nodata values and are treated as missing.
5. **The 24 time steps are not calendar-aligned.** Each field's observations sit in a block of the 24 slots,
   padded with NaN on both sides. `times` holds each slot's date as days since one reference date per country
   file, exactly as the CF `units` attribute says (e.g. "days since 2019-09-22" for Germany), so offsets can be
   negative. (An earlier version of this note called the attribute misleading; that was wrong. Checked on
   2026-11-09: reference + offset falls on the Raw S2 acquisition dates.)

---

## 2. Evaluation protocol

### 2.1 Unit of evaluation [paper]

- Everything is done **separately per subset** (the nine above). There is no pooled score.
- Models predict **per pixel**. Each pixel is treated as an independent training sample, as in the paper.
- All pixels of a field always stay on the same side of a split. The split unit is the **field**
  (`field_shared_name`).

### 2.2 Standard cross-validation, CV10 [paper]

- Use **10-fold cross-validation, stratified and grouped**: pixels are grouped by field and stratified by
  region (farm).
- Implementation: on the table of fields in the subset, `StratifiedGroupKFold(n_splits=10, shuffle=True,
  random_state=42)`, with `y = farm_identifier` and `groups = field_shared_name`. Folds are made once, frozen
  to a file, and reused by every model (2.6).

### 2.3 Out-of-distribution splits [paper]

- **Leave-one-year-out (LOYO):** one fold per year in the subset. Train on all other years, test on the
  held-out year. That's 5–8 folds, depending on the subset.
- **Leave-one-region-out (LORO):** one fold per region. A region is a **group of neighbouring farms**. The
  paper defines a region as "a set of fields belonging to a single farmer or to a local data provider", and
  its weight-space figure shows 8 regions for Argentina. Using every farm as its own region would give up to
  44 folds, many with only a handful of test fields, so [AgriBench] farms are grouped by location:
  1. For each subset, take each farm's location as the mean centroid of its fields, in km (equirectangular
     projection around the subset's mean latitude).
  2. If the subset has more than 8 farms, cluster the farms into 8 groups with Ward agglomerative clustering
     (deterministic, no random seed). Otherwise each farm starts as its own region.
  3. While any region has fewer than 10 fields (and more than 2 regions remain), merge the smallest such
     region into the region whose centroid is nearest.
  4. Farms are never split, so all of a farm's fields are always in the same region.

  This gives 3–8 regions per subset, each with at least 12 fields (checked on the data, 2026-11-09):

  | Subset | Farms | Regions | Fields per region |
  |---|---:|---:|---|
  | ARG-C | 29 | 6 | 52, 49, 24, 24, 23, 13 |
  | ARG-S | 44 | 8 | 115, 94, 84, 55, 39, 24, 17, 12 |
  | ARG-W | 21 | 3 | 58, 55, 13 |
  | BRA-C | 7 | 4 | 56, 28, 18, 16 |
  | BRA-S | 9 | 6 | 94, 77, 42, 35, 27, 18 |
  | BRA-W | 6 | 5 | 47, 41, 22, 18, 12 |
  | GER-R | 6 | 6 | 30, 22, 17, 15, 15, 12 |
  | GER-W | 6 | 3 | 124, 48, 16 |
  | URG-S | 10 | 6 | 283, 114, 59, 55, 35, 26 |

  The region assignment is frozen in the fold files (2.8), so it never changes between runs.

### 2.3b Final evaluation: leave-one-country-out (LOCO) [AgriBench: beyond the paper]

This is AgriBench's **headline YieldSAT result**: can a model trained in some countries predict yield in a
country it has never seen? The paper doesn't report it. It says cross-country settings "led to model collapse"
without domain adaptation, which is exactly why it's the most informative test for foundation-model inputs.

- **Per crop, across countries.** Only crops grown in more than one country can be tested:

  | Crop | Countries (fields) | Folds | Training countries per fold |
  |---|---|---:|---:|
  | Soybean | Argentina (440), Brazil (293), Uruguay (572) | 3 | 2 |
  | Wheat | Argentina (126), Brazil (140), Germany (188) | 3 | 2 |
  | Corn | Argentina (185), Brazil (118) | 2 | 1 |
  | Rapeseed | Germany only | – | – |

  Each fold holds out one country, trains on all fields of that crop from the other countries, and scores
  every pixel of the held-out country.
- **Same metrics as 2.4, but reported per held-out country.** With only 2–3 folds, a mean ± std hides too
  much. The leaderboard shows R² and RMSE for each held-out country at field and pixel level, plus the mean
  across countries.
- **Reference row.** Every LOCO table includes a "training-mean" baseline: predict the training countries'
  mean yield everywhere. This shows the size of the country shift, and a model that doesn't beat it has
  collapsed.
- **Nothing from the held-out country may leak into training.** Any normalisation or imputation is fitted on
  the training countries' pixels only. The per-country `stats-*` variables in the files must not be used,
  because each one includes the held-out country's own data.
- **Treated as a final test, like PASTIS-R fold 5.**
  - Model and input choices are made on CV10, LOYO and LORO. LOCO is never used to choose.
  - A LOCO run needs `experiment.confirm_test=true`.
  - Each run is appended to `/lustre/shared/agriscience-benchmark/runs/yieldsat/loco-ledger.jsonl`, so we
    can see how many configurations were tried.
- **Known differences to keep in mind when reading results:**
  - Seasons are not aligned: German wheat is sown in autumn and harvested in summer, Argentine wheat runs
    roughly May to December, and the 24 time steps are relative to each field's own season.
  - Yield levels differ by country (Fig. 4 of the paper).
  - The input includes coordinates, which point outside the training range for a new country.
  - Data quality differs a lot. Argentina is 93% "good"; Uruguay and Brazil are mostly "average" or "bad".
- **How it is run (from 2026-11-09).**
  - `python -m tools.sweep_yieldsat --name loco1 --schemes loco --inputs ... --models ...` plans one run per
    crop × input × head under `<sweep>/<crop>/loco/<input>/<head>/`, adds `experiment.confirm_test=true`, and
    accepts `--resource-scale 2`, since training sets pool countries.
  - We run LOCO for **every** input and head in the development tables, not a shortlist. All of them are
    reported, so no LOCO number is picked after the fact, and the ledger still logs every run.
  - Inputs on a per-subset time grid (s2-aligned, s2-gdd, biophys, ARC) have different columns per country,
    e.g. German wheat's season is longer. LOCO keeps only the columns that every country of the crop shares, by
    name (the common days after sowing). `n_features` in the provenance records the width actually used.
  - The aggregate table has one row per crop (pooled over held-out countries) and one per held-out country,
    labelled `<crop>><COUNTRY>`, e.g. `wheat>GER`.

### 2.4 Metrics [paper]

For every fold, compute both metrics at both levels:

- **Pixel level:** RMSE (t/ha) and R² over all test pixels.
- **Field level:** average predicted and true yield over each test field's pixels, then compute RMSE and R²
  over the test fields.

**Report** the mean ± standard deviation across folds for each metric (`np.std`, ddof=0). This matches the
paper's "metrics are presented as the average across the folds" and its ± values. Pooled metrics, computed over
all out-of-fold predictions at once, are saved alongside for reference but not ranked.

R² uses the test fold's own mean (`sklearn.metrics.r2_score`). With few test fields per fold, field-level R²
can swing widely or go negative. The paper shows this too (e.g. ±5.28).

### 2.5 Inputs [paper + AgriBench]

| Input name | Bands | Matches |
|---|---|---|
| `yieldsat/s2` | 12 S2 bands × 24 steps | the paper's "Sentinel-2" rows |
| `yieldsat/s2-adm` | all 120 bands × 24 steps | the paper's "S2 + ADM, input fusion" rows |
| `latlon` | pixel latitude, longitude (WGS84) | AgriBench: plain-coordinate control for the location encoders |
| `mind` | MIND location encoder, 3072-d (Corley et al. 2026) | AgriBench |
| `satclip` | SatCLIP ViT16-L40 location encoder, 256-d (Klemmer et al.) | AgriBench |
| `galileo` | Galileo-Base, 768-d, from the pixel's S2 series | AgriBench |
| `aef` | AlphaEarth Foundations annual embedding, 64-d | AgriBench |
| `tessera` | TESSERA annual embedding, 128-d | AgriBench |
| `presto` | Presto, 128-d, from the pixel's in-season S2 series (monthly composites) | AgriBench |
| `olmoearth` | OlmoEarth-v1.2-Base, 768-d, from the pixel's in-season S2 acquisitions at their real dates | AgriBench |
| `anysat` | AnySat-Base, 768-d, from in-season S2 tiles of the field (pixel token with field context) | AgriBench |
| `prithvi` | Prithvi-EO-2.0-300M-TL, 1024-d, from 4 in-season S2 dates of the field's Raw raster (224 px windows) | AgriBench |
| `s2-aligned` | 12 S2 bands interpolated onto a fixed grid of days since sowing (every 15 days) | AgriBench: season-aligned version of `s2` |
| `s2-gdd` | 12 S2 bands interpolated onto a fixed grid of growing degree days since sowing | AgriBench: thermal-time version of `s2-aligned` |
| `s2-harmonic` | per band, 5 harmonic-regression coefficients (2 harmonics) over days since sowing | AgriBench: compact gap-robust summary |
| `biophys` | LAI, FAPAR, FCOVER, canopy chlorophyll, canopy water (SNAP S2 biophysical processor) on days since sowing | AgriBench: physically based canopy variables |
| `arc` | ARC retrieval: 15 crop-model parameters, peak LAI and the LAI trajectory on days since sowing | AgriBench, internal testing: physics-based season retrieval |
| `arcvae` | 11 ARC crop-model parameters (canopy scaling and phenology timing), posterior mean and SD from ARC-VAE | AgriBench, internal testing: physics-guided season summary |
| `weather-pheno` | 69 agronomic weather summaries per field in thermal-time crop-stage windows from sowing (GDD, heat, frost, rain, water balance) | AgriBench: year-level weather keyed to the crop calendar |
| `wofost` | WOFOST 7.2 (PCSE) per field: potential and water-limited yield and biomass, max LAI, flowering and maturity, water stress (13) | AgriBench: process-based crop model |
| `wofost-da`, `wofost-da-snap` | WOFOST 7.2 water-limited ensemble with field LAI assimilated by an EnKF (ARC or SNAP LAI), 12 | AgriBench: crop model + satellite hybrid |
| combinations | blocks joined with `+`, e.g. `galileo+mind`, `s2+mind` | AgriBench |

Tabular heads get each pixel as a flat vector (24 × bands). The static bands are taken once instead of 24
times. Missing values stay NaN: heads that can't handle NaN get training-fold median imputation inside the
pipeline, so nothing is fitted on test data.

**Feature caches.** `tools/build_yieldsat_inputs.py` (run with `sbatch prepare.sbatch
tools.build_yieldsat_inputs --country <Country>`) streams each country file once and writes, per subset, to
`/lustre/shared/agriscience-benchmark/features/yieldsat/v1/`:
- `<subset>.npy`: float32, pixels × 488;
- `<subset>-meta.npz`: field, target, row/col and pixel UUID per row;
- `<subset>.json`: feature names, the column lists for `s2` (288) and `s2-adm` (488), and provenance.

Columns are the 16 time-varying bands at each of the 24 steps (`<band>_t<step>`), then the 104 static bands.
S2 values of 65,535 become NaN. Caches are never overwritten. On Germany, every one of 12,000 spot-checked
values matched the source file.

**Representation caches** (2026-11-09). Every representation lives in its own folder under
`features/yieldsat/`, with rows aligned to `v1/<subset>-meta.npz` (each `<subset>.json` stores that file's
SHA-256 and the runner refuses a mismatch):

- `latlon/` (`tools/build_yieldsat_latlon.py`): the preprocessed file has no usable coordinates (`coord_x/y/z`
  are a transformed version), so each pixel centre (row + 0.5, col + 0.5) is mapped through its field's Raw S2
  GeoTIFF transform and reprojected to WGS84. Check: every field's mean pixel position lies within 1 km of the
  metadata centroid (worst case 0.95 km, in ARG-W, where yield maps cover only part of some fields; Brazil and
  Germany are within 90 m).
- `mind/`, `satclip/` (`tools/embed_yieldsat_location.py`): location encoders only see coordinates, so pixels
  are snapped to a 0.001° grid (about 110 × 70 m), each of the 73k occupied cells is embedded once, and pixels
  index into that table. MIND uses its full 3072-d trunk features (upstream eb25937, weights
  `taylor-geospatial/MIND@1437c4d`); SatCLIP uses the location encoder of `microsoft/SatCLIP-ViT16-L40`.
  Neither takes a date, so a location gets the same embedding in every year.
- `galileo/` (`tools/embed_yieldsat_galileo.py`): Galileo-Base on each pixel's 24-slot S2 series as a 1 × 1
  time series (the 10 S2 bands Galileo knows, plus NDVI), with each slot's month token taken from
  the file's time reference + slot offset (checked against the Raw S2 acquisition dates on 25 fields per subset). Weather,
  topography and location are left out on purpose, so `galileo` is comparable to `s2` and location only enters
  through the location encoders. Mean of the encoder's tokens (Galileo `average_tokens`), 768-d.
- **Time-series foundation models, in-season** [AgriBench]. `aef` and `tessera` are annual products: one
  embedding per calendar year, computed by their providers from the whole year. The models below instead see only
  the field's own season, i.e. the S2 observations in its 24 slots, at their real dates (file time reference +
  slot offset, checked against the Raw acquisition dates on 25 fields per subset, as for `galileo`). This is the
  "annual summary vs in-season embedding" comparison. Only S2 goes in: no weather, topography or location, so
  they compare with `s2` and `galileo`. Upstream code and weights are pinned under `upstream/<name>/` and loaded
  with `weights_only=True`.
  - `presto/` (`tools/embed_yieldsat_presto.py`): Presto (nasaharvest/presto 11e207a, `default_model.pt`), pixel.
    Presto was pretrained on monthly composites, so each pixel's valid observations are averaged per calendar month
    over the field's span of months (median 8–12, at most 13). Each month gets its calendar month token, and months
    without an observation are masked. Input: 10 S2 bands / 10,000 plus NDVI. S1, ERA5, SRTM and Dynamic World are
    masked. Presto always adds a location token that cannot be masked, so lat/lon is fixed at (0, 0) for every
    pixel. Output: the encoder's mean token, 128-d.
  - `olmoearth/` (`tools/embed_yieldsat_olmoearth.py`): OlmoEarth-v1.2-Base (`allenai/OlmoEarth-v1_2-Base` at Hub
    revision 2e99a73, code allenai/olmoearth_pretrain 13ca06f), pixel. Each pixel is a 1 × 1 image with patch size 1,
    which the upstream inference guide supports. Its valid acquisitions (median 6–8 per pixel) go in at their real
    dates, which the model reads through temporal RoPE. Input: all 12 L2A bands with the upstream "computed"
    normalisation. The model takes at most 12 timesteps; the 0.8% of ARG-C pixels with 13 have their two
    closest-in-time observations averaged. Only valid observations are passed: the encoder's inference path attends
    to every token it is given, including masked ones. Pixels are batched by their number of observations. Output:
    mean of the output tokens over time, 768-d.
  - `anysat/` (`tools/embed_yieldsat_anysat.py`): AnySat-Base (gastruc/AnySat 5f6f475, `g-astruc/AnySat` revision
    63f2521, `models/AnySat.pth`), patch with field context. Each field's raster is rebuilt from the v1 row/col and
    cut into 24 × 24-pixel tiles (240 m) with AnySat patch size 10 m: a per-pixel temporal encoder (L-TAE), then
    a spatial transformer across the tile. The output is the pixel's own patch token (768-d). The preprocessed
    file only has yield-map pixels, so the context is the field itself; cells outside the yield map get the field's
    mean series. AnySat takes one date list per tile and has no per-timestep mask. So slots where fewer than 10% of
    the field's pixels are valid are dropped, and the remaining gaps in each pixel are filled by linear
    interpolation in time. Dates are days since the 1st of the month of the field's first kept slot (as AnySat's
    PASTIS loader counts from a season start). Band normalisation: mean and std of valid S2 values from a seeded
    sample pooled over all nine subsets, because AnySat normalises each dataset with its own statistics. The
    sample includes cloudy observations (the preprocessed file has no cloud mask), which inflates the std of the
    visible bands; AnySat's input MLP applies GroupNorm after its first layer, so a uniform rescale mostly cancels.
  - `prithvi/` (`tools/embed_yieldsat_prithvi.py`): Prithvi-EO-2.0-300M-TL (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`
    revision 63adbd3), a multi-temporal patch model: 3D patches of 1 frame × 16 × 16 px, pretrained on 4-frame HLS
    stacks with (year, day of year) per frame. Space: the field's Raw S2 raster (the bounding box with its real
    surroundings) via `tools/yieldsat_patches.py`, in 224 × 224 windows at native 10 m. Each labelled pixel uses the
    window whose centre is nearest, and takes the last-layer token grid averaged over the 4 frames, sampled
    bilinearly at the pixel centre. One token covers 160 m, so within-field detail comes only from the
    interpolation. Time: 4 label-free dates from the field's season, sorted: the first clear in-season acquisition
    plus the helper's green-up, peak and senescence dates (clear = SCL clear on ≥ 90% of the labelled pixels; the
    date can repeat when there are few clear dates). In GER-R the median dates are 8, 136, 265 and 295 days after
    sowing. Prithvi applies its temporal encoding only for its pretraining frame count of 4. Bands: B02, B03, B04,
    B8A, B11, B12 as HLS B02–B07, in DN, with the checkpoint's mean and std. The TL variant's location input is not
    used. Output: 1024-d.
- `aef/` (`tools/extract_yieldsat_aef.py`): the 64-d AlphaEarth annual embedding of the calendar year of the
  field's harvest date, read with HTTP range requests from the Source Cooperative mirror. AEF starts in 2017,
  so the 48 German fields harvested in 2016 have no AEF value (imputed by the runner, and counted in the json).
- `s2-aligned/` (`tools/build_yieldsat_s2_aligned.py`) [AgriBench]: the raw `s2` slots are not tied to the season
  (1, note 5). Checked on 2026-11-09: in BRA-C, slot 12 is 30 days before sowing in 2017 but 77 days after it in
  2023; in ARG-C it is about 110 days after sowing in most years but 24–42 days in 2021 and 2023. A model that
  learns from slot positions can then misread a new year, which matters for LOYO (not for CV10, where test
  fields share years with training fields). `s2-aligned` puts each pixel's series on one axis, days since the
  field's sowing date: per pixel and band, linear interpolation between its valid observations onto -15, 0, 15,
  … days, up to the subset's 95th percentile of the last observation day (capped at 330). Grid points outside
  the pixel's observed range stay NaN. Only the sowing date is used, never the harvest date or yield. It is an
  extra input beside `s2`, not a replacement, so the paper's input is still reported as the paper defines it.
- `s2-gdd/` (`tools/build_yieldsat_s2_gdd.py`) [AgriBench]: as `s2-aligned`, but the axis is growing degree days
  since sowing, from each field's daily ERA5 file in `Raw/.../weather/`. Daily GDD = max(0, (clip(Tmax) +
  clip(Tmin)) / 2 - Tbase), clipped to [Tbase, 30 °C], with Tbase 0 °C (wheat), 5 °C (rapeseed), 10 °C (corn,
  soybean). The grid starts at -50 °C·d and has about 16 equal steps up to the subset's 95th percentile of the
  last observation (steps of 125–225 °C·d). Motivation: crop development follows heat accumulation more closely
  than the calendar (Nyborg et al. 2022, thermal positional encoding; Franch et al. 2026, GDD-normalised S2 yield
  models; Osei et al. 2026 found days-after-planting slightly better than GDD with 1 km temperatures and a fixed
  planting date). Every field's weather record covers its season.
- `s2-harmonic/` (`tools/build_yieldsat_s2_harmonic.py`) [AgriBench]: per pixel and band, least-squares fit of
  c0 + a1 cos + b1 sin + a2 cos + b2 sin (period 365 days, t = days since sowing), with a ridge penalty of 0.5
  (in observations) on the harmonic terms; 60 features. A band needs 5 valid observations, otherwise NaN
  (77–98% of pixels fitted, lowest in GER-W). On simulated noisy seasons the ridge makes the coefficients 2–4×
  more stable than plain least squares at a small cost in fit.
- `biophys/` (`tools/build_yieldsat_biophys.py`) [AgriBench]: each observation converted to LAI, FAPAR, FCOVER,
  CCC and CWC with the ESA SNAP Sentinel-2 Toolbox biophysical networks (s2tbx 3.0 "S2A" coefficients, GPL-3.0,
  pinned commit, downloaded to `upstream/snap-biophysical/`), then placed on the `s2-aligned` grid. The
  implementation matches SNAP's own test cases (max error 3e-4 for LAI). YieldSAT has no acquisition geometry,
  so the sun angles are computed for 10:30 local solar time, the view zenith is set to 5° and the relative
  azimuth to 90°. This is the operational BVNET baseline; PROSAIL-VAE (Zérah et al. 2024; De Sousa & Inglada
  2026) would be the physics-guided successor. A Transformer-PROSAILVAE checkpoint is public, but the code for
  its latent head is not, and our reconstruction did not recover simulated LAI (R² 0.05), so it is not used.
- `arc/` (`tools/build_yieldsat_arc.py`) [AgriBench, internal testing]: ARC retrieval (MarcYin/ARC, pinned
  commit, no licence) following the AgriBench author's county-level ARC pipeline, adapted to pixels. Per field:
  Whittaker IRLS outlier removal on B03, B8A and B11 (d = 2, 3 iterations, Tukey bisquare, lambda by V-curve over
  10–1000, median of the three bands). The cleaned observations are then interpolated onto the `s2-aligned` grid,
  and ARC sees day of year = 30 + days since sowing, so one reference library (2^17 simulated seasons) and one
  nearest-neighbour index serve a whole subset; the original pipeline built both per county-year. Start of season
  and season length come from the subset's median EVI2 curve (threshold 0.5). Retrieval: k = 300 neighbours, then
  ARC's assimilation (best 50, inverse-distance weights), band uncertainty 5%. Features: 7 canopy, 4 phenology and 4
  soil parameters, peak LAI and LAI per grid step. Archetypes: corn → maize, soybean → soybean, wheat → wheat,
  rapeseed → wheat (ARC has no rapeseed archetype). Same geometry approximation as `biophys`, with the sun angle
  for the subset's median latitude and sowing date. About 3 min per 100k pixels on 8 CPUs.
- `arcvae/` (`tools/build_yieldsat_arcvae.py`) [AgriBench, internal testing]: the encoder of ARC-VAE
  (harryfyjiswalker/arc-vae, GPL-3.0, maize checkpoint), a transformer VAE trained on synthetic Sentinel-2
  seasons from the ARC archetype crop model. Each pixel's whole season goes in, and the posterior mean and SD of
  11 ARC parameters come out (N, Cab, Cm, Cw, LAI, ALA, Cbrown scaling; growth speed, start, senescence speed,
  end): 22 features. Same geometry approximation as `biophys`. The model expects a northern-hemisphere maize
  calendar, so each observation's day of year is replaced by 120 + days since sowing (observations outside
  1–365 dropped), which makes the phenology outputs relative to sowing. Crops other than maize are outside its
  training distribution.
- `weather-pheno/` (`tools/build_yieldsat_weather_pheno.py`) [AgriBench]: does year-level weather fix the poor
  LOYO/LORO transfer of imagery? `s2-adm` holds weather only as calendar-month slots; this input summarises each
  field's daily ERA5 file (`Raw/.../weather/`) in agronomic terms keyed to its sowing date. Crop stages come from
  thermal time since sowing (the `s2-gdd` GDD definition), with one constant per crop for flowering and maturity
  (wheat 1650/2400 °C·d at base 0 °C, maize 800/1500 and soybean 750/1650 at base 10 °C, rapeseed 850/1750 at
  base 5 °C; literature ranges, chosen so maturity falls 1–3 weeks before each subset's median sowing-to-harvest
  thermal time, a per-subset check, never a field's own harvest date). Windows: 60 days before sowing,
  early and late vegetative, flowering (±10 % around the flowering constant), grain fill, whole season. Per window:
  length, mean Tmean and Tmax, heat degree-days above a crop threshold on Tmax (wheat 32, maize and soybean 35,
  rapeseed 29 °C), frost days (Tmin < 0 and < −5 °C), rainfall, Hargreaves ET0 (radiation exists only in Germany
  and Uruguay, so every country uses Hargreaves), rainfall − ET0, longest dry spell and the water-stress index
  1 − AET/ETc of a 150 mm bucket with FAO-56-style crop coefficients. Plus sowing day of year, days to flowering
  and maturity, soil water at sowing and flowering, GDD in the first 30/60/90 days and the season's minimum Tmin.
  **Leakage guard:** every weather file runs from 60 days before sowing to exactly 60 days after harvest (all
  2,173 fields), so a field's record end would reveal its harvest date. The builder reads weather only up to
  sowing + a cap that is one constant per subset (the shortest record in the subset: ARG-C 188, ARG-S 141, ARG-W
  205, BRA-C 177, BRA-S 171, BRA-W 185, GER-R 368, GER-W 318, URG-S 183 days), so windows are truncated in the same
  way for every field. Estimated maturity falls within the cap for every field except 33 % of ARG-S (where the
  grain-fill window ends at the cap). Pixels of a field share its values.
- `wofost/` (`tools/build_yieldsat_wofost.py`, shared set-up in `tools/yieldsat_wofost.py`) [AgriBench]: WOFOST
  7.2 through PCSE 6.0.13 (layered venv `envs/pcse`, `tools/setup_pcse_env.sh`), potential (`Wofost72_PP`) and
  water-limited (`Wofost72_WLP_CWB`) runs per field from sowing to maturity or the subset cap above. Weather: the
  same ERA5 files and cap, radiation by Hargreaves, vapour pressure from Tmin, wind 2 m/s, E0/ES0/ET0 by PCSE's
  Penman(-Monteith). Soil: the SoilGrids layers already in the v1 cache (the `s2-adm` static bands), field mean of
  sand, clay, organic carbon and coarse fragments over 0–100 cm → wilting point, field capacity, saturation and
  conductivity by Saxton & Rawls (2006); rooting depth 100 cm; root zone half full 60 days before sowing. Crop:
  default WOFOST 7.2 parameter sets (ajwdewit/WOFOST_crop_parameters, branch `wofost72`, pinned): Winter_wheat_101,
  Grain_maize_201, Soybean_901, Oilseed_rape_1001. These are European calibrations, so phenology is replaced by the
  `weather-pheno` thermal-time constants (temperature-only development, IDSL = 0; DTSMTB linear from the crop's
  base temperature to 30 °C; TSUM1 = flowering − TSUMEM, TSUM2 = maturity − flowering); all other crop parameters
  stay default and nothing is calibrated on yield, so absolute yields are indicative and the features act as
  relative, year-sensitive indices. 13 features: potential and water-limited grain yield, biomass and max LAI,
  water-limited LAI at flowering and transpiration, their yield ratio, a transpiration-deficit water-stress index,
  simulated days to flowering and maturity, and root-zone available water. Slurm array of 60-field chunks, about
  1 s per field.
- `wofost-da/`, `wofost-da-snap/` (`tools/build_yieldsat_wofost_da.py`) [AgriBench]: the classic crop-model +
  satellite hybrid. WOFOST water-limited as `wofost`, as an ensemble of 32 members with perturbed LAI-related
  parameters (TDWI × lognormal(0, 0.3); SPAN × N(1, 0.1); SLATB and AMAXTB × N(1, 0.15); TSUM1 × N(1, 0.05); initial
  water × U(0.5, 1.5)), corrected by a stochastic ensemble Kalman filter on the LAI state (PCSE `set_variable`,
  which rescales leaf biomass) at each observation of the field-median satellite LAI on the `s2-aligned` grid (days
  1 to the cap, at least 5 valid pixels; observation SD max(0.3, 0.15 LAI)). LAI source: `arc` → `wofost-da`, SNAP
  `biophys` → `wofost-da-snap`. 12 features: ensemble mean and SD of grain yield, mean biomass, max LAI, LAI at
  flowering, transpiration, days to flowering and maturity, the number of observations assimilated, the mean and
  mean absolute innovation (observed − forecast LAI) and the largest observed LAI. Per-field seeds make it
  reproducible.

### 2.6 Hyperparameter tuning [AgriBench: the paper doesn't say]

**Linear probe** (`model=probe`, `tools/probe.py`), the main head for comparing representations. It follows the
regression probe of the TorchGeo evaluation harness (geo-eval-harness, CoordBench `linear_probe_score`):
closed-form ridge in float64 on features z-scored with the training fold's statistics, unpenalised intercept,
penalty chosen by cross-validation, and a warning when the best penalty is on the grid edge. Two AgriBench
differences: the inner folds are the scheme-mirroring folds below (not a single random split), and the penalty
is per training pixel (MSE + λ‖w‖², λ in 19 log-spaced values from 1e-6 to 1e3), so the same grid means the same
shrinkage at any training size. The harness's image-backbone probe is logistic and classification-only, so it
doesn't apply to yield.

**TabPFN and TabICL** need no tuning: one fit per outer fold, on 32 training pixels per field (TabICL v2 is
pre-trained on up to ~48k rows), with PCA to 500 (TabPFN) or 100 (TabICL) columns, fitted on training pixels
only, when an input is wider than that.

The paper doesn't describe tuning. To keep it honest and the same for every head:

- **Nested tuning, inside each outer fold.** The grid in the model's YAML is searched with an inner split
  over that fold's **training fields only**. The best setting is then refit on all training fields, and the
  test fold is scored once.
- **The inner split mirrors the outer scheme:**
  - CV10: 3-fold `StratifiedGroupKFold` (stratified by farm, grouped by field);
  - LOYO: `GroupKFold` on year, with k = min(3, number of training years);
  - LORO: `GroupKFold` on region, with k = min(3, number of training regions);
  - LOCO, soybean and wheat: leave one *training* country out (2 inner folds);
  - LOCO, corn: there's only one training country, so use `GroupKFold` on farm, k = 3, within it.
- **Selection metric:** pixel-level RMSE, averaged over the inner folds. This is the quantity the models are
  trained to reduce.
- Heads without a grid are fit once with their defaults. Neural heads take their early-stopping set from the
  training fields, grouped by field.

### 2.7 Training-pixel cap [AgriBench: needed for tabular heads]

Argentina alone has 5.3 M pixels × 2,880 features (61 GB as float32). That's too much for RF or TabPFN, and
pixels within a field are strongly autocorrelated anyway.

- **Training:** at most **256 pixels per field**, sampled uniformly with the fold's seed. This keeps the
  same field mix and caps the largest subset (URG-S, 572 fields) at about 146k training rows.
- **Evaluation:** always uses **all** pixels of the test fields.
- The cap is part of the protocol, so every head uses it. A run may raise it, but it is then marked as a
  separate configuration.

### 2.8 Frozen fold files and provenance [AgriBench]

- `tools/make_yieldsat_folds.py` builds the field table and writes
  `/lustre/shared/agriscience-benchmark/splits/yieldsat/<subset>/{cv10,loyo,loro}.csv` and
  `/lustre/shared/agriscience-benchmark/splits/yieldsat/loco/<crop>.csv`, with columns
  `field_shared_name, fold`, plus `fields.csv` (with each field's region) and `manifest.json` (SHA-256 of
  every file and of the code). It writes them once and refuses to overwrite. `--check` rebuilds them in
  memory and confirms they match. **Frozen on 2026-11-09.**
- Each run records the SHA-256 of its fold file, the input file and the config, as for PASTIS-R.
- **Development:** CV10, LOYO and LORO are reported as cross-validation, like the paper. Model choices are
  made on CV10. LOYO and LORO are reported, not tuned on.
- **[AgriBench] Pooled R² for LOYO and LORO** (2026-11-09). Each run reports R² and RMSE both as the mean over
  folds and pooled (all held-out predictions of the run scored together). For LOYO and LORO the dashboard shows
  pooled R² by default: R² within one held-out year or region is measured against that fold's own mean, so a
  model that ranks fields well but misses the fold's overall level scores far below zero, and one such fold
  dominates the mean (ARG-W, LightGBM on `s2`, LOYO: −2.91 as a mean of folds, 0.67 pooled).
- **Final:** LOCO (2.3b) is run only for configurations chosen on the development schemes, and every run is
  logged in the ledger.

### 2.9 What a result contains

**Running sweeps.** `python -m tools.sweep_yieldsat --name <sweep> --inputs ... --models ...` submits one Slurm
array per head and input width: CPU heads run one task per outer fold (the folds of a run execute in
parallel), GPU heads one task per run (the model loads once). Resources per head are set from measured use,
the sweep-wide concurrency cap defaults to 200 tasks, finished folds are skipped on re-runs, and a dependent job
runs `tools.collect_yieldsat`, which combines folds into `metrics.json` and writes the aggregate table
`results/yieldsat/<sweep>.csv` that the dashboard reads. `--efficiency` reports requested vs used time, CPU and
memory per array.

For each model × input × subset × scheme:

- R² and RMSE, as mean ± std over folds, at pixel and field level (for LOCO, one value per held-out country);
- the number of folds, test fields and test pixels;
- the pixel cap, seed, and fold-file and input hashes.

The leaderboard has:
- **the final LOCO table first**, with one column per crop × held-out country (8 columns) and the
  training-mean baseline as a reference row;
- one table per development scheme (CV10, LOYO, LORO), with the nine subsets as columns, laid out like the
  paper's Tables 13–18.

### 2.10 Uncertainty, error decomposition and rank persistence [AgriBench]

`python -m tools.yieldsat_diagnostics --sweep <name>` works from the saved predictions, so it needs no reruns. It
works at field level (pixel predictions averaged per field) and writes `results/yieldsat/diagnostics/<sweep>-*.csv`.

- **Bootstrap intervals (`-ci.csv`).** Pooled field R² and RMSE, with 95% percentile intervals from 1,000
  resamples of test fields within each held-out fold. Two configurations whose intervals overlap heavily are
  not ranked against each other in the write-up.
- **Error decomposition per held-out group (`-decomp.csv`).** The group is the year (LOYO), region (LORO, by
  fold number) or country (LOCO). The mean squared deviation is split following Kobayashi & Salam (2000),
  normalised by the group's yield variance (population SDs):
  - b = (mean ŷ − mean y) / s_y, κ = s_ŷ / s_y, and r is the within-group Pearson correlation.
  - 1 − R² = B + V + A, where B = b² (level bias), V = (κ − 1)² (spread mismatch) and A = 2κ(1 − r)
    (lack of correlation).
  - Also reported: MBE in t/ha, the shares of B, V and A, and R² after removing the group's mean bias
    (= 1 − V − A).
  - The bias-removed R² is a diagnostic only, never a score: in deployment the held-out year's mean is unknown.
    It says whether a model ranks fields within a year correctly but misses the year's level (B dominates),
    which points to a missing year-level signal (weather, management) rather than poor field-level features.
  - The same decomposition of the pooled predictions is reported as group `pooled`.
- **Rank persistence (`-ranks.csv`).** Within each subset × scheme, every input × head is ranked by its R² in
  each held-out group. We report the mean rank, the rank variance across groups, and the worst-group R². A
  representation that is best on average but has a high rank variance is not a dependable choice.

### 2.11 Early season and label efficiency [AgriBench]

Both are variants of a normal run, selected with `tools.sweep_yieldsat` options. They get their own run
directories (`<input>@d060`, `<input>@f0.10-s1`) and extra columns in the aggregate table (`season_cutoff`, or
`train_fraction` and `seed`). The dashboard keeps them out of the main comparison.

- **Early season (`--season-cutoffs 30 60 90 120 150`).**
  - Only columns at or before the cutoff, in days after sowing, are used. For the full season, see the
    normal runs, which go to day +210.
  - This needs an input whose columns are all indexed by days after sowing: s2-aligned (raw S2 on the sowing
    grid) and biophys.
  - Inputs computed from the whole season cannot be truncated after the fact. These are the annual
    embeddings (AEF, TESSERA), Galileo, and ARC and ARC-VAE (fitted on the full season). They need
    re-encoding on truncated time series, which is listed as a follow-up for the foundation-model runs.
- **Label efficiency (`--train-fractions 0.05 0.1 0.25 0.5 --seeds 0 1 2`).**
  - In each outer fold, a random subset of that fraction of the training fields is kept, with at least 10.
    The test fold is unchanged.
  - The subset is drawn from its own random stream (seed, fold), so it is the same for every input and head.
  - Three seeds per fraction give the spread caused by which fields happen to be labelled.
  - This is run on CV10 first (LOYO to follow). The question is how quickly a representation reaches its full-data score:
    a good foundation representation should need fewer labelled fields.

---

## 3. Paper reference numbers (sanity checks)

These are CV10 results from the paper's Tables 13 and 14, R² / RMSE (t/ha), for the pixel-wise LSTM with S2
only. The LSTM is the closest of the paper's models to a pixel-wise tabular head.

| | ARG-S | BRA-C | GER-W | URG-S |
|---|---|---|---|---|
| Field level | 0.72 / 0.64 | 0.75 / 0.88 | 0.55 / 2.60 | 0.66 / 0.62 |
| Pixel level | 0.60 / 0.96 | 0.42 / 2.20 | −0.32 / 3.38 | 0.37 / 1.26 |

The best overall model in the paper, AFF (feature fusion of S2 and ADM, 3D context), gets 0.84 / 0.49 for
ARG-S at field level.

## 4. Decisions

All agreed on 2026-11-09:

- **Leave-one-country-out is the final evaluation** and AgriBench's headline YieldSAT result, run per crop
  across countries (2.3b).
- **Region for LORO = a group of neighbouring farms**, 3–8 per subset, built by the rule in 2.3.
- **Nested tuning**, selected by pixel RMSE (2.6).
- **Training cap of 256 pixels per field** (2.7).
- **All quality levels are included**, as in the paper. A "good only" ablation could come later.
