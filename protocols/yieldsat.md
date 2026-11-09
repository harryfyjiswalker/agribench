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
   padded with NaN on both sides. In the rows checked, `times` counts days from the field's first slot, so the
   file-level `units` attribute ("days since <date>") looks misleading. This will be confirmed when the input
   builder is written.
6. **Admin unit names are spelled inconsistently** (e.g. `Santa Fe` and `Santa_Fe`). Don't group by them
   without cleaning.
7. **The raw weather CSV has 10 variables, but the preprocessed file keeps 4.** Wind, humidity, radiation and
   dew point are only in `Raw/`.

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
  seeding date + slot offset (checked against the Raw S2 acquisition dates on 25 fields per subset). Weather,
  topography and location are left out on purpose, so `galileo` is comparable to `s2` and location only enters
  through the location encoders. Mean of the encoder's tokens (Galileo `average_tokens`), 768-d.
- `aef/` (`tools/extract_yieldsat_aef.py`): the 64-d AlphaEarth annual embedding of the calendar year of the
  field's harvest date, read with HTTP range requests from the Source Cooperative mirror. AEF starts in 2017,
  so the 48 German fields harvested in 2016 have no AEF value (imputed by the runner, and counted in the json).

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
