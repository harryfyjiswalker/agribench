# YieldSAT evaluation protocol (draft for review)

Status: **draft, nothing run yet.** This page sets how every AgriBench model is scored on YieldSAT so results are
comparable with the paper and with each other. Sections marked **[paper]** follow Miranda et al. (2026)
exactly; sections marked **[AgriBench]** fill gaps the paper leaves open, and need the group's agreement before
the first run.

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
- **Leave-one-region-out (LORO):** one fold per region. A region is `farm_identifier`, following the
  authors' notebook ("farm_identifier: unique ID for a farm (region)") and the paper's definition: "a set of
  fields belonging to a single farmer or to a local data provider". That's 6–44 folds, depending on the subset.
- The paper doesn't do leave-one-country-out. It says cross-country and cross-crop settings "led to model
  collapse" and leaves them to future work. AgriBench can add this later as an **extra track**, labelled as not
  from the paper.

**Open point:** the paper's weight-space figure for Argentina shows 8 LORO regions, but ARG-S has 44 farms.
Argentina has 5 provinces, or 8 if you count the inconsistent spellings separately. So the paper may have
grouped farms differently for that figure. We follow the documented definition (farm) and will ask the authors.

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
| embeddings, location encoders, crop-model outputs | from extracted feature tables, keyed by pixel `index` | AgriBench additions |

Tabular heads get each pixel as a flat vector (24 × bands). The static bands are taken once instead of 24
times. Missing values stay NaN: heads that can't handle NaN get training-fold median imputation inside the
pipeline, so nothing is fitted on test data.

### 2.6 Hyperparameter tuning [AgriBench: the paper doesn't say]

The paper doesn't describe tuning. To keep it honest and the same for every head:

- **Nested tuning, inside each outer fold.** The grid in the model's YAML is searched with an inner split
  over that fold's **training fields only**. The best setting is then refit on all training fields, and the
  test fold is scored once.
- **The inner split mirrors the outer scheme:**
  - CV10: 3-fold `StratifiedGroupKFold` (stratified by farm, grouped by field);
  - LOYO: `GroupKFold` on year, with k = min(3, number of training years);
  - LORO: `GroupKFold` on farm, with k = min(3, number of training farms).
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

- `tools/make_yieldsat_folds.py` (still to write) builds the field tables and writes
  `/lustre/shared/agriscience-benchmark/splits/yieldsat/<subset>/{cv10,loyo,loro}.csv`, with columns
  `field_shared_name, fold`. It writes them once and refuses to overwrite.
- Each run records the SHA-256 of its fold file, the input file and the config, as for PASTIS-R.
- There's no held-out test fold. CV10, LOYO and LORO are all reported as cross-validation, like the paper.
  Model choices should be made on CV10. LOYO and LORO are reported, not tuned on.

### 2.9 What a result contains

For each model × input × subset × scheme:

- R² and RMSE, as mean ± std over folds, at pixel and field level;
- the number of folds, test fields and test pixels;
- the pixel cap, seed, and fold-file and input hashes.

The leaderboard has one table per scheme, with the nine subsets as columns, laid out like the paper's
Tables 13–18.

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

## 4. Decisions to confirm before running

1. Region for LORO = `farm_identifier` (2.3).
2. Nested tuning, selected by pixel RMSE (2.6).
3. Training cap of 256 pixels per field (2.7).
4. Include all quality levels, as the paper does. An optional "good only" ablation could come later.
5. Leave-one-country-out only as a separate, clearly labelled extra track.
