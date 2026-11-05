# PASTIS-R parcel classification 

## Task and splits

The PASTIS-R PixelSet v1 ([Zenodo release](https://doi.org/10.5281/zenodo.5745151)) allows 








## Data and splits

Use PASTIS-R PixelSet v1 
and its published parcel IDs, labels and folds. Use Sentinel-2 only, with bands
B02, B03, B04, B05, B06, B07, B08, B8A, B11 and B12 in that order.

| Stage | Folds | Expected parcels |
|---|---|---:|
| Fit each candidate probe | 1–3 | 51,833 |
| Select regularisation by accuracy | 4 | 17,478 |
| Refit the selected probe | 1–4 | 69,311 |
| Score the final predictions | 5 | 16,640 |

Use all parcels in each split. Match representations by parcel ID and verify
unique IDs, identical labels, identical split membership and complete coverage.
Fix the input and probe rules before scoring fold 5. Keep fold-5 labels out of
feature generation, preprocessing fits and model selection.

## Shared input decision

Use **fixed 30-day parcel medians** for v1. This is a controlled representation
comparison. Both entries see the same Sentinel-2 observations, with no new
scene selection, additional sensors or spatial imagery. The cadence is close
to Galileo's published pipeline, and the existing AgriBench cache and Galileo
embeddings can be reused after checking their provenance. This choice makes
no claim about the best cadence for crop classification.

1. At each acquisition, take the median of the stored pixels for each parcel
   and band. Within each 30-day bin, take the median of those acquisition-level
   values, separately for each band. Keep empty bins missing.
2. Use bins **7–20 inclusive** from AgriBench's existing grid. The grid is
   anchored at day `-200` relative to **2018-09-01**, with left-inclusive,
   right-exclusive bins. The retained 14 slots cover **2018-09-11 inclusive
   through 2019-11-05 exclusive**. Use the same slots for every parcel.
3. Before scoring, audit fold-5 coverage and report acquisition counts outside
   the fixed window and missing bins within it. The fixed window applies to
   both entries even if fold 5 has observations outside it. If any parcel has
   no valid retained bin, or any observed bin has only some bands present,
   stop and review the protocol before evaluating. Keep all test parcels in
   scope during that review.

The 14 slots cover the observed acquisition span in folds 1–4 and fit within
Galileo Base's 24 temporal positions. Our parcel medians are a benchmark input
choice. Galileo's upstream data pipeline uses a quality mosaic for Sentinel-2.
Any comparison using finer bins, new cloud screening or gridded imagery gets
its own protocol. [AgriBench extractor](../tools/extract_galileo.py) ·
[Galileo Sentinel-2 compositing](https://github.com/nasaharvest/galileo/blob/0f0b5b95ac81acef4b74cf4686dc877202f4541b/src/data/earthengine/s2.py)

## Representations

- **30-day parcel medians:** Flatten the 14 slots and 10 bands in time-major
  order into 140 features. Use per-feature median imputation. Fit it on folds
  1–3 for the sweep and on folds 1–4 for the refit. Fill a column that is
  entirely missing in the fitting split with zero. Apply the fitted
  transformation to the corresponding evaluation split. Use no standardisation
  or additional features.
- **Galileo Base:** Use the existing frozen S2-only encoder and checkpoint,
  the same 14-slot input and the same one-spatial-location-per-parcel setup.
  Use the calendar month of each bin midpoint, mask missing bins, apply the
  existing checkpoint normalization, and average unmasked tokens with
  upstream `average_tokens`. This gives 768 features per parcel. Use the
  resulting embeddings directly in the probe. Verify the existing folds 1–4
  cache against the recipe and extract only fold 5 when it matches.

## Geo Eval Harness linear evaluation

Pin [Geo Eval Harness](https://github.com/torchgeo/geo-eval-harness/tree/a782d730ab4dbdb9f9c0ae05ac7fe893e739be03)
at `a782d730ab4dbdb9f9c0ae05ac7fe893e739be03`. Use its single-label linear
probe. Keep AgriBench's dataset and feature loaders and reuse the harness's
selection, estimator and accuracy interval routines directly. The adapter
provides parcel features, stage-specific imputation for the medians, and
AgriBench result export. The harness's built-in raster PASTIS preset is a
separate task.

- Fit L2-regularised multinomial logistic regression with upstream L-BFGS.
  Use the default 40 log-spaced values of `C` from 1e-6 to 1e4 in ascending
  order. Fit each candidate on folds 1–3 and select by fold-4 accuracy.
  On a tie, the first candidate wins, as in the upstream implementation.
- Use the upstream tolerance of 1e-6 and iteration limits of 2,000 for the
  sweep and 4,000 for the final refit. Refit on folds 1–4 using the selected
  `C`. Use seed 0, the upstream default, and the same CUDA device policy
  and solver settings for both representations.
- Score the final predictions on fold 5. Rank by unrounded test accuracy.
  Use the upstream 95% percentile bootstrap interval with 200 resamples and
  seed 0. The interval resamples parcels and describes uncertainty on this
  split. It does not estimate uncertainty across geographic regions.
- Also report parcel mIoU and macro F1 as AgriBench diagnostics, with the
  same 18-class mapping for both entries. Use test metrics for reporting.

[Upstream methodology](https://github.com/torchgeo/geo-eval-harness/blob/a782d730ab4dbdb9f9c0ae05ac7fe893e739be03/docs/user/methodology.rst) ·
[Probe implementation](https://github.com/torchgeo/geo-eval-harness/blob/a782d730ab4dbdb9f9c0ae05ac7fe893e739be03/src/geo_eval_harness/main.py)

## Run record and leaderboard

Record the protocol ID, dataset and split identifiers, parcel counts and ID
hashes, representation and feature hashes, selected `C`, fold-4 selection
score, test metrics and interval, code revisions, environment and device.
For Galileo, include the encoder revision, checkpoint hash and preprocessing
recipe. Save parcel predictions on Anunna and link the result JSON from the
leaderboard.

Publish these rows in a **fold-5 linear-probe** table. Existing fold-4 pilots,
including the standardised `C=1` logistic regression and TabPFN runs, retain
their validation labels and their own table.
