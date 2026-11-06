# Benchmarks and Scores

## Crop Classification

### In-Distribution (PASTIS-R)

- [PASTIS-R](https://zenodo.org/records/5745151) (Garnot and Landrieu, 2021)
  - A benchmark that supports parcel-based crop type classification, comprising 85,951 individual parcels and 18 crop types in metropolitan France. The benchmark covers one year, the September 2018 to November 2019 growing season. PASTIS can be used for in-distribution scoring, using the pre-determined five folds provided by the dataset maintainers. 


<!-- leaderboard:start -->


| Run | Model | Representation | Sensors | Train → validation folds | Train parcels | Validation parcels | Seed | Epoch | Parcel mIoU | Accuracy | Macro F1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [`s2-tabpfn35-fast-full-pilot`](results/s2-tabpfn35-fast-full-pilot.json) | tabpfn | Parcel summaries | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.702 | 0.906 | 0.807 |
| [`s2-galileo-base-logistic-regression-full-pilot`](results/s2-galileo-base-logistic-regression-full-pilot.json) | logistic_regression | Galileo Base embeddings | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.605 | 0.873 | 0.726 |
| [`s2-galileo-base-tabpfn35-fast-full-pilot`](results/s2-galileo-base-tabpfn35-fast-full-pilot.json) | tabpfn | Galileo Base embeddings | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.596 | 0.874 | 0.717 |
| [`s2-logistic-regression-full-pilot-43601671`](results/s2-logistic-regression-full-pilot-43601671.json) | logistic_regression | Parcel summaries | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.596 | 0.865 | 0.717 |
| [`s2-rf-full-pilot`](results/s2-rf-full-pilot.json) | rf | Parcel summaries | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.508 | 0.840 | 0.618 |
| [`s2-pse-ltae-1epoch-seed42`](results/s2-pse-ltae-1epoch-seed42.json) | adapted_pse_ltae2d | Parcel pixel sets | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | 1 | 0.308 | 0.737 | 0.384 |


<!-- leaderboard:end -->

### Out-Of-Distribution (PLANTEUR)

## Yield Prediction

## Phenology Detection

## Field Delineation


_____________________________________________________________________________________________________________________________________________________________________

# Setup

## Setup from scratch

1. Login to Annuna
```
ssh USERNAME@login.anunna.wur.nl
```

2. Clone AgriBench and set paths.
```
mkdir -p "$HOME/projects"
git clone https://github.com/harryfyjiswalker/agribench.git "$HOME/projects/agribench"
```

3. Create environment
```
cd "$HOME/projects/agribench"
export project=/lustre/shared/agriscience-benchmark
```

## Returning to Anunna

1. Log in, and run:

```
cd "$HOME/projects/agribench"
export project=/lustre/shared/agriscience-benchmark
```

# Using the benchmark with presets

## PASTIS-R

1. Set dataset and environment paths,
```
export data="${AGRIBENCH_DATA:-/lustre/nobackup/WUR/AIN/fyjis001/datasets/pastis-r-pixelset/extracted/PASTIS-R_PixelSet}"
export work="${AGRIBENCH_WORK:-/lustre/nobackup/WUR/AIN/$USER/agribench-probe}"
export bench="$work/venv/bin/python"
```
2. Then, by running the following, you can view the possible,
- Input feature representations: ```find conf/input/pastis-r -name '*.yaml'```   ;
- Encoders, if desired: ```find conf/encoder -name '*.yaml' ```;
- and, classifiers: ? ;


3. You can then select choices by running
```
selection=(encoder=XXX input=XXX)
representation=galileo-base--pastis-r-s2-parcel-medians
features_dir="$work/pastis-r/features"

"$bench" -m tools.extract_representation "${selection[@]}" --check-config
```

For example, for encoder = galileo-base and feature input = s2-parcel-medians, you can run:
```
selection=(encoder=galileo-base input=pastis-r/s2-parcel-medians)
representation=galileo-base--pastis-r-s2-parcel-medians
features_dir="$work/pastis-r/features"

"$bench" -m tools.extract_representation "${selection[@]}" --check-config
```

4. Check if the encoder has already been run:
```
for dir in "$features_dir" "$project/features/pastis-r/parcel-probe-v1"; do
  if [ -r "$dir/$representation.npz" ] && [ -r "$dir/$representation.json" ]; then
    features_dir="$dir"
    printf 'Found feature pair in %s\n' "$features_dir"
    break
  fi
done
```

If so, skip to step 6. Otherwise,

5. 



To run a smoke test to check all is running, you can run:



And then to run a full experiment, you can run:




by running the following:

1. To view th





1. Select a dataset


# Contributing

To contribute an encoder 







