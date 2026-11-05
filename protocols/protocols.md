# Datasets and Evaluation Protocols

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

## Crop Type Classification

### In-Domain Crop Type Classification

- [PASTIS-R](https://zenodo.org/records/5745151) (Garnot and Landrieu, 2021)
  - A benchmark that supports parcel-based crop type classification, comprising 124,422 individual parcels and 18 crop types in metropolitan France. The benchmark covers one year, the September 2018 to November 2019 growing season. PASTIS can be used for in-distribution scoring, using the pre-determined five folds provided by the dataset maintainers. 

### Evaluation 

Cross-validation: 
- In-domain performance: PASTIS-R (5-fold CV, using pre-determined folds):
  - Frozen Feature Set/ Embedding: Uses [Geo Eval Harness](https://github.com/torchgeo/geo-eval-harness) configuration, with an L2-regularised logistic probe trained on folds 1-3 with fold 4 as validation and fold 5 as testing.

This uses harness defaults of 40 log-spaced regularisation values in range [1e-6, 1e4]; 200 parcel bootstrap resampling for 95% confidence interval.

### Code

1. Select PASTIS-R
```
export data="${AGRIBENCH_DATA:-/lustre/nobackup/WUR/AIN/fyjis001/datasets/pastis-r-pixelset/extracted/PASTIS-R_PixelSet}"
export work="${AGRIBENCH_WORK:-/lustre/nobackup/WUR/AIN/$USER/agribench-probe}"
export bench="$work/venv/bin/python"
```

2. If you have not installed probe previously, run this:
```
(
  set -eu
  mkdir -p "$work"

  if [ ! -x "$bench" ]; then
    if command -v python3.12 >/dev/null; then
      python3.12 -m venv "$work/venv"
    else
      "$project/envs/benchmark-v1/bin/python" -m venv "$work/venv"
    fi
  fi

  "$bench" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.8.0' 'torchvision==0.23.0'

  "$bench" -m pip install geopandas scipy \
    'geo-eval-harness @ git+https://github.com/torchgeo/geo-eval-harness.git@a782d730ab4dbdb9f9c0ae05ac7fe893e739be03'

  "$bench" -m pip check
)
```

3. Verify working,.
```
(
  set -eu

  head -n 1 "$data/metadata_parcel.csv"
  test -r "$data/metadata.geojson"
  test -r "$project/upstream/pastis-benchmark/code/dataloader_pixelset.py"
  "$bench" -c 'from geo_eval_harness.main import select_logistic_c; print("PASTIS-R setup OK")'

  mkdir -p "$work/pastis-r"/{features,runs,logs}
  logs="$work/pastis-r/logs"

  status=0
  job=$(sbatch --wait --parsable --partition=main --time=00:10:00 --mem=4G \
    --job-name=pastis-load --output="$logs/load-%j.log" <<'SLURM'
#!/bin/bash
set -eu
export PYTHONPATH="$project/upstream/pastis-benchmark/code"

"$bench" -u - <<'PY'
import os

from dataloader_pixelset import PASTIS_Dataset_PixelSet

dataset = PASTIS_Dataset_PixelSet(
    folder=os.environ["data"],
    sats=["S2"],
    folds=[1],
    norm=False,
    geom_features=False,
)
sample = dataset[0]
pixels, _ = sample["S2"]
print(f"Fold 1 parcels: {len(dataset):,}")
print(f"Loaded Sentinel-2 parcel: {tuple(pixels.shape)} (acquisitions, bands, pixels)")
PY
SLURM
  ) || status=$?

  if [ -n "$job" ]; then
    cat "$logs/load-${job%%;*}.log"
  fi
  exit "$status"
)
```

  

2. If "PASTIS-R setup OK"


### Out-of-Distribution Crop Type Classification

- [PLANTEUR](https://github.com/allenai/planteur-benchmark) (Wolters _et al._, 2026)
  - A complementary benchmark comprising the same content but for eight of France's overseas territories, enabling analysis of smallholder and tropical crop type mapping performance and supporting analysis of geographic generalisation.
 

- Out-of-distribution: PLATEUR (leave-one-region-out CV)

Evaluation:
- Frozen: 
For evaluation on frozen features,

### Limitations

- Currently lacking capacity for understanding of temporal generalisation (both are 2018-2019 growing seasons)

## Crop Yield Prediction

### Datasets

- [YieldSAT](https://yieldsat.github.io) (Miranda _et al._, 2026)




end of season?






### PLANTEUR
