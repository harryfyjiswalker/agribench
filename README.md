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

3. Open AgriBench.
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

Example: Generating Presto embeddings and training a random forest on them.

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
selection=(encoder=presto-base input=pastis-r/s2-parcel-medians-location)
representation=presto-base--pastis-r-s2-parcel-medians-location
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

5. If no presto embeddings available, load presto
```
source_dir="$HOME/projects/presto"
revision=11e207a668a34336ced1d8e492a1bd5849b96c4a

if [ ! -d "$source_dir/.git" ]; then
  git clone https://github.com/nasaharvest/presto.git "$source_dir"
fi

git -C "$source_dir" checkout --detach "$revision" &&
"$bench" -c 'import einops' || "$bench" -m pip install einops
```

and check for prepared input file:
```
export bench data
export inputs="$work/pastis-r/inputs/pastis-s2-and-location.npz"
mkdir -p "$work/pastis-r/logs"

if [ -s "$inputs" ]; then
  ls -lh "$inputs"
elif [ -n "$(squeue -h -u "$USER" -n pastis-inputs)" ]; then
  squeue -u "$USER" -n pastis-inputs
else
  sbatch --partition=main --job-name=pastis-inputs \
    --cpus-per-task=1 --mem=8G --time=04:00:00 \
    --output="$work/pastis-r/logs/inputs-%j.log" \
    --wrap='exec "$bench" -u tools/build_pastis_inputs.py --data "$data" --output "$inputs"'
fi
```

5. Extract embeddings:
```
source_dir="$HOME/projects/presto"
inputs="$work/pastis-r/inputs/pastis-s2-and-location.npz"

ls -lh "$inputs" &&
weights=$("$bench" tools/prepare_weights.py encoder=presto-base \
  --local-dir "$source_dir/data") &&
extract_args=(
  "${selection[@]}"
  --inputs "$inputs"
  --metadata "$data/metadata_parcel.csv"
  --source-dir "$source_dir"
  --weights-dir "$weights"
) &&
sbatch extract.sbatch "${extract_args[@]}" --smoke
```

6. If "Smoke passed" in log, submit full extraction:
```
sbatch extract.sbatch "${extract_args[@]}" --output-dir "$features_dir"
```

7. Then create a launcher for the classifier:

```
cat > evaluate_rf.sbatch <<'SLURM'
#!/bin/bash
#SBATCH --job-name=pastis-rf
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -eu

work=${AGRIBENCH_WORK:-/lustre/nobackup/WUR/AIN/$USER/agribench-probe}
bench=${BENCHMARK_PYTHON:-$work/venv/bin/python}

exec "$bench" -u - "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

feature_path, metadata_path, runs_dir = map(Path, sys.argv[1:])
representation = json.loads(feature_path.with_suffix(".json").read_text())
meta = pd.read_csv(
    metadata_path, usecols=["ID_PARCEL", "Fold", "Label"], dtype="Int64",
).set_index("ID_PARCEL")

with np.load(feature_path, allow_pickle=False) as table:
    ids, features = table["ids"], table["features"]

if (
    len(meta) != 85_951
    or not meta.index.is_unique
    or ids.shape != (85_951,)
    or ids.dtype.kind not in "iu"
    or not np.array_equal(np.sort(ids), np.sort(meta.index.to_numpy(dtype=np.int64)))
):
    raise ValueError("Expected exactly one feature row per PASTIS-R parcel ID")
if features.ndim != 2 or features.shape[0] != len(ids) or not np.isfinite(features).all():
    raise ValueError("Expected one finite feature vector per parcel")

rows = meta.loc[ids]
fit = rows.Fold.isin([1, 2, 3, 4]).to_numpy()
test = rows.Fold.eq(5).to_numpy(dtype=bool)
if fit.sum() != 69_311 or test.sum() != 16_640:
    raise ValueError("Unexpected train or test fold sizes")

labels = rows.Label.to_numpy(dtype=np.int64)
classes = np.arange(1, 19)
settings = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "random_state": 0,
    "n_jobs": int(os.environ["SLURM_CPUS_PER_TASK"]),
}
forest = RandomForestClassifier(**settings)
forest.fit(features[fit], labels[fit])
predicted = forest.predict(features[test])
truth = labels[test]

out = runs_dir / f"{feature_path.stem}-rf-{os.environ['SLURM_JOB_ID']}"
out.mkdir(parents=True)
metrics = {
    "classifier": "random-forest",
    "settings": settings,
    "representation": representation,
    "test_parcels": int(test.sum()),
    "accuracy": accuracy_score(truth, predicted),
    "macro_f1": f1_score(truth, predicted, labels=classes, average="macro", zero_division=0),
    "macro_iou": jaccard_score(truth, predicted, labels=classes, average="macro", zero_division=0),
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
pd.DataFrame({"parcel_id": ids[test], "label": truth, "prediction": predicted}).to_csv(
    out / "test-predictions.csv", index=False,
)
print(out / "metrics.json")
PY
SLURM

bash -n evaluate_rf.sbatch
```


8. Set up trianing job:

```
features="$features_dir/$representation.npz"
mkdir -p "$work/pastis-r/logs"

test -r "$features" &&
test -r "${features%.npz}.json" &&
rf_job=$(BENCHMARK_PYTHON="$bench" sbatch --parsable \
  --output="$work/pastis-r/logs/rf-%j.log" \
  evaluate_rf.sbatch "$features" "$data/metadata_parcel.csv" "$work/pastis-r/runs") &&
echo "Random Forest job $rf_job"
```

9. Read metrics
```
cat "$work/pastis-r/runs/$representation-rf-${rf_job%%;*}/metrics.json"
```





