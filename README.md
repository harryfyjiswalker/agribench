# Scores on the doors

## Crop Classification

### In-Distribution (PASTIS-R)

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

# Usage

## Setup/ Access checks

After logging in,

1. Set paths
```
project=/lustre/shared/agriscience-benchmark
data=/lustre/nobackup/WUR/AIN/fyjis001/datasets/pastis-r-pixelset/extracted/PASTIS-R_PixelSet
```

2. Check if can read dataset
```
head -n 1 "$data/metadata_parcel.csv"
```
If prints CSV header with ID_PARCEL, fold etc. then all good.

3. Check if can use shared python environment (if prints python OK):
```
"$project/envs/benchmark-v1/bin/python" -c 'import hydra, torch, sklearn, geopandas; print("Python OK")'
```

4. Check if github code can be used (should print three git commit IDs; need to call them for access):

```
git -C "$project/upstream/pastis-benchmark" rev-parse HEAD
git -C "$project/upstream/ltae" rev-parse HEAD
git -C "$project/upstream/utae-paps" rev-parse HEAD
```

5. Can read sentinel-2 cache?
```
find "$project/features/pastis" -maxdepth 1 -type f -name 'S2-*.npz' -readable -print -quit
```

6. Can write results onto file? Need to add file with username /pastis.
```
probe=$(mktemp "$project/runs/pastis/$USER/.access-check.XXXXXX") &&
rm -- "$probe" &&
echo "Run directory OK"
```

## Run baseline test

7. Clone agribench
```
mkdir -p ~/projects &&
git clone https://github.com/harryfyjiswalker/agribench.git ~/projects/agribench &&
cd ~/projects/agribench
```

[if already cloned: 
```
cd ~/projects/agribench
```
]

8. Submit baseline:
```
sbatch cpu.sbatch validation=pilot \
  upstream_root="$project/upstream" \
  dataset.root="$data" \
  cache_dir="$project/features/pastis"
```
and check (replacing id):
```
job=123456
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed
tail -n 20 "slurm-$job.out"
find "$project/runs/pastis/$USER" -maxdepth 1 -type d -name "*-$job"
```

__________________________________________________________________________

# Adding a model

## Adding a generic sci-kit learn model

1. After logging in, open agribench and set paths:
```
cd ~/projects/agribench
project=/lustre/shared/agriscience-benchmark
data=/lustre/nobackup/WUR/AIN/fyjis001/datasets/pastis-r-pixelset/extracted/PASTIS-R_PixelSet
```

2. Add model config (logistic reg in this case)
```
cat > conf/model/logistic_regression.yaml <<'YAML'
name: logistic_regression
estimator:
  _target_: sklearn.pipeline.make_pipeline
  _args_:
    - _target_: sklearn.preprocessing.StandardScaler
    - _target_: sklearn.linear_model.LogisticRegression
      solver: lbfgs
      C: 1.0
      max_iter: 1000
YAML
```

3. Submit the model

```
model=logistic_regression

sbatch cpu.sbatch model="$model" validation=pilot \
  upstream_root="$project/upstream" \
  dataset.root="$data" \
  cache_dir="$project/features/pastis"
```

4. Check result 

```
job=123456
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed
tail -n 30 "slurm-$job.out"
find "$project/runs/pastis/$USER" -maxdepth 1 -type d -name "*-$job"
```
5. Add to leaderboard list (change name)
```
job=123456
project=/lustre/shared/agriscience-benchmark
name="s2-logistic-regression-full-pilot-$job"

run_dir=$(find "$project/runs/pastis/$USER" \
  -maxdepth 1 -type d -name "*-$job" -print -quit) &&
test -n "$run_dir" &&
"$project/envs/benchmark-v1/bin/python" tools/export_result.py "$run_dir" "$name" &&
cat "results/$name.json"
```

6. Open a pull request
```
git add "results/$name.json"
git add conf/model/logistic_regression.yaml run_experiments.py
```

and commit and push branch:
```
git diff --cached --stat
git commit -m "Record full S2 logistic regression validation result" &&
git push -u origin HEAD
```

## Adding a GeoFM

Current weights uploaded:



a. Standard ML

b. DL (LTSM, etc)

c. Location encoder

d. Geospatial FM

e. Add your own model



To add:
- Inputs etc.





