# Scores on the doors

## Best Performers

## Crop Classification

### In-Distribution (PASTIS-R)

<!-- leaderboard:start -->

Parcel-level crop type classification on PASTIS-R PixelSet (Zenodo 5745151). Scores are from validation. Fold 5 is reserved for testing.

| Run | Model | Sensors | Train → validation folds | Train parcels | Validation parcels | Seed | Epoch | Parcel mIoU | Accuracy | Macro F1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [`s2-tabpfn35-fast-full-pilot`](results/s2-tabpfn35-fast-full-pilot.json) | tabpfn | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.702 | 0.906 | 0.807 |
| [`s2-rf-full-pilot`](results/s2-rf-full-pilot.json) | rf | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.508 | 0.840 | 0.618 |
| [`s2-pse-ltae-1epoch-seed42`](results/s2-pse-ltae-1epoch-seed42.json) | adapted_pse_ltae2d | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | 1 | 0.308 | 0.737 | 0.384 |

Run links contain the recorded parcel and feature fingerprints. Sequence runs lack a training parcel fingerprint. Check inputs and training budgets before comparing scores.

<!-- leaderboard:end -->

### Out-Of-Distribution (PLANTEUR)

## Yield Prediction

## Phenology Detection

## Field Delineation


_____________________________________________________________________________________________________________________________________________________________________

#Usage

#Setup/ Access checks

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

4. Check if code can be used (should print three git commit IDs):

```
git -C "$project/upstream/pastis-benchmark" rev-parse HEAD
git -C "$project/upstream/ltae" rev-parse HEAD
git -C "$project/upstream/utae-paps" rev-parse HEAD
```

5. Can read sentinel-2 cache?
```
find "$project/features/pastis" -maxdepth 1 -type f -name 'S2-*.npz' -readable -print -quit
```

6. Can write results onto file?
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

# General Use

