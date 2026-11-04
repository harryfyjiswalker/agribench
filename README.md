# Scores on the doors

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

## Run a model on Anunna

Run these commands in your **Anunna SSH terminal**. All Slurm commands below assume you are in the AgriBench checkout.

### 1. Get the code and check access

For a first checkout:

```bash
mkdir -p ~/projects
git clone https://github.com/harryfyjiswalker/agribench.git ~/projects/agribench
cd ~/projects/agribench
```

If you already have a checkout, enter it with `cd ~/projects/agribench`.

Use the shared Python environment and set `data` to the extracted PixelSet directory you can read. The path below is Harry’s current copy. Thijs and Vishal need access to an extracted copy and permission to read the environment and cache and write their run directories.

```bash
project=/lustre/shared/agriscience-benchmark
bench="$project/envs/benchmark-v1/bin/python"
data=/lustre/nobackup/WUR/AIN/fyjis001/datasets/pastis-r-pixelset/extracted/PASTIS-R_PixelSet
test -r "$data/metadata_parcel.csv" &&
  "$bench" -c 'import geopandas, hydra, numpy, pandas, scipy, sklearn, torch, yaml'
```

No output means both checks passed. If the metadata check fails, set `data` to an accessible copy. If an import fails, use a working benchmark environment before submitting. Set `BENCHMARK_PYTHON` to use a different Python interpreter for Slurm jobs.

### 2. Prepare the upstream code once

Skip the clone commands for any checkout you already have. Use these directory names because the runner looks for all three, including for RF.

```bash
git clone https://github.com/VSainteuf/pastis-benchmark.git ~/projects/pastis-benchmark
git clone https://github.com/VSainteuf/lightweight-temporal-attention-pytorch.git ~/projects/ltae
git clone https://github.com/VSainteuf/utae-paps.git ~/projects/utae-paps
```

For new checkouts, pin the tested revisions and apply the pandas compatibility fix:

```bash
git -C ~/projects/pastis-benchmark checkout 43a0eceb9f40a6e6b0863b1a591407877812efcc
git -C ~/projects/ltae checkout d8f99bd5bc9746d2df25351f0fe156b36af35c11
git -C ~/projects/utae-paps checkout 987874e27a98bb399277f2d55eb010744d26f881
sed -i 's/dates\.iteritems()/dates.items()/' ~/projects/pastis-benchmark/code/dataloader_pixelset.py
```

### 3. Check the configuration and submit

```bash
"$bench" run_experiments.py --cfg job --resolve model=rf dataset.root="$data"
```

Choose one job:

| Job | Command |
|---|---|
| S2 random forest on CPU | `sbatch cpu.sbatch dataset.root="$data"` |
| S2 PSE + L-TAE GPU smoke check | `sbatch --time=00:15:00 train.sbatch dataset.root="$data" experiment.max_batches=8` |
| S2 PSE + L-TAE, one full epoch on GPU | `sbatch train.sbatch dataset.root="$data"` |

Run the GPU smoke check before the full GPU job. The first RF run builds the shared S2 feature cache. Later runs with the same inputs and split reuse it.

### 4. Check the result

Replace `123456` with the job ID printed by `sbatch`.

```bash
job=123456
sacct -j "$job" --format=JobID,State,ExitCode,Elapsed
tail -n 20 "slurm-$job.out"
find "$project/runs/pastis/$USER" -maxdepth 1 -type d -name "*-$job"
```

Look for `COMPLETED`, exit code `0:0`, and validation scores in the log. The last command gives the run directory containing `metrics.json`, predictions and the resolved configuration.

## Publish a validation result

From a results branch, pass the completed run directory and a unique run name to the exporter:

```bash
"$bench" tools/export_result.py /absolute/path/to/run s2-rf-full-seed42
```

Commit the new JSON under `results/`, push your branch, and open a pull request into `main`. After merge, GitHub Actions is configured to regenerate this table and [LEADERBOARD.md](LEADERBOARD.md). To regenerate both files locally, run `python3 tools/render_leaderboard.py`. Keep predictions, checkpoints and data on Anunna.

See the [benchmark protocol](BENCHMARK.md) for sensor ablations, matched RF and TabPFN runs, and preprocessing details.
