#!/bin/bash -l
#SBATCH --job-name=pastis-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number

if [[ $# -ne 2 || $1 != representation=* || $2 != classifier=linear-probe ]]; then
  echo "Usage: sbatch evaluate.sbatch representation=NAME classifier=linear-probe"
  exit 2
fi

representation=${1#representation=}
if [[ ! $representation =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Use letters, numbers, hyphens or underscores in the representation name."
  exit 2
fi

project=/lustre/shared/agriscience-benchmark
work=${AGRIBENCH_WORK:-/lustre/nobackup/WUR/AIN/$USER/agribench-probe}
data=${AGRIBENCH_DATA:-/lustre/nobackup/WUR/AIN/fyjis001/datasets/pastis-r-pixelset/extracted/PASTIS-R_PixelSet}
bench=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
features=${AGRIBENCH_FEATURES:-$project/features/pastis-r/parcel-probe-v1}

for file in "$features/$representation.npz" "$features/$representation.json"; do
  if [[ ! -r $file ]]; then
    printf 'Cannot read %s. Extract the features or set AGRIBENCH_FEATURES.\n' "$file"
    exit 1
  fi
done

exec "$bench" -u "$SLURM_SUBMIT_DIR/tools/evaluate_parcel_probe.py" \
  --metadata "$data/metadata_parcel.csv" \
  --features "$features/$representation.npz" \
  --recipe "$features/$representation.json" \
  --out "$work/pastis-r/runs/$representation-$SLURM_JOB_ID"
