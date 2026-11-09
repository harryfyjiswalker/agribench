#!/bin/bash -l
#SBATCH --job-name=pastis-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number

work=${AGRIBENCH_WORK:-/lustre/nobackup/WUR/AIN/$USER/agribench-probe}
bench=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
cd "$SLURM_SUBMIT_DIR"
exec "$bench" -u -m tools.extract_representation "$@" --device cuda:0
