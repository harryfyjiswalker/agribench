#!/bin/bash -l
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=agribench
# GPU models (e.g. TabPFN): sbatch --partition=gpu --gres=gpu:1 --mem=32G bench.sbatch ...
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
exec "$py" -u "$SLURM_SUBMIT_DIR/run_benchmark.py" "$@"
