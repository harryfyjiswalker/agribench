#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=agribench
# GPU models (e.g. TabPFN): sbatch --partition=gpu --gres=gpu:1 --mem=32G bench.sbatch ...
set -eu
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
exec "$py" -u "$SLURM_SUBMIT_DIR/run_benchmark.py" "$@"
