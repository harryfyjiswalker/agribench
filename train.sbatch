#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --job-name=pastis-train
set -eu
benchmark_python=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v1/bin/python}
exec "$benchmark_python" -u "$SLURM_SUBMIT_DIR/run_experiments.py" "$@"
