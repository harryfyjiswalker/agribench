#!/bin/bash -l
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --job-name=pastis-rf
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number
benchmark_python=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
exec "$benchmark_python" -u "$SLURM_SUBMIT_DIR/run_experiments.py" model=rf "$@"
