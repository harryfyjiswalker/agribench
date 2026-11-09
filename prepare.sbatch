#!/bin/bash -l
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=agribench-prep
# One-off data preparation steps, run as a Python module from the repo root, e.g.
#   sbatch prepare.sbatch tools.make_yieldsat_folds
#   sbatch prepare.sbatch tools.build_yieldsat_inputs --country Argentina
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
cd "$SLURM_SUBMIT_DIR"
exec "$py" -u -m "$@"
