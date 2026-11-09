#!/bin/bash -l
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=yieldsat
# YieldSAT runner (protocols/yieldsat.md), e.g.
#   sbatch yieldsat.sbatch subset=ARG-S scheme=cv10 input=s2 model=rf
# GPU models (TabPFN): sbatch --partition=gpu --gres=gpu:1 --mem=64G yieldsat.sbatch ... model=tabpfn
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
cd "$SLURM_SUBMIT_DIR"
exec "$py" -u "$SLURM_SUBMIT_DIR/run_yieldsat.py" "$@"
