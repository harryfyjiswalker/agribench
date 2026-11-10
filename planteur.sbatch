#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=planteur
# PLANTEUR runner (protocols/planteur.md), e.g.
#   sbatch planteur.sbatch split=default input=emb-aef model=probe experiment.confirm_test=true
# CPU heads (LightGBM, background): sbatch --partition=main --gres=none --mem=24G planteur.sbatch ... model=lightgbm
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
cd "$SLURM_SUBMIT_DIR"
exec "$py" -u "$SLURM_SUBMIT_DIR/run_planteur.py" "$@"
