#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=earthshift
# EarthShift field-delineation training and scoring (protocols/earthshift.md). Submit from the repo root, e.g.
#   sbatch earthshift.sbatch --model resnet50-imagenet --setting frozen --task de-geo --mode sweep --lr 1e-4 3e-4
# All arguments go to tools/earthshift_train.py (see its --help).
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
cd "$SLURM_SUBMIT_DIR"
exec "$py" -u -m tools.earthshift_train "$@"
