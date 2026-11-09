#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=agribench-prep-gpu
# GPU data preparation steps, run as a Python module from the repo root, e.g.
#   sbatch prepare-gpu.sbatch tools.embed_yieldsat_location --encoder mind
# For one task per YieldSAT subset, submit an array and pass SUBSET:
#   sbatch --array=0-8 prepare-gpu.sbatch tools.embed_yieldsat_galileo --subset SUBSET
# (SUBSET is replaced by the subset with index $SLURM_ARRAY_TASK_ID in tools/yieldsat.py SUBSETS.)
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
cd "$SLURM_SUBMIT_DIR"
args=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  subset=$("$py" -c "from tools.yieldsat import SUBSETS; print(SUBSETS[$SLURM_ARRAY_TASK_ID])")
  args=("${args[@]/#SUBSET/$subset}")
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "$py" -u -m "${args[@]}"
