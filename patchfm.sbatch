#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=patchfm
# Patch foundation-model embeddings for YieldSAT (tools/patch_embed.py), one subset per array task:
#   sbatch --array=0-8 patchfm.sbatch dates                      # peak-greenness dates (use --partition=main, no GPU)
#   sbatch --array=0-8 patchfm.sbatch embed_yieldsat_croma       # a model's cache (needs the dates)
# Array index -> subset in tools/yieldsat.py SUBSETS order.
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
cd "$SLURM_SUBMIT_DIR"
subsets=(ARG-C ARG-S ARG-W BRA-C BRA-S BRA-W GER-R GER-W URG-S)
subset=${subsets[$SLURM_ARRAY_TASK_ID]}
if [[ $1 == dates ]]; then
  exec "$py" -u -m tools.patch_embed --subset "$subset" --workers "${SLURM_CPUS_PER_TASK:-4}"
fi
exec "$py" -u -m "tools.$1" --subset "$subset"
