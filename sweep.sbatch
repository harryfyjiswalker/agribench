#!/bin/bash -l
#SBATCH --job-name=ys-sweep
# One task of a YieldSAT sweep array (written by tools/sweep_yieldsat.py; don't submit by hand). Line
# $SLURM_ARRAY_TASK_ID of the task file holds the Hydra overrides for one run_yieldsat.py call. Resources
# (partition, CPUs, memory, time, GPU) come from the sweep tool's per-head classes.
set -eu
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
cd "$SLURM_SUBMIT_DIR"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
[ -n "$line" ] || { echo "No task $SLURM_ARRAY_TASK_ID in $1" >&2; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: $line"
eval "set -- $line"
exec "$py" -u "$SLURM_SUBMIT_DIR/run_yieldsat.py" "$@" hydra.output_subdir=null \
  hydra/job_logging=disabled hydra/hydra_logging=disabled
