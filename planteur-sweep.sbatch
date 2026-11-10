#!/bin/bash -l
#SBATCH --job-name=pl-sweep
# One task of a PLANTEUR sweep array (written by tools/sweep_planteur.py; don't submit by hand). Line
# $SLURM_ARRAY_TASK_ID of the task file holds the Hydra overrides for one run_planteur.py call. Resources come from
# the sweep tool's per-head classes.
set -eu
# The sweep tool submits with --chdir=<frozen code copy>, so the job starts there. Run that copy, never
# $SLURM_SUBMIT_DIR (the live checkout the sweep was submitted from, which may have moved on since).
code=$PWD
[ -f "$code/COMMIT" ] || { echo "Not a frozen code copy: $code (submit through tools/sweep_planteur.py)" >&2; exit 1; }
source "$code/tools/slurm_env.sh"   # threads, optional project number
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
[ -n "$line" ] || { echo "No task $SLURM_ARRAY_TASK_ID in $1" >&2; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: $line"
eval "set -- $line"
exec "$py" -u "$code/run_planteur.py" "$@" hydra.output_subdir=null \
  hydra/job_logging=disabled hydra/hydra_logging=disabled
