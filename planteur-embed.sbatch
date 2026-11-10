#!/bin/bash -l
#SBATCH --job-name=pl-embed
# One part (array task) or the combine step of a PLANTEUR input build (tools/planteur_embed.py --submit writes the
# jobs; don't submit by hand). Runs the frozen code copy it was submitted with (--chdir), never the live checkout.
# BENCHMARK_PYTHON picks the env (envs/patchfm for the terratorch-based encoders).
set -eu
code=$PWD
[ -f "$code/COMMIT" ] || { echo "Not a frozen code copy: $code (submit through tools/planteur_embed.py --submit)" >&2; exit 1; }
source "$code/tools/slurm_env.sh"   # threads, optional project number
py=${BENCHMARK_PYTHON:-/lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/python}
exec "$py" -u -m tools.planteur_embed "$@"
