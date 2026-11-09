# Shared settings for every AgriBench Slurm job. Each *.sbatch script sources this file first:
#     source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"
# It never stops a job: anything optional that fails only prints a warning.

# 1. Threads. Numerical libraries (NumPy/MKL/OpenBLAS, PyTorch) otherwise start one thread per core of the
#    whole node, far more than the cores this job asked for, which slows everything down.
threads=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads \
       NUMEXPR_NUM_THREADS=$threads
unset threads

# 2. Project number (optional). Anunna asks every job to carry the project number for billing as its
#    Slurm comment. If one is configured and the job was submitted without --comment, tag the job here, so
#    nobody has to remember the flag. Set it once for everyone in the shared file below
#    (a line ANUNNA_PROJECT=<number>), or per person with: export ANUNNA_PROJECT=<number>
#    The file is read as data, never executed.
agribench_project_env=${AGRIBENCH_PROJECT_ENV:-/lustre/shared/agriscience-benchmark/project.env}
if [[ -z ${ANUNNA_PROJECT:-} && -r $agribench_project_env ]]; then
  ANUNNA_PROJECT=$(sed -n 's/^[[:space:]]*ANUNNA_PROJECT=["'\'']\{0,1\}\([^"'\'' ]*\).*/\1/p' \
                     "$agribench_project_env" | head -n 1) || ANUNNA_PROJECT=
fi
if [[ -n ${ANUNNA_PROJECT:-} && -n ${SLURM_JOB_ID:-} ]]; then
  agribench_scontrol=/cm/shared/apps/slurm/current/bin/scontrol
  [[ -x $agribench_scontrol ]] || agribench_scontrol=scontrol
  agribench_comment=$("$agribench_scontrol" show job "$SLURM_JOB_ID" 2>/dev/null \
                      | sed -n 's/.*Comment=\([^ ]*\).*/\1/p' | head -n 1) || agribench_comment=
  if [[ -z $agribench_comment || $agribench_comment == "(null)" ]]; then
    "$agribench_scontrol" update JobId="$SLURM_JOB_ID" Comment="$ANUNNA_PROJECT" 2>/dev/null \
      || echo "WARNING: could not tag job $SLURM_JOB_ID with project number $ANUNNA_PROJECT" >&2
  fi
  unset agribench_scontrol agribench_comment
fi
unset agribench_project_env
