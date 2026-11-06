#!/bin/bash
# Usage:  bash tools/sweep.sh "<encoders>" "<models>" [hydra overrides...]
# e.g.    bash tools/sweep.sh "galileo-base summaries" "rf xgboost" experiment.smoke=true
# GPU:    SBATCH_ARGS="--partition=gpu --gres=gpu:1 --mem=32G" bash tools/sweep.sh "galileo-base" "mlp" ...
# NOTE: the default protocol is validation=final, which scores the reserved test fold 5 once and
#       logs it in test-ledger.jsonl. Add experiment.smoke=true for plumbing checks.
set -eu
cd "$(dirname "$0")/.."
encoders=$1; models=$2; shift 2
for enc in $encoders; do for model in $models; do
  id=$(sbatch --parsable ${SBATCH_ARGS:-} bench.sbatch encoder=$enc input=${INPUT:-pastis-r/s2-parcel-medians} model=$model "$@")
  echo "$enc / $model -> job $id"
done; done
