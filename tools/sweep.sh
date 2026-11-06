#!/bin/bash
# Usage:  bash tools/sweep.sh "<encoders>" "<models>" [hydra overrides...]
# e.g.    bash tools/sweep.sh "galileo-base summaries" "rf xgboost" experiment.smoke=true
# GPU:    SBATCH_ARGS="--partition=gpu --gres=gpu:1 --mem=32G" bash tools/sweep.sh "galileo-base" "mlp" ...
# NOTE: the default protocol is validation=pilot (train folds 1-3, score fold 4). validation=final scores
#       the reserved test fold 5 and needs experiment.confirm_test=true (or experiment.smoke=true).
set -eu
cd "$(dirname "$0")/.."
encoders=$1; models=$2; shift 2
for enc in $encoders; do for model in $models; do
  id=$(sbatch --parsable ${SBATCH_ARGS:-} bench.sbatch encoder=$enc input=${INPUT:-pastis-r/s2-parcel-medians} model=$model "$@")
  echo "$enc / $model -> job $id"
done; done
