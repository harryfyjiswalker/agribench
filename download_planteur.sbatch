#!/bin/bash -l
# Download PLANTEUR (allenai/planteur-benchmark) from the Hugging Face Hub and verify every file.
# Submit from the repository root:  sbatch download_planteur.sbatch
# Core files only (no monthly mosaics, no embeddings):  sbatch download_planteur.sbatch --core
# See protocols/planteur.md, "Getting the data". Safe to resubmit: finished files are skipped.
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --job-name=planteur-download
set -euo pipefail
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number

project=/lustre/shared/agriscience-benchmark
python=${BENCHMARK_PYTHON:-$project/envs/benchmark-v2/bin/python}
hf=$(dirname "$python")/hf
repo=allenai/planteur-benchmark
revision=51b74a91424e7dc9ea1fe0042390353658af8185   # Hub commit of 2026-10-22; change only on purpose
target=${PLANTEUR_DIR:-$project/datasets/planteur}

exclude=()
if [[ ${1:-} == --core ]]; then
  exclude=(--exclude "DATA_*_MONTHLY/*" --exclude "EMB_*/*")
fi

mkdir -p "$target"
echo "$(date -Is) downloading $repo@$revision to $target ${exclude[*]:-}"
# --max-workers 8: downloads wait on the network, not the CPU, so one core keeps 8 transfers busy.
"$hf" download "$repo" --repo-type dataset --revision "$revision" --local-dir "$target" \
  --max-workers 8 "${exclude[@]}"

echo "$(date -Is) verifying"
"$python" -u "$SLURM_SUBMIT_DIR/tools/verify_hf_snapshot.py" "$repo" --repo-type dataset \
  --revision "$revision" --local-dir "$target" "${exclude[@]}"
# hf keeps a .lock and a .metadata file per downloaded file (about 105,000 small files for the full
# set) only to resume an interrupted download; drop them once everything is verified.
rm -rf "$target/.cache"
echo "$(date -Is) files: $(lfs find "$target" -type f --lazy | wc -l), size: $(du -sh "$target" | cut -f1)"
