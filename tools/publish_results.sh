#!/usr/bin/env bash
# Publish the latest aggregate results to the dashboard, finished or not (run from a laptop checkout of this repo).
#
#   tools/publish_results.sh            # collect on Anunna, open a results PR and merge it when GitHub says it's clean
#   tools/publish_results.sh --no-merge # stop after opening the PR
#
# 1. On Anunna: runs tools.collect_yieldsat for every YieldSAT sweep (complete runs only; one small CPU job each)
#    and tools.earthshift_results --csv for EarthShift (final runs so far, with their seed counts). YieldSAT
#    diagnostics (tools.yieldsat_diagnostics) for every finished run, cached per run so only new runs are computed.
#    PLANTEUR: tools.collect_planteur for every PLANTEUR sweep (finished test scores so far).
# 2. Copies the aggregate CSVs into results/, rebuilds results/index.json, and opens a PR if anything changed.
# Only aggregates are published (the repo is public): no field, tile or fold-level data.
set -euo pipefail
HOST=${ANUNNA_HOST:-anunna}
SWEEP_CHECKOUT=${SWEEP_CHECKOUT:-/lustre/nobackup/WUR/AIN/fyjis001/agribench-sweep}
PROJECT=/lustre/shared/agriscience-benchmark
MERGE=1; [ "${1:-}" = "--no-merge" ] && MERGE=0
cd "$(git rev-parse --show-toplevel)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "Commit or stash local changes first"; exit 1; }

ssh -o BatchMode=yes "$HOST" bash -s -- "$SWEEP_CHECKOUT" "$PROJECT" <<'REMOTE'
set -euo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
checkout=$1; project=$2; py=$project/envs/benchmark-v2/bin/python
cd "$checkout"
git checkout -q results 2>/dev/null || true
rm -f results/yieldsat/*.csv.tmp
ids=()
for d in "$project"/runs/yieldsat/sweeps/*/; do
  s=$(basename "$d"); [[ $s == _* ]] && continue
  ids+=("$(sbatch --parsable --job-name=ys-pub-collect-$s --time=00:30:00 --mem=8G --cpus-per-task=1 \
    --output="$d/_logs/pubcollect-%j.log" prepare.sbatch tools.collect_yieldsat --sweep "$s")")
done
rm -f results/planteur/*.csv.tmp
for d in "$project"/runs/planteur/sweeps/*/; do
  [ -d "$d" ] || continue
  s=$(basename "$d"); [[ $s == _* ]] && continue
  sbatch --parsable --job-name=ys-pub-collect-pl-$s --time=00:20:00 --mem=4G --cpus-per-task=1 \
    --output="$d/_logs/pubcollect-%j.log" prepare.sbatch tools.collect_planteur --sweep "$s" >/dev/null
done
for _ in $(seq 1 90); do squeue --me -h -o %j | grep -q ys-pub-collect || break; sleep 10; done
mkdir -p "$HOME/agribench-publish" && rm -rf "$HOME/agribench-publish"/* && mkdir -p "$HOME/agribench-publish/yieldsat" "$HOME/agribench-publish/earthshift" "$HOME/agribench-publish/planteur"
cp results/yieldsat/*.csv "$HOME/agribench-publish/yieldsat/" 2>/dev/null || true
cp results/planteur/*.csv "$HOME/agribench-publish/planteur/" 2>/dev/null || true
git checkout -q results; git clean -fq results 2>/dev/null || true
# EarthShift: always the latest main's tool, in its own detached worktree. The sweep checkout stays pinned for
# YieldSAT (code-version rule), so its copy of tools/earthshift_results.py can be out of date.
es=$checkout/../agribench-es-results
git fetch -q origin main
[ -d "$es" ] || git worktree add -q --detach "$es" FETCH_HEAD
git -C "$es" checkout -q --detach "$(git rev-parse FETCH_HEAD)"
(cd "$es" && "$py" -m tools.earthshift_results --csv "$HOME/agribench-publish/earthshift/final.csv" >/dev/null) || echo "EarthShift export failed"
# Diagnostics (bootstrap CIs, error decomposition, rank persistence; protocols/yieldsat.md 2.10) for every finished
# run of every sweep, so every new result gets its decomposition in the same publish. Rows are cached per run, so
# only runs new since the last pass are computed. One job at a time (singleton): a pass already running is waited for.
diag=$HOME/agribench-diagnostics
mkdir -p "$diag" "$HOME/agribench-publish/yieldsat/diagnostics"
args=(); for d in "$project"/runs/yieldsat/sweeps/*/; do s=$(basename "$d"); [[ $s == _* ]] || args+=(--sweep "$s"); done
sbatch --wait --dependency=singleton --job-name=ys-diagnostics --time=02:00:00 --mem=16G --cpus-per-task=2 \
  --output="$diag/diagnostics-%j.log" prepare.sbatch tools.yieldsat_diagnostics "${args[@]}" --out "$diag" >/dev/null \
  || echo "Diagnostics job failed; publishing the last finished output"
cp "$diag"/*.csv "$HOME/agribench-publish/yieldsat/diagnostics/" 2>/dev/null || true
ls -l "$HOME"/agribench-publish/*/
REMOTE

stamp=$(date +%Y%m%d-%H%M)
branch=results/auto-$stamp
git checkout -q main && git pull -q
git checkout -q -b "$branch"
for ds in yieldsat earthshift planteur yieldsat/diagnostics; do
  mkdir -p results/$ds
  scp -q "$HOST:agribench-publish/$ds/*.csv" results/$ds/ 2>/dev/null || true
done
python3 - <<'PY'
import datetime, json, pathlib
files = sorted(str(p.relative_to("results")) for p in pathlib.Path("results").glob("*/*.csv"))
now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
diag = sorted(str(p.relative_to("results")) for p in pathlib.Path("results").glob("yieldsat/diagnostics/*-decomp.csv"))
ci = sorted(str(p.relative_to("results")) for p in pathlib.Path("results").glob("yieldsat/diagnostics/*-ci.csv"))
pathlib.Path("results/index.json").write_text(json.dumps({"updated": now, "files": [{"path": f} for f in files],
                                                          "diagnostics": diag, "intervals": ci}, indent=2) + "\n")
PY
git add results
if git diff --cached --quiet -- results/*/*.csv results/yieldsat/diagnostics; then
  echo "No new results since the last publish."; git reset -q --hard; git checkout -q main; git branch -q -D "$branch"; exit 0
fi
git commit -q -m "Results: automatic publish $stamp (aggregates only, finished runs so far)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q -u origin "$branch"
summary=$(git diff --stat HEAD~1 -- results | tail -1)
url=$(gh pr create --title "Results: publish $stamp" --body "Latest aggregate results, including sweeps that are still running (complete runs only; EarthShift rows show their seed count). $summary

🤖 Generated with [Claude Code](https://claude.com/claude-code)")
echo "$url"
if [ $MERGE = 1 ]; then
  for _ in $(seq 1 30); do
    state=$(gh pr view "$url" --json mergeStateStatus --jq .mergeStateStatus)
    [ "$state" = CLEAN ] && break; sleep 10
  done
  [ "$state" = CLEAN ] && gh pr merge "$url" --merge --delete-branch && git checkout -q main && git pull -q || echo "PR not clean ($state); left open"
fi
