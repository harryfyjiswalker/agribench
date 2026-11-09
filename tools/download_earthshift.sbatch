#!/bin/bash -l
# Download the Fields of The World (FTW) country archives that EarthShift's field-delineation tasks use,
# and check them. See protocols/earthshift.md, section 0. Submit from the repo root:
#
#     sbatch tools/download_earthshift.sbatch                      # the 4 countries the paper uses (4.3 GB)
#     COUNTRIES=all sbatch --time=12:00:00 tools/download_earthshift.sbatch   # all 25 countries (56.8 GB)
#
# The zips are kept as they are (one file per country, not ~26,000 small GeoTIFFs on Lustre). Readers open
# chips inside them with zipfile or GDAL's /vsizip/. Re-running skips zips that are already complete and
# resumes a partial one.
#SBATCH --partition=main
#SBATCH --qos=std
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --job-name=earthshift-download
set -euo pipefail
source "$SLURM_SUBMIT_DIR/tools/slurm_env.sh"   # threads, optional project number

dest=${EARTHSHIFT_DIR:-/lustre/shared/agriscience-benchmark/datasets/earthshift}/ftw-zips
base=https://data.source.coop/kerner-lab/fields-of-the-world-archive
countries=${COUNTRIES:-germany cambodia denmark south_africa}

# Files per zip (GeoTIFFs + chips_<country>.parquet + data_config_<country>.json), read from each
# archive's index on 2026-11-09. A different count means the upstream archive changed.
declare -A expected_files=(
  [austria]=33428 [belgium]=9707 [brazil]=8037 [cambodia]=1534 [corsica]=12362 [croatia]=17412
  [denmark]=17802 [estonia]=33567 [finland]=28327 [france]=17778 [germany]=3432 [india]=9932
  [kenya]=1957 [latvia]=34692 [lithuania]=26292 [luxembourg]=4042 [netherlands]=19397 [portugal]=352
  [rwanda]=352 [slovakia]=20367 [slovenia]=10887 [south_africa]=3737 [spain]=12170 [sweden]=23802
  [vietnam]=1438
)
[[ $countries == all ]] && countries=$(printf '%s\n' "${!expected_files[@]}" | sort | tr '\n' ' ')

mkdir -p "$dest"
cd "$dest"
curl -fsSL --retry 5 --retry-all-errors -o checksum.md5 "$base/checksum.md5"

status=0
for c in $countries; do
  if [[ -z ${expected_files[$c]:-} ]]; then echo "unknown country: $c" >&2; status=1; continue; fi
  if [[ ! -s $c.zip ]]; then
    echo "$(date +%T) downloading $c.zip"
    # A connection can stall at a few KB/s while a fresh one runs at MB/s, so give up on any transfer
    # slower than 100 KB/s for 60 s and resume the .part file on a new connection (up to 30 tries).
    for try in $(seq 30); do
      curl -fsSL --speed-limit 100000 --speed-time 60 -C - -o "$c.zip.part" "$base/$c.zip" && break
      echo "$(date +%T) $c: attempt $try stopped at $(stat -c %s "$c.zip.part" 2>/dev/null || echo 0) bytes, resuming"
      sleep 10
    done
    want_size=$(curl -fsSIL "$base/$c.zip" | awk 'tolower($1) == "content-length:" {n = $2} END {print n + 0}')
    [[ $(stat -c %s "$c.zip.part") == "$want_size" ]] || { echo "FAIL $c: incomplete download" >&2; status=1; continue; }
    mv "$c.zip.part" "$c.zip"
  fi
  want_md5=$(awk -F, -v c="$c" '$1 == c {print $2}' checksum.md5)
  have_md5=$(md5sum "$c.zip" | cut -d' ' -f1)
  have_files=$(unzip -Z1 "$c.zip" | grep -vc '/$' || true)
  if [[ $have_md5 == "$want_md5" && $have_files == "${expected_files[$c]}" ]]; then
    echo "OK   $c  $(du -h --apparent-size "$c.zip" | cut -f1)  $have_files files  md5 $have_md5"
  else
    echo "FAIL $c  md5 $have_md5 (want $want_md5), $have_files files (want ${expected_files[$c]})" >&2
    status=1
  fi
done
chmod -R g+rwX "$dest" 2>/dev/null || true
exit $status
