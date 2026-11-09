#!/bin/bash
# Layered environment for WOFOST (tools/build_yieldsat_wofost.py, tools/build_yieldsat_wofost_da.py): a venv on top
# of benchmark-v2 with PCSE, so the shared env is never modified, plus a pinned copy of the WOFOST crop parameter
# files (PCSE would otherwise download them at run time). Run once on the Anunna login node (needs internet):
#     bash tools/setup_pcse_env.sh
set -euo pipefail
project=/lustre/shared/agriscience-benchmark
base=$project/envs/benchmark-v2/bin/python
venv=$project/envs/pcse
PCSE_VERSION=6.0.13
CROP_PARAMS_COMMIT=${CROP_PARAMS_COMMIT:-master}
params=$project/upstream/wofost-crop-parameters
[ -d "$venv" ] || "$base" -m venv "$venv"
# benchmark-v2 is itself a venv, so --system-site-packages would not see its packages: link them with a .pth file.
"$base" -c "import site; print(site.getsitepackages()[0])" > "$("$venv/bin/python" -c "import site; print(site.getsitepackages()[0])")/benchmark-v2.pth"
"$venv/bin/pip" install -q "pcse==$PCSE_VERSION"
if [ ! -d "$params" ]; then
  git clone -q https://github.com/ajwdewit/WOFOST_crop_parameters.git "$params"
fi
git -C "$params" checkout -q "$CROP_PARAMS_COMMIT"
git -C "$params" rev-parse HEAD
"$venv/bin/python" -c "import pcse; print('PCSE env ok', pcse.__version__)"
