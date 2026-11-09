#!/bin/bash
# Layered environment for ARC (tools/build_yieldsat_arc.py): a venv on top of benchmark-v2, so the shared env is
# never modified. Run once on the Anunna login node (needs internet):
#     bash tools/setup_arc_env.sh
set -euo pipefail
project=/lustre/shared/agriscience-benchmark
base=$project/envs/benchmark-v2/bin/python
venv=$project/envs/arc
ARC_COMMIT=7a200007b0d48df22dc2b0b5b382ee84675730e1
EOF_COMMIT=eebd01b9f1c914cc1a6029b3d246b739d29c4ac9
[ -d "$venv" ] || "$base" -m venv --system-site-packages "$venv"
"$venv/bin/pip" install -q "jax[cpu]" pynndescent numba tqdm "whitsmooth-rust==0.1.3" \
  "eof @ https://github.com/profLewis/eof/archive/$EOF_COMMIT.zip" \
  "ARC @ https://github.com/MarcYin/ARC/archive/$ARC_COMMIT.zip"
"$venv/bin/python" -c "import jax, pynndescent, whitsmooth_rust, arc.assimilate_jax; print('ARC env ok, jax', jax.__version__)"
