#!/bin/bash
# Layered environment for ARC (tools/build_yieldsat_arc.py): a venv on top of benchmark-v2, so the shared env is
# never modified. Run once on the Anunna login node (needs internet):
#     bash tools/setup_arc_env.sh
set -euo pipefail
project=/lustre/shared/agriscience-benchmark
base=$project/envs/benchmark-v2/bin/python
venv=$project/envs/arc
ARC_COMMIT=7a200007b0d48df22dc2b0b5b382ee84675730e1
[ -d "$venv" ] || "$base" -m venv "$venv"
# benchmark-v2 is itself a venv, so --system-site-packages would not see its packages: link them with a .pth file
# (packages installed here take precedence).
"$base" -c "import site; print(site.getsitepackages()[0])" > "$("$venv/bin/python" -c "import site; print(site.getsitepackages()[0])")/benchmark-v2.pth"
"$venv/bin/pip" install -q "jax[cpu]" pynndescent numba tqdm "whitsmooth-rust==0.1.3"
# ARC without its dependencies: its `eof` dependency only downloads imagery and needs the GDAL system library.
"$venv/bin/pip" install -q --no-deps "ARC @ https://github.com/MarcYin/ARC/archive/$ARC_COMMIT.zip"
"$venv/bin/python" -c "import jax, pynndescent, whitsmooth_rust, arc.assimilate_jax; print('ARC env ok, jax', jax.__version__)"
