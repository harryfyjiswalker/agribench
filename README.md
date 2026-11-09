# AgriBench

A shared benchmark for the AIN group: pick a **dataset**, an **input** (raw satellite data, a location encoder, a
geospatial foundation model's embedding, …) and a **head** (linear probe, LightGBM, TabPFN, …), and AgriBench
scores it with that dataset's official cross-validation and a fixed tuning procedure. Everything runs on
[Anunna](https://wiki.anunna.wur.nl) with shared data, features and environments, so results from different
people are directly comparable.

**Results:** [dashboard](https://harryfyjiswalker.github.io/agribench/), updated whenever results are pushed.

## Datasets

| Task | Dataset | Scored on | Status | Protocol |
|---|---|---|---|---|
| Crop classification | [PLANTEUR](https://huggingface.co/datasets/allenai/planteur-benchmark) (French overseas territories) | mIoU-8, leave one island out | protocol agreed, data downloading, no runner yet | [protocols/planteur.md](protocols/planteur.md) |
| Yield prediction | [YieldSAT](https://yieldsat.github.io/) (Argentina, Brazil, Germany, Uruguay; 2,173 fields, 10 m yield maps) | R² and RMSE per field and pixel; 10-fold, leave one year out, leave one region out; final: leave one country out | runner, sweeps, 8 representations; first sweep running | [protocols/yieldsat.md](protocols/yieldsat.md) |
| Field delineation | [EarthShift](https://earthshift.github.io/) (location and temporal shift tasks) | in- and out-of-distribution mIoU | splits and inputs built; training code under review, learning-rate sweeps running | [protocols/earthshift.md](protocols/earthshift.md) |
| Phenology | – | – | not started | – |

## Quick start

    ssh USERNAME@login.anunna.wur.nl          # WUR username; off campus you may need the VPN; needs the AIN group
    git clone --recurse-submodules https://github.com/harryfyjiswalker/agribench.git
    cd agribench
    export project=/lustre/shared/agriscience-benchmark

    # one YieldSAT run: Argentina soybean, standard 10-fold CV, raw Sentinel-2, linear probe
    sbatch yieldsat.sbatch subset=ARG-S scheme=cv10 input=s2 model=probe

Follow it with `squeue --me`. The run prints its output folder; `metrics.json` there has the scores. The login node
is for light work only: anything heavy goes through `sbatch`.

## YieldSAT

### One run

    sbatch yieldsat.sbatch subset=<subset> scheme=<scheme> input=<input> model=<head>

| Choice | Options |
|---|---|
| `subset=` | `ARG-C`, `ARG-S`, `ARG-W`, `BRA-C`, `BRA-S`, `BRA-W`, `GER-R`, `GER-W`, `URG-S` (country–crop: C corn, R rapeseed, S soybean, W wheat) |
| `scheme=` | `cv10` (the paper's grouped 10-fold), `loyo` (leave one year out), `loro` (leave one region out), `loco` (leave one country out: the final evaluation, see below) |
| `input=` | any input below, or several joined with `+` (e.g. `s2+mind`, `galileo+mind`) |
| `model=` | `probe` (linear probe: the main head for comparing inputs), `lightgbm`, `tabpfn` and `tabicl` (GPU), `mean` (reference); also `rf`, `extra_trees`, `xgboost` |

**Inputs** (all pre-computed on Anunna, one row per 10 m pixel):

| Input | What it is | Size |
|---|---|---|
| `s2` | raw Sentinel-2: 12 bands × 24 time steps | 288 |
| `s2-adm` | `s2` plus weather, soil, topography and coordinates (the paper's "S2 + ADM") | 488 |
| `latlon` | the pixel's latitude and longitude: the plain control for the location encoders | 2 |
| `mind` | [MIND](https://arxiv.org/abs/2609.25454) location encoder; location only | 3072 |
| `satclip` | [SatCLIP](https://arxiv.org/abs/2311.17179) location encoder; location only | 256 |
| `galileo` | [Galileo](https://github.com/nasaharvest/galileo)-Base embedding of the pixel's Sentinel-2 series | 768 |
| `aef` | [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) annual embedding for the harvest year | 64 |
| `tessera` | [TESSERA](https://arxiv.org/abs/2506.20380) annual embeddin for the harvest year | 128 |

Location encoders give the same embedding in every year, so on their own they can't follow year-to-year yield
changes; combine them with a time-varying input. How each input was built and checked: protocol section 2.5.

**Heads.** `probe` is a closed-form ridge on standardised features, as in the TorchGeo evaluation harness, with
its penalty tuned inside each training fold. Tree heads tune a small grid the same way. TabPFN and TabICL don't
tune: they fit once per fold on 32 pixels per field. Details: protocol section 2.6.

**What a run gives you.** R² and RMSE at field and pixel level, per fold and as mean ± std over folds, plus
per-pixel predictions and the tuning results, in `$project/runs/yieldsat/$USER/<time>-<jobid>/`.

**Final evaluation.** `scheme=loco` takes `crop=corn|soybean|wheat` instead of `subset=` and needs
`experiment.confirm_test=true`; every such run is logged in `$project/runs/yieldsat/loco-ledger.jsonl`. Choose
models on `cv10`, `loyo` and `loro` first.

Add `experiment.smoke=true` for a two-minute plumbing check (scores meaningless).

### Many runs: sweeps

    $project/envs/benchmark-v2/bin/python -m tools.sweep_yieldsat --name mysweep \
        --inputs s2 galileo mind galileo+mind --models probe lightgbm
    # add --dry-run to see the plan first; add --subsets / --schemes to narrow it

The sweep tool submits every subset × scheme × input × head as Slurm job arrays and takes care of efficiency:
- CPU heads run one task per fold, so the folds of a run go in parallel. GPU heads run one task per run, so
  each loads its model once.
- CPUs, memory and time limits are set per head from measured use. Anunna bills what you request, so check
  `tools.collect_yieldsat --sweep mysweep --efficiency` after a sweep and tighten them in
  `tools/sweep_yieldsat.py`.
- At most 200 tasks run at once (Anunna's guidance for jobs that read Lustre).
- The sweep runs from a frozen copy of the code at your current commit. Editing your checkout never changes
  queued tasks, and every result records the commit it came from. Commit before you sweep.
- Re-running the same command resumes: finished folds are skipped.
- When the arrays finish, a collect job combines the folds and writes `results/yieldsat/mysweep.csv`. That table
  holds aggregate scores only, safe to commit; the dashboard reads it.

## EarthShift and PLANTEUR

Both protocols are agreed: [protocols/earthshift.md](protocols/earthshift.md) and
[protocols/planteur.md](protocols/planteur.md). EarthShift's splits and inputs are built on Anunna, and its
training code is on the branch `earthshift/training`, to be merged with its first results. PLANTEUR is still
downloading and has no runner yet.

## Results

All results go to the [dashboard](https://harryfyjiswalker.github.io/agribench/), one view per dataset. It reads
the aggregate tables in `results/<dataset>/`, which the sweep tools write; commit and push a table to publish it.
YieldSAT is there now; EarthShift and PLANTEUR join when their first results are in.

The repo is public. Commit aggregate scores only: never field names, locations, fold lists or predictions (the
datasets have their own licences).

## Where things live on Anunna

`$project` is `/lustre/shared/agriscience-benchmark`.

| Path | Contents |
|---|---|
| `$project/datasets/` | raw datasets (YieldSAT, PLANTEUR, EarthShift; PASTIS-R archived) |
| `$project/splits/` | frozen fold files per dataset (never edited by hand) |
| `$project/features/yieldsat/` | YieldSAT inputs: `v1/` (raw), and one folder per representation (`mind/`, `galileo/`, `aef/`, …) |
| `$project/runs/` | run outputs per dataset; `runs/yieldsat/sweeps/<name>/` for sweeps; test ledgers |
| `$project/checkpoints/`, `$project/upstream/` | model weights and pinned model source code |
| `$project/envs/` | shared Python environments: `benchmark-v2` (everything), `tessera` (adds geotessera) |
| `$project/project.env` | optional Anunna project number for billing: one line `ANUNNA_PROJECT=<number>` |

Every batch script loads `tools/slurm_env.sh` first. It limits library threads to the cores the job asked for and,
if `$project/project.env` (or your own `export ANUNNA_PROJECT=<number>`) gives a project number, tags the job with
it for billing.

## Extending AgriBench

- **A new head:** copy a file in `conf/yieldsat/model/`. Set `name`, an
  `estimator` (any scikit-learn-compatible class via `_target_`) and optionally `search.param_grid`; then use
  `model=<name>`. Add its resources to `RESOURCES` in `tools/sweep_yieldsat.py` to sweep it.
- **A new YieldSAT input:** write a script that saves `features/yieldsat/<name>/<subset>.npy` with rows in the
  order of `v1/<subset>-meta.npz`, plus a `<subset>.json` with `kind`, `feature_names`, `rows` and that meta
  file's `v1_meta_sha256`. `tools/extract_yieldsat_aef.py` is a short example. It's then usable as
  `input=<name>`, alone or with `+`.

## Python environment

Everyone uses the shared read-only `$project/envs/benchmark-v2`; the batch scripts already point at it. Don't
pip install into it: to add a package, add it to `envs/benchmark-v2.lock.txt` in a pull request. For your own copy:

    python3 -m venv ~/myenv
    ~/myenv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r envs/benchmark-v2.lock.txt
    BENCHMARK_PYTHON=~/myenv/bin/python sbatch yieldsat.sbatch ...

## Working with an AI agent (Claude Code and others)

The repo includes the [Anunna skill](https://git.wur.nl/anunna/anunna-skill) as a submodule at
`.claude/skills/anunna`, which teaches an agent the cluster (ssh, Slurm, Lustre, environments, costs). Claude Code
picks it up automatically. If you cloned without `--recurse-submodules`, run `git submodule update --init`; to
update it, `git submodule update --remote .claude/skills/anunna` and commit the new pointer.

For an agent on your laptop to reach Anunna, add connection sharing to the `Host anunna` block of
`~/.ssh/config`, so it reuses a login you opened yourself and needs no key of its own:

    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h-%p
    ControlPersist 8h

Open the connection with `ssh -fN anunna` before starting the agent; close it with `ssh -O exit anunna`.
