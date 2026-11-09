# AgriBench

A shared benchmark for the AIN group: pick a **dataset**, an **input** (raw satellite data, a location encoder, a
geospatial foundation model's embedding, …) and a **head** (linear probe, LightGBM, TabPFN, …), and AgriBench
scores it with that dataset's official cross-validation and a fixed tuning procedure. Everything runs on
[Anunna](https://wiki.anunna.wur.nl) with shared data, features and environments, so results from different
people are directly comparable.

**Results:** [dashboard](https://harryfyjiswalker.github.io/agribench/) (YieldSAT; updates whenever results
are pushed) · PASTIS-R table [below](#results).

## Datasets

| Task | Dataset | Scored on | Status | Protocol |
|---|---|---|---|---|
| Crop classification, in distribution | [PASTIS-R](https://zenodo.org/records/5745151) (France, 18 crops, 85,951 parcels) | parcel mIoU, official 5 folds (fold 5 reserved) | runner and first results | [protocols/protocols.md](protocols/protocols.md) |
| Crop classification, out of distribution | [PLANTEUR](https://huggingface.co/datasets/allenai/planteur-benchmark) (French overseas territories) | mIoU-8, leave one island out | protocol agreed, data downloading, no runner yet | [protocols/planteur.md](protocols/planteur.md) |
| Yield prediction | [YieldSAT](https://yieldsat.github.io/) (Argentina, Brazil, Germany, Uruguay; 2,173 fields, 10 m yield maps) | R² and RMSE per field and pixel; 10-fold, leave one year out, leave one region out; final: leave one country out | runner, sweeps, 8 representations; first sweep running | [protocols/yieldsat.md](protocols/yieldsat.md) |
| Field delineation | [EarthShift](https://earthshift.github.io/) (location and temporal shift tasks) | in- and out-of-distribution mIoU | splits and inputs built; training code under review, learning-rate sweeps running | [protocols/earthshift.md](protocols/earthshift.md) |
| Phenology | – | – | not started | – |

Every protocol follows the dataset paper exactly where the paper is specific, and marks where AgriBench had to
decide something itself.

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
| `mind` | [MIND](https://arxiv.org/abs/2609.25454) location encoder (Corley, …, Klemmer et al., 2026); location only | 3072 |
| `satclip` | [SatCLIP](https://arxiv.org/abs/2311.17179) location encoder (Klemmer et al.); location only | 256 |
| `galileo` | [Galileo](https://github.com/nasaharvest/galileo)-Base embedding of the pixel's Sentinel-2 series | 768 |
| `aef` | [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) annual embedding (Google DeepMind) for the harvest year | 64 |
| `tessera` | [TESSERA](https://arxiv.org/abs/2506.20380) annual embedding (Cambridge) for the harvest year | 128 |

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

## PASTIS-R

    sbatch bench.sbatch encoder=galileo-base input=pastis-r/s2-parcel-medians model=xgboost validation=pilot

| Choice | Options |
|---|---|
| `encoder=` | `galileo-base`, `summaries` (raw monthly bands), `presto-base` |
| `input=` | `pastis-r/s2-parcel-medians` (galileo-base, summaries); `pastis-r/s2-parcel-medians-location` (presto-base) |
| `model=` | `logistic_regression`, `rf`, `extra_trees`, `xgboost`, `lightgbm`, `mlp`, `tabpfn` |
| `validation=` | `pilot` (default), `rehearsal`, `final` |

- `pilot`: train on folds 1–3 with default settings, score fold 4.
- `rehearsal`: tune on fold 3 (macro IoU), refit on folds 1–3, score fold 4: the `final` procedure one fold
  earlier, so it never touches fold 5.
- `final`: tune on fold 4, refit on folds 1–4, score fold 5 once. Needs `experiment.confirm_test=true`; logged in
  `$project/runs/pastis/test-ledger.jsonl`. Fold 5 is reserved: never use it to choose models.
- `tabpfn` needs a GPU: add `--partition=gpu --gres=gpu:1 --mem=32G` after `sbatch`.
- Many at once: `bash tools/sweep.sh "galileo-base summaries" "logistic_regression rf xgboost" validation=pilot`.
- Everyone's runs in one table: `$project/envs/benchmark-v2/bin/python tools/results.py`.

## EarthShift and PLANTEUR

Both protocols are agreed: [protocols/earthshift.md](protocols/earthshift.md) and
[protocols/planteur.md](protocols/planteur.md). EarthShift's splits and inputs are built on Anunna, and its
training code is on the branch `earthshift/training`, to be merged with its first results. PLANTEUR is still
downloading and has no runner yet.

## Results

- **YieldSAT and future datasets:** the [dashboard](https://harryfyjiswalker.github.io/agribench/). It reads
  the tables in `results/<dataset>/`, which sweeps write; commit and push a table to publish it.
- **PASTIS-R** (validation fold 4; publish a run with `tools/export_result.py <run folder> <short-name>`, then
  commit `results/<short-name>.json`; this table rebuilds automatically):

<!-- leaderboard:start -->


| Run | Model | Representation | Sensors | Train → validation folds | Train parcels | Validation parcels | Seed | Epoch | Parcel mIoU | Accuracy | Macro F1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [`s2-tabpfn35-fast-full-pilot`](results/s2-tabpfn35-fast-full-pilot.json) | tabpfn | Parcel summaries | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.702 | 0.906 | 0.807 |
| [`s2-galileo-base-logistic-regression-full-pilot`](results/s2-galileo-base-logistic-regression-full-pilot.json) | logistic_regression | Galileo Base embeddings | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.605 | 0.873 | 0.726 |
| [`s2-galileo-base-tabpfn35-fast-full-pilot`](results/s2-galileo-base-tabpfn35-fast-full-pilot.json) | tabpfn | Galileo Base embeddings | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.596 | 0.874 | 0.717 |
| [`s2-logistic-regression-full-pilot-43601671`](results/s2-logistic-regression-full-pilot-43601671.json) | logistic_regression | Parcel summaries | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.596 | 0.865 | 0.717 |
| [`s2-rf-full-pilot`](results/s2-rf-full-pilot.json) | rf | Parcel summaries | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.508 | 0.840 | 0.618 |
| [`s2-pse-ltae-1epoch-seed42`](results/s2-pse-ltae-1epoch-seed42.json) | adapted_pse_ltae2d | Parcel pixel sets | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | 1 | 0.308 | 0.737 | 0.384 |


<!-- leaderboard:end -->

The repo is public. Commit aggregate scores only: never field names, locations, fold lists or predictions (the
datasets have their own licences).

## Where things live on Anunna

`$project` is `/lustre/shared/agriscience-benchmark`.

| Path | Contents |
|---|---|
| `$project/datasets/` | raw datasets (YieldSAT, PASTIS-R, PLANTEUR, EarthShift) |
| `$project/splits/` | frozen fold files per dataset (never edited by hand) |
| `$project/features/yieldsat/` | YieldSAT inputs: `v1/` (raw), and one folder per representation (`mind/`, `galileo/`, `aef/`, …) |
| `$project/features/pastis-r/` | PASTIS-R encoder features |
| `$project/runs/` | run outputs per dataset; `runs/yieldsat/sweeps/<name>/` for sweeps; test ledgers |
| `$project/checkpoints/`, `$project/upstream/` | model weights and pinned model source code |
| `$project/envs/` | shared Python environments: `benchmark-v2` (everything), `tessera` (adds geotessera) |
| `$project/project.env` | optional Anunna project number for billing: one line `ANUNNA_PROJECT=<number>` |

Every batch script loads `tools/slurm_env.sh` first. It limits library threads to the cores the job asked for and,
if `$project/project.env` (or your own `export ANUNNA_PROJECT=<number>`) gives a project number, tags the job with
it for billing.

## Extending AgriBench

- **A new head:** copy a file in `conf/yieldsat/model/` (or `conf/model/` for PASTIS-R). Set `name`, an
  `estimator` (any scikit-learn-compatible class via `_target_`) and optionally `search.param_grid`; then use
  `model=<name>`. Add its resources to `RESOURCES` in `tools/sweep_yieldsat.py` to sweep it.
- **A new YieldSAT input:** write a script that saves `features/yieldsat/<name>/<subset>.npy` with rows in the
  order of `v1/<subset>-meta.npz`, plus a `<subset>.json` with `kind`, `feature_names`, `rows` and that meta
  file's `v1_meta_sha256`. `tools/extract_yieldsat_aef.py` is a short example. It's then usable as
  `input=<name>`, alone or with `+`.
- **A new PASTIS-R encoder:** copy `tools/adapters/template.py`, add `conf/encoder/<name>.yaml` with the source
  revision and weights hash pinned, smoke-test on 8 parcels, then extract once with `extract.sbatch`.

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
