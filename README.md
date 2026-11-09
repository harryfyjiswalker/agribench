# Benchmarks and Scores

## Crop Classification

### In-Distribution (PASTIS-R)

- [PASTIS-R](https://zenodo.org/records/5745151) (Garnot and Landrieu, 2021)
  - A benchmark that supports parcel-based crop type classification, comprising 85,951 individual parcels and 18 crop types in metropolitan France. The benchmark covers one year, the September 2018 to November 2019 growing season. PASTIS can be used for in-distribution scoring, using the pre-determined five folds provided by the dataset maintainers. 


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

### Out-Of-Distribution (PLANTEUR)

- [PLANTEUR](https://huggingface.co/datasets/allenai/planteur-benchmark) (Wolters _et al._, 2026): crop-type segmentation in France's overseas territories, scored by mIoU-8 with leave-one-island-out; protocol and download in [protocols/planteur.md](protocols/planteur.md).

## Yield Prediction

- [YieldSAT](https://yieldsat.github.io/) (Miranda _et al._, 2026)
  - Pixel-level (10 m) yield maps for 2,173 fields in Argentina, Brazil, Germany and Uruguay (corn, rapeseed,
    soybean, wheat, 2016–2024), with Sentinel-2, weather, soil and topography inputs. Scored per country–crop
    subset with the paper's stratified grouped 10-fold CV, leave-one-year-out and leave-one-region-out splits;
    the final evaluation is leave-one-country-out per crop. Protocol:
    [protocols/yieldsat.md](protocols/yieldsat.md).

## Phenology Detection

## Field Delineation

- [EarthShift](https://earthshift.github.io/) (Doerksen and Kerner, 2026), field-delineation tasks
  - 3-class field segmentation (background, field interior, boundary) on Fields of The World Sentinel-2
    chips (256 × 256 px, RGB + NIR), scored with ID and OOD mIoU. Covers the location-shift tasks
    (train in Germany, test in Cambodia and in Denmark) and the temporal-shift tasks (South Africa season
    to season, Germany 2018 to 2019). Protocol: [protocols/earthshift.md](protocols/earthshift.md).


_____________________________________________________________________________________________________________________________________________________________________

# Setup

## 1. Log in

    ssh USERNAME@login.anunna.wur.nl

Replace `USERNAME` with your WUR username (off campus you may need the WUR VPN). You need access to the
`AIN` group. The login node is for light work only: run anything heavy with `sbatch`.

## 2. Get the code

    mkdir -p ~/projects && cd ~/projects
    git clone --recurse-submodules https://github.com/harryfyjiswalker/agribench.git
    cd agribench
    export project=/lustre/shared/agriscience-benchmark    # shared files; set again in each new session

## 3. Run an experiment (PASTIS-R)

    sbatch bench.sbatch encoder=galileo-base input=pastis-r/s2-parcel-medians model=xgboost validation=pilot

| Choice | Options |
|---|---|
| `encoder=` | `galileo-base`, `summaries` (raw monthly bands), `presto-base` |
| `input=` | `pastis-r/s2-parcel-medians` (galileo-base, summaries); `pastis-r/s2-parcel-medians-location` (presto-base) |
| `model=` | `logistic_regression`, `rf`, `extra_trees`, `xgboost`, `lightgbm`, `mlp`, `tabpfn` |
| `validation=` | `pilot` (default), `rehearsal`, `final` |

- `pilot`: train on folds 1-3 with default settings, score fold 4.
- `rehearsal`: tune on fold 3 (grid in the model's YAML, macro IoU), refit on folds 1-3, score fold 4. The `final`
  procedure shifted one fold earlier, so it never touches fold 5.
- `final`: tune on fold 4, refit on folds 1-4, score fold 5 once. Needs `experiment.confirm_test=true`; each run is
  logged in `$project/runs/pastis/test-ledger.jsonl`. Fold 5 is reserved: never use it to choose models.
- Add `experiment.smoke=true` to check the plumbing on a few thousand parcels (scores are meaningless).

`tabpfn` needs a GPU (`mlp` can use one): add `--partition=gpu --gres=gpu:1 --mem=32G` after `sbatch`.

Many at once (every encoder x every model, in parallel; set `INPUT=` for Presto, `SBATCH_ARGS=` for GPUs):

    bash tools/sweep.sh "galileo-base summaries" "logistic_regression rf xgboost" validation=pilot

Follow jobs with `squeue --me`; the log is `slurm-<jobid>.out` in the folder you submitted from. Most pilot runs
take under 10 minutes. Everyone's runs in one table (add `--smoke` to include smoke runs):

    $project/envs/benchmark-v2/bin/python tools/results.py

Each run writes `$project/runs/pastis/$USER/<time>-<jobid>/` with `metrics.json` (accuracy, macro IoU, macro F1),
predictions and the resolved config.

### Run an experiment (YieldSAT)

    sbatch yieldsat.sbatch subset=ARG-S scheme=cv10 input=s2 model=rf

| Choice | Options |
|---|---|
| `subset=` | `ARG-C`, `ARG-S`, `ARG-W`, `BRA-C`, `BRA-S`, `BRA-W`, `GER-R`, `GER-W`, `URG-S` (country–crop) |
| `scheme=` | `cv10` (standard 10-fold), `loyo` (leave one year out), `loro` (leave one region out), `loco` (leave one country out, the final evaluation) |
| `input=` | `s2` (Sentinel-2 only), `s2-adm` (plus weather, soil, topography, coordinates) |
| `model=` | `mean` (reference), `ridge`, `rf`, `extra_trees`, `xgboost`, `lightgbm` |

- `scheme=loco` takes `crop=corn|soybean|wheat` instead of `subset=`. It needs `experiment.confirm_test=true`,
  and each run is logged in `$project/runs/yieldsat/loco-ledger.jsonl`. Pick models on the other schemes first.
- Add `experiment.smoke=true` for a quick plumbing check (2 folds, a few pixels per field, scores meaningless).
- Each run writes `$project/runs/yieldsat/$USER/<time>-<jobid>/metrics.json`, with R² and RMSE at pixel and
  field level per fold, mean ± std over folds and pooled. It also saves per-pixel predictions and the tuning
  results.
- How folds, tuning, the 256-pixel training cap and the metrics work: [protocols/yieldsat.md](protocols/yieldsat.md).
  Folds and feature caches are already built on Anunna; `prepare.sbatch` rebuilds them if needed.

## 4. Where things live

| Path | Contents |
|---|---|
| `$project/envs/benchmark-v2` | shared Python environment (read-only) |
| `$project/features/pastis-r/parcel-probe-v1` | extracted encoder features, one `.npz` per encoder and input |
| `$project/inputs/pastis-r` | prepared inputs the encoders read |
| `$project/checkpoints`, `$project/upstream` | model weights and pinned encoder source code |
| `$project/runs/pastis` | run outputs and the fold-5 ledger |
| `$project/project.env` | optional Anunna project number for billing, one line `ANUNNA_PROJECT=<number>` |

Every batch script first loads `tools/slurm_env.sh`. It limits library threads to the cores the job asked for. If
`$project/project.env` (or your own `export ANUNNA_PROJECT=<number>`) gives a project number, it also tags the job
with it, as Anunna asks for billing, unless you passed `--comment` yourself. Without a number, jobs run as before.

The PASTIS-R data path is set in `conf/dataset/pastis_s2.yaml`. Environment overrides: `AGRIBENCH_DATA`,
`AGRIBENCH_FEATURES`, `AGRIBENCH_RUNS`, `AGRIBENCH_PROJECT`, `AGRIBENCH_TABPFN_CKPT`, `BENCHMARK_PYTHON`.

## 5. Add a model (classifier head)

Copy `conf/model/lightgbm.yaml`: set `name`, an `estimator` (any scikit-learn-compatible class via `_target_`) and
optionally `search.param_grid` (used by `rehearsal` and `final`). Then run it with `model=<name>`. The package must
be in the shared environment (section 7). Neural heads: see `agribench_heads.py`.

## 6. Add an encoder

1. Copy `tools/adapters/template.py` and implement `extract(...) -> (parcel_ids, features)` (see `presto.py`).
2. Add `conf/encoder/<name>.yaml`; pin the source revision and weights hash as the existing ones do.
3. Check 8 parcels (add `--source-dir`/`--weights-dir` if the encoder needs them), then extract everything once
   on a GPU. Presto example (`<DATA>` is the PASTIS-R_PixelSet folder):

       python -m tools.extract_representation encoder=presto-base input=pastis-r/s2-parcel-medians-location \
         --inputs $project/inputs/pastis-r/pastis-s2-and-location.npz --metadata <DATA>/metadata_parcel.csv \
         --source-dir $project/upstream/presto --device cpu --smoke

       sbatch --partition=gpu --gres=gpu:1 --mem=32G extract.sbatch encoder=presto-base \
         input=pastis-r/s2-parcel-medians-location --inputs $project/inputs/pastis-r/pastis-s2-and-location.npz \
         --metadata <DATA>/metadata_parcel.csv --source-dir $project/upstream/presto \
         --output-dir $project/features/pastis-r/parcel-probe-v1

   Galileo also needs `--weights-dir $project/checkpoints/galileo/base`. Extraction refuses to overwrite an
   existing table. The result is shared, so every encoder after that runs with `encoder=<name>`.

## 7. Python environment

Everyone uses the shared read-only `$project/envs/benchmark-v2`; the batch scripts already point at it. Do not pip
install into it. To add a package, edit `envs/benchmark-v2.lock.txt` in a pull request; the maintainer builds the
next versioned environment. Your own copy:

    python3 -m venv ~/myenv
    ~/myenv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r envs/benchmark-v2.lock.txt
    BENCHMARK_PYTHON=~/myenv/bin/python sbatch bench.sbatch ...

## 8. Publish a validation result

    $project/envs/benchmark-v2/bin/python tools/export_result.py <run folder> <short-name>
    git add results/<short-name>.json && git commit -m "Add <short-name>" && git push

Only `pilot` runs can be exported (not smoke runs). The leaderboard above rebuilds automatically after the push.
Without push access, fork the repo and open a pull request.

## 9. Working with an AI agent (Claude Code and others)

The repo includes the [Anunna skill](https://git.wur.nl/anunna/anunna-skill) as a git submodule at
`.claude/skills/anunna`. It teaches the agent how to use the cluster: ssh, Slurm, Lustre, environments and job
costs. Claude Code picks it up automatically when you open this repo. If you cloned without
`--recurse-submodules`, fetch it with:

    git submodule update --init

Update it to the latest version (then commit the new pointer):

    git submodule update --remote .claude/skills/anunna

For an agent on your laptop to reach Anunna, add connection sharing to the `Host anunna` block in
`~/.ssh/config`. The agent then reuses a login you opened yourself and needs no key of its own (see
`.claude/skills/anunna/references/windows-and-agents.md`):

    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h-%p
    ControlPersist 8h

Open the connection with `ssh -fN anunna` before starting the agent, and close it with `ssh -O exit anunna`.
