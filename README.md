# AgriBench

- a **dataset** (a task, such as yield prediction or crop classification);
- an **input**: what the model sees for each pixel. This can be raw Sentinel-2 data, a location encoder's
  embedding, or a geospatial foundation model's embedding;
- a **head**: the model trained on that input, such as a linear probe, LightGBM or TabPFN.

AgriBench then trains and scores it with that dataset's official cross-validation and a fixed tuning procedure.
Everything runs on [Anunna](https://wiki.anunna.wur.nl) with shared data, pre-computed inputs and Python
environments, so results from different people can be compared directly.

**Results:** [the dashboard](https://harryfyjiswalker.github.io/agribench/) is updated whenever results are pushed.

## Contents

- [Tasks](#tasks-at-a-glance)
- [Getting started](#getting-started)
- [YieldSAT: yield prediction](#yieldsat-yield-prediction)
- [EarthShift: field delineation under shift](#earthshift-field-delineation-under-shift)
- [PLANTEUR: crop classification](#planteur-crop-classification)
- [Results and the dashboard](#results-and-the-dashboard)
- [Where things live on Anunna](#where-things-live-on-anunna)
- [Adding an input or a head](#adding-an-input-or-a-head)
- [Python environment](#python-environment)
- [Working with an AI agent](#working-with-an-ai-agent)

## Tasks at a glance

| Task | Dataset | What it tests | Main score | Status (2026-11-09) |
|---|---|---|---|---|
| Yield prediction | [YieldSAT](https://yieldsat.github.io/) | predicting 10 m yield maps in four countries, including in unseen years and regions | R² and RMSE (t/ha), per field and per pixel | **Running.** Runner, sweeps and 8 inputs ready; first sweep under way |
| Field delineation | [EarthShift](https://earthshift.github.io/) | finding field boundaries in a new country or a new season | mIoU, in and out of distribution | **First results.** ResNet-50 and Galileo Base, frozen and full, 5 seeds |
| Crop classification | [PLANTEUR](https://huggingface.co/datasets/allenai/planteur-benchmark) | classifying crops on tropical islands, including an island never seen in training | mIoU over 8 classes | **Protocol agreed.** Data downloading; runner not written yet |
| Phenology | – | – | – | Not started |

Each task has a protocol in [`protocols/`](protocols/).

## Getting started

You need an Anunna account in the AIN group. Off campus you may need the WUR VPN.

    ssh USERNAME@login.anunna.wur.nl
    git clone --recurse-submodules https://github.com/harryfyjiswalker/agribench.git
    cd agribench
    export project=/lustre/shared/agriscience-benchmark

    # One YieldSAT run: Argentina soybean, standard 10-fold CV, raw Sentinel-2, linear probe
    sbatch yieldsat.sbatch subset=ARG-S scheme=cv10 input=s2 model=probe

Follow the job with `squeue --me`. When it finishes, its log names the output folder, and `metrics.json` in that
folder has the scores.

Two rules for the cluster: the login node is for light work only, so anything heavy goes through `sbatch`; and
the shared environments are read-only (see [Python environment](#python-environment)).

## YieldSAT: yield prediction

**The data.** 2,173 fields in Argentina, Brazil, Germany and Uruguay, each with a 10 m yield map from a combine
harvester and a Sentinel-2 time series of the growing season. It is split into nine **subsets**, one per
country and crop. The model predicts yield for every 10 m pixel; scores are reported per pixel and per field
(the field's mean yield). Full protocol: [protocols/yieldsat.md](protocols/yieldsat.md).

### One run

    sbatch yieldsat.sbatch subset=<subset> scheme=<scheme> input=<input> model=<head>

| Option | Choices |
|---|---|
| `subset=` | `ARG-C`, `ARG-S`, `ARG-W`, `BRA-C`, `BRA-S`, `BRA-W`, `GER-R`, `GER-W`, `URG-S`. The letters are the country, then the crop: C corn, R rapeseed, S soybean, W wheat. |
| `scheme=` | how fields are split into training and test folds. `cv10`: the paper's 10-fold CV, grouped by field. `loyo`: leave one year out. `loro`: leave one region out. `loco`: leave one country out, which is the final evaluation (see below). |
| `input=` | any input from the table below, or several joined with `+`, e.g. `s2+mind` or `galileo+mind` |
| `model=` | `probe` (the main head for comparing inputs), `lightgbm`, `tabpfn` and `tabicl` (these two need a GPU), `mean` (predicts the training mean: the floor every model must beat). Also available: `rf`, `extra_trees`, `xgboost`. |

`cv10` tells you how well a model fits fields like the ones it was trained on. `loyo` and `loro` are harder and
more useful: they test whether it still works in a year or region it has never seen.

### Inputs

All inputs are pre-computed on Anunna, with one row per 10 m pixel.

| Input | What it is | Features |
|---|---|---|
| `s2` | raw Sentinel-2: 12 bands × 24 time steps over the season | 288 |
| `s2-adm` | `s2` plus weather, soil, topography and coordinates (the paper's "S2 + ADM") | 488 |
| `s2-aligned` | `s2` put on a fixed season axis: each band interpolated to every 15 days since the field's sowing date. The raw `s2` time steps are not tied to the season, so the same step can be a different growth stage in different years | ~250 (varies by subset) |
| `s2-gdd` | like `s2-aligned`, but the time axis is growing degree days (accumulated heat) since sowing, from each field's daily weather | ~200 (varies) |
| `s2-harmonic` | each band's season summarised by a smooth curve (5 harmonic-regression coefficients per band); robust to cloud gaps | 60 |
| `biophys` | canopy variables per date (leaf area index, FAPAR, ground cover, canopy chlorophyll and water) from ESA SNAP's biophysical processor, on the `s2-aligned` grid | ~85 |
| `arc` | ARC retrieval per pixel (crop-model parameters, peak LAI and the LAI curve since sowing), after Whittaker outlier removal; internal testing | ~40 |
| `arcvae` | ARC crop-model parameters per pixel (canopy scaling and phenology timing, mean and uncertainty) from the ARC-VAE encoder; internal testing | 22 |
| `latlon` | the pixel's latitude and longitude: the simple control for the location encoders | 2 |
| `mind` | [MIND](https://arxiv.org/abs/2609.25454) location encoder. Sees only the location. | 3072 |
| `satclip` | [SatCLIP](https://arxiv.org/abs/2311.17179) location encoder. Sees only the location. | 256 |
| `galileo` | [Galileo](https://github.com/nasaharvest/galileo)-Base embedding of the pixel's Sentinel-2 time series | 768 |
| `aef` | [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) annual embedding for the harvest year | 64 |
| `tessera` | [TESSERA](https://arxiv.org/abs/2506.20380) annual embedding for the harvest year | 128 |
| `copernicusfm` | [Copernicus-FM](https://arxiv.org/abs/2503.11849) image embedding of the field's peak-greenness Sentinel-2 scene, sampled at the pixel | 768 |
| `dofa` | [DOFA](https://arxiv.org/abs/2403.15356) image embedding of the peak-greenness scene, sampled at the pixel | 768 |
| `croma` | [CROMA](https://arxiv.org/abs/2311.00566) optical-encoder embedding of the peak-greenness scene, sampled at the pixel | 768 |
| `softcon` | [SoftCon](https://arxiv.org/abs/2405.20462) image embedding of the peak-greenness scene, sampled at the pixel | 768 |

A location encoder gives a pixel the same embedding every year, so on its own it can't follow year-to-year
changes in yield. It is meant to be combined with an input that changes over time, such as `s2+mind`. How each
input was built and checked is in protocol section 2.5.

### Heads

- **`probe`** is a linear model (ridge regression) on standardised features, as in the TorchGeo evaluation
  harness. Because it is simple, differences between inputs show up clearly, which makes it the main head for
  comparing inputs. Its penalty is tuned inside each training fold.
- **Tree heads** (`lightgbm`, `rf`, `extra_trees`, `xgboost`) tune a small grid the same way.
- **`tabpfn` and `tabicl`** are pretrained tabular models. They need no tuning, but can only take small
  training sets, so each fold trains on 32 pixels per field.

"Tuned inside each training fold" means the test fold is never used to choose settings: each training fold is
split again, the same way as the outer scheme, and the best setting is then refit on the whole training fold.
Details: protocol section 2.6.

### What a run produces

Each run writes to `$project/runs/yieldsat/$USER/<time>-<jobid>/`:

- `metrics.json`: R² and RMSE at field and pixel level, per fold and as mean ± std over folds;
- `predictions.npz`: every test pixel's prediction;
- `tuning.csv`: the score of every setting tried.

### Final evaluation (leave one country out)

`scheme=loco` trains on some countries and tests on another. It is the final test, so it is used only after
models have been chosen on `cv10`, `loyo` and `loro`. It takes `crop=corn|soybean|wheat` instead of `subset=`,
must be confirmed with `experiment.confirm_test=true`, and every such run is logged in
`$project/runs/yieldsat/loco-ledger.jsonl`.

To check that everything works without waiting for a full run, add `experiment.smoke=true`. It finishes in about
two minutes; its scores mean nothing.

### Many runs at once: sweeps

    $project/envs/benchmark-v2/bin/python -m tools.sweep_yieldsat --name mysweep \
        --inputs s2 galileo mind galileo+mind --models probe lightgbm

This submits every subset × scheme × input × head as Slurm job arrays. Add `--dry-run` to see the plan first,
and `--subsets` or `--schemes` to narrow it. The sweep tool:

- **runs the folds of a run in parallel** for CPU heads (one task per fold). GPU heads run one task per run, so
  each task loads its model once;
- **requests resources per head**, from measured use. Anunna bills what you request, not what you use, so after
  a sweep run `tools.collect_yieldsat --sweep mysweep --efficiency` and tighten the numbers in
  `tools/sweep_yieldsat.py` if they are too generous;
- **keeps at most 200 tasks running at once**, Anunna's guidance for jobs that read Lustre;
- **runs from a frozen copy of your current commit**. Editing your checkout never changes queued tasks, and every
  result records the commit it came from. So commit before you sweep;
- **resumes** when you run the same command again: finished folds are skipped;
- **collects the results** when the arrays finish, into `results/yieldsat/mysweep.csv`. That table holds aggregate
  scores only, so it is safe to commit; the dashboard reads it.

## EarthShift: field delineation under shift

**The task.** Find field boundaries in 4-band (RGB + near-infrared) Sentinel-2 image chips of 256 × 256 pixels.
Each pixel is labelled field, boundary or background, and models are scored with mIoU. The point of EarthShift
is **distribution shift**: each task trains in one setting and tests both in that setting (in distribution, ID)
and in a different one (out of distribution, OOD). The gap between the two is what the benchmark measures.
Full protocol: [protocols/earthshift.md](protocols/earthshift.md).

| Task | Shift | Train and ID test | OOD test |
|---|---|---|---|
| DE→KH | location, far | Germany | Cambodia (other continent, small plots) |
| DE→DK | location, near | Germany | Denmark |
| ZA A→B | season | South Africa, image window A | the same chips in window B, another time of the season |
| DE 2018→2019 | year | Germany | Germany in another year |

**Splits.** The paper's splits let neighbouring chips fall into both train and test, which inflates the ID
score. AgriBench re-draws train, validation and ID test so that they never share or touch a chip, with a gap of
at least one chip (about 1.5 km) between them. The OOD test sets are the paper's. Protocol section 2.1b has the
details.

**Models.** Each model is trained in the paper's two settings: **frozen** (the backbone is fixed and only a
light segmentation head is trained, 40 epochs) and **full** fine-tuning (50 epochs).

- `resnet50-imagenet`: ResNet-50 pretrained on ImageNet, the paper's most robust model;
- `resnet50-random`: the same network with random weights, as a floor;
- `galileo-base`: the full Galileo-Base encoder.

**Running it.** From the repo root:

    # 1. Learning-rate sweep, scored on ID validation only
    sbatch earthshift.sbatch --model resnet50-imagenet --setting frozen --task de-geo --mode sweep
    # 2. Pick the best learning rate per model, setting and shift
    $project/envs/benchmark-v2/bin/python -m tools.earthshift_results --best-lr
    # 3. Final runs on the test sets (5 seeds, logged in a test ledger)
    sbatch earthshift.sbatch --model resnet50-imagenet --setting frozen --task de-geo --mode final \
        --lr <chosen> --confirm-test
    # 4. Result tables
    $project/envs/benchmark-v2/bin/python -m tools.earthshift_results

`--task de-geo` trains once on Germany and scores the Germany ID test, Cambodia and Denmark together. The other
tasks are `de-year` and `za-season`. One run takes about 50 s for ResNet-50, 5 min for frozen Galileo and 14 min
for fully fine-tuned Galileo on an A100.

**Inputs.** EarthShift needs a label for every pixel, so the per-pixel inputs and tabular heads used for YieldSAT
don't carry over. Location encoders don't help either: the location barely changes within a chip. Foundation
models are compared as backbones instead (protocol section 2.5).

**Status.** First results are in (protocol section 4b; `results/earthshift/final.csv`, on the dashboard), mIoU
at the model's input size, 5 seeds:

| Model | Setting | Germany ID | → Cambodia | → Denmark | DE 2018 → 2019 | ZA A → B |
|---|---|---:|---:|---:|---:|---:|
| Galileo Base | frozen | 55.0 | 21.2 | 41.5 | 53.7 → 54.8 | 49.9 → 46.5 |
| Galileo Base | full | 55.3 | 23.1 | 41.2 | 53.6 → 54.9 | 51.6 → 50.1 |
| ResNet-50 ImageNet | frozen | 48.3 | 18.8 | 33.5 | 44.2 → 47.0 | 39.3 → 37.0 |
| ResNet-50 ImageNet | full | 53.1 | 13.4 | 39.1 | 50.7 → 52.3 | 46.2 → 45.5 |
| ResNet-50 random | frozen | 45.0 | 20.6 | 29.5 | 38.7 → 42.5 | 38.9 → 37.0 |
| ResNet-50 random | full | 52.6 | 16.4 | 36.8 | 50.7 → 52.1 | 45.7 → 44.6 |

Cambodia is the hard shift: every model loses 24–40 points, and a frozen random ResNet-50 does about as well as
frozen Galileo. The temporal shifts barely matter. The boundary class is never predicted under the paper's recipe,
so mIoU tops out near 67 (protocol 4b).

## PLANTEUR: crop classification

**The task.** Classify every pixel into one of 8 crop and land-cover classes, from Sentinel-2 time series of the
French overseas territories (September 2018 to December 2019). There are 2,750 patches of 128 × 128 pixels:
Réunion 813, Guadeloupe 735, Martinique 550, Guyane 506, Mayotte 140 and Saint-Martin 6. Smallholder tropical
agriculture with heavy cloud cover makes it much harder than European benchmarks. Full protocol:
[protocols/planteur.md](protocols/planteur.md).

**Splits and score.** Two official splits:

- **leave one island out (LOIO)**, 5 folds: train on four islands, test on the fifth. This is the headline score:
  the mean mIoU-8 over the five islands;
- **the default split**: patches from all islands mixed, reported alongside.

**Inputs.** The dataset ships ready-made embeddings, and AgriBench adds raw data and two encoders:

| Input | What it is |
|---|---|
| `planteur/s2-monthly` | raw Sentinel-2: 12 monthly mosaics × 10 bands |
| `planteur/emb-aef` | AlphaEarth annual embedding (shipped) |
| `planteur/emb-tessera-v1`, `-v1.1`, `-v2` | TESSERA embeddings (shipped) |
| `planteur/emb-olmoearth-v1.3` | OlmoEarth embedding (shipped) |
| `planteur/galileo-base`, `planteur/presto-base` | extracted by AgriBench with the paper's settings |

**Heads.** The linear probe follows the paper's recipe exactly, on all training pixels, so its numbers can be
compared with the paper's. Other heads train on up to 256 pixels per patch; TabPFN uses the official few-shot set
of 1,000 pixels per class. Every test pixel is scored, for every head. All normalisation is fitted on the
training fold only: the normalisation files shipped with the dataset include every island, so they are not used.

**Status.** Protocol agreed. The data is downloading to Anunna. The runner is the next thing to write. Still open:
whether to keep the full download (258 GB, which includes the monthly mosaics and embeddings) or only the core
(133 GB).

## Results and the dashboard

All results go to [the dashboard](https://harryfyjiswalker.github.io/agribench/), one view per dataset. It shows
each input × head combination, a table per subset, and lets you switch scheme and metric. It reads the aggregate
tables in `results/<dataset>/`, which the sweep tools write: commit and push a table to publish it. YieldSAT and
EarthShift are there now; PLANTEUR joins when its first results are in.

**The repo is public.** Commit aggregate scores only: never field names, locations, fold lists or predictions.
The datasets have their own licences.

## Where things live on Anunna

`$project` is `/lustre/shared/agriscience-benchmark`.

| Path | Contents |
|---|---|
| `$project/datasets/` | the raw datasets |
| `$project/splits/` | the frozen fold files for each dataset (never edited by hand) |
| `$project/features/yieldsat/` | YieldSAT inputs: `v1/` holds `s2` and `s2-adm`; every other input has its own folder (`mind/`, `galileo/`, `aef/`, …) |
| `$project/runs/` | run outputs per dataset; sweeps under `runs/yieldsat/sweeps/<name>/`; test ledgers |
| `$project/checkpoints/`, `$project/upstream/` | model weights, and pinned copies of the models' source code |
| `$project/envs/` | shared Python environments: `benchmark-v2` (everything) and `tessera` (adds geotessera) |
| `$project/project.env` | optional Anunna project number for billing: one line `ANUNNA_PROJECT=<number>` |

Every batch script first loads `tools/slurm_env.sh`. It limits library threads to the cores the job asked for
and, if a project number is set in `$project/project.env` (or in your own `export ANUNNA_PROJECT=<number>`), tags
the job with it for billing.

## Adding an input or a head

- **A new head:** copy a file in `conf/yieldsat/model/` and set `name`, an `estimator` (any
  scikit-learn-compatible class, via `_target_`) and optionally a tuning grid in `search.param_grid`. Then use
  `model=<name>`. To sweep it, add its resources to `RESOURCES` in `tools/sweep_yieldsat.py`.
- **A new YieldSAT input:** write a script that saves `features/yieldsat/<name>/<subset>.npy`, with rows in the
  order of `v1/<subset>-meta.npz`, plus a `<subset>.json` with `kind`, `feature_names`, `rows` and that meta
  file's `v1_meta_sha256`. `tools/extract_yieldsat_aef.py` is a short example. The input is then usable as
  `input=<name>`, alone or joined with `+`.

## Python environment

Everyone uses the shared, read-only environment `$project/envs/benchmark-v2`; the batch scripts already point at
it. Don't pip install into it. To add a package, add it to `envs/benchmark-v2.lock.txt` in a pull request. To
make your own copy:

    python3 -m venv ~/myenv
    ~/myenv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r envs/benchmark-v2.lock.txt
    BENCHMARK_PYTHON=~/myenv/bin/python sbatch yieldsat.sbatch ...

## Working with an AI agent

The repo includes the [Anunna skill](https://git.wur.nl/anunna/anunna-skill) as a submodule at
`.claude/skills/anunna`. It teaches an agent how to use the cluster: ssh, Slurm, Lustre, environments and costs.
Claude Code picks it up automatically. If you cloned without `--recurse-submodules`, run
`git submodule update --init`. To update it, run `git submodule update --remote .claude/skills/anunna` and commit
the new pointer.

For an agent on your laptop to reach Anunna, add connection sharing to the `Host anunna` block of
`~/.ssh/config`. The agent then reuses a login you opened yourself and needs no key of its own:

    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h-%p
    ControlPersist 8h

Open the connection with `ssh -fN anunna` before starting the agent, and close it with `ssh -O exit anunna`.
