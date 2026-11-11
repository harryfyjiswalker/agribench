# AgriBench

AgriBench compares frozen representations of satellite and agronomic data on agricultural tasks. You choose:

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
- [Python environment](#python-environment)
- [Working with an AI agent](#working-with-an-ai-agent)
- [Add your own model: step by step](#add-your-own-model-step-by-step)

## Tasks at a glance

| Task | Dataset | What it tests | Main score | Status (2026-11-11) |
|---|---|---|---|---|
| Yield prediction | [YieldSAT](https://yieldsat.github.io/) | predicting 10 m yield maps in four countries, including in unseen years, regions and countries | R² and RMSE (t/ha), per field and per pixel | **Running.** 30 inputs × 5 heads × 4 split schemes; CPU heads done, GPU heads (TabPFN, TabICL) catching up; label efficiency and early season under way |
| Field delineation | [EarthShift](https://earthshift.github.io/) | finding field boundaries in a new country or a new season | mIoU, in and out of distribution | **First results.** ResNet-50 and Galileo Base, frozen and full, 5 seeds; a CPU route for frozen backbones is being validated |
| Crop classification | [PLANTEUR](https://huggingface.co/datasets/allenai/planteur-benchmark) | classifying crops on tropical islands, including an island never seen in training | mIoU over 8 classes | **Running.** The 6 shipped inputs are done for every CPU head; 18 inputs built by AgriBench are finishing; GPU heads queued |
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
| `weather-pheno` | the field's season weather in agronomic terms, keyed to sowing and to crop stages from accumulated heat: growing degree days, heat stress around flowering, frost, rainfall and a simple soil water balance in the vegetative, flowering and grain-fill windows. One value per field | 69 |
| `wofost` | the WOFOST crop model run for each field (PCSE): simulated potential and water-limited yield and biomass, peak LAI, flowering and maturity dates, water stress. One value per field | 13 |
| `wofost-da` | WOFOST corrected during the season by the field's ARC leaf area index (ensemble Kalman filter); `wofost-da-snap` uses the SNAP LAI instead. One value per field | 12 |
| `latlon` | the pixel's latitude and longitude: the simple control for the location encoders | 2 |
| `mind` | [MIND](https://arxiv.org/abs/2609.25454) location encoder. Sees only the location. | 3072 |
| `satclip` | [SatCLIP](https://arxiv.org/abs/2311.17179) location encoder. Sees only the location. | 256 |
| `galileo` | [Galileo](https://github.com/nasaharvest/galileo)-Base embedding of the pixel's Sentinel-2 time series | 768 |
| `aef` | [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) annual embedding for the harvest year | 64 |
| `tessera` | [TESSERA](https://arxiv.org/abs/2506.20380) annual embedding for the harvest year | 128 |
| `presto` | [Presto](https://github.com/nasaharvest/presto) embedding of the pixel's in-season Sentinel-2 series (monthly composites at real dates) | 128 |
| `olmoearth` | [OlmoEarth](https://github.com/allenai/olmoearth_pretrain)-v1.2-Base embedding of the pixel's in-season Sentinel-2 acquisitions at their real dates | 768 |
| `anysat` | [AnySat](https://github.com/gastruc/AnySat)-Base embedding of the pixel, from in-season Sentinel-2 tiles of its field (spatial context) | 768 |
| `prithvi` | [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL)-300M-TL embedding at the pixel, from 4 in-season Sentinel-2 dates of the field's image (224-px windows) | 1024 |
| `copernicusfm` | [Copernicus-FM](https://arxiv.org/abs/2503.11849) image embedding of the field's peak-greenness Sentinel-2 scene, sampled at the pixel | 768 |
| `dofa` | [DOFA](https://arxiv.org/abs/2403.15356) image embedding of the peak-greenness scene, sampled at the pixel | 768 |
| `croma` | [CROMA](https://arxiv.org/abs/2311.00566) optical-encoder embedding of the peak-greenness scene, sampled at the pixel | 768 |
| `softcon` | [SoftCon](https://arxiv.org/abs/2405.20462) image embedding of the peak-greenness scene, sampled at the pixel | 768 |
| `terramind` | [TerraMind](https://arxiv.org/abs/2504.11171) embedding of the peak-greenness scene plus elevation (DEM), sampled at the pixel | 768 |
| `clay` | [Clay](https://github.com/Clay-foundation/model) v1.5 image embedding of the peak-greenness scene, sampled at the pixel | 1024 |
| `satmae` | [SatMAE](https://arxiv.org/abs/2207.08051) multispectral image embedding of the peak-greenness scene, sampled at the pixel | 1024 |
| `thor` | [THOR](https://arxiv.org/abs/2601.16011) image embedding of the peak-greenness scene, sampled at the pixel | 768 |

A location encoder gives a pixel the same embedding every year, so on its own it can't follow year-to-year
changes in yield. It is meant to be combined with an input that changes over time, such as `s2+mind`. How each
input was built and checked is in protocol section 2.5. To add your own model as an input, follow [Add your own model](#add-your-own-model-step-by-step).

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

**The task.** Find field boundaries in 4-band (red, green, blue, near-infrared) Sentinel-2 chips of 256 × 256
pixels. Every pixel is labelled field, boundary or background, and the score is mIoU over the three classes.
EarthShift is about **distribution shift**: each task trains in one setting and is tested both in that setting
(in distribution, ID) and in another one (out of distribution, OOD). The gap between the two is what it measures.
Full protocol: [protocols/earthshift.md](protocols/earthshift.md).

| Task | Shift | Train and ID test | OOD test |
|---|---|---|---|
| DE→KH | location, far | Germany | Cambodia (another continent, small plots) |
| DE→DK | location, near | Germany | Denmark |
| ZA A→B | season | South Africa, image window A | the same chips in window B, another part of the season |
| DE 2018→2019 | year | Germany | Germany in the next year |

**Splits.** In the paper's splits, neighbouring chips can land in both train and test, which inflates the ID
score. AgriBench redraws train, validation and ID test so that no two of them share or touch a chip. The OOD test
sets are the paper's. Details: protocol section 2.1b.

**Models.** A model here is a **backbone**: an image encoder whose patch features feed the paper's small
segmentation head (conv → ReLU → conv). Each backbone is trained in the paper's two settings:

- **frozen**: the backbone's weights stay fixed and only the head trains (40 epochs);
- **full**: everything trains (50 epochs).

The learning rate is chosen from the paper's grid on ID validation, then 5 seeds are run on the test sets.

| Backbone | What it is |
|---|---|
| `resnet50-imagenet` | ResNet-50 pretrained on ImageNet; the paper's most robust model |
| `resnet50-random` | the same network with random weights, as a floor |
| `galileo-base` | the full Galileo-Base encoder, with R, G, B and NIR in its own band slots |

The per-pixel inputs and tabular heads used for YieldSAT and PLANTEUR don't carry over: a boundary needs spatial
context, and a location encoder can't tell field from boundary within one chip. The other foundation models are
added as backbones instead.

**Running it.** From the repo root (GPU jobs):

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
tasks are `de-year` and `za-season`. Add `--smoke` for a one-epoch plumbing check. On an A100, one run takes
about 50 s for ResNet-50, 5 min for frozen Galileo and 14 min for fully fine-tuned Galileo.

**Frozen backbones on CPU (being validated).** A frozen backbone without a trainable input adapter gives the same
features every epoch, so they can be computed once and the head trained on them on CPU. This is being checked
against the GPU results for Galileo before it is used for other backbones (protocol section 4).

**Results so far** (protocol section 4b; `results/earthshift/final.csv`; on the dashboard). mIoU at the model's
input size, mean of 5 seeds:

| Model | Setting | Germany ID | → Cambodia | → Denmark | DE 2018 → 2019 | ZA A → B |
|---|---|---:|---:|---:|---:|---:|
| Galileo Base | frozen | 55.0 | 21.2 | 41.5 | 53.7 → 54.8 | 49.9 → 46.5 |
| Galileo Base | full | 55.3 | 23.1 | 41.2 | 53.6 → 54.9 | 51.6 → 50.1 |
| ResNet-50 ImageNet | frozen | 48.3 | 18.8 | 33.5 | 44.2 → 47.0 | 39.3 → 37.0 |
| ResNet-50 ImageNet | full | 53.1 | 13.4 | 39.1 | 50.7 → 52.3 | 46.2 → 45.5 |
| ResNet-50 random | frozen | 45.0 | 20.6 | 29.5 | 38.7 → 42.5 | 38.9 → 37.0 |
| ResNet-50 random | full | 52.6 | 16.4 | 36.8 | 50.7 → 52.1 | 45.7 → 44.6 |

- Cambodia is the hard shift: every model loses 24–40 points, and a frozen ResNet-50 with random weights does
  about as well as frozen Galileo.
- The season and year shifts barely matter.
- The boundary class is never predicted under the paper's recipe, so mIoU tops out near 67.

## PLANTEUR: crop classification

**The task.** Classify every pixel into one of 8 crop and land-cover classes, from Sentinel-2 data of the French
overseas territories (September 2018 to December 2019). There are 2,750 patches of 128 × 128 pixels: Réunion 813,
Guadeloupe 735, Martinique 550, Guyane 506, Mayotte 140 and Saint-Martin 6. Smallholder tropical farming under
heavy cloud makes it much harder than European crop maps. Full protocol: [protocols/planteur.md](protocols/planteur.md).

**Splits and score.** The official splits, unchanged:

- **leave one island out (LOIO)**, 5 folds: train on four islands and test on the fifth. This is the headline:
  the mean mIoU over the five islands;
- **the default split**: patches from every island mixed, reported alongside.

mIoU is over all 8 classes, Background included, from one confusion matrix over every labelled test pixel.

**Inputs.** One row per labelled pixel, like YieldSAT.

| Input | What it is |
|---|---|
| `s2-monthly` | raw Sentinel-2: 12 monthly mosaics × 10 bands |
| `emb-aef`, `emb-tessera-v1`, `emb-tessera-v1.1`, `emb-tessera-v2`, `emb-olmoearth-v1.3` | the embeddings shipped with the dataset |
| `galileo`, `presto`, `olmoearth`, `prithvi`, `anysat` | time-series encoders, run by AgriBench on the monthly mosaics with the same code as for YieldSAT |
| `copernicusfm`, `dofa`, `croma`, `softcon`, `clay`, `satmae`, `thor`, `terramind` | single-date encoders on the patch's greenest month |
| `s2-harmonic`, `biophys` | harmonic fit of each band's year; SNAP canopy variables per month |
| `latlon`, `satclip`, `mind` | location only |

Inputs that need the crop type or sowing date (crop-calendar alignment, WOFOST, ARC, crop-stage weather) are left
out: the crop type is what is being predicted. How each input is built: protocol section 7.

**Heads.**

- `probe`: the paper's linear probe on every training pixel (50 epochs, learning rate picked on validation), so
  its numbers can be compared with the paper's. Sweeps run it on CPU;
- `lightgbm`: at most 256 pixels per training patch, a small grid picked on validation;
- `tabpfn`, `tabicl` (GPU): 1,000 training pixels per class;
- `background`: predicts Background everywhere; the floor.

All normalisation is fitted on the training pixels of the fold. The normalisation files shipped with the dataset
include every island, so they are never used.

**Running it.**

    # One input x head x fold. planteur.sbatch asks for a GPU; for CPU heads add --partition=main --gres=none
    sbatch --partition=main --gres=none planteur.sbatch split=loio fold=reunion input=emb-tessera-v2 \
        model=lightgbm experiment.confirm_test=true
    # Many runs: every fold of every input x head, as Slurm arrays (commit first)
    $project/envs/benchmark-v2/bin/python -m tools.sweep_planteur --name mysweep \
        --inputs emb-aef galileo --models probe lightgbm

The folds are `default`, or `reunion`, `guadeloupe`, `martinique`, `guyane`, `mayotte` with `split=loio`. Without
`experiment.confirm_test=true` a run stops after validation. Add `experiment.smoke=true` for a quick check.

**Results so far** (mIoU, default split / mean of the five held-out islands; best of probe and LightGBM;
`results/planteur/`):

| Input | Default | Held-out island |
|---|---:|---:|
| OlmoEarth v1.3 (shipped) | 34.6 | **20.6** |
| TESSERA v1.1 (shipped) | 33.3 | 19.7 |
| AlphaEarth (shipped) | 33.3 | 17.5 |
| OlmoEarth (AgriBench, monthly mosaics) | 28.0 | 18.5 |
| TerraMind | 26.7 | 18.0 |
| CROMA | 27.4 | 17.6 |
| Galileo | 28.0 | 16.1 |
| Presto | 23.1 | 14.3 |
| S2 monthly (raw) | 25.4 | 11.9 |
| Background only | 10.1 | 10.5 |

- Moving to an island the model has never seen roughly halves mIoU for every input.
- The shipped embeddings are ahead of every encoder AgriBench runs itself.
- The probe beats LightGBM on most foundation-model embeddings; LightGBM is far better on raw bands.
- The probe lands within a few points of the paper's numbers, and the Background row matches it exactly.

## Results and the dashboard

All results are on [the dashboard](https://harryfyjiswalker.github.io/agribench/), with one view per dataset. It
reads the aggregate tables in `results/<dataset>/`, which `tools/publish_results.sh` refreshes from Anunna every
couple of hours: it collects every sweep, computes the YieldSAT diagnostics (bootstrap intervals, error
decomposition, rank persistence) and opens a pull request.

**The repo is public.** Commit aggregate scores only: never field or patch identifiers, locations, fold lists or
predictions. The datasets have their own licences.

## Where things live on Anunna

`$project` is `/lustre/shared/agriscience-benchmark`.

| Path | Contents |
|---|---|
| `$project/datasets/` | the raw datasets |
| `$project/splits/` | the frozen fold files for each dataset (never edited by hand) |
| `$project/features/yieldsat/` | YieldSAT inputs: `v1/` holds `s2` and `s2-adm`; every other input has its own folder (`mind/`, `galileo/`, `aef/`, …) |
| `$project/runs/` | run outputs per dataset; sweeps under `runs/yieldsat/sweeps/<name>/`; test ledgers |
| `$project/checkpoints/`, `$project/upstream/` | model weights, and pinned copies of the models' source code |
| `$project/envs/` | shared Python environments: `benchmark-v2` (everything), `tessera` (adds geotessera) and `patchfm` (adds terratorch for TerraMind, Clay and THOR; `tools/setup_patchfm_env.sh`) |
| `$project/project.env` | optional Anunna project number for billing: one line `ANUNNA_PROJECT=<number>` |

Every batch script first loads `tools/slurm_env.sh`. It limits library threads to the cores the job asked for
and, if a project number is set in `$project/project.env` (or in your own `export ANUNNA_PROJECT=<number>`), tags
the job with it for billing.

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

## Add your own model: step by step

This walks through adding a new model to YieldSAT and comparing it with everything else. YieldSAT is the easiest
place to start: a model only has to produce one feature vector per 10 m pixel, and the runner does the rest
(splits, tuning, heads, scoring). PLANTEUR and EarthShift follow at the end.

The worked example is [`tools/example_yieldsat_input.py`](tools/example_yieldsat_input.py). As shipped, it turns
each pixel's Sentinel-2 series into an NDVI series, so you can run every step below before touching your own
model. Each step says what you should see.

### 0. Set up (once)

    ssh USERNAME@login.anunna.wur.nl
    cd /lustre/nobackup/WUR/AIN/$USER          # your own space on Lustre
    git clone --recurse-submodules https://github.com/harryfyjiswalker/agribench.git
    cd agribench
    export project=/lustre/shared/agriscience-benchmark
    export py=$project/envs/benchmark-v2/bin/python

Run everything from this folder. Heavy work always goes through `sbatch`, never on the login node.

### 1. Build the example input for one subset

    sbatch prepare.sbatch tools.example_yieldsat_input --name $USER-ndvi --subset URG-S

`--name` is your input's name. Prefix it with your username until it is reviewed, so it can't clash with anyone
else's. Check with `squeue --me`; it takes under a minute. The log (`slurm-<jobid>.out`) ends with:

    Wrote /lustre/shared/agriscience-benchmark/features/yieldsat/<you>-ndvi/URG-S.npy (2,177,206 x 24) and .../URG-S.json

So the input is two files per subset: the features (one row per pixel, in the cache's pixel order) and a small
JSON that records the feature names, the row count and a hash of the pixel list. The runner checks both, so a file
built for the wrong pixels is refused rather than silently misread.

### 2. Check that the runner accepts it

    sbatch yieldsat.sbatch subset=URG-S scheme=loyo input=$USER-ndvi model=probe experiment.smoke=true

`experiment.smoke=true` runs a cut-down version in about a minute; its scores mean nothing. The log ends with
`Outputs: .../runs/yieldsat/<you>/<time>-<jobid>`. If it fails, see [Troubleshooting](#troubleshooting).

### 3. One real run

    sbatch yieldsat.sbatch subset=URG-S scheme=loyo input=$USER-ndvi model=probe

That is Uruguay soybean, leave one year out, linear probe. It takes a minute or two. Read the headline scores:

    $py -c "import json,sys; m=json.load(open(sys.argv[1]+'/metrics.json')); print(m['pooled'])" <output folder>

For the example you should get a pooled field R² of about 0.22. For comparison, on this subset and scheme the
training-mean floor (`model=mean`) gets −0.61 and raw `s2` with the probe gets 0.60, the best of all inputs so
far: NDVI alone throws away most of what the bands carry.

### 4. Put in your own model

Copy the example and edit only the `embed()` function:

    cp tools/example_yieldsat_input.py tools/embed_yieldsat_mymodel.py

- **What `embed()` gets:** a batch of pixels' raw Sentinel-2 values, `(pixels, 288)` float32, with NaN where
  there was no observation, and the column names `<band>_t<step>` (12 bands B01–B12 including B8A, 24 time steps
  over the season).
- **What it returns:** `(pixels, d)` features and a list of `d` names. Any `d` works. Values are stored as float16,
  so keep them below about 65,000 in size (embeddings always are).
- **Set `METHOD`** at the top of the file to one line that describes your input.
- **If your model needs more than the pixel's own series** (image patches around it, dates, the field outline),
  load that inside `embed()` from `$project/datasets/yieldsat/`. The meta file
  `$project/features/yieldsat/v1/<subset>-meta.npz` says which field (`field_idx`) and which pixel of that field's
  raster (`row`, `col`) each row is. The existing wrappers (`tools/embed_yieldsat_*.py`) show how each foundation
  model is fed; reuse them where you can.
- **Model weights** go in your own space or in `$project/checkpoints/`. Load PyTorch checkpoints with
  `torch.load(..., weights_only=True)`.
- **If it needs a GPU,** use `prepare-gpu.sbatch` instead of `prepare.sbatch` (same arguments).
- **If it needs a package the shared environment lacks,** make your own environment (see
  [Python environment](#python-environment)) and put `BENCHMARK_PYTHON=~/myenv/bin/python` in front of `sbatch`.

Test on one subset as in steps 1–3, using your new script's module name (`tools.embed_yieldsat_mymodel`).

### 5. Build all nine subsets

    for s in ARG-C ARG-S ARG-W BRA-C BRA-S BRA-W GER-R GER-W URG-S; do
        sbatch prepare.sbatch tools.embed_yieldsat_mymodel --name $USER-mymodel --subset $s
    done

Add `--overwrite` to rebuild a subset that already exists. Check that all nine are there:

    ls $project/features/yieldsat/$USER-mymodel/        # 9 .npy and 9 .json files

### 6. Run the full comparison (a sweep)

A sweep runs every subset × split scheme × head as Slurm job arrays. Look at the plan first:

    $py -m tools.sweep_yieldsat --name $USER-mymodel --inputs $USER-mymodel --models probe lightgbm --dry-run

Then run it without `--dry-run`. Notes:

- **Commit your changes first.** A sweep runs a frozen copy of your current commit and refuses to start if
  tracked files have uncommitted edits. Every result records the commit it came from.
- **Heads:** `probe` and `lightgbm` run on CPU and are the ones to start with. `tabpfn` and `tabicl` need GPUs,
  which are often busy. `mean` is the floor every input must beat.
- **Schemes:** by default `cv10`, `loyo` and `loro`. Leave one year out (`loyo`) is the headline. Leave one
  country out is the final test: add `--schemes loco` only once you are done choosing.
- **Combinations:** `--inputs $USER-mymodel+mind` joins your input with another one, column by column.
- **Resuming:** run the same command again; finished folds are skipped.
- **Size:** for one input with `probe` and `lightgbm` the plan is 400 tasks (one per fold) in 2 arrays, most of
  them a few minutes each. Follow them with `squeue --me`.

When the arrays finish, the sweep writes `results/yieldsat/$USER-mymodel.csv` in your checkout: one row per
subset × scheme × input × head, with field and pixel R² and RMSE.

### 7. Compare with everything else

- **On the dashboard.** The regular publish collects every sweep under `$project/runs/yieldsat/sweeps/`, so your
  input appears on [the dashboard](https://harryfyjiswalker.github.io/agribench/) next to all the others, with
  bootstrap intervals and the error decomposition, within a couple of hours. The dashboard is public, and only
  aggregate scores are shown.
- **Straight away.** Open `results/yieldsat/$USER-mymodel.csv`. For a like-for-like comparison, use
  `pooled_field_r2` for `loyo` and `loro` (and `field_r2_mean` for `cv10`), averaged over the nine subsets, as
  the dashboard does. The other inputs' tables are in `results/yieldsat/` of the main branch.

### 8. Make it an official input

Open a pull request with your `tools/embed_yieldsat_<model>.py`, a row in the YieldSAT input table above, and a
short entry in protocol section 2.5: which weights (and their version), what the model sees, and how features are
taken per pixel. Keep the repo public-safe: no field names, locations, fold lists or predictions.

### Troubleshooting

| Message | What it means |
|---|---|
| `Unknown input '<name>': no .../<subset>.json` | That subset hasn't been built yet (step 5), or the name is misspelt |
| `Input block '<name>' has no cache yet (.../ARG-C.json)` | The sweep tool reads the first subset (ARG-C) to size the jobs: build all nine first |
| `... was built for different v1 cache rows` | The features were built for another version of the pixel cache: rebuild with `--overwrite` |
| `row count differs from the stored array` | `embed()` returned a different number of rows than it was given |
| `Values too large for float16` | rescale your features in `embed()` |
| `The checkout has uncommitted changes` | commit before sweeping (step 6) |
| A job is killed for memory (`OUT_OF_MEMORY` in `sacct -j <jobid>`) | ask for more: `sbatch --mem=64G prepare.sbatch ...` |

### PLANTEUR

An input is one folder in `$project/features/planteur/<name>/` with `X.npy` (one row per labelled pixel, in the
row order of `$project/features/planteur/meta.npz`), `valid.npy` (one boolean per row: False where your model has
no value for that pixel) and `info.json` (at least `kind: "dense"`, `feature_names`, `rows` and the `meta_sha256`
of `meta.npz`). The raw data is in `$project/datasets/planteur/` (monthly mosaics in `DATA_S2_MONTHLY`).

If your model already has a YieldSAT wrapper, [`tools/planteur_embed.py`](tools/planteur_embed.py) can run it on
PLANTEUR in the same way as the other encoders (protocol section 7). Then:

    # Single run on CPU (LightGBM), one held-out island
    sbatch --partition=main --gres=none planteur.sbatch split=loio fold=reunion input=<name> model=lightgbm \
        experiment.confirm_test=true
    # Every fold and head: add the name to INPUTS in tools/planteur.py, commit, then
    $py -m tools.sweep_planteur --name $USER-<name> --inputs <name> --models probe lightgbm

### EarthShift

A backbone is a small PyTorch class in [`tools/earthshift_train.py`](tools/earthshift_train.py), added to `MODELS`.
It needs:

- `input_size`: the image size it expects (chips are resized to it);
- `out_channels`: the number of feature channels it returns;
- `prepare(dn, months)`: turns raw 4-band digital numbers (R, G, B, NIR) into the model's input (scaling and
  normalisation);
- `features(x, months)`: returns a feature map of shape `(batch, out_channels, h, w)`;
- `self.adapter`, a learnable 1 × 1 convolution, only if the model can't take R, G, B and NIR directly.

`ResNet50` and `Galileo` in that file are the two patterns to copy. Then check it and run it:

    sbatch earthshift.sbatch --model <name> --setting frozen --task de-geo --mode sweep --smoke
    sbatch earthshift.sbatch --model <name> --setting frozen --task de-geo --mode sweep
    $py -m tools.earthshift_results --best-lr
    sbatch earthshift.sbatch --model <name> --setting frozen --task de-geo --mode final --lr <chosen> --confirm-test

Repeat for `--task de-year` and `--task za-season`, and for `--setting full`.

### A new head

Copy a file in `conf/yieldsat/model/` and set `name`, an `estimator` (any scikit-learn-compatible class, via
`_target_`) and, optionally, a tuning grid in `search.param_grid`. Then use `model=<name>`. To sweep it, add its
resources to `RESOURCES` in `tools/sweep_yieldsat.py`.
