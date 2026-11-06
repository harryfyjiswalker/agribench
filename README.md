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

## Yield Prediction

## Phenology Detection

## Field Delineation


_____________________________________________________________________________________________________________________________________________________________________

# Setup

You need an Anunna account in the AIN group. Shared files live in
`/lustre/shared/agriscience-benchmark` (called `$project` below): the Python environment, feature
tables, checkpoints, upstream code and run outputs.

## Get the code

    mkdir -p ~/projects && cd ~/projects
    git clone https://github.com/harryfyjiswalker/agribench.git
    cd agribench

## Run an experiment (PASTIS-R)

An experiment is four choices, each a config group:

| Group | Meaning | Options today |
|---|---|---|
| `encoder=` | how parcels become feature vectors | `galileo-base`, `summaries` |
| `input=` | which prepared input the encoder reads | `pastis-r/s2-parcel-medians` |
| `model=` | the classifier head | `logistic_regression`, `rf`, `extra_trees`, `xgboost`, `mlp` (Lightning), `tabpfn` (GPU) |
| `validation=` | the protocol | `pilot` (default), `final` |

    sbatch bench.sbatch encoder=galileo-base input=pastis-r/s2-parcel-medians model=xgboost validation=pilot

`validation=pilot` is the development protocol: train on folds 1-3, score fold 4.

`validation=final` is the test protocol: tune on fold 4 (macro IoU, small grid in the model's YAML), refit
on folds 1-4, score the reserved fold 5 once. It needs `experiment.confirm_test=true`, and each run is
appended to `$project/runs/pastis/test-ledger.jsonl`. Never use fold 5 to choose models.

Add `experiment.smoke=true` to check the plumbing on a few thousand parcels. Smoke runs never read fold-5
labels, and their scores are meaningless.

GPU models:

    sbatch --partition=gpu --gres=gpu:1 --mem=32G bench.sbatch encoder=galileo-base input=pastis-r/s2-parcel-medians model=tabpfn validation=pilot

Many at once (every encoder x every model, in parallel):

    bash tools/sweep.sh "galileo-base summaries" "logistic_regression rf xgboost" validation=pilot

Check jobs with `squeue --me`. Results go to `$project/runs/pastis/$USER/<time>-<jobid>/` (`metrics.json`,
predictions, resolved config). The Slurm log is `slurm-<jobid>.out` in the folder you submitted from.
If the feature table for your encoder/input pair does not exist, the runner stops and prints the extraction command.

## Python environment

Everyone uses the shared read-only environment `$project/envs/benchmark-v2`; the batch scripts already point
at it. Do not pip install into it. To add a package, edit `envs/benchmark-v2.lock.txt` in a pull request.
To build your own copy:

    python3 -m venv ~/myenv
    ~/myenv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r envs/benchmark-v2.lock.txt

## Add an encoder

1. Copy `tools/adapters/template.py` to `tools/adapters/<name>.py` and implement `extract` (see `galileo.py`).
2. Add `conf/encoder/<name>.yaml` (adapter, input format, pinned source and weights if any).
3. Extract once; the table is then shared. Try `--smoke` first (8 parcels, seconds). Example for Galileo:

       data=/lustre/nobackup/WUR/AIN/fyjis001/datasets/pastis-r-pixelset/extracted/PASTIS-R_PixelSet
       sbatch --partition=gpu --gres=gpu:1 --mem=32G extract.sbatch encoder=galileo-base input=pastis-r/s2-parcel-medians \
         --inputs $project/inputs/pastis-r/pastis-s2-and-location.npz --metadata $data/metadata_parcel.csv \
         --source-dir $project/upstream/galileo --weights-dir $project/checkpoints/galileo/base \
         --output-dir $project/features/pastis-r/parcel-probe-v1

## Add a model

Add `conf/model/<name>.yaml` with `name`, an `estimator` (any scikit-learn-compatible class via `_target_`) and
optionally `search.param_grid` (used by `validation=final`). For neural heads see `agribench_heads.py`.

## Publish a validation result

    $project/envs/benchmark-v2/bin/python tools/export_result.py <run folder> <short-name>
    git add results/<short-name>.json && git commit -m "Add <short-name>" && git push

Use lowercase letters, digits, - or _. The leaderboard above is rebuilt automatically after the push.

## Other tools

`evaluate.sbatch` runs the geo-eval-harness linear probe (`protocols/protocols.md`). It scores the reserved
fold 5, so use it sparingly. The older `run_experiments.py` (PSE+L-TAE, the original leaderboard runs) is
described in `BENCHMARK.md`.
