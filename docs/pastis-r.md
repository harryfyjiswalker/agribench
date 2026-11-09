# PASTIS-R (archived)

**Retired from active development on 2026-11-09.** PASTIS-R is in-distribution (one year, metropolitan France)
and close to saturated, so it tells us little about how representations generalise. PLANTEUR, with
leave-one-island-out, is AgriBench's crop-classification benchmark now. Everything below still works and is kept
for reference and for comparison with papers that report PASTIS-R; no new sweeps are planned.

- Dataset: [PASTIS-R](https://zenodo.org/records/5745151) (Garnot and Landrieu, 2021), 85,951 parcels, 18 crops,
  official five folds (fold 5 reserved for the final test).
- Protocol notes: [protocols/protocols.md](../protocols/protocols.md), [BENCHMARK.md](../BENCHMARK.md).
- Pilot results (validation fold 4): [LEADERBOARD.md](../LEADERBOARD.md).
- Data and features on Anunna: `$project/datasets/`, `$project/features/pastis-r/`, runs in `$project/runs/pastis/`.

## Running it

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

- **A new encoder:** copy `tools/adapters/template.py`, add `conf/encoder/<name>.yaml` with the source revision
  and weights hash pinned, smoke-test on 8 parcels, then extract once with `extract.sbatch`.

- Publish a pilot run to LEADERBOARD.md: `tools/export_result.py <run folder> <short-name>`, then commit
  `results/<short-name>.json`.
