# PASTIS-R parcel leaderboard


| Run | Model | Representation | Sensors | Train → validation folds | Train parcels | Validation parcels | Seed | Epoch | Parcel mIoU | Accuracy | Macro F1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [`s2-tabpfn35-fast-full-pilot`](results/s2-tabpfn35-fast-full-pilot.json) | tabpfn | Parcel summaries | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.702 | 0.906 | 0.807 |
| [`s2-galileo-base-logistic-regression-full-pilot`](results/s2-galileo-base-logistic-regression-full-pilot.json) | logistic_regression | Galileo Base embeddings | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.605 | 0.873 | 0.726 |
| [`s2-galileo-base-tabpfn35-fast-full-pilot`](results/s2-galileo-base-tabpfn35-fast-full-pilot.json) | tabpfn | Galileo Base embeddings | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.596 | 0.874 | 0.717 |
| [`s2-logistic-regression-full-pilot-43601671`](results/s2-logistic-regression-full-pilot-43601671.json) | logistic_regression | Parcel summaries | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.596 | 0.865 | 0.717 |
| [`s2-rf-full-pilot`](results/s2-rf-full-pilot.json) | rf | Parcel summaries | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 0.508 | 0.840 | 0.618 |
| [`s2-pse-ltae-1epoch-seed42`](results/s2-pse-ltae-1epoch-seed42.json) | adapted_pse_ltae2d | Parcel pixel sets | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | 1 | 0.308 | 0.737 | 0.384 |

