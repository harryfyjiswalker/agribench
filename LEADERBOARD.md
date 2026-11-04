# PASTIS-R parcel leaderboard

Parcel-level crop type classification on PASTIS-R PixelSet (Zenodo 5745151). Scores are from validation. Fold 5 is reserved for testing.

| Run | Model | Sensors | Train → validation folds | Train parcels | Validation parcels | Seed | Epoch | Parcel mIoU | Accuracy | Macro F1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [`s2-rf-full-pilot`](results/s2-rf-full-pilot.json) | rf | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | — | 50.82% | 83.97% | — |
| [`s2-pse-ltae-1epoch-seed42`](results/s2-pse-ltae-1epoch-seed42.json) | adapted_pse_ltae2d | S2 | 1, 2, 3 → 4 | 51,833 | 17,478 | 42 | 1 | 30.75% | 73.73% | — |

Run links contain the recorded parcel and feature fingerprints. Sequence runs lack a training parcel fingerprint. Check inputs and training budgets before comparing scores.
