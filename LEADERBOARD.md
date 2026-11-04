# PASTIS-R parcel leaderboard

Validation results. Fold 5 is reserved for test. Smoke checks are excluded.
Rows are grouped by sensors, split, training budget and available data fingerprints.
Sequence runs lack an exact training ID fingerprint in the current runner.

## PASTIS-R PixelSet (Zenodo 5745151) · S2 · 51,833 training parcels

Train folds [1, 2, 3], validation folds [4]. Fingerprints `65f6c84b / ee4215d3 / 13a14f68`.

| Run | Model | Seed | Epoch | Validation parcels | mIoU | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| s2-rf-full-pilot | rf | 42 | — | 17,478 | 50.82% | 83.97% |

## PASTIS-R PixelSet (Zenodo 5745151) · S2 · 51,833 training parcels

Train folds [1, 2, 3], validation folds [4]. Fingerprints `65f6c84b`.

| Run | Model | Seed | Epoch | Validation parcels | mIoU | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| s2-pse-ltae-1epoch-seed42 | adapted_pse_ltae2d | 42 | 1 | 17,478 | 30.75% | 73.73% |
