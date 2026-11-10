# PLANTEUR evaluation protocol

Status: **all decisions agreed (D1 on 2026-11-10: full download, done and verified); Phase A pipeline built and its
first sweep running (section 6); Phase B planned (section 7).** This page sets how
every AgriBench model is scored on PLANTEUR, so results are comparable with the paper and with each other.
Sections marked **[paper]** follow the authors exactly, citing the paper, the dataset card or the code. Sections
marked **[AgriBench]** fill gaps the authors leave open, or go beyond them.

- Paper: Wolters, Tseng, Bastani, Redmon, Zhang, Sablon and Shah, *PLANTEUR: A Crop-Type Segmentation
  Benchmark for Tropical Regions*, arXiv:2611.05223v1 (2026). Cited below as "paper §x / Table y".
- Data: https://huggingface.co/datasets/allenai/planteur-benchmark, pinned at Hub commit
  `51b74a91424e7dc9ea1fe0042390353658af8185` (2026-10-22). Cited as `card:<line>` (the dataset card,
  `README.md` at that commit).
- Code: https://github.com/allenai/planteur-benchmark, read at commit `9f860ee` (2026-10-14). Cited as
  `<file>:<line>`. The repository has **data loaders, split-building scripts and a demo notebook only**. It has
  no training, evaluation or metric code, so the paper's training recipes cannot be rerun from it.

> The licence is ODC-BY (`README.md:271-272`), so the data could be shared with attribution. AgriBench still
> commits only aggregate metrics and hashes, as for the other datasets. Patch IDs and fold lists are public
> upstream (`splits/*.json`), so referring to them is fine.

---

## 1. Getting the data

### What you need to do yourself

- **Nothing to accept.** The dataset is public and not gated (Hub API: `"gated": false`), with no licence
  click-through.
- **A Hugging Face token is optional, but recommended.** The download is 52,263 files, and each file is one
  Hub request. Anonymous requests have lower rate limits, so the download may stall and retry for longer. With
  a token, run this once on the Anunna login node; the job then picks it up from `~/.cache/huggingface/token`:

      /lustre/shared/agriscience-benchmark/envs/benchmark-v2/bin/hf auth login

  A free account and a read-only token are enough. Never put the token in the repository or in a job script.

### Tools

`hf` (huggingface_hub 2.4.0, with `hf-xet` 1.8.0) is already in the shared `benchmark-v2` environment. **No
`git lfs` is needed**, and it shouldn't be used. A `git clone` would keep a second copy of every file under
`.git/lfs/objects`, doubling both size and file count. Compute nodes have internet access, so the download
runs as an ordinary Slurm job.

### Size and file count (from the Hub file list at the pinned commit)

| Part | Folders | Files | Size |
|---|---|---:|---:|
| Core: per-observation time series, labels, metadata, splits, normalisation | `DATA_S2`, `DATA_S1`, `DATA_LS`, `DATA_LS_PAN`, `ANNOTATIONS*`, `INSTANCE_ANNOTATIONS`, root files, `splits/`, `MONTHLY/` | 27,519 | 133.1 GB |
| Monthly mosaics (the paper's inputs for Galileo, Presto, OlmoEarth, U-TAE and TSViT) | `DATA_{S2,S1,LS,LS_PAN}_MONTHLY` | 11,000 | 21.1 GB |
| Precomputed embeddings (AlphaEarth, Tessera v1/v1.1/v2, OlmoEarth v1.3) | `EMB_*` | 13,744 | 103.8 GB |
| **Total** | | **52,263** | **258.0 GB** (240 GiB) |

The largest single file is 49 MB, and the average is about 5 MB. That's tens of thousands of files rather
than millions, so it's fine on Lustre as a plain folder. While it runs, `hf download` also writes a `.lock`
and a `.metadata` file per file under `<target>/.cache/` (about 105,000 small files), which it needs only to
resume. The job deletes that folder once verification passes. If we read the data often, it can later be
packed into one SquashFS image (`.claude/skills/anunna/references/file-counts.md`). This is not needed to
start.

Lustre space: `/lustre` was 98% full (104 TB free, checked 2026-11-09). AIN uses 144 GB. There is no group
quota.

### Commands

From the repository root on Anunna:

    sbatch download_planteur.sbatch            # everything: 258 GB, 52,263 files
    sbatch download_planteur.sbatch --core     # core only: 133 GB, 27,519 files (no monthly mosaics, no embeddings)

[`download_planteur.sbatch`](../download_planteur.sbatch) runs on 1 core with 4 GB (downloads wait on the
network, not the CPU), on the `main` partition with a 24-hour limit. It does three things:

1. `hf download allenai/planteur-benchmark --repo-type dataset --revision 51b74a9… --local-dir
   /lustre/shared/agriscience-benchmark/datasets/planteur --max-workers 8`, with
   `--exclude "DATA_*_MONTHLY/*" --exclude "EMB_*/*"` added for `--core`.
2. Verification: [`tools/verify_hf_snapshot.py`](../tools/verify_hf_snapshot.py) gets the Hub's file list for
   the pinned commit, and checks every local file's size and hash against it. Large files are checked with
   SHA-256 against the Hub's LFS/Xet hash (49,497 files). Small files are checked with their git blob SHA-1.
   The script also reports missing and extra files, and writes `MANIFEST.sha256` next to the data. Later checks
   need no network:

       cd /lustre/shared/agriscience-benchmark/datasets/planteur && sha256sum -c --quiet MANIFEST.sha256

3. It deletes `.cache/` and prints the final file count and size.

The job can be resubmitted after a failure or timeout, and files already downloaded are skipped. To add the
rest after a `--core` run, submit it again without `--core`. Follow it with `squeue --me`. The log is
`slurm-<jobid>.out`.

Expected result: 52,263 files plus `MANIFEST.sha256` (27,519 + 1 for `--core`), and 258.0 GB (133.1 GB). The
download time depends on the Hub, and wasn't measured: at 50–100 MB/s it's 45–90 minutes. Hashing 258 GB on
one core adds roughly another hour.

---

## 2. What is in the dataset

Location on Anunna (once downloaded): `/lustre/shared/agriscience-benchmark/datasets/planteur/`

- **2,750 patches of 128 × 128 px at 10 m** (1.28 km square) in France's overseas territories: Réunion 813,
  Guadeloupe 735, Martinique 550, Guyane 506, Mayotte 140, and Saint-Martin 6 (`card:293-301`).
- **Sept 2018 – Dec 2019** (`README.md:36`).
- **Labels:** RPG 2019 parcels, which gives 47,256 parcels in total (`documentation/planteur-documentation.md:127-130`).

### Arrays (PASTIS format, one `.npy` per patch)

| Path | Shape, dtype | Contents |
|---|---|---|
| `DATA_S2/S2_<id>.npy` | (T, 10, 128, 128) uint16 | Sentinel-2 L2A: B02–B08, B8A, B11, B12. T = 40–150 dates, median 87 |
| `DATA_S1/S1_<id>.npy` | (T, 2, 128, 128) float32 | Sentinel-1 RTC VV, VH, linear (not dB). `0` = no data. Ascending and descending in one series, orbit per step in `metadata.geojson` `orbits-S1` |
| `DATA_LS/LS_<id>.npy`, `DATA_LS_PAN/LSPAN_<id>.npy` | (T, 10, 64, 64), (T, 1, 128, 128) uint16 | Landsat-8 C2 **Level-1** (DN, not reflectance), 20 m multispectral, 10 m pan |
| `DATA_*_MONTHLY/` | (12, C, H, W) | 12 monthly 2019 mosaics. S2 has 12 bands: the 10 above, then B01, B09 |
| `ANNOTATIONS/TARGET_<id>.npy` | (1, 128, 128) uint8 | semantic label, classes 0–7, 255 = void |
| `ANNOTATIONS_BALANCED/` | same | training labels with Background thinned to equal the crop pixels (paper's "balanced" protocol) |
| `INSTANCE_ANNOTATIONS/` | (128, 128) | parcel IDs, heatmaps and zones for panoptic segmentation |
| `EMB_<product>/<name>_<id>.npy` | (C, 128, 128) float32 | embeddings on the label grid: AEF 64-d, Tessera v1, v1.1 and v2 128-d, OlmoEarth v1.3 128-d |
| `metadata.geojson` | 2,750 rows | `ID_PATCH`, dates per sensor, `Fold`, `split`, `Island`, footprint |
| `splits/443.json`, `splits/loio.json`, `splits/fewshot_pixels.json` | | official splits (2.2, 2.3) and few-shot pixels |
| `NORM_<sensor>_patch.json` | | per-band training mean and std |

Sources: `documentation/planteur-documentation.md:15-33`, `README.md:115-120`, `card:146-157`.

### Classes [paper]

| id | class | id | class |
|--:|---|--:|---|
| 0 | Background | 4 | Sugarcane |
| 1 | Meadow | 5 | Banana |
| 2 | Fruits, vegetables, flowers | 6 | Pineapple |
| 3 | Orchard | 7 | Tropical tuber |

**255 = void**: crops outside the taxonomy, such as corn, vanilla, coffee and cacao. Void is 92,145 px (0.20%).
It is excluded from loss and metrics, and is **not** Background. Background (0) is a real class
(`documentation/planteur-documentation.md:47-63`). Sugarcane dominates Réunion and Guadeloupe, banana dominates
Martinique and Mayotte, and Guyane is mostly meadow and orchard (`:132-133`).

### Things that differ from expectations or need care

1. **The out-of-distribution split is leave-one-*island*-out (LOIO), with 5 folds.** Each fold holds out one
   territory. There is no other "region" grouping. Saint-Martin is never held out, and stays in `train` in
   every fold (`README.md:208-223`). The default split is a separate, in-domain one (2.2).
2. **The shipped normalisation statistics leak in LOIO.** `NORM_*_patch.json` holds the default training fold's
   statistics. The same values are copied under `Fold_1`…`Fold_5` (`card:100-101`). That fold includes patches
   of every island, so the authors note the statistics "are not held-out statistics for a LOIO run"
   (`README.md:235-236`). The loader still uses them for any fold list (`code/dataloader.py:170-184`). See 2.6.
3. **The loader standardises no-data zeros.** S1 and Landsat use `0` for no observation, but the loader
   standardises those pixels like any other (`code/dataloader.py:64-67`). They must be masked on the raw arrays.
4. **The pixel-set loader normalises with statistics of whatever folds it loads.** Its geometric features are
   standardised with the mean and std of the loaded folds (`code/dataloader_pixelset.py:186-189`). On the test
   fold, that means test statistics.
5. **Parcels can cross split boundaries.** Splits are by 0.1° block, but 592 of the 47,256 parcels have
   pixels in more than one split: 321 in train and test, 240 in train and val, 30 in val and test, and 1 in all
   three (`documentation/planteur-documentation.md:101-103`). This is harmless for pixel segmentation, but
   matters for a parcel task (5, D6).
6. **LOIO validation is optimistic.** Inside each LOIO fold, `train` and `val` share 0.1° blocks
   (`splits/loio.json:4`). The `test` island never shares a block with either.
7. **Mayotte carries little signal.** Its test patches are 99.45% Background, so predicting Background
   everywhere already scores 12.42 mIoU there, and no model in the paper beats that meaningfully (paper §3.5,
   Tables 1 and 6). The authors still keep it in the LOIO mean.
8. **The monthly mosaics are separate products.** Each month is the least-cloudy scene of that 30-day period,
   so they can't be rebuilt from `DATA_*` (`README.md:105-113`, `card:115-140`).
9. **Some embeddings have no-data pixels.** Tessera v1 is missing for patches 1260–1265, and has zero or NaN
   vectors in 155 more. Tessera v1.1 has NaN in 4 patches. AEF, Tessera v2 and OlmoEarth have none
   (`documentation/planteur-documentation.md:33`, `card:165-178`).
10. **The paper is inconsistent once.** OlmoEarth v1.2 FT with S2 input has an LOIO mean of 25.28 in Table 6,
    but 23.88 in Table 11. Use Table 6.

---

## 3. Evaluation protocol

### 3.1 Task and unit [paper]

- **Task:** per-pixel semantic segmentation into the 8 classes. Panoptic metrics are not part of the benchmark:
  "we leave panoptic evaluation to users interested in that task" (paper §2.2).
- **Void** (255) is excluded from training and scoring (`README.md:177`, paper §3.2).
- **Unit of splitting:** the patch. Each split is a list of `ID_PATCH`.

### 3.2 Default split: in-domain [paper]

- Fold `1` = train (1,896 patches), `4` = val (411), `5` = test (443). These are in `metadata.geojson` `Fold`
  and `splits/443.json` (`README.md:193-204`, `code/dataloader.py:26`).
- Whole 0.1° (~11 km) blocks are assigned to a single split, aiming for 70/15/15 within each territory, so every
  territory appears in train, val and test (`data_generation/assign_splits.py:33-34, 80-107`). Mayotte's
  realised split is 39/15/46.
- The paper says "the 443-window test partition is used for all in-domain evaluation" (paper §2.4).

### 3.3 Leave-one-island-out: out-of-distribution [paper]

- **5 folds**, from `splits/loio.json`, in the order Guadeloupe, Martinique, Guyane, Réunion, Mayotte. For each
  fold:
  - `test` = every patch of the held-out territory;
  - `train` and `val` split the other patches about 88/12, and Saint-Martin is always in `train`
    (`README.md:208-223`).
- **Checked:** in every fold, train, val and test are disjoint and together cover all 2,750 patches.

| Held out | train | val | test |
|---|--:|--:|--:|
| Guadeloupe | 1,750 | 265 | 735 |
| Martinique | 1,950 | 250 | 550 |
| Guyane | 1,960 | 284 | 506 |
| Réunion | 1,720 | 217 | 813 |
| Mayotte | 2,296 | 314 | 140 |

### 3.4 Metrics [paper, with one AgriBench detail]

- **mIoU-8**, the headline metric. It is the mean IoU over all 8 classes, Background included, with void
  ignored. A class absent from the evaluated ground truth (for example a crop that doesn't grow on the held-out
  island) counts as **IoU 0** in the mean (`README.md:177-179`, `card:238-240`, paper §3.2).
- **Overall pixel accuracy (OA)** is reported alongside it, though "this metric can be deceiving" (paper §3.2).
  Predicting Background everywhere gives 80.56% OA on the test split.
- **[AgriBench] How IoU is computed.** The paper doesn't give a formula. AgriBench builds one confusion matrix
  over **all non-void pixels of the test set** of a fold. For each class, IoU = TP / (TP + FP + FN), and IoU is
  0 when the class is absent from the ground truth, as stated above. AgriBench does not average per patch.
  This is the usual PASTIS / U-TAE convention.
- **LOIO aggregate** [paper]: the unweighted mean of the 5 per-island mIoU-8 values. "Mean is across the 5
  folds" (paper Table 6). The paper doesn't pool pixels across islands.
- **Reported exactly as the paper does** (D5): one value per island and the 5-island mean, Mayotte included.
  No standard deviations and no extra means. Per-class IoU is saved in each run's `metrics.json` for
  diagnosis, but it isn't on the leaderboard.

### 3.5 How the paper trains and selects [paper]

There is no training code (see the header), so this comes from the paper only.

- **Linear probes on precomputed embeddings.** A linear head on one embedding per pixel, for 50 epochs. The
  learning rate is chosen from "an 8-point sweep (10⁻⁴ to 0.5) on the validation set" (paper §3.1).
- **Linear probes on frozen encoders.** The same head and the same sweep, on per-pixel features. Features
  come from mean pooling over a 16 × 16 window, upsampled bilinearly from each encoder's token grid
  (paper §3.1).
- **Supervised baselines.** Each uses its original recipe, with no tuning:
  - U-TAE: Adam 10⁻³, batch 4, 100 epochs;
  - TSViT: AdamW 10⁻³ with cosine schedule, batch 16, 100 epochs.
- **Fine-tuning.** OlmoEarth v1.2: 50 epochs, backbone frozen for the first 10, learning rate 10⁻⁴, batch 8.
- **Inputs.** Galileo, Presto, OlmoEarth, U-TAE and TSViT use the **12 monthly S2 mosaics**. AnySat uses the
  first 96 S2 observations of 2019 (`card:326-327`). Tessera v2 uses all S2 and S1 observations of 2019. The
  embeddings are annual 2019 products.
- **Validation is used only for selection.** Training is on `train` only.
- **The paper gives no seeds, no repeats and no ± values.** Every number is a single run. The learning rate
  changes probe mIoU by at most 2.3 points (paper §3.1).
- **The paper does not state** normalisation, augmentation (beyond cropping in few-shot runs), or the probe's
  optimiser and regularisation.

### 3.6 AgriBench inputs and heads [AgriBench]

AgriBench heads are per-pixel tabular models: one row per labelled, non-void pixel.

| Input name | Features per pixel | Matches |
|---|---|---|
| `planteur/s2-monthly` | 12 months × 10 bands from `DATA_S2_MONTHLY` (120) | the raw-band analogue of the paper's monthly input |
| `planteur/emb-aef`, `emb-tessera-v1`, `emb-tessera-v1.1`, `emb-tessera-v2`, `emb-olmoearth-v1.3` | the shipped `EMB_*` vector (64 or 128) | paper Table 2 "released embeddings" and OlmoEarth v1.3 (128d) |
| `planteur/galileo-base`, `planteur/presto-base` | extracted by AgriBench from `DATA_S2_MONTHLY`, with the paper's settings (Galileo patch 4, window 16; Presto patch 1, window 16) | paper Table 2 frozen-encoder rows |
| S1 and Landsat combinations | later | paper Table 3 |

- **[AgriBench] no leakage, anywhere** (D3). Every statistic a model uses is fitted on the training pixels of
  the current fold only: input normalisation, imputation, and anything else fitted to data. The shipped
  `NORM_*_patch.json` files are never used, on any split, because they include every island (2, item 2).
  Selection uses only the fold's own `val` list, which never contains the held-out island or test patches.
- **The linear probe follows the paper** (D4). This is the reference head, and the one the paper's numbers
  compare with:
  - one linear layer per pixel, trained with cross-entropy on **all** non-void training pixels of the fold,
    with no cap and no class weighting (paper §3.1, and Table 8 keeps the default loss);
  - 50 epochs;
  - the learning rate chosen from an 8-point sweep from 10⁻⁴ to 0.5, by mIoU-8 on `val`, then that model
    scores `test` once. Training is on `train` only.

  [AgriBench] Where the paper is silent, AgriBench uses 8 log-spaced points (`np.geomspace(1e-4, 0.5, 8)`),
  AdamW, batches of 4,096 pixels, and the final-epoch weights. Features are standardised with training-fold
  statistics.
- **[AgriBench] other heads** (RF, XGBoost, LightGBM, MLP, TabPFN). The paper has no counterpart for these, so
  they can't follow it exactly. They can't fit 31 million rows either. For these heads:
  - training uses at most **256 non-void pixels per training patch**, sampled uniformly with seed 0 (about
    485,000 rows on the default split);
  - TabPFN, which is limited to small tables, trains on the official **few-shot set with X = 1000 pixels per
    class** (`splits/fewshot_pixels.json`, `card:146-150`). Its results are labelled few-shot, and compared
    with paper Table 7, not Table 2;
  - the grid in the model's YAML is searched on the fold's `val` list by mIoU-8, and the best setting is refit
    on `train` only.

  Evaluation always uses **every** test pixel, for every head.
- **[AgriBench] missing embeddings** (D7). Pixels with no embedding (2, item 9) are left out of training. At
  test time they are scored as Background, the trivial prediction, so every model is scored on the same pixels.

### 3.7 Final tests and provenance [AgriBench]

- **Both test sets are final tests**, as with PASTIS-R fold 5:
  - default split: fold 5;
  - LOIO: each held-out island.

  Model, input and grid choices are made on validation only. A test run needs `experiment.confirm_test=true`,
  and is appended to `/lustre/shared/agriscience-benchmark/runs/planteur/test-ledger.jsonl`.
- **Headline:** the **LOIO mean mIoU-8** (D2), matching the README's "Out-Of-Distribution" placement. The
  default-split test mIoU-8 is reported beside it.
- **Single runs, as in the paper** (D5). Each configuration runs once per split, with seed 0 (the seed of the
  official few-shot pixels). There is no ± in the tables.
- **Provenance.** No fold files need building: the official `splits/443.json` and `splits/loio.json` are used
  as shipped. Each run records the SHA-256 of the split file (from `MANIFEST.sha256`), the input file and the
  config.

### 3.8 What a result contains

For each model × input × split (default, or LOIO island):

- mIoU-8, OA and per-class IoU;
- the number of training rows, and test patches and pixels;
- the head's training regime (all pixels, 256-per-patch cap, or few-shot X), the chosen hyperparameters,
  the seed, and the hashes of the split and input files.

The leaderboard shows the LOIO table first (5 islands plus the mean, like paper Table 6), then the default-split
table (like paper Table 2). Both include the trivial all-Background row.

---

## 4. Paper reference numbers (sanity checks)

All values are mIoU-8 (%), from single runs. **Default split, test fold 5** (paper Table 2), with OA in
brackets:

| Family | Model | mIoU-8 (OA) |
|---|---|---|
| Trivial | all Background | 10.07 (80.56) |
| Linear probe, released embeddings | AEF | 27.68 (89.50) |
| | Tessera v1 / v1.1 | 33.13 (89.77) / 32.78 (89.84) |
| Linear probe, frozen encoder | Presto | 16.25 (80.15) |
| | Galileo | 34.71 (89.91) |
| | Tessera v2 | 34.20 (90.26) |
| | OlmoEarth v1.3, 64d / 128d / 768d | 35.17 / 36.88 / 38.94 |
| Supervised | U-TAE / TSViT | 35.82 (90.62) / 36.33 (90.27) |
| Fine-tuned | Galileo / AnySat | 38.25 (91.27) / 38.35 (90.58) |
| | OlmoEarth v1.2 FT (best) | 40.82 (91.80) |

**LOIO** (paper Table 6):

| Model | Réunion | Guadeloupe | Martinique | Guyane | Mayotte | mean |
|---|--:|--:|--:|--:|--:|--:|
| Trivial (all Background) | 9.33 | 10.17 | 10.18 | 10.33 | 12.42 | 10.49 |
| AEF probe | 9.88 | 23.08 | 20.77 | 12.75 | 12.42 | 15.78 |
| Tessera v2 probe | 11.50 | 28.79 | 29.57 | 12.26 | 12.43 | 18.91 |
| OlmoEarth v1.3 (128d) probe | 26.43 | 31.16 | 35.16 | 15.50 | 12.42 | 24.13 |
| U-TAE | 22.34 | 28.43 | 30.25 | 14.52 | 12.52 | 21.61 |
| TSViT | 18.72 | 28.33 | 33.86 | 12.23 | 12.37 | 21.10 |
| OlmoEarth v1.2 FT (best) | 27.77 | 32.87 | 36.54 | 16.82 | 12.42 | 25.28 |

Every method keeps roughly 55–65% of its in-domain mIoU under LOIO (paper §3.5). Few-shot, with X = 1000
pixels per class (paper Table 7): AEF 24.25, Tessera v2 27.23, U-TAE 22.59, TSViT 27.44.

For AgriBench, the closest rows are the **linear probes on AEF, Tessera and OlmoEarth v1.3 (128d)**. A
per-pixel logistic regression on the same `EMB_*` files should land near them. If it is far below, check the
pipeline before trying other heads.

---

## 5. Decisions

Agreed on 2026-11-09:

- **D2. Headline:** the LOIO mean mIoU-8, with the default-split test beside it. Both are final tests with a
  ledger (3.7).
- **D3. No leakage:** every statistic is fitted on the fold's own training pixels. The shipped `NORM_*` files
  are never used (3.6).
- **D4. Training follows the authors:** the linear probe uses all training pixels with the paper's recipe.
  Only heads the paper doesn't have get the 256-per-patch cap, and TabPFN gets the official few-shot set (3.6).
- **D5. Reporting follows the authors:** single runs with seed 0. Tables show per-island values and the
  5-island mean, Mayotte included, with no ± and no extra columns (3.4, 3.7).
- **D6. Parcel classification:** not now. It isn't in the paper, so there would be no reference numbers.
- **D7. Missing embedding pixels:** left out of training, and scored as Background at test time (3.6).

Settled on 2026-11-10:

- **D1. Download scope:** everything (258 GB, 52,263 files), downloaded at the pinned commit and verified against
  the Hub's hashes (`MANIFEST.sha256`).

---

## 6. AgriBench implementation (Phase A)

### 6.1 The pixel cache

[`tools/build_planteur_features.py`](../tools/build_planteur_features.py) (a Slurm job, about 2 minutes per input
on 8 cores) turns the patches into one table per input in `features/planteur/` (59 GB in all):

- `meta.npz`: one row per **non-void** pixel of every patch (44,963,855 rows; 92,145 void pixels dropped, as the
  card says), patches in ID order, pixels row-major. It holds the class, the pixel position, and each patch's row
  range. Every split selects rows of the same tables, so the default split and the five LOIO folds share one copy.
- `<input>/X.npy` + `valid.npy` + `info.json`. `s2-monthly` is the 10 `DATA_S2` bands of the 12 monthly mosaics,
  month-major (`m00_B02` … `m11_B12`), as raw uint16 with 0 = no data (read as missing). The embeddings are the
  shipped vectors stored as float16 (the largest |value| is 15.6, so float16 loses nothing that matters).
  `valid` is False where the embedding is missing, NaN or all zero (D7). Counted: Tessera v1 842,326 pixels in
  157 patches (6 with no file), Tessera v1.1 5,662 pixels in 4 patches, none for AEF, Tessera v2 and OlmoEarth.
- The shipped `NORM_*` files are never read (D3).

### 6.2 The runner

[`run_planteur.py`](../run_planteur.py) (Hydra, `conf/planteur/`) runs one input × head × fold:
`split=default fold=default`, or `split=loio fold=<reunion|guadeloupe|martinique|guyane|mayotte>`.

1. Training pixels by the head's regime: `all` (probe), `cap-256` (LightGBM; at most 256 valid pixels per
   training patch, seed 0), or `fewshot-X` (TabPFN, TabICL; `experiment.fewshot=X` for any head). Pixels without
   an input value are dropped (D7).
   - **[AgriBench] few-shot on LOIO.** The official few-shot pixels come from default-split `train` patches, and
     these include every island, so they can't be used for a LOIO fold. For LOIO, X pixels per class are drawn
     uniformly (seed 0) from the fold's own valid training pixels.
2. The head's setting is chosen by mIoU-8 on **all** val pixels: the probe's learning rate, LightGBM's grid
   (`num_leaves` 31/127 × `learning_rate` 0.05/0.1, 300 trees). TabPFN and TabICL have no grid.
3. Without `experiment.confirm_test=true`, the run stops at val (development). With it, every test pixel is
   predicted (pixels without input value → Background, D7), one confusion matrix gives mIoU-8, OA and per-class
   IoU, and the run is appended to `runs/planteur/test-ledger.jsonl`.

Each run writes `fold-<fold>.json` (val and test scores, the tuning table, the regime, pixel counts, and the
hashes of the split file, few-shot file, `meta.npz`, input `info.json` and resolved config, plus the git commit),
`fold-<fold>-confusion.npy` and `fold-<fold>-predictions.npz`. The last two stay on the cluster.

**The probe** ([`tools/planteur_probe.py`](../tools/planteur_probe.py)) follows 3.6 exactly. Its 8 learning rates
train together as 8 independent linear heads that see the same batches (one forward pass, one AdamW parameter
group each). This is the same as 8 separate runs with one data order, at about the cost of one. AdamW's default
weight decay (0.01) is used; the paper states none. The standardised training table sits on the GPU in float16.

### 6.3 Sweeps, results and the dashboard

    python -m tools.sweep_planteur --name pa1 --inputs s2-monthly emb-aef emb-tessera-v1 emb-tessera-v1.1 \
        emb-tessera-v2 emb-olmoearth-v1.3 --models background probe lightgbm tabpfn tabicl

[`tools/sweep_planteur.py`](../tools/sweep_planteur.py) runs one Slurm array task per fold (6 per input × head:
default + 5 islands) from a frozen copy of the commit, at low priority on the GPU, and resumes where it stopped.
Every sweep task is a final test (`confirm_test=true`), because the heads and grids are fixed in advance and every
choice inside a run uses val only. [`tools/collect_planteur.py`](../tools/collect_planteur.py) writes
`results/planteur/<sweep>.csv`: one row per default test, per island, and the LOIO mean (only once all five
islands are in). [`tools/publish_results.sh`](../tools/publish_results.sh) collects every PLANTEUR sweep, and the
dashboard's PLANTEUR view shows the default test beside the LOIO mean, the per-island table, and the paper's
numbers for the same rows.

### 6.4 First results

Filled in as `pa1` finishes; see the dashboard for the live table.

---

## 7. Phase B plan [AgriBench]

Phase B adds the frozen encoders AgriBench already wraps for YieldSAT, then early season and label efficiency. Every
new input is written as a `features/planteur/<name>/` table aligned with `meta.npz`, so the runner, the heads and
the sweep tool need no change beyond a line in `tools/planteur.INPUTS`, and the probe is the reference head for
all of them.

### 7.1 Encoders the paper also reports (Table 2 frozen-encoder rows)

Input: the 12 monthly S2 mosaics, the 10 `DATA_S2` bands, months dated at their mid-points in 2019 (`MONTHLY/months.json`).
Pixel features as the paper makes them (3.5): the encoder's token grid, **mean-pooled over 16 × 16 pixel windows**,
then upsampled bilinearly to the 128 × 128 label grid.

| Input | Encoder and settings | Reuses | Paper |
|---|---|---|---|
| `galileo-base` | Galileo base, patch size 4, S2 only, 12 monthly steps | `tools/embed_yieldsat_galileo.py`, `tools/extract_galileo.py` | 34.71 |
| `presto-base` | Presto, patch size 1 (per pixel), S2 only, 12 monthly steps, the patch centre's lat/lon | `tools/embed_yieldsat_presto.py` | 16.25 |

Presto's per-pixel features are pooled over the same 16 × 16 windows, to match the paper's setting. Both are
cross-checked against the paper's numbers before anything else uses them.

### 7.2 Further time-series encoders

Same input and pooling, adapted from the YieldSAT wrappers:

- **OlmoEarth v1.2 Base**, pixel patch 1 with the real month dates (`tools/embed_yieldsat_olmoearth.py`). This
  can be checked against the shipped v1.3 embedding.
- **AnySat** on the first 96 S2 observations of 2019 (`DATA_S2`, as in `card:326-327`), so it follows the paper's
  input instead of the monthly one (`tools/embed_yieldsat_anysat.py`).
- **Prithvi-EO-2.0 300M-TL**, 224-px windows reflect-padded from each 128-px patch, 4 monthly steps (the first
  month, then the months of the patch's NDVI rise, peak and decline, chosen from the imagery only, not the labels)
  (`tools/embed_yieldsat_prithvi.py`).

### 7.3 Single-date (patch) encoders

Copernicus-FM, DOFA, CROMA and SoftCon via `tools/patch_embed.py` (`tools/yieldsat_patches.py` supplies the
YieldSAT windows; a PLANTEUR adapter supplies one image per patch). The date is the patch's **peak-greenness
month** (highest median NDVI over the patch), chosen from the imagery only. The token grid is upsampled
bilinearly to pixels, with no extra pooling (these encoders' tokens are already 8–16 px). TerraMind, THOR, Clay and
SatMAE follow once the single-date session's terratorch environment exists.

### 7.4 Early season

`experiment.months=k` keeps the first k monthly mosaics (January to month k of 2019) of `s2-monthly`, already
implemented (`--months` in the sweep tool, runs marked `@m04`). Plan: k = 2, 4, 6, 8, 10, and 12 = the full run,
with the probe and LightGBM, on both splits. Annual embeddings can't be truncated; the time-series encoders of 7.1
and 7.2 are re-embedded on the first k months (k = 4, 6, 8) once their full-year runs check out. The dashboard
gets a PLANTEUR early-season view: mIoU-8 (LOIO mean and default test) against k.

### 7.5 Label efficiency

`experiment.fewshot=X` trains any head on the official few-shot pixels, X = 10, 25, 100, 1000 per class, already
implemented (`--fewshot` in the sweep tool, runs marked `@x0100`). On the default split these are the authors'
pixels, so the results compare with paper Table 7 (AEF 24.25, Tessera v2 27.23 at X = 1000). Plan: probe,
LightGBM and TabPFN on every input, default split, single run (the official pixels have one seed), plus the full
run as the last point. LOIO uses the AgriBench draw (6.2). With X = 10, the per-island scores are very noisy, so
LOIO starts at X = 100. The dashboard gets a PLANTEUR label-efficiency view: mIoU-8 against X, log scale.
