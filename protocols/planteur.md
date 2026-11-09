# PLANTEUR evaluation protocol

Status: **draft 2026-11-09. Not downloaded, nothing run, decisions in section 5 still open.** This page sets how
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
- **[AgriBench] extras, never ranked:**
  - the standard deviation across the 5 islands;
  - the mean over the 4 islands without Mayotte (2, item 7);
  - per-class IoU for every fold.

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

- **[AgriBench] normalisation:** fit inside the head's pipeline, on the training pixels of the current fold
  only. AgriBench never uses `NORM_*_patch.json` for LOIO (2, item 2; D3).
- **[AgriBench] tuning, replacing the paper's learning-rate sweep:**
  1. Search the grid in the model's YAML on the fold's own `val` list (fold 4, or the LOIO fold's `val`).
  2. Select by **mIoU-8** on val.
  3. Refit the best setting on that fold's `train` only, as the paper does.
  4. Score `test` once.

  For logistic regression this is the analogue of the paper's sweep: it tunes `C` instead of the learning
  rate.
- **[AgriBench] training-pixel cap** (D4). The default training fold has 31 million pixels. AgriBench samples
  at most **256 non-void pixels per training patch**, uniformly, with the run's seed (about 485,000 rows).
  Heads limited to small tables (TabPFN) are instead trained on the official **few-shot set with X = 1000
  pixels per class** (`splits/fewshot_pixels.json`, `card:146-150`). Those results are labelled few-shot, and
  compared with paper Table 7, not Table 2. Evaluation always uses **every** test pixel.
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
- **Seeds** (D5): stochastic heads run 3 seeds on each split. AgriBench reports the mean ± std over seeds
  (`np.std`, ddof=0). The paper reports single runs, so its numbers compare with our mean.
- **Provenance.** No fold files need building: the official `splits/443.json` and `splits/loio.json` are used
  as shipped. Each run records the SHA-256 of the split file (from `MANIFEST.sha256`), the input file and the
  config.

### 3.8 What a result contains

For each model × input × split (default, or LOIO island):

- mIoU-8, OA and per-class IoU;
- the number of training rows, and test patches and pixels;
- the pixel cap or few-shot X, the seed, and the hashes of the split and input files.

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

## 5. Decisions to confirm

Recommended options first.

- **D1. Download scope.** Download everything, 258 GB (recommended). The shipped embeddings are exactly the
  kind of input AgriBench compares, and the monthly mosaics feed Galileo and Presto. The alternative is core
  only, 133 GB, with the rest added later.
- **D2. Headline.** The LOIO mean mIoU-8 is the headline (recommended), with the default-split test beside it.
  Both are treated as final tests with a ledger (3.7).
- **D3. Normalisation for LOIO.** Fit on each fold's own training pixels (recommended). The alternative is the
  shipped `NORM_*` files, which the authors say are not held out.
- **D4. Training-pixel cap.** At most 256 non-void pixels per training patch. TabPFN-type heads use the
  official X = 1000 few-shot set instead.
- **D5. Seeds.** 3 seeds per configuration, reported as mean ± std. LOIO also gets the across-island std and a
  no-Mayotte mean as unranked extras.
- **D6. Parcel classification.** A parcel-level track like PASTIS-R, using `PLANTEUR_Dataset_PixelSet`,
  could follow later. It is not in the paper, so there would be no reference numbers. It would also need a rule
  for the 592 parcels that cross splits. Recommended: not now.
- **D7. Missing embedding pixels.** Leave them out of training, and score them as Background at test time.
