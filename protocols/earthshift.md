# EarthShift field-delineation evaluation protocol

Status: **agreed 2026-11-09 (section 5). Data downloaded and verified, split files frozen and input cache built
on Anunna (2026-11-09); the training and evaluation code is not written yet.** This page sets
how AgriBench scores a model on EarthShift's field-delineation tasks, so results are comparable with the
paper and with each other. Sections marked **[paper]** follow Doerksen and Kerner (2026) or their released
code exactly. Sections marked **[AgriBench]** fill gaps the paper leaves open or go beyond it.

**Scope: only the location (geographic) shift and temporal shift settings.** EarthShift also has sensor,
spatial-scale and data-source shifts. They are out of scope here, and none of them uses field-delineation
data anyway (they use RESISC45/UCMerced, DeepGlobe/DFC2022, m-EuroSAT, BigEarthNet v2 and Sen1Floods11).

Sources. Each rule below cites one of these:

- **Paper:** Doerksen and Kerner, *EarthShift: a benchmark for measuring robustness to real-world
  distribution shifts in Earth observation*, arXiv:2605.29330 (2026), 27 pages. Cited as "paper p.N".
  Website: https://earthshift.github.io/.
- **Code:** https://github.com/kerner-lab/earthshift at commit `39d77e3` (2026-10-02). It is identical to the
  tag `v1.0-arxiv`, which the README names as the code behind the paper, apart from the README. Cited as
  `file:line`.
- **Data:** EarthShift ships no data of its own for these tasks. It uses **Fields of The World (FTW)**
  (Kerner et al., AAAI 2025, arXiv:2409.16252), downloaded from Source Cooperative as country zips. Facts
  about the files were read from the archives on 2026-11-09 (section 1).

> Public repo: nothing chip-level (predictions, per-chip scores) is committed here. Split files and run
> outputs live under `/lustre/shared/agriscience-benchmark/`. Results JSON in `results/` only carries
> aggregate metrics and hashes, as for YieldSAT.

---

## 0. Getting the data onto Anunna

**No account and no licence click-through are needed.** The archive is public on Source Cooperative
(https://source.coop/kerner-lab/fields-of-the-world). The download URL pattern is the one FTW's own CLI uses:
`https://data.source.coop/kerner-lab/fields-of-the-world-archive/<country>.zip`, plus a `checksum.md5`.

**The licence is per country** (source.coop README, "Source Data License" column). The ones we need:

| Country | Used for | Licence |
|---|---|---|
| Germany | training set of every task below | DL-DE/BY-2.0 |
| Cambodia | location-shift test | CC-BY-4.0 |
| Denmark | location-shift test | CC0-1.0 |
| South Africa | temporal-shift task | **CC-BY-NC-SA-4.0** (non-commercial) |

Research use at WUR is fine under all four. Latvia, Lithuania and Portugal (only needed for the optional
FTW-All task) are also non-commercial. The EarthShift paper lists FTW as "CC BY-SA 4.0" (paper p.14,
Table 3). The per-country licences above are the more precise statement.

### Size and file counts

The archives keep every chip as a separate small GeoTIFF:

| Download | Countries | Zip size | Unpacked | Files |
|---|---|---:|---:|---:|
| **Core (downloaded)** | germany, cambodia, denmark, south_africa | **4.3 GB** | 9.1 GB | **26,505** |
| Everything (only for FTW-All, 2.3c) | all 25 | 56.8 GB | 120.5 GB | 352,803 |

Per country (zip GB / files): germany 0.54 / 3,432; cambodia 0.28 / 1,534; denmark 2.90 / 17,802;
south_africa 0.61 / 3,737.

**Raw zips stay as they are; models read a packed input cache built once from them** [AgriBench]:

- **The zips are the raw copy.** A zip is one file on Lustre, and the Anunna skill asks us to keep finished
  read-only data as archives (`.claude/skills/anunna/references/file-counts.md`). Each chip is a separately
  compressed entry, so any chip can be read without unpacking the rest: `zipfile.ZipFile(z).open(name)` or
  `rasterio.open("/vsizip/<zip>/<inner path>")`. Decompressing one chip takes milliseconds.
- **Nothing unzips per run.** `tools/build_earthshift_inputs.py` reads the zips once and writes, per
  country, every sample the split files use outside the gaps, to `$project/inputs/earthshift/v1/`:
  - `<country>-images.npy`: uint16 `(N, 4, 256, 256)`, bands R, G, B, NIR, reflectance × 10,000;
  - `<country>-masks.npy`: uint8 `(N, 256, 256)`;
  - `<country>-index.csv`: row, `aoi_id`, window;
  - `<country>.json`: shapes, label pixel counts, zip and split hashes.

  Training jobs memory-map the arrays or load them into RAM: a few large sequential reads, which is the
  fastest pattern on Lustre. Run it as `sbatch prepare.sbatch tools.build_earthshift_inputs`.
- **The cache is small**, because only the samples the tasks use are kept: all used Germany chips, the
  Cambodia and Denmark test chips, and the used South Africa chips. That's about 2,600 images, roughly
  1.5 GB in 4 files. Denmark's 2.9 GB zip shrinks to about 0.4 GB, because only its 332 test chips are
  needed.
- **Encoder features are cached once** per encoder × country in `$project/features/earthshift/`, as for
  PASTIS-R. The frozen-backbone runs and their learning-rate sweeps read only those.
- **Unpacking the zips into folders on Lustre gains nothing:** 26,505 small files, slower to list and
  read, and no faster than the packed cache. If a tool really needs a folder tree (e.g. to rerun the
  EarthShift code unchanged), unpack to `$TMPDIR` on the compute node inside that job.

### Commands

From the repo root on Anunna (1 core, 2 GB, a network-bound job):

    sbatch tools/download_earthshift.sbatch

It writes `/lustre/shared/agriscience-benchmark/datasets/earthshift/ftw-zips/{germany,cambodia,denmark,south_africa}.zip`
and `checksum.md5`. For the full set, use
`COUNTRIES=all sbatch --time=12:00:00 tools/download_earthshift.sbatch`. Re-running skips finished zips and
resumes a partial one.

**Verification** is built in. For each zip, the job checks:

- the MD5 against the upstream `checksum.md5`;
- the number of files against the expected count (table above).

It prints `OK` or `FAIL` per country and exits non-zero on any failure. To check again later:

    cd /lustre/shared/agriscience-benchmark/datasets/earthshift/ftw-zips
    for c in germany cambodia denmark south_africa; do
      echo "$c $(md5sum $c.zip | cut -c1-32) $(grep "^$c," checksum.md5 | cut -d, -f2) $(unzip -Z1 $c.zip | grep -vc '/$')"
    done
    # expect the two hashes to match, and 3432 / 1534 / 17802 / 3737 files

---

## 1. What is in the data

Each country zip unpacks to (no country folder inside the zip):

| Path | Contents |
|---|---|
| `chips_<country>.parquet` | one row per chip: `aoi_id`, `split` (train / val / test, FTW's official split), `geometry` |
| `data_config_<country>.json` | dates of window A and window B, collection year, grid polygons |
| `s2_images/window_a/<aoi_id>.tif` | Sentinel-2 L2A, 4 bands (R, G, B, NIR), uint16 reflectance × 10,000, 256 × 256 px in EPSG:4326. A chip covers about 1.5 km north–south and less east–west (≈ 0.95 km in Germany, ≈ 1.27 km in South Africa) |
| `s2_images/window_b/<aoi_id>.tif` | the same chip on a second date |
| `label_masks/semantic_3class/<aoi_id>.tif` | uint8: 0 background, 1 field interior, 2 field boundary |
| `label_masks/semantic_2class/`, `label_masks/instance/` | other label forms (not used by EarthShift) |

EarthShift reads `semantic_3class` and both windows as **separate samples**: one 4-band image each. It
never stacks the windows into 8 bands (`data_manager.py:219-221`, `:240-256`). It keeps a sample only if
both its image and its label exist (`data_manager.py:270`), and divides by 10,000 (`data_manager.py:305`).

### Chips per country and split (read from the archives)

| Country | Train | Val | Test | Grids | Windows A / B |
|---|---:|---:|---:|---|---|
| Germany | 306 | 30 | 350 | train + val in grid g2, test in grid g1 | A Jul–Sep, B Apr–Jun; **2018 and 2019** |
| Cambodia | 245 (274) | 27 (36) | 25 (34) | 32 small grids | A Oct–Nov 2021, B May–Jul 2021 |
| Denmark | 2,868 | 360 | 332 | g6, g22 | A Jul–Oct 2021, B Jan–Jun 2021 |
| South Africa | 590 | 72 | 85 | g1–g3, all splits in each | A Jul–Sep 2018, B Apr–Jun 2018 |

The Cambodia numbers in brackets are chips in the parquet. Only 297 of Cambodia's 344 chips have a label
and a window-A image, so EarthShift silently uses fewer.

### Things to know

1. **The Germany "year" shift is also a location shift.** FTW's Germany train and val chips are all in grid
   g2, and its test chips are all in grid g1, a separate area to the north-west (each grid is about 25 × 25 km;
   the nearest g1 and g2 chips are 2.4 km apart). `data_config_germany.json`
   lists the 2019 season first and the 2018 season second, and grids g1 then g2. The paper says the training
   data is 2018 and the test data is 2019 (paper p.4–5). So "2018 → 2019" is also "grid g2 → grid g1".
2. **The location-shift ID test set is the same Germany test split (g1, 2019).** So the "ID" test is already
   shifted in year and grid from training. The ID numbers in the paper's Table 12 aren't a pure
   in-distribution score.
3. **The paper's splits leak inside Germany grid g2 and inside South Africa** (AgriBench replaces them, 2.1b):
   - **Germany year task, ID test.** It's a random 412 / 200 split of the 612 training samples
     (`run_pipeline.py:203-210`). A chip's window A and window B are separate samples with the *same* label
     mask, so one can land in train and the other in the ID test. Neighbouring chips also share fields. (The
     paper's Table 1 says 413 / 200; the code gives 412 / 200.)
   - **Germany validation.** FTW's 30 Germany val chips are single chips scattered among the training chips
     of grid g2, each one touching training chips on all sides.
   - **South Africa.** FTW's val and test chips are 3 × 3 blocks inside the same grids as training, with no
     gap: 180 of the 590 training chips touch a val or test chip, and 9 val chips touch a test chip.
   - Fields cross chip edges, so a chip next to a test chip shows part of the same fields. Those ID and val
     scores are optimistic.
4. **Denmark's OOD test has 664 samples here** (332 chips × 2 windows), but the paper lists 663 (paper p.4,
   Table 1). The difference is probably one unreadable file in the authors' copy. It matters little.
5. **Window B comes before window A in time** in all four countries (data configs). The paper's "planting
   and harvesting" wording (paper p.4) isn't literal for every country.

---

## 2. Evaluation protocol

### 2.1 The tasks [paper]

Four field-delineation tasks, all 3-class semantic segmentation (paper p.4, Table 1;
`configs/geo-shift-exp.json`, `configs/temporal-shift-exp.json`):

| Task | Shift | Code name | Train | Val | ID test | OOD test |
|---|---|---|---|---|---|---|
| **DE→KH** | location | `ftw-germany-cambodia` | DE train, A+B (612) | DE val, A+B (60) | DE test, A+B (700) | KH test, A+B (50) |
| **DE→DK** | location | `ftw-germany-denmark` | DE train, A+B (612) | DE val, A+B (60) | DE test, A+B (700) | DK test, A+B (664) |
| **ZA A→B** | temporal (season) | `ftw-south_africa-window` | ZA train, A (590) | ZA val, A (72) | ZA test, A (85) | ZA test, **B** (85) |
| **DE 2018→2019** | temporal (year) | `ftw-germany-year` | 412 of DE train, A+B | DE val, A+B (60) | other 200 of DE train, A+B | DE test, A+B (700) |

These are the paper's definitions. **AgriBench uses the leak-free version in 2.1b**, which keeps the same
countries, windows and OOD test sets but re-draws train, val and ID test inside Germany grid g2 and South
Africa.

Sources for the sample definitions:

- **Both windows** are used for the location tasks and the Germany year task: `data_manager.py:252-256`
  (the default case) and `:238-241`.
- **Season tasks** use window A for train, val and ID test, and window B of the test chips for the OOD
  test: `data_manager.py:244-250`. The paper says the same: "We selected window A as our ID set and window B
  as our OOD set" (paper p.5).
- **The OOD test is always the OOD country's `test` split**: `run_pipeline.py:157-164`.
- **The Germany year split** is `random_split(train, [412, 200], generator=torch.Generator().manual_seed(42))`:
  `run_pipeline.py:203-210`.

The paper's pairings were chosen "because of their differences in geographic location, field boundary size,
and shape" (paper p.5). Cambodia is the far shift (other continent, small plots); Denmark is the near one.

**Not included:**

- `ftw-germany-brazil`, `-kenya` and `-south_africa` are in the code config (`configs/geo-shift-exp.json:15-44`)
  but not in the paper, and have no reference numbers. Brazil and Kenya also only have presence labels in FTW
  (they aren't in FTW's `FULL_DATA_COUNTRIES`, `ftw_tools/settings.py:54`).
- The single-country `-window` season tasks for Germany, Cambodia, Brazil and Kenya are in the config but
  not in the paper.
- **FTW-All A→B** is in the paper; see 2.3c.

### 2.1b Leak-free splits [AgriBench: replaces the paper's train / val / ID test]

**Rule:** within a task, no two of train, val and ID test may share a chip or touch each other. Both windows
of a chip always go to the same side, and at least one whole chip (about 1.5 km) separates train, val and ID
test. "Touching" is measured on the chip footprints in `chips_<country>.parquet` (within 50 m, edges and
corners included), not on chip ids. Chips in the gap are dropped. OOD test sets are untouched: they are other
countries, a different grid (Germany g1 is 2.4 km from g2), or the other window of the ID-test chips. That
last one is the shift being measured, not a leak.

**Germany grid g2 (all three Germany tasks).** The 336 train + val chips of g2 lie in exactly **14 latitude
bands of 24 chips**, 1.535 km apart (chip centroids). Band 1 is the northernmost. The chip ids'
`<row>` number is *not* a clean band (some id rows wrap onto two latitudes), so bands come from the
footprints. g2 is cut into whole bands, ignoring FTW's train/val labels:

| Bands | Chips | Role in DE→KH, DE→DK | Role in DE 2018→2019 |
|---|---:|---|---|
| 1–7 | 168 | train | train |
| 8 | 24 | gap (dropped) | gap (dropped) |
| 9 | 24 | val | val |
| 10 | 24 | gap (dropped) | gap (dropped) |
| 11–14 | 96 | train | ID test |

So, in samples (chips × 2 windows):

| Task | Train | Val | ID test | OOD test |
|---|---:|---:|---:|---:|
| DE→KH, DE→DK | 528 (bands 1–7 + 11–14) | 48 | 700 (DE test, grid g1, unchanged) | 50 / 664 (unchanged) |
| DE 2018→2019 | 336 (bands 1–7) | 48 | 192 (bands 11–14) | 700 (DE test, grid g1, unchanged) |

Checked on the footprints (2026-11-09): train ↔ val and val ↔ ID test are at least 1,538 m apart, and train ↔
ID test at least 4,607 m. The location tasks share one validation set, so one learning-rate sweep still
covers both.

**South Africa.** FTW's 3 × 3 val and test blocks are kept. Training chips that touch a val or test chip are
dropped, and so are val chips that touch a test chip:

| Task | Train | Val | ID test | OOD test |
|---|---:|---:|---:|---:|
| ZA A→B | 410 (window A) | 63 (window A) | 85 (window A, unchanged) | 85 (window B, unchanged) |

**Cost:** fewer training chips than the paper: about 30% fewer in South Africa (410 vs 590), 14% fewer in the
Germany location tasks (264 vs 306) and 18% fewer in the Germany year task (168 vs about 206).
**Consequence:** the paper's numbers in section 3 are no longer like-for-like for any of the four tasks.
They stay as rough sanity checks. Models in AgriBench are compared only with each other.

### 2.2 Training rules [paper]

These come from paper p.6–8 and match the code unless noted.

- **Two settings:** the **frozen backbone**, where only the new head trains, and **full fine-tuning**.
  Frozen runs train for **40 epochs**, full runs for **50**. Both use batch size 32, AdamW with no weight
  decay (`run_pipeline.py:231`, `training.py:431`), and cross-entropy loss with `ignore_index=255`
  (`training.py:390`).
- **Schedule:** a 20-epoch linear warm-up, then cosine decay over the remaining epochs (`training.py:440-448`).
- **No early stopping or checkpoint choice.** The model after the last epoch is scored. Validation mIoU is
  only logged (`training.py:256-340`).
- **The head** is a 3×3 conv (256 channels) → ReLU → 1×1 conv on the backbone's patch features, then bilinear
  upsampling to the input size (`utils.py:99-108`, e.g. `models/galileo_model.py:71-103`). The paper calls
  it "a simple decoder head consisting of a convolution - ReLU - convolution sequential module" (paper p.6).
- **Input adaptation:** a learnable 1×1 conv maps the 4 bands to the channels the backbone expects
  (`utils.py:110-124`; paper p.6).
- **Images are resized** to each backbone's pre-training size: 224 for most, 120 for CROMA, 256 for Clay
  (`data_manager.py:734-787`). Masks are resized with nearest neighbour (`data_manager.py:906-910`).
- **Normalisation:** per channel, with FTW mean `[0.1552, 0.1355, 0.1105, 0.2743]` and std
  `[0.1888, 0.1757, 0.1809, 0.1742]` (Clay's Sentinel-2 statistics; `data_manager.py:641-642`, paper p.8).
- **No data augmentation** (none in `data_manager.py`).

### 2.3 Tuning [paper]

- **Learning rate is the only thing tuned.** Grids (paper p.6):
  - full fine-tuning: {1, 3, 6} × {1e-5, 1e-4, 1e-3};
  - frozen backbone: {1, 3, 5} × {1e-4, 1e-3, 1e-2}.
- **One sweep per shift type, reused for the other tasks of that type.** Location uses DE→KH, temporal uses
  ZA A→B. "We used the best learning rate from the model validation score and implemented this learning rate
  in any other tasks of the same shift type" (paper p.6).
- **Selection is on the ID validation set only:** the best learning rate by mean validation mIoU over 5
  seeds (paper p.14, A.4). The OOD data is never used for selection. The chosen rates are in paper Tables 17
  (location) and 23 (temporal), pp.25–27.
- **Final scores:** each model is retrained with the chosen rate on **5 random seeds**, and reported as
  mean ± std over those seeds (paper p.14; Tables 12–13).

AgriBench follows this, on the leak-free validation sets of 2.1b. DE→KH and DE→DK share the same training
and validation chips, so one sweep covers both.

### 2.3c FTW-All season task [paper, optional for AgriBench]

The paper also runs A→B on all FTW countries pooled (paper p.4, Table 1: 49,333 / 7,041 / 14,110 / 14,110).
The code ignores FTW's official split here. It re-splits chips 70 / 10 / 20 per country with a seed
(`data_manager.py:314-400`), so neighbouring chips end up in train and test. The paper gives only effective
robustness for it (Table 5, row "Season-to-Season"), not mIoU. It needs the 56.8 GB, 352,803-file download.
**Left out** (decision 5.1).

### 2.4 Metric [paper]

- **mIoU over the 3 classes** (background, field interior, boundary), computed with
  `torchmetrics.JaccardIndex(task="multiclass", num_classes=3, ignore_index=255)` (`testing.py:118`).
- **Aggregation:** the metric uses the default macro average. Intersections and unions are summed over
  *every pixel of the whole test set* first, then IoU is taken per class and averaged over the 3 classes. It
  is not a mean of per-chip scores.
- **Reported:** ID mIoU, OOD mIoU and Δ (the absolute difference) per task and seed, as mean ± std over 5
  seeds (paper pp.21–22, Tables 12–13).
- **Effective robustness ρ = OOD − β(ID)** (paper p.3, Eq. 1). β is a straight line fitted to the
  (ID, OOD) mIoU of *all models and all tasks within a shift type*, separately for frozen and full fine-tuning.
  The paper's fitted lines, in mIoU percentage points (paper pp.15–16, Tables 5 and 7):

  | Shift | Frozen | Full fine-tuning |
  |---|---|---|
  | Temporal | OOD = 1.020 × ID − 2.2 | OOD = 0.880 × ID + 4.3 |
  | Location | OOD = 0.800 × ID − 14.6 | OOD = 0.725 × ID − 12.7 |

  The temporal lines include the FTW-All task.

**Headline metric = the paper's**: 3-class mIoU at each backbone's resized input size, as above, plus ID,
OOD, Δ and ρ.

**[AgriBench] secondary columns** (reported, never used to rank or select):

- **Native-resolution mIoU.** Logits are upsampled to the native 256 × 256 mask and scored there, so models
  with different input sizes (224, 120, 256 px) are scored on the same pixels.
- **Per-class IoU**, and the **boundary-class IoU** (class 2) on its own. Boundaries are the hard part of
  delineation, and a 3-class mean hides them.
- **ρ uses the paper's fixed lines** in the table above, not a fit to AgriBench's own models. That keeps ρ
  stable as models are added and comparable with the paper.
- **Object-level scores** (FTW's per-field precision and recall at IoU ≥ 0.5) are **not** included: the
  EarthShift paper doesn't use them.

### 2.5 Inputs and heads that apply [AgriBench]

Field delineation needs a label for every pixel. The tabular heads AgriBench uses for PASTIS-R and YieldSAT
predict one value per row, so most of them don't carry over directly:

| AgriBench piece | Fits? | How |
|---|---|---|
| **Frozen encoder + light decoder** (Galileo, Presto, any GFM) | **Yes, main route** | Extract dense patch features once per chip and window, store them, and train the paper's conv-ReLU-conv head on them. This is exactly the paper's frozen-backbone setting, cheap enough to sweep, and needs no backbone gradients. |
| **Full fine-tuning** | Yes, more expensive | Needs a GPU training loop over images. Only for encoders that are worth it. |
| **Per-pixel tabular heads** (logistic regression, RF, LightGBM, TabPFN) | Only as a baseline | Classify each pixel from its 4 bands, possibly with its dense encoder feature. There is no spatial context, so boundaries will be poor. Train on a capped random sample of pixels per chip. Useful as a floor. |
| **Presto** | Weak fit | It is a per-pixel time-series model. Here each sample is one date with 4 bands, so Presto sees one time step without SWIR or red-edge. |
| **Location encoders** | No | The location is almost constant within a 256 × 256 chip, so it cannot separate field from boundary. As a chip-level input it would just encode the country, which is the shift being tested. |
| **Crop models** | No | No yield or phenology target here. |

**Input names:**

- `ftw/rgbn-window`: one window, 4 bands, the paper's input.
- An 8-band stacked A+B input is FTW's own default, but it would break the temporal tasks, which need one
  window per sample. It is **not** part of this protocol.

**Training follows the paper's two settings exactly** (2.2–2.3): frozen backbone (head only, 40 epochs) and
full fine-tuning (50 epochs). Both use the same decoder, optimiser, schedule, batch size and learning-rate
grids, no augmentation, and score the last epoch. Per-pixel tabular heads are an optional extra baseline,
not part of the core benchmark.

**Where the paper's text and its code disagree, AgriBench follows the text** [AgriBench]:

- **Input adapter.** The paper calls the 1×1 input adapter "learnable" (paper p.6). In frozen mode the code
  unfreezes only the decoder (`training.py:408-410`), so the adapter stays at its random initialisation.
  AgriBench trains the adapter together with the head.
- **Galileo.** The paper evaluates Galileo as a backbone, but its segmentation wrapper runs only the
  patch-embedding layer, not the transformer (`models/galileo_model.py:73-103`). AgriBench uses the full
  encoder.

Expect AgriBench's Galileo numbers to differ from the paper's.

### 2.6 Splits, seeds and provenance [AgriBench]

- **Frozen split files.** `tools/make_earthshift_splits.py` (`sbatch prepare.sbatch tools.make_earthshift_splits`;
  `--check` rebuilds them in memory and compares) reads the four
  `chips_<country>.parquet` files from inside the zips. It writes one file per task to
  `/lustre/shared/agriscience-benchmark/splits/earthshift/<task>.csv`, with columns
  `country, aoi_id, window, role` (role = train / val / id_test / ood_test / gap). It applies the rules in
  2.1b, keeps only samples whose image and label exist, as the code does, and refuses to overwrite.
  Beside the task files it writes `chips.csv` (every chip: FTW split, grid, Germany band, which files exist)
  and `manifest.json` (SHA-256 of every file, of the four zips and of the code).
- **It checks the rule** before writing: no `aoi_id` in two roles, and no train, val or ID-test chip within
  50 m of a chip of another of those roles (footprints from the parquet, in UTM). It stops if either check
  fails, and prints the counts in 2.1b.
- **No randomness:** every split is fixed by chip position, so no seed is needed.
- **Seeds:** five fixed seeds, `0 1 2 3 4`. The paper draws them at random (`run_pipeline.py:66-70`) and
  doesn't publish them.
- Each run records the SHA-256 of the split file, the zip and the config, as for PASTIS-R and YieldSAT.

### 2.7 Development vs final test [AgriBench]

- **Development:** learning-rate sweeps and every model choice use the **ID validation set only**, as in
  the paper.
- **Final:** the ID test and OOD test sets are scored only for configurations fixed on validation. Every run
  that touches them is logged in `$project/runs/earthshift/test-ledger.jsonl`. Test-set scores are never
  used to choose anything.

### 2.8 What a result contains

For each model × setting (frozen / full) × task:

- headline: ID mIoU, OOD mIoU, Δ and ρ (paper-style, resized resolution), as mean ± std over 5 seeds;
- secondary: native-resolution mIoU and per-class IoU, boundary IoU included;
- the learning rate, the number of samples in each role, seeds, and split-file and zip hashes.

**Leaderboard:** one table for location (DE→KH, DE→DK) and one for temporal (ZA A→B, DE 2018→2019),
laid out like the paper's Tables 12–13, frozen and full fine-tuning in separate blocks.

---

## 3. Paper reference numbers (sanity checks)

mIoU (%), mean ± std over 5 seeds, at each backbone's resized input size. From paper p.21 (Table 12) and p.22
(Table 13). ∗ = frozen backbone. These use the paper's splits, so they are **not like-for-like** with
AgriBench's leak-free splits (2.1b), especially the ID columns. The OOD test sets are the same, but models
train on fewer chips.

| Model | DE→KH ID | DE→KH OOD | DE→DK ID | DE→DK OOD | ZA A→B ID | ZA A→B OOD | DE year ID | DE year OOD |
|---|---|---|---|---|---|---|---|---|
| Galileo∗ | 49.47 ± 1.05 | 17.73 ± 1.28 | 53.12 ± 0.50 | 33.75 ± 0.43 | 44.83 ± 1.45 | 42.28 ± 2.42 | 50.39 ± 0.63 | 52.83 ± 0.56 |
| Clay∗ | 54.47 ± 0.02 | 20.81 ± 0.05 | 54.75 ± 0.08 | 39.23 ± 0.11 | 50.65 ± 0.23 | 48.88 ± 0.22 | 53.60 ± 0.11 | 54.62 ± 0.03 |
| ResNet50-ImageNet∗ | 46.96 ± 0.90 | 15.17 ± 2.63 | 47.64 ± 0.45 | 32.91 ± 0.47 | 40.21 ± 0.84 | 35.87 ± 3.19 | 45.58 ± 1.47 | 47.58 ± 0.29 |
| ResNet50-random∗ | 41.49 ± 1.72 | 19.40 ± 0.97 | 42.23 ± 3.06 | 27.84 ± 2.23 | 39.16 ± 0.51 | 37.99 ± 1.18 | 41.87 ± 2.68 | 44.23 ± 1.84 |
| Galileo (full) | 51.63 ± 0.82 | 17.21 ± 0.97 | 53.75 ± 0.43 | 33.77 ± 1.74 | 47.30 ± 3.17 | 45.71 ± 3.73 | 52.35 ± 0.43 | 54.18 ± 0.27 |
| Clay (full) | 54.12 ± 0.18 | 15.88 ± 0.78 | 55.33 ± 0.13 | 40.71 ± 0.64 | 51.48 ± 0.49 | 48.68 ± 1.24 | 55.56 ± 0.21 | 54.91 ± 0.36 |

What these show:

- **Location shift is the real test.** Every model loses 30–40 points from Germany to Cambodia and 12–20 to
  Denmark.
- **The temporal tasks barely move:** Δ is 0–7 points, and the Germany OOD set often scores *higher* than ID.
  The paper's fitted slope is about 1 ("near-perfect robustness", paper p.8), and it calls this "a flag for
  including additional temporal shift samples for future work" (paper p.8).
- **On the far shift (Cambodia), foundation models beat a randomly initialised ResNet-50 by only a few
  points** (frozen: Clay 20.8, Galileo 17.7, ResNet-random 19.4). That makes it a useful floor.

Chosen learning rates for Galileo (paper Tables 17 and 23): location 3e-3 frozen / 6e-3 full; temporal 5e-3
frozen / 3e-3 full.

---

## 4. Out of scope

- **Sensor, spatial-scale and data-source shifts** (paper pp.5–6): different tasks, not field delineation.
- **The FTW-All season task** for now (2.3c).
- **Code-only FTW pairs** without paper numbers (2.1).
- **FTW's own benchmark protocol** (2-window stacked input, object-level metrics, all-country training). It
  could be a separate AgriBench entry later.

## 5. Decisions

All agreed on 2026-11-09. The rule is to follow the paper's conditions unless there's a good reason not to.

1. **Scope = the paper's four tasks** (DE→KH, DE→DK, ZA A→B, DE 2018→2019). FTW-All is left out: it has no
   mIoU in the paper and a random chip split that leaks.
2. **Data handling:** the raw zips stay on Lustre unchanged. A packed per-country input cache (about 1.5 GB,
   4 files) is built once from them, and encoder features are cached once per encoder (section 0). Nothing
   is unpacked per run.
3. **No leaky splits.** AgriBench uses the leak-free splits in 2.1b for all four tasks. The paper's numbers
   stay as rough sanity checks only.
4. **Headline metric = the paper's** (3-class mIoU at the resized input size, ID / OOD / Δ / ρ with the
   paper's fixed lines). Native-resolution and per-class / boundary IoU are secondary columns (2.4).
5. **No object-level metrics**, as in the paper.
6. **The paper's two training settings exactly** (frozen backbone and full fine-tuning, 2.2–2.3), following
   the paper's text where its code disagrees: the input adapter is trained, and Galileo runs its full
   encoder (2.5). Per-pixel tabular heads are an optional extra.
7. **Five seeds, mean ± std**, as in the paper, fixed to 0–4 so runs can be repeated. Test-set runs go in the
   ledger (2.6–2.7).
