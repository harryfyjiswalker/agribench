# PASTIS-R parcel pilot

Run from this directory on Anunna. Reuses the tested PASTIS-R loader, PSE, LTAE2d and parcel mIoU. Hydra composes dataset, model, validation and evaluation settings. Slurm launches the plain PyTorch loop.

Install Hydra once as environment maintainer:

```bash
bench=/lustre/shared/agriscience-benchmark/envs/benchmark-v1/bin/python
"$bench" -m pip install 'hydra-core==1.3.2'
"$bench" -m pip check
"$bench" run_experiments.py --cfg job --resolve
```

Submit a quick check and inspect its log before submitting one full epoch:

```bash
sbatch --time=00:15:00 train.sbatch experiment.max_batches=8
sbatch train.sbatch
```

Slurm writes the terminal log to slurm-JOBID.out. Run outputs are under the shared runs/pastis directory, grouped by user and job. Override hydra.run.dir if needed. The runner writes the resolved settings, upstream commit IDs and tracked patches, installed packages, epoch metrics, and the best validation checkpoint and parcel predictions. Prediction class indices are zero based. The test fold is reserved and is never loaded.

Expected upstream checkouts under ~/projects:

- pastis-benchmark at 43a0eceb9f40a6e6b0863b1a591407877812efcc, with dates.iteritems() changed to dates.items() in code/dataloader_pixelset.py
- ltae (lightweight-temporal-attention-pytorch) at d8f99bd5bc9746d2df25351f0fe156b36af35c11
- utae-paps at 987874e27a98bb399277f2d55eb010744d26f881

Override upstream_root=/path/to/checkouts and dataset.root=/path/to/PASTIS-R_PixelSet as needed. Colleagues need read access to the shared environment and dataset, and write access to their run directories. A dataset in Harry's personal Lustre directory may need an accessible shared path. Each user can keep their own upstream checkouts. Use BENCHMARK_PYTHON for a different interpreter.

This is the adapted baseline tested in the earlier smoke job, with official fold membership and a pilot rotation. Train on folds 1–3, validate on 4, reserve 5. The loader supplies training-fold normalization for both splits. Batches have equal acquisition counts and retain individual dates. Training merges a one-parcel remainder into its preceding batch and skips buckets containing only one parcel because of BatchNorm. Metrics record the actual train count. Validation keeps all parcels unless max_batches is explicitly set. Full real-data runtime on Anunna remains to be measured.

## Standardised tabular comparison

Use the same Hydra runner from this directory. These jobs score validation only. Fold 5 stays reserved, and the feature cache reads only folds 1–4 for the supplied split. The existing GPU command still runs the sequence model.

```bash
# Full-data Random Forest on S2, then the sensor ablation.
sbatch cpu.sbatch
sbatch cpu.sbatch inputs=s2_s1

# Compare estimators on identical training IDs and S2 features.
sbatch cpu.sbatch validation=matched
sbatch train.sbatch model=tabpfn validation=matched \
  model.estimator.model_path=/absolute/path/to/classifier.ckpt
```

Run the first RF job before starting the others. It builds the S2 cache under the shared project. The multimodal job reuses S2 and builds one table per radar orbit. For a short fit check, use `experiment.train_limit=500`; the first invocation still builds the full train/validation feature cache so subsequent runs can reuse it. The matched config uses a stratified 5,000-row training subset with the same seed for both classifiers. Compare the saved training IDs and feature table hashes across the two runs. Both use the same training-only median imputation and the common scorer. The CPU and GPU launchers request different resources, independently of the split.

The TabPFN command needs an installed TabPFN release and an accessible classifier checkpoint supporting 18 classes and the selected table size. Choose these together before installing into a versioned environment. The runner records installed packages and the checkpoint SHA-256. This package does not select or download a TabPFN checkpoint.

The tabular features are our baseline, separate from the authors' sequence protocol. For each sensor, pool every stored parcel pixel by median at each acquisition, then take per-band medians in 30-day bins over the loader's day-offset range. Empty bins stay missing in the cache. Both estimators get the same imputed table. Each run saves actual training IDs, parcel predictions, feature hashes and config. Treat the extracted dataset as immutable while using the cache, or use a fresh cache directory after changing the raw arrays. The first cache build reads tens of thousands of files per sensor. Do not submit multiple builds of the same table simultaneously.

Keep the same data path, cache path and configuration for shared comparisons. Ensure collaborators can read the data and cache and write to their own run directory.

## Team repository and leaderboard

Keep code, configs and small result summaries in a private GitHub repository. Keep the dataset, environments, feature caches, predictions and checkpoints on Anunna. Each collaborator clones the code and uses the shared paths documented above.

After a completed Slurm validation run, use the shared benchmark Python to export a small result JSON. Pass the run directory containing resolved.yaml, metrics.json and validation-predictions.csv, and a unique descriptive name.

```bash
bench=/lustre/shared/agriscience-benchmark/envs/benchmark-v1/bin/python
"$bench" tools/export_result.py /absolute/path/to/run s2-rf-full-seed42
git add results/s2-rf-full-seed42.json
git commit -m "Record S2 Random Forest validation result"
git push
```

When the summary reaches main, GitHub Actions regenerates [LEADERBOARD.md](LEADERBOARD.md). If the team uses pull requests, the update runs after merge. The workflow needs permission to push its generated page to main. If branch rules prevent bot pushes, regenerate it locally with `python3 tools/render_leaderboard.py` and include the page in the same pull request. The exporter leaves raw parcel IDs, private paths, checkpoints and dependencies on the cluster.
