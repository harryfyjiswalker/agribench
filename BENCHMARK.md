# Benchmark protocol

The adapted PSE + L-TAE baseline uses official fold membership and a pilot rotation. Train on folds 1–3, validate on 4, reserve 5. The loader supplies training-fold normalization for both splits. Batches have equal acquisition counts and retain individual dates. Training merges a one-parcel remainder into its preceding batch and skips buckets containing only one parcel because of BatchNorm. Metrics record the actual train count. Validation keeps all parcels unless max_batches is explicitly set.

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
