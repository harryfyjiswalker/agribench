# Datasets and Evaluation Protocols

## Setup from scratch

1. Login to Annuna
```
ssh USERNAME@login.anunna.wur.nl
```

2. Clone AgriBench and set paths.
```
mkdir -p "$HOME/projects"
git clone https://github.com/harryfyjiswalker/agribench.git "$HOME/projects/agribench"
cd "$HOME/projects/agribench"

export work="${AGRIBENCH_WORK:-/lustre/nobackup/WUR/AIN/$USER/agribench}"
export bench="$work/venv/bin/python"
```

3. Create environment
```
mkdir -p "$work"

if command -v python3.12 >/dev/null; then
  python3.12 -m venv "$work/venv"
else
  /lustre/shared/agriscience-benchmark/envs/benchmark-v1/bin/python -m venv "$work/venv"
fi
```

## Returning to Anunna

1. Log in, and run:

```
cd "$HOME/projects/agribench"

export work="${AGRIBENCH_WORK:-/lustre/nobackup/WUR/AIN/$USER/agribench}"
export bench="$work/venv/bin/python"

"$bench" --version
```

## Crop Type Classification

### Datasets

- [PASTIS-R](https://zenodo.org/records/5745151) (Garnot and Landrieu, 2021)
  - A benchmark that supports parcel-based crop type classification, comprising 124,422 individual parcels and 18 crop types in metropolitan France. The benchmark covers one year, the September 2018 to November 2019 growing season. PASTIS can be used for in-distribution scoring, using the pre-determined five folds provided by the dataset maintainers. 
- [PLANTEUR](https://github.com/allenai/planteur-benchmark) (Wolters _et al._, 2026)
  - A complementary benchmark comprising the same content but for eight of France's overseas territories, enabling analysis of smallholder and tropical crop type mapping performance and supporting analysis of geographic generalisation. 

### Evaluation 

Cross-validation: 
- In-domain performance: PASTIS-R (5-fold CV, using pre-determined folds):
  - Frozen Feature Set/ Embedding: Uses [Geo Eval Harness](https://github.com/torchgeo/geo-eval-harness) configuration, with an L2-regularised logistic probe trained on folds 1-3 with fold 4 as validation and fold 5 as testing.
    
        ```
        python -m pip install \
          'geo-eval-harness @ git+https://github.com/torchgeo/geo-eval-harness.git@a782d730ab4dbdb9f9c0ae05ac7fe893e739be03'
        ```
         ```
        export metadata=/absolute/path/to/PASTIS-R_PixelSet/metadata_parcel.csv
        export features=/absolute/path/to/parcel-features.npz
        export recipe=/absolute/path/to/representation.json
        export out="$PWD/runs/my-representation"
        sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=02:00:00 \
          --wrap='python tools/evaluate_parcel_probe.py --metadata "$metadata" --features "$features" --recipe "$recipe" --out "$out"'
         ```
This uses harness defaults of 40 log-spaced regularisation values in range [1e-6, 1e4]; 200 parcel bootstrap resampling for 95% confidence interval.

- Out-of-distribution: PLATEUR (leave-one-region-out CV)

Evaluation:
- Frozen: 
For evaluation on frozen features, 

### Limitations

- Currently lacking capacity for understanding of temporal generalisation (both are 2018-2019 growing seasons)

## Crop Yield Prediction

### Datasets

- [YieldSAT](https://yieldsat.github.io) (Miranda _et al._, 2026)




end of season?






### PLANTEUR
