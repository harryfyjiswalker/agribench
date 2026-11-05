# Datasets and Evaluation Protocols

## Crop Type Classification

### Datasets

- [PASTIS-R](https://zenodo.org/records/5745151) (Garnot and Landrieu, 2021) is a benchmark that supports parcel-based crop type classification, comprising 124,422 individual parcels and 18 crop types in metropolitan France. The benchmark covers one year, the September 2018 to November 2019 growing season. PASTIS can be used for in-distribution scoring, using the pre-determined five folds provided by the dataset maintainers. 
- [PLANTEUR](https://github.com/allenai/planteur-benchmark) (Wolters _et al._, 2026) is a complementary benchmark comprising the same content but for eight of France's overseas territories, enabling analysis of smallholder and tropical crop type mapping performance and supporting analysis of geographic generalisation. 

### Evaluation 

Cross-validation: 
- In-domain performance: PASTIS-R (5-fold CV, using pre-determined folds):
  - Frozen: Uses [Geo Eval Harness](https://github.com/torchgeo/geo-eval-harness) configuration, with an L2-regularised logistic probe trained on folds 1-3 with fold 4 as validation and fold 5 as testing.
 



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
