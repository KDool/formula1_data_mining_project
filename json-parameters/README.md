# json-parameters

Contains JSON parameter files for configuring experiments, model hyperparameters, dataset splits, and other settings used by training and evaluation scripts.

## Structure

```text
json-parameters/
├── random-forest/
│   ├── random_forest_params.json
│   ├── random_forest_best_params.json
│   ├── random_forest_optimization_params.json
│   └── random_forest_best_optimization_params.json
├── svm/
│   ├── svm_optimization_params.json
│   └── svm_best_optimization_params.json
├── gradient-boosting/
│   ├── gradient_boosting_optimization_params.json
│   └── gradient_boosting_best_optimization_params.json
└── ensemble/
    ├── ensemble_optimization_params.json
    └── ensemble_optimization_results.json
```

Each algorithm folder keeps both the input search configuration and the saved
best-result JSON for that algorithm. Ensemble configuration points to the best
result files in the model-specific folders.
