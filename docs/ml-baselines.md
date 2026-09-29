# FD001 ML baseline reproduction

This run tunes Random Forest, XGBoost, and Elastic Net on the FD001 training
engines. It uses cycle, three operating settings, and 21 raw sensors. A seeded
engine-level split reserves 20% of engines for internal validation. Three
`GroupKFold` folds tune each model on the other 80%; the shared metrics score
the held-out engines after tuning.

These validation scores cover every held-out **cycle**. They are not the
official C-MAPSS test-set scores, which use the last cycle of each test engine.
Use the official test protocol when comparing against published results.

## Run

Install the project with `pip install -r requirements.txt && pip install -e .`.
Place NASA C-MAPSS `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`
under `data/raw/CMAPSSData/`. Raw data is not committed.

Run this from the repository root:

```python
from rul.data.labels import add_train_rul
from rul.data.load import SENSOR_COLS, SETTING_COLS, load_subset
from rul.models.ml import run_baselines, save_best_parameters

data = add_train_rul(load_subset("FD001").train)
features = ["cycle", *SETTING_COLS, *SENSOR_COLS]
report = run_baselines(data, features, validation_size=0.2, n_splits=3, seed=42)
save_best_parameters(report, "configs/ml_baseline.yaml", subset="FD001", feature_columns=features)
print({name: result["holdout"] for name, result in report["models"].items()})
```

The selected parameters and run settings are in [`configs/ml_baseline.yaml`](../configs/ml_baseline.yaml).
