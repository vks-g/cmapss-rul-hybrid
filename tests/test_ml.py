"""ML baseline behavior on small, engine-grouped datasets."""

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import rul.models.ml as ml
from rul.models.ml import make_models, run_baselines, save_best_parameters, tune_models


def sample_engines() -> pd.DataFrame:
    return pd.DataFrame({
        "unit": [unit for unit in range(1, 7) for _ in range(3)],
        "sensor": [cycle + unit for unit in range(1, 7) for cycle in range(3)],
        "rul": [2 - cycle for _ in range(1, 7) for cycle in range(3)],
    })


def test_model_wrappers_use_fixed_seed_and_scale_elastic_net():
    models = make_models(seed=17)

    assert set(models) == {"random_forest", "xgboost", "elastic_net"}
    assert models["random_forest"].random_state == 17
    assert models["xgboost"].random_state == 17
    assert isinstance(models["elastic_net"], Pipeline)
    assert models["elastic_net"].named_steps["model"].random_state == 17
    assert "scaler" in models["elastic_net"].named_steps


def test_tuning_uses_engine_grouped_folds_and_best_parameters():
    data = sample_engines()
    searches = tune_models(
        data, ["sensor"],
        models={"random_forest": RandomForestRegressor(random_state=9)},
        param_grids={"random_forest": {"n_estimators": [2, 3]}},
        n_splits=3, seed=9,
    )

    search = searches["random_forest"]
    assert search.best_params_["n_estimators"] in {2, 3}
    assert search.scoring == "neg_root_mean_squared_error"
    assert len(search.cv) == 3
    for train_idx, valid_idx in search.cv:
        train_units = set(data.iloc[train_idx]["unit"])
        valid_units = set(data.iloc[valid_idx]["unit"])
        assert train_units.isdisjoint(valid_units)


def test_run_holds_out_engines_before_tuning_and_uses_shared_metrics(monkeypatch):
    data = sample_engines()
    seen_units = []
    real_tune = ml.tune_models

    def capture_training_data(training_data, *args, **kwargs):
        seen_units.extend(training_data["unit"].unique())
        return real_tune(training_data, *args, **kwargs)

    monkeypatch.setattr(ml, "tune_models", capture_training_data)
    report = run_baselines(
        data, ["sensor"],
        models={"random_forest": RandomForestRegressor(random_state=9)},
        param_grids={"random_forest": {"n_estimators": [2]}},
        validation_size=2, n_splits=2, seed=9,
    )

    assert set(seen_units) == set(report["split"]["train_units"])
    assert set(seen_units).isdisjoint(report["split"]["validation_units"])
    assert len(report["split"]["validation_units"]) == 2
    assert report["models"]["random_forest"]["holdout"]["n"] == 6
    assert report["models"]["random_forest"]["best_params"] == {"n_estimators": 2}


def test_best_parameter_config_records_run_context(tmp_path):
    report = {"seed": 9, "models": {
        "random_forest": {"best_params": {"n_estimators": 2}, "cv_rmse": 4.2}
    }}
    path = tmp_path / "ml_baseline.yaml"

    save_best_parameters(report, path, subset="FD001", feature_columns=["sensor"])

    saved = yaml.safe_load(path.read_text())
    assert saved["subset"] == "FD001"
    assert saved["seed"] == 9
    assert saved["feature_columns"] == ["sensor"]
    assert saved["models"]["random_forest"]["best_params"] == {"n_estimators": 2}
