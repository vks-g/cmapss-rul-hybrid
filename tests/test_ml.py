"""ML baseline behavior on small, engine-grouped datasets."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from rul.models.ml import make_models, tune_models


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
