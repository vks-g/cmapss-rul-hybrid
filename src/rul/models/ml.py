"""Reproducible tabular baselines for engine-level RUL prediction."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from rul.data.split import group_kfold_splits


DEFAULT_PARAM_GRIDS = {
    "random_forest": {"n_estimators": [100, 200], "min_samples_leaf": [1, 3]},
    "xgboost": {"n_estimators": [100, 200], "max_depth": [3, 5]},
    "elastic_net": {"model__alpha": [0.1, 1.0], "model__l1_ratio": [0.2, 0.8]},
}


def make_models(seed: int = 42) -> dict:
    """Create unfitted regressors with a shared random seed."""
    return {
        "random_forest": RandomForestRegressor(random_state=seed, n_jobs=1),
        "xgboost": XGBRegressor(
            objective="reg:squarederror", random_state=seed, n_jobs=1,
            tree_method="hist",
        ),
        "elastic_net": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(random_state=seed)),
        ]),
    }


def tune_models(
    data,
    feature_columns,
    *,
    models=None,
    param_grids=None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, GridSearchCV]:
    """Fit each model using folds that never split an engine across sets."""
    if not feature_columns or {"unit", "rul"}.intersection(feature_columns):
        raise ValueError("feature_columns must contain predictors only")
    models = make_models(seed) if models is None else models
    param_grids = DEFAULT_PARAM_GRIDS if param_grids is None else param_grids
    folds = list(group_kfold_splits(data, n_splits=n_splits, seed=seed))
    searches = {}
    for name, model in models.items():
        search = GridSearchCV(
            model, param_grids[name], cv=folds,
            scoring="neg_root_mean_squared_error", n_jobs=1, refit=True,
        )
        search.fit(data.loc[:, feature_columns], data["rul"])
        searches[name] = search
    return searches
