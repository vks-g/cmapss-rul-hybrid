"""Reproducible tabular baselines for engine-level RUL prediction."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


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
