"""ML baseline behavior on small, engine-grouped datasets."""

from sklearn.pipeline import Pipeline

from rul.models.ml import make_models


def test_model_wrappers_use_fixed_seed_and_scale_elastic_net():
    models = make_models(seed=17)

    assert set(models) == {"random_forest", "xgboost", "elastic_net"}
    assert models["random_forest"].random_state == 17
    assert models["xgboost"].random_state == 17
    assert isinstance(models["elastic_net"], Pipeline)
    assert models["elastic_net"].named_steps["model"].random_state == 17
    assert "scaler" in models["elastic_net"].named_steps
