"""Engine-aware preprocessing is fitted independently by each model pipeline."""

import numpy as np
import pandas as pd

from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.models.experiment import EngineFeatures


def sample_cycles(offset=0.0):
    data = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2]})
    for name in SETTING_COLS:
        data[name] = 0.0
    for name in SENSOR_COLS:
        data[name] = 1.0
    data["s_2"] = np.array([1, 2, 3, 4]) + offset
    data["rul"] = [1, 0, 1, 0]
    return data


def test_engineered_transform_uses_training_statistics_and_sensor_selection():
    train = sample_cycles()
    validation = sample_cycles(100)
    validation["s_3"] = [1, 2, 3, 4]
    transformer = EngineFeatures(engineered=True, windows=(2,)).fit(train)

    transformed = transformer.transform(validation)

    assert "s_3" not in transformed
    assert transformed["s_2"].min() > 50
    assert {"unit", "rul"}.isdisjoint(transformed.columns)
    assert (transformed.loc[[0, 2], "s_2_health_2"] == 0).all()
    assert list(transformer.get_feature_names_out()) == list(transformed.columns)


def test_raw_transform_preserves_sensor_values_without_metadata_or_target():
    data = sample_cycles()
    result = EngineFeatures(engineered=False).fit_transform(data)

    assert result["s_2"].tolist() == data["s_2"].tolist()
    assert list(result.columns) == ["cycle", *SETTING_COLS, *SENSOR_COLS]
