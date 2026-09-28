# Rolling degradation features

`rul.features.rolling.add_rolling_features` adds trailing features for selected
sensors. It sorts rows by `unit` and `cycle` for calculation, computes each
engine separately, then restores the input row and index order. The original
`cycle` column remains available to the model.

For each sensor `s` and window `w`:

| Column | Definition |
|---|---|
| `s_mean_w` | Mean of the current and previous `w - 1` readings. |
| `s_slope_w` | Least-squares slope per cycle over the same readings; zero when only one cycle exists. |
| `s_health_w` | `s_mean_w` minus the sensor value at that engine's first observed cycle. Signed change, not a calibrated health score. |

At the start of an engine history, the window uses the cycles available so
far (`min_periods=1`). No feature reads later cycles or rows from another
engine. Window sizes are distinct positive integers; the default is
`(5, 10, 20)`.

```python
from rul.features.rolling import add_rolling_features

train_features = add_rolling_features(train_scaled, sensors=kept_sensors)
valid_features = add_rolling_features(valid_scaled, sensors=kept_sensors)
test_features = add_rolling_features(test_scaled, sensors=kept_sensors)
```

Select `kept_sensors` on raw training-fold rows before fitting any regime
normalizer. Fit the normalizer on that same training fold and transform the
other splits with it. Pass the same sensor list and window sizes to every
split. The feature calculation itself has no fitted state.
