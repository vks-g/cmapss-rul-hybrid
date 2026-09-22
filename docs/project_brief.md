# Project 7: Remaining Useful Life Prediction for Industrial Engines

**Title:** Predictive Maintenance via Sensor Features and Sequence Models

## Project Overview & Abstract

Estimate the remaining operating cycles before an engine fails from multivariate sensor readings. The key comparison is between engineered degradation features and a model that learns directly from sensor sequences.

### The "Hybrid" Logic: Degradation Feature Engineering to Sequence Based Remaining Life Prediction

**AML baseline:** Create cycle, rolling mean, rolling slope, operating regime and sensor health features; train Random Forest, XGBoost or Elastic Net regression.

**Identify the gap:** Fixed rolling features may miss long degradation trajectories and interactions whose timing differs across engines.

**DL model:** Train an LSTM, temporal CNN or Transformer encoder on fixed length sensor windows to predict RUL.

**Fair comparison:** Split by engine ID, not individual rows. Report RMSE and MAE on RUL, plus error by early versus late degradation stage.

## Suggested Pre-reads

- Saxena and Goebel (2008). *Turbofan Engine Degradation Simulation Data Set.*
- Hochreiter and Schmidhuber (1997). *Long Short Term Memory.*

## Datasets & Stack

**Recommended starter dataset:** NASA C-MAPSS Turbofan Engine Degradation Dataset, starting with subset FD001.

**Dataset Link:** [NASA C-MAPSS Turbofan Engine Degradation](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulated-data)

**Suggested stack:** Python, pandas, scikit-learn, PyTorch or TensorFlow, matplotlib and optional SHAP.

## Feature Engineering & Setup Tips

Compute RUL only from the final cycle of each training engine, cap the target only if stated, normalize sensors by training operating regime, remove flat sensors after inspection, and make windows within engine boundaries.

## Expected Deliverable

A predictive maintenance notebook comparing engineered feature regression and sequence learning with engine level test results.
