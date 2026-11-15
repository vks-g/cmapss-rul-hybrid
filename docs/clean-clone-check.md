# Phase 2 clean-clone reproducibility check

Issue [#82](https://github.com/vks-g/cmapss-rul-hybrid/issues/82), checked by
`affan80` on **15 November 2026**. A new remote clone of `dev` at
`e27db92731800b32ad8a1b7bc43643d10c3590e0` and a new virtual environment were used.
No data, checkpoints, environment or processed artifacts were copied from another
checkout.

**Result:** installation, fresh download, the full test suite, an unchanged LSTM
CV/refit run, checkpoint reload, notebook output checks and repository hygiene
checks passed on the platform below. The current `dev` README still describes
Phase 2 as planned and lacks its training commands; issue
[#77](https://github.com/vks-g/cmapss-rul-hybrid/issues/77) and
[PR #121](https://github.com/vks-g/cmapss-rul-hybrid/pull/121) track that documentation
gap. Merge that update before the final release check in
[#83](https://github.com/vks-g/cmapss-rul-hybrid/issues/83).

> **Update, 15 November 2026:** PR #121 merged the Phase 2 README and diagrams
> into `dev` after this check, so the gap above is closed. The rest of this record
> describes `e27db92` as checked.

## Environment

| Component | Tested version |
|---|---|
| OS / architecture | macOS 15.5 / arm64 |
| Python | 3.11.15, fresh `.venv`, system site packages disabled |
| PyTorch | 2.15.0, CPU execution |
| NumPy / pandas | 2.4.6 / 3.1.1 |
| scikit-learn / XGBoost | 1.9.1 / 3.2.0 |
| pytest / nbconvert | 9.2.0 / 7.17.2 |

Dependencies were installed from `requirements.txt`, followed by `pip install -e .`.
`pip check` reported **No broken requirements found**. The project uses minimum
versions rather than a lockfile; this check establishes that this environment
runs successfully. Other operating systems and Python versions were not tested
in this check.

## Commands and results

Run from a new checkout, following [Setup & reproduce](../README.md#setup--reproduce).
Python 3.11 was selected from the README's supported Python 3.10+ range:

```sh
git clone --branch dev https://github.com/vks-g/cmapss-rul-hybrid.git
cd cmapss-rul-hybrid
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -c "import torch, rul"
python scripts/download_data.py
pytest -q -ra
pip check
```

- Imports resolve to the package in the new checkout.
- First download: **PASS**, all 12 files fetched and extracted into the previously
  empty `data/raw/CMAPSSData/`.
- Second `python scripts/download_data.py`: **PASS**, complete local data detected
  and the download skipped.
- Tests: **304 passed in 42.55s**, with no skipped tests.
- All four subsets load with 26 train/test columns and one RUL label per test engine:

| Subset | Training cycles / engines | Test cycles / engines | RUL labels |
|---|---:|---:|---:|
| FD001 | 20,631 / 100 | 13,096 / 100 | 100 |
| FD002 | 53,759 / 260 | 33,991 / 259 | 259 |
| FD003 | 24,720 / 100 | 16,596 / 100 | 100 |
| FD004 | 61,249 / 249 | 41,214 / 248 | 248 |

### Phase 2 training, refit and saved outputs

The training command below supplements the tested `dev` README's missing Phase 2
commands. It uses the original `configs/lstm.yaml` and inherited defaults, with
no model, seed, epoch-limit or data-size overrides. Output directories are moved
to gitignored `build/` so committed tuning records remain unchanged.

```sh
python scripts/train_dl.py --config configs/lstm.yaml --refit \
  --metrics-dir build/clean-clone/metrics \
  --models-dir build/clean-clone/models
```

Observed output:

```text
lstm: selection RMSE 15.78 ± 0.44, all-cycle RMSE 48.07, best epochs [11, 20, 20], 55 s
saved build/clean-clone/metrics/fd001_cv_lstm.json
refit in 23 s (networks: 20 epochs), saved build/clean-clone/models/fd001_lstm_seed42.pt
```

Checks on the generated artifacts:

- **PASS:** JSON top-level keys, fold keys and score keys match
  [`fd001_cv_lstm.json`](../results/metrics/fd001_cv_lstm.json), and the saved config
  equals the committed config record. These are training-engine CV scores.
- **PASS:** CSV columns match
  [`fd001_cv_lstm_predictions.csv`](../results/metrics/fd001_cv_lstm_predictions.csv):
  `unit, cycle, rul, fold, prediction`. It contains 20,631 unique engine/cycle rows,
  100 engines, three folds, and finite predictions. Each engine belongs to one
  held-out fold, disjoint from that fold's training and early-stopping engines.
- **PASS:** recomputing each fold's selection RMSE from its saved CSV predictions
  with true RUL ≤ 130 agrees with the JSON.
- **PASS:** `TrainedModel.load` reloads the new checkpoint, records all 100 training
  engines and 20 refit epochs, and `predict` returns 192 finite predictions for
  training engine 1.

The official test was not rescored. Files under `src/`, `configs/`, `notebooks/`
and `results/` remained unchanged. Training outputs and raw data stay local and
are not included in this documentation PR.

### Phase 2 notebooks

All notebook files were read and validated with `nbformat`, then exported without
executing their code:

```sh
jupyter nbconvert --to html --output-dir build/clean-clone/notebooks \
  notebooks/phase2/*.ipynb
```

**PASS:** all 11 HTML exports were produced, all 80 code cells have execution
counts, 79 code cells have saved outputs, and none has an error output. The one
output-free cell defines functions in notebook 09. There are 20 embedded figure
outputs across these notebooks. No notebook was edited or re-executed.

| Notebook | Executed code cells | Cells with saved outputs |
|---|---:|---:|
| 05 RUL cap | 7 | 7 |
| 06 LSTM | 7 | 7 |
| 07 TCN | 8 | 8 |
| 08 Transformer | 8 | 8 |
| 09 Hybrid | 9 | 8 |
| 10 Regularization | 6 | 6 |
| 11 Ablations | 7 | 7 |
| 12 Multi-regime | 9 | 9 |
| 13 Validation | 5 | 5 |
| 14 Interpretability | 6 | 6 |
| 15 Final comparison | 8 | 8 |

The exporter emitted missing-image-alt-text warnings for all 11 notebooks, but
completed every conversion. This is an accessibility improvement for the notebook
owners to consider; it did not prevent reading or rendering the committed outputs.

### Paths, links and repository hygiene

The scan covered all 297 tracked files at the checked commit. Markdown links,
HTML image/link attributes, and notebook Markdown cells were checked for relative
path existence; Markdown heading anchors were checked where linked.

- **PASS:** all 55 relative links at the checked commit resolve.
- **PASS:** no developer-specific absolute paths in executable source, config or
  notebook source cells. Historical output text in Phase 1 notebook 03 and Phase 2
  notebooks 05 and 11 contains local paths printed by earlier runs. Those strings
  are display output, not inputs to execution. The notebooks were left unchanged
  under the ownership rules in [CONTRIBUTING.md](../CONTRIBUTING.md).
- **PASS:** no tracked file is 10 MB or larger. The largest is the intentional
  compressed regularization prediction table, 6,577,389 bytes. The Phase 2 report
  is 1,459,089 bytes and the deck is 642,407 bytes.
- **PASS:** no raw/processed data, virtual environment, build output or model
  checkpoint is tracked. `git status --short` was empty after all runtime checks,
  before this report and its README link were added.

The frozen experiment files were preserved. Remaining release integration work
is tracked in #83; the pending front-page update is covered by #77 / PR #121.
