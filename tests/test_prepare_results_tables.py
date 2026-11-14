"""Tests for scripts/prepare_results_tables.py (#81)."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_results_tables.py"


def load():
    spec = importlib.util.spec_from_file_location("prepare_results_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["hybrid_value_cv.tex", "final_results.tex"])
def test_committed_tables_match_the_results(name):
    module = load()
    assert (module.TABLES / name).read_text() == module.BUILDERS[name]()


def test_every_capped_model_has_a_cross_validation_score():
    module = load()
    assert set(module.test_scores()) == set(module.cv_selection_rmse())


def test_test_scores_repeat_the_final_comparison():
    module = load()
    scores = module.test_scores()
    assert round(scores["Hybrid B (two-stage)"]["rmse"][0], 2) == 15.18
    assert round(scores["Hybrid B (two-stage)"]["nasa"][0]) == 356
    assert round(scores["Random Forest (engineered)"]["rmse"][0], 2) == 16.21  # the Phase 1 capped result
    assert scores["XGBoost (engineered)"]["rmse"][1] is None  # one run, no seed spread
