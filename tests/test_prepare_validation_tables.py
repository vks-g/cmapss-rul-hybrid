"""Tests for scripts/prepare_validation_tables.py (#76)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_validation_tables.py"


def load():
    spec = importlib.util.spec_from_file_location("prepare_validation_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["validation_models.tex", "validation_paired.tex", "multi_regime_test.tex"])
def test_committed_tables_match_the_results(name):
    module = load()
    assert (module.TABLES / name).read_text() == module.BUILDERS[name]()


def test_formatting_rounds_halves_away_from_zero_and_keeps_math_signs():
    module = load()
    assert module.num(-0.545, sign=True) == "$-$0.55"
    assert module.num(15.945) == "15.95"
    assert module.pm(15.4, 0.459) == r"15.40 $\pm$ 0.46"
    assert module.pm(15.71, float("nan")) == "15.71"
    assert module.pvalue(1.0) == "1.00" and module.pvalue(0.0004) == "$<$0.001"
    assert module.latex_name("XGBoost (engineered)") == "XGBoost, engineered"
