"""Tests for scripts/prepare_ablations_table.py (#75)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_ablations_table.py"


def load():
    spec = importlib.util.spec_from_file_location("prepare_ablations_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_table_matches_the_result_csvs():
    module = load()
    assert module.OUT.read_text() == module.build()


def test_numbers_round_halves_away_from_zero_and_keep_math_signs():
    module = load()
    assert module.num(0.795, sign=True) == "$+$0.80"
    assert module.num(-0.5335, sign=True) == "$-$0.53"
    assert module.num(16.6079) == "16.61"
    assert module.pvalue(0.0004) == "$<$0.001" and module.pvalue(0.0507) == "0.051"
