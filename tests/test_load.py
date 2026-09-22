"""Tests for rul.data.load."""

from pathlib import Path

import pytest

from rul.data.load import load_subset

# First row of train_FD001.txt after the unit and cycle columns, including the
# trailing double space the raw files end every line with.
RAW_TAIL = (
    "-0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 "
    "2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 "
    "100.00 39.06 23.4190  "
)
EXPECTED_COLUMNS = ["unit", "cycle", "op_1", "op_2", "op_3"] + [f"s_{i}" for i in range(1, 22)]


def write_subset(folder: Path, train_cycles: dict, test_cycles: dict, rul: list, name: str = "FD001") -> Path:
    """Write a tiny fake subset: {unit: n_cycles} per split, plus a RUL file."""

    def lines(cycles: dict) -> str:
        return "".join(f"{u} {c} {RAW_TAIL}\n" for u, n in cycles.items() for c in range(1, n + 1))

    (folder / f"train_{name}.txt").write_text(lines(train_cycles))
    (folder / f"test_{name}.txt").write_text(lines(test_cycles))
    # Real RUL files also end each line with a trailing space.
    (folder / f"RUL_{name}.txt").write_text("".join(f"{r} \n" for r in rul))
    return folder


def test_cycles_are_parsed_into_named_columns(tmp_path):
    write_subset(tmp_path, train_cycles={1: 3, 2: 2}, test_cycles={1: 2, 2: 1}, rul=[112, 98])

    train, test, _ = load_subset("FD001", data_dir=tmp_path)

    assert list(train.columns) == EXPECTED_COLUMNS
    assert list(test.columns) == EXPECTED_COLUMNS
    assert train["unit"].tolist() == [1, 1, 1, 2, 2]
    assert train["cycle"].tolist() == [1, 2, 3, 1, 2]
    assert train["unit"].dtype == "int64" and train["cycle"].dtype == "int64"
    assert train.loc[0, "op_3"] == 100.0
    assert train.loc[0, "s_1"] == 518.67
    assert train.loc[0, "s_21"] == pytest.approx(23.419)
    assert len(test) == 3


def test_rul_targets_are_indexed_by_test_unit(tmp_path):
    write_subset(tmp_path, train_cycles={1: 3, 2: 2}, test_cycles={1: 2, 2: 1}, rul=[112, 98])

    _, _, rul_test = load_subset("FD001", data_dir=tmp_path)

    assert rul_test.to_dict() == {1: 112, 2: 98}
    assert rul_test.index.name == "unit"
