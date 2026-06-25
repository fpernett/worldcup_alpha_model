from __future__ import annotations

import pandas as pd

from src.backtesting import PREDICTION_LOG_COLUMNS, evaluate_predictions, load_prediction_log
from src.prediction_log_repair import inspect_prediction_log_schema, repair_prediction_log_by_quarantine


def _write_mixed_schema_log(path) -> None:
    old_header = PREDICTION_LOG_COLUMNS[:22]
    old_row = ["old"] * 22
    new_row = ["new"] * len(PREDICTION_LOG_COLUMNS)
    path.write_text(
        ",".join(old_header) + "\n"
        + ",".join(old_row) + "\n"
        + ",".join(new_row) + "\n",
        encoding="utf-8",
    )


def test_malformed_prediction_log_does_not_crash_loader(tmp_path, monkeypatch):
    monkeypatch.setattr("src.storage.DATA_DIR", tmp_path)
    path = tmp_path / "prediction_log.csv"
    _write_mixed_schema_log(path)

    log = load_prediction_log()

    assert log.empty
    assert list(log.columns) == PREDICTION_LOG_COLUMNS
    assert "schema mismatch" in log.attrs.get("warning", "")


def test_repair_quarantines_and_recreates_current_schema_log(tmp_path):
    path = tmp_path / "prediction_log.csv"
    _write_mixed_schema_log(path)

    result = repair_prediction_log_by_quarantine(path, PREDICTION_LOG_COLUMNS)
    repaired = pd.read_csv(path)

    assert result["repaired"] is True
    assert result["rows_quarantined"] == 2
    assert result["backup_path"]
    assert list(repaired.columns) == PREDICTION_LOG_COLUMNS
    assert repaired.empty


def test_prediction_log_inspection_detects_mixed_field_counts(tmp_path):
    path = tmp_path / "prediction_log.csv"
    _write_mixed_schema_log(path)

    diagnostics = inspect_prediction_log_schema(path, PREDICTION_LOG_COLUMNS)

    assert diagnostics["is_corrupt"] is True
    assert diagnostics["header_columns"] == 22
    assert diagnostics["expected_columns"] == len(PREDICTION_LOG_COLUMNS)
    assert diagnostics["malformed_rows"] == 1


def test_empty_repaired_prediction_log_evaluates_gracefully():
    evaluation = evaluate_predictions(pd.DataFrame(columns=PREDICTION_LOG_COLUMNS), pd.DataFrame())
    assert evaluation.iloc[0]["notes"] == "No saved predictions."
