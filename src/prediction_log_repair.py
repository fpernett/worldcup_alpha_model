from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR


def inspect_prediction_log_schema(
    path: str | Path | None = None,
    expected_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Return structural diagnostics for the legacy Polymarket prediction log."""
    log_path = Path(path) if path is not None else DATA_DIR / "prediction_log.csv"
    expected = list(expected_columns or [])
    if not log_path.exists():
        return {
            "path": str(log_path),
            "exists": False,
            "header_columns": 0,
            "expected_columns": len(expected),
            "data_rows": 0,
            "malformed_rows": 0,
            "field_count_distribution": {},
            "is_corrupt": False,
            "warning": "prediction_log.csv does not exist.",
        }

    rows = _read_csv_rows(log_path)
    if not rows:
        return {
            "path": str(log_path),
            "exists": True,
            "header_columns": 0,
            "expected_columns": len(expected),
            "data_rows": 0,
            "malformed_rows": 0,
            "field_count_distribution": {},
            "is_corrupt": False,
            "warning": "prediction_log.csv is empty.",
        }

    header = rows[0]
    expected_count = len(expected) or len(header)
    counts: dict[int, int] = {}
    malformed_lines: list[int] = []
    for line_number, row in enumerate(rows[1:], start=2):
        count = len(row)
        counts[count] = counts.get(count, 0) + 1
        if count != expected_count:
            malformed_lines.append(line_number)

    header_mismatch = bool(expected) and header != expected
    is_corrupt = header_mismatch or bool(malformed_lines)
    warning = ""
    if is_corrupt:
        warning = (
            "prediction_log.csv schema mismatch detected. "
            f"Header has {len(header)} columns; expected {expected_count}. "
            f"Malformed data rows: {len(malformed_lines)}."
        )
    return {
        "path": str(log_path),
        "exists": True,
        "header_columns": len(header),
        "expected_columns": expected_count,
        "data_rows": max(len(rows) - 1, 0),
        "malformed_rows": len(malformed_lines),
        "malformed_line_numbers": malformed_lines[:50],
        "field_count_distribution": counts,
        "is_corrupt": is_corrupt,
        "warning": warning,
    }


def repair_prediction_log_by_quarantine(
    path: str | Path | None = None,
    expected_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Move a malformed prediction log aside and recreate it with the current schema."""
    log_path = Path(path) if path is not None else DATA_DIR / "prediction_log.csv"
    expected = list(expected_columns or [])
    diagnostics = inspect_prediction_log_schema(log_path, expected)
    if not diagnostics["exists"]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=expected).to_csv(log_path, index=False)
        return {**diagnostics, "repaired": True, "backup_path": "", "rows_quarantined": 0}

    if not diagnostics["is_corrupt"]:
        return {**diagnostics, "repaired": False, "backup_path": "", "rows_quarantined": 0}

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    backup_path = log_path.with_name(f"{log_path.stem}_corrupt_backup_{timestamp}{log_path.suffix}")
    shutil.copy2(log_path, backup_path)
    pd.DataFrame(columns=expected).to_csv(log_path, index=False)
    return {
        **diagnostics,
        "repaired": True,
        "backup_path": str(backup_path),
        "rows_quarantined": int(diagnostics.get("data_rows", 0)),
    }


def empty_prediction_log_with_warning(columns: list[str], warning: str) -> pd.DataFrame:
    out = pd.DataFrame(columns=columns)
    out.attrs["warning"] = warning
    return out


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))
