from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction_ledger import (  # noqa: E402
    MANUAL_MISSING_RESULTS_PATH,
    RESULTS_LEDGER_PATH,
    import_completed_results,
    load_manual_missing_results,
)


HOME_SCORE_COLUMNS = ["home_goals", "home_score_90", "home_goals_90"]
AWAY_SCORE_COLUMNS = ["away_goals", "away_score_90", "away_goals_90"]


def _first_present(row: pd.Series, columns: list[str]) -> object:
    for col in columns:
        if col not in row:
            continue
        value = row.get(col)
        if pd.isna(value) or str(value).strip() == "":
            continue
        return value
    return pd.NA


def _manual_rows_loaded(path: Path) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        return pd.DataFrame(), 0
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(), 0
    return raw, int(len(raw))


def main() -> None:
    raw, manual_rows = _manual_rows_loaded(MANUAL_MISSING_RESULTS_PATH)
    if raw.empty:
        valid_rows = pd.DataFrame()
        skipped_missing_scores = 0
    else:
        home_scores = raw.apply(lambda row: _first_present(row, HOME_SCORE_COLUMNS), axis=1)
        away_scores = raw.apply(lambda row: _first_present(row, AWAY_SCORE_COLUMNS), axis=1)
        valid_mask = (
            home_scores.notna()
            & away_scores.notna()
            & (home_scores.astype(str).str.strip() != "")
            & (away_scores.astype(str).str.strip() != "")
        )
        skipped_missing_scores = int((~valid_mask).sum())
        valid_rows = load_manual_missing_results()

    results = import_completed_results(
        valid_rows,
        path=RESULTS_LEDGER_PATH,
        result_source="manual_verified_missing_results",
    )

    print(f"manual rows loaded: {manual_rows:,}")
    print(f"valid scored rows: {len(valid_rows):,}")
    print(f"skipped missing-score rows: {skipped_missing_scores:,}")
    print(f"final results ledger rows: {len(results):,}")


if __name__ == "__main__":
    main()
