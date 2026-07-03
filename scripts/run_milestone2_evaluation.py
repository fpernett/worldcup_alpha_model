from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import build_formal_evaluation_dataset, build_prediction_source_dataset, write_milestone2_reports  # noqa: E402
from src.odds import load_market_odds  # noqa: E402
from src.prediction_ledger import load_prediction_ledger, load_results_ledger  # noqa: E402
from src.storage import data_path  # noqa: E402


def main() -> None:
    prediction_ledger = load_prediction_ledger()
    results_ledger = load_results_ledger()
    prediction_log = _read_csv_if_exists("prediction_log.csv")
    completed_results = _read_csv_if_exists("completed_results.csv")
    market_odds = load_market_odds()
    evaluation = build_formal_evaluation_dataset(
        prediction_ledger,
        results_ledger,
        completed_results_df=completed_results,
        prediction_log_df=prediction_log,
        market_odds_df=market_odds,
    )
    prediction_sources = build_prediction_source_dataset(
        prediction_ledger,
        prediction_log,
        market_odds,
    )
    paths = write_milestone2_reports(
        evaluation,
        reports_dir=PROJECT_ROOT / "reports",
        prediction_sources_df=prediction_sources,
    )

    print(f"prediction ledger rows: {len(prediction_ledger):,}")
    print(f"prediction log rows: {len(prediction_log):,}")
    print(f"results ledger rows: {len(results_ledger):,}")
    print(f"completed-result rows: {len(completed_results):,}")
    print(f"evaluation rows: {len(evaluation):,}")
    for key, path in paths.items():
        print(f"{key}: {path.relative_to(PROJECT_ROOT)}")


def _read_csv_if_exists(filename: str) -> pd.DataFrame:
    path = data_path(filename)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
