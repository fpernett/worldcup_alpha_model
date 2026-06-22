from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import load_completed_matches_for_backtest  # noqa: E402
from src.prediction_ledger import RESULTS_LEDGER_PATH, import_completed_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or update data/results_ledger.csv from completed matches.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--teams", nargs="*", default=None)
    args = parser.parse_args()

    matches = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date, teams=args.teams)
    results = import_completed_results(matches, competition=args.competition, path=RESULTS_LEDGER_PATH)

    print(f"completed matches read: {len(matches):,}")
    print(f"results ledger rows: {len(results):,}")
    print(f"ledger: {RESULTS_LEDGER_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
