from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_sources import update_all_sources  # noqa: E402
from src.historical_ingestion import update_historical_matches  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh all configured data sources into data/cache.")
    parser.add_argument("--start-date", default=date.today().isoformat())
    parser.add_argument("--end-date", default=(date.today() + timedelta(days=7)).isoformat())
    parser.add_argument("--update-historical", action="store_true", help="Fetch historical matches and rebuild team behavior.")
    parser.add_argument("--teams", nargs="*", default=None, help="Optional team list for historical ingestion.")
    args = parser.parse_args()

    summary = update_all_sources(
        date.fromisoformat(args.start_date),
        date.fromisoformat(args.end_date),
        update_historical_behavior=not args.update_historical,
    )
    if args.update_historical:
        summary["historical_ingestion"] = update_historical_matches(
            start_date=args.start_date,
            end_date=args.end_date,
            teams=args.teams,
            rebuild_behavior=True,
        )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
