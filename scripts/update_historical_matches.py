from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.historical_ingestion import update_historical_matches  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and merge recent national-team match history.")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--teams", nargs="*", default=None, help="Optional explicit team list.")
    parser.add_argument("--rebuild-behavior", action="store_true", default=True)
    parser.add_argument("--skip-rebuild-behavior", action="store_false", dest="rebuild_behavior")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore fresh API cache when API key is configured.")
    args = parser.parse_args()

    summary = update_historical_matches(
        start_date=args.start_date,
        end_date=args.end_date,
        teams=args.teams,
        rebuild_behavior=args.rebuild_behavior,
        force_refresh=args.force_refresh,
    )
    for key, value in summary.items():
        if key == "teams":
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
