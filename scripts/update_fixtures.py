from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache import write_dataframe_cache  # noqa: E402
from src.data_sources import get_upcoming_fixtures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh upcoming fixtures into data/cache.")
    parser.add_argument("--start-date", default=date.today().isoformat())
    parser.add_argument("--end-date", default=(date.today() + timedelta(days=7)).isoformat())
    args = parser.parse_args()

    fixtures = get_upcoming_fixtures(date.fromisoformat(args.start_date), date.fromisoformat(args.end_date), force_refresh=True)
    write_dataframe_cache(fixtures, "fixtures_latest.csv", "fixture update script")
    print(f"Cached {len(fixtures)} fixture rows.")


if __name__ == "__main__":
    main()
