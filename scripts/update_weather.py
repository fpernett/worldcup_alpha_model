from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache import write_dataframe_cache  # noqa: E402
from src.weather import get_venue_environment, load_venues  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh venue weather/environment into data/cache.")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--time", default="12:00")
    args = parser.parse_args()

    venues = load_venues()
    rows = [
        get_venue_environment(row["venue"], args.date, args.time, force_refresh=True)
        for _, row in venues.iterrows()
    ]
    write_dataframe_cache(pd.DataFrame(rows), "venue_environment_latest.csv", "weather update script")
    print(f"Cached {len(rows)} venue environment rows.")


if __name__ == "__main__":
    main()
