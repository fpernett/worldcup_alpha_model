from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache import write_dataframe_cache  # noqa: E402
from src.ratings import get_team_ratings  # noqa: E402


def main() -> None:
    ratings = get_team_ratings(force_refresh=True)
    write_dataframe_cache(ratings, "team_ratings_latest.csv", "team rating update script")
    print(f"Cached {len(ratings)} team rating rows.")


if __name__ == "__main__":
    main()
