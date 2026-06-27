from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_event_registry import load_polymarket_event_registry, match_fixture_to_polymarket_registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local Polymarket sports event registry matching.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--fixture-date", required=True)
    parser.add_argument("--competition", default="World Cup")
    args = parser.parse_args()

    registry = load_polymarket_event_registry()
    match = match_fixture_to_polymarket_registry(args.home, args.away, args.fixture_date, args.competition, registry)
    print(f"registry rows: {len(registry)}")
    print(f"fixture: {args.home} vs {args.away} {args.fixture_date}")
    print(f"resolved_slug: {match.get('resolved_slug', '')}")
    print(f"resolution_status: {match.get('resolution_status', '')}")
    print(f"confidence: {match.get('confidence', '')}")
    print(f"matched_home: {match.get('matched_home', False)}")
    print(f"matched_away: {match.get('matched_away', False)}")
    print(f"matched_date: {match.get('matched_date', False)}")
    print(f"team_order: {match.get('team_order', '')}")
    if match.get("warning"):
        print(f"warning: {match.get('warning', '')}")
    if not registry.empty:
        display = registry.loc[registry["event_date"].astype(str) == args.fixture_date].copy()
        if display.empty:
            display = registry.head(20)
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()
