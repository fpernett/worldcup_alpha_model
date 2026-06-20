from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.elo import add_elo_to_historical_matches  # noqa: E402
from src.historical_data import HISTORICAL_MATCH_COLUMNS  # noqa: E402
from src.team_behavior import rebuild_team_behavior_csv  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild rolling Elo columns on historical match rows.")
    parser.add_argument("--input", default="data/historical_matches.csv", help="Input historical long-format CSV.")
    parser.add_argument("--output", default="data/historical_matches.csv", help="Output historical long-format CSV.")
    parser.add_argument("--rebuild-behavior", action="store_true", help="Rebuild data/team_behavior.csv after Elo update.")
    parser.add_argument("--reference-date", default=today_iso(), help="Reference date for behavior rebuild.")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    historical = read_csv_with_columns(input_path, HISTORICAL_MATCH_COLUMNS)
    enhanced = add_elo_to_historical_matches(historical)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.to_csv(output_path, index=False)

    print(f"input_rows: {len(historical)}")
    print(f"output_rows: {len(enhanced)}")
    print(f"team_elo_pre_populated: {int(pd.to_numeric(enhanced.get('team_elo_pre'), errors='coerce').notna().sum())}")
    print(f"opponent_elo_populated: {int(pd.to_numeric(enhanced.get('opponent_elo'), errors='coerce').notna().sum())}")
    print(f"output: {_display_path(output_path)}")

    if args.rebuild_behavior:
        behavior = rebuild_team_behavior_csv(matches_df=enhanced, reference_date=args.reference_date)
        print(f"team_behavior_rows: {len(behavior)}")

def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
