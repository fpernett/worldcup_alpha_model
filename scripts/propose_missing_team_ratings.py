from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import load_completed_matches_for_backtest  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.historical_data import load_historical_matches  # noqa: E402
from src.rating_coverage import (  # noqa: E402
    append_missing_aliases,
    append_missing_team_ratings,
    audit_rating_coverage,
    build_missing_rating_proposals,
    collect_required_teams,
    load_team_ratings_csv,
    propose_team_aliases,
)
from src.team_behavior import load_team_behavior  # noqa: E402
from src.team_names import TEAM_NAME_ALIAS_COLUMNS, load_team_name_aliases_df  # noqa: E402
from src.utils import read_csv_with_columns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose conservative generated rating rows for required teams missing from team_ratings.csv.")
    parser.add_argument("--write", action="store_true", help="Append missing proposals to data/team_ratings.csv.")
    parser.add_argument("--write-aliases", action="store_true", help="Append missing alias proposals to data/team_name_aliases.csv.")
    parser.add_argument("--output", default=str(DATA_DIR / "team_ratings_proposed.csv"))
    parser.add_argument("--include-backtest", action="store_true")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    args = parser.parse_args()

    fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", ["home", "away"])
    historical = load_historical_matches(use_cache=False)
    behavior = load_team_behavior()
    ratings_path = DATA_DIR / "team_ratings.csv"
    ratings = load_team_ratings_csv(ratings_path)
    aliases_path = DATA_DIR / "team_name_aliases.csv"
    aliases = load_team_name_aliases_df(include_defaults=False)
    backtest = pd.DataFrame()
    if args.include_backtest:
        backtest = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date)
        if args.competition and not backtest.empty and "competition" in backtest.columns:
            needle = str(args.competition).strip().lower()
            backtest = backtest.loc[backtest["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()

    required = collect_required_teams(fixtures, historical, behavior, backtest)
    audit = audit_rating_coverage(required, ratings, aliases)
    proposals = build_missing_rating_proposals(audit, behavior, historical, ratings)
    alias_proposals = propose_team_aliases(required, ratings, aliases)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(output_path, index=False)
    alias_output = DATA_DIR / "team_name_aliases_proposed.csv"
    alias_proposals.to_csv(alias_output, index=False)

    print(f"missing teams: {_team_list(proposals)}")
    print("\nProposed rating rows")
    print(_printable(proposals).to_string(index=False) if not proposals.empty else "No missing required teams.")
    print("\nProposed aliases")
    print(_printable(alias_proposals).to_string(index=False) if not alias_proposals.empty else "No alias proposals.")
    print(f"\nproposal output: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"alias proposal output: {alias_output.relative_to(PROJECT_ROOT)}")

    if args.write:
        updated = append_missing_team_ratings(ratings, proposals)
        updated.to_csv(ratings_path, index=False)
        print(f"appended rating rows: {max(len(updated) - len(ratings), 0)}")
        print(f"updated: {ratings_path.relative_to(PROJECT_ROOT)}")
    else:
        print("data/team_ratings.csv was not modified. Pass --write to append missing rows.")

    if args.write_aliases:
        existing_aliases = read_csv_with_columns(aliases_path, TEAM_NAME_ALIAS_COLUMNS)
        updated_aliases = append_missing_aliases(existing_aliases, alias_proposals)
        updated_aliases.to_csv(aliases_path, index=False)
        print(f"appended alias rows: {max(len(updated_aliases) - len(existing_aliases), 0)}")
        print(f"updated aliases: {aliases_path.relative_to(PROJECT_ROOT)}")
    else:
        print("data/team_name_aliases.csv was not modified. Pass --write-aliases to append missing aliases.")


def _team_list(df: pd.DataFrame) -> str:
    if df.empty or "team" not in df.columns:
        return "None"
    return ", ".join(df["team"].dropna().astype(str).tolist()) or "None"


def _printable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else round(float(value), 4))
    return out


if __name__ == "__main__":
    main()
