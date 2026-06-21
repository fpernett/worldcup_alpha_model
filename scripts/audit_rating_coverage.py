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
    audit_rating_coverage,
    collect_required_teams,
    load_team_ratings_csv,
    propose_team_aliases,
    rating_coverage_summary,
)
from src.team_behavior import load_team_behavior  # noqa: E402
from src.team_names import load_team_name_aliases_df  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit required-team coverage in data/team_ratings.csv.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    args = parser.parse_args()

    fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", ["home", "away"])
    historical = load_historical_matches(use_cache=False)
    behavior = load_team_behavior()
    backtest = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date)
    if args.competition and not backtest.empty and "competition" in backtest.columns:
        needle = str(args.competition).strip().lower()
        backtest = backtest.loc[backtest["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    ratings = load_team_ratings_csv()
    aliases = load_team_name_aliases_df(include_defaults=False)

    required = collect_required_teams(fixtures, historical, behavior, backtest)
    audit = audit_rating_coverage(required, ratings, aliases)
    alias_proposals = propose_team_aliases(required, ratings, aliases)
    summary = rating_coverage_summary(audit, alias_proposals)

    missing = audit.loc[~audit["in_team_ratings"].astype(bool)] if not audit.empty else pd.DataFrame()
    alias_mismatches = audit.loc[audit["possible_alias_match"].astype(str) != ""] if not audit.empty else pd.DataFrame()
    behavior_missing = audit.loc[audit["warning"].astype(str).str.contains("behavior exists but manual rating missing", case=False, na=False)] if not audit.empty else pd.DataFrame()

    print(f"required teams: {summary['required_teams']}")
    print(f"teams in team_ratings.csv: {summary['teams_in_team_ratings']}")
    print(f"missing teams: {_team_list(missing)}")
    print(f"possible alias mismatches: {_team_list(alias_mismatches)}")
    print(f"teams with behavior but no manual rating: {_team_list(behavior_missing)}")
    print(f"neutral fallback risk count: {summary['neutral_fallback_risk_count']}")
    print(f"recommended next actions: {summary['recommended_fixes']}")

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"rating_coverage_teams_{today}.csv"
    report_path = reports / f"rating_coverage_audit_{today}.md"
    audit.to_csv(csv_path, index=False)
    report_path.write_text(
        "\n\n".join(
            [
                f"# Rating Coverage Audit - {today}",
                "",
                "This audit checks whether teams required by fixtures and completed backtests have transparent manual rating rows.",
                "",
                "## Summary",
                "",
                _to_markdown(pd.DataFrame([summary])),
                "",
                "## Missing Ratings",
                "",
                _to_markdown(_printable(missing)),
                "",
                "## Possible Alias Mismatches",
                "",
                _to_markdown(_printable(alias_mismatches)),
                "",
                "## Proposed Aliases",
                "",
                _to_markdown(_printable(alias_proposals)),
                "",
                "Backtest metrics should not be trusted while required teams use neutral fallback ratings.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nteams CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


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


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
