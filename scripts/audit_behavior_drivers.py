from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.behavior_driver_report import (  # noqa: E402
    behavior_quality_warnings,
    calculate_competition_breakdown,
    calculate_opponent_tier_breakdown,
    get_behavior_driver_matches,
)
from src.config import DATA_DIR  # noqa: E402
from src.historical_data import load_historical_matches  # noqa: E402
from src.ratings import TEAM_RATING_COLUMNS, _normalise_ratings  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


DEFAULT_TEAMS = [
    "Norway",
    "England",
    "Argentina",
    "Brazil",
    "Uruguay",
    "Croatia",
    "Morocco",
    "United States",
    "Mexico",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit which matches drive behavior scores.")
    parser.add_argument("--teams", nargs="*", default=DEFAULT_TEAMS)
    parser.add_argument("--reference-date", default=today_iso())
    args = parser.parse_args()

    historical = load_historical_matches(use_cache=False)
    behavior = load_team_behavior()
    manual = _normalise_ratings(read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS), "manual_csv")
    sections: list[str] = []

    for team in args.teams:
        behavior_row = _matching_row(behavior, team)
        manual_row = _matching_row(manual, team)
        attack_drivers = get_behavior_driver_matches(historical, team, args.reference_date, n=10, sort_by="attack")
        defense_drivers = get_behavior_driver_matches(historical, team, args.reference_date, n=10, sort_by="defense")
        opponent_breakdown = calculate_opponent_tier_breakdown(historical, team, args.reference_date)
        competition_breakdown = calculate_competition_breakdown(historical, team, args.reference_date)
        warnings = behavior_quality_warnings(behavior_row, manual_row, pd.Series(dtype="object"), opponent_breakdown, competition_breakdown)

        print(f"\n=== {team} ===")
        print(
            "attack_index_raw=",
            _fmt(behavior_row.get("attack_index_raw")),
            "attack_index_residual_robust=",
            _fmt(behavior_row.get("attack_index_residual_robust", behavior_row.get("attack_index_residual"))),
            "attack_index_final=",
            _fmt(behavior_row.get("attack_index_final", behavior_row.get("attack_index"))),
        )
        print("warnings:", warnings or "")
        print("\nTop 10 attack driver matches")
        print(_printable(attack_drivers).to_string(index=False))
        print("\nTop 10 defense driver matches")
        print(_printable(defense_drivers).to_string(index=False))
        print("\nOpponent-tier breakdown")
        print(_printable(opponent_breakdown).to_string(index=False))
        print("\nCompetition-type breakdown")
        print(_printable(competition_breakdown).to_string(index=False))

        sections.append(
            "\n".join(
                [
                    f"## {team}",
                    "",
                    f"- attack_index_raw: `{_fmt(behavior_row.get('attack_index_raw'))}`",
                    f"- attack_index_residual_robust: `{_fmt(behavior_row.get('attack_index_residual_robust', behavior_row.get('attack_index_residual')))}`",
                    f"- attack_index_final: `{_fmt(behavior_row.get('attack_index_final', behavior_row.get('attack_index')))}`",
                    f"- warnings: {warnings or '_None._'}",
                    "",
                    "### Top 10 Attack Driver Matches",
                    "",
                    _to_markdown(_printable(attack_drivers)),
                    "",
                    "### Top 10 Defense Driver Matches",
                    "",
                    _to_markdown(_printable(defense_drivers)),
                    "",
                    "### Opponent-Tier Breakdown",
                    "",
                    _to_markdown(_printable(opponent_breakdown)),
                    "",
                    "### Competition-Type Breakdown",
                    "",
                    _to_markdown(_printable(competition_breakdown)),
                ]
            )
        )

    report_path = PROJECT_ROOT / "reports" / f"behavior_driver_audit_{today_iso()}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n\n".join([f"# Behavior Driver Audit - {today_iso()}", *sections]), encoding="utf-8")
    print(f"\nreport: {report_path.relative_to(PROJECT_ROOT)}")


def _matching_row(df: pd.DataFrame, team: str) -> pd.Series:
    if df is None or df.empty or "team" not in df.columns:
        return pd.Series(dtype="object")
    rows = df.loc[df["team"].astype(str).str.lower() == str(team).lower()]
    if rows.empty:
        return pd.Series(dtype="object")
    return rows.iloc[0]


def _printable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else round(float(value), 3))
    return out


def _fmt(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value or "")


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
