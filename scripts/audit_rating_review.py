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
from src.rating_coverage import load_team_ratings_csv  # noqa: E402
from src.rating_review import audit_generated_ratings, rating_review_summary  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated team ratings that need manual review.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    args = parser.parse_args()

    ratings = load_team_ratings_csv()
    behavior = load_team_behavior()
    historical = load_historical_matches(use_cache=False)
    backtest = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date)
    if args.competition and not backtest.empty and "competition" in backtest.columns:
        needle = str(args.competition).strip().lower()
        backtest = backtest.loc[backtest["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()

    audit = audit_generated_ratings(ratings, behavior, historical, backtest)
    summary = rating_review_summary(audit)
    generated = audit.loc[audit["rating_status"] == "generated_from_behavior"] if not audit.empty else pd.DataFrame()
    high_priority = generated.loc[generated["priority"] == "high"] if not generated.empty else pd.DataFrame()
    reviewed = audit.loc[audit["rating_status"] == "manual_reviewed"] if not audit.empty else pd.DataFrame()
    disagreement = audit.loc[
        audit["warning"].astype(str).str.contains("differs from behavior", case=False, na=False)
    ] if not audit.empty else pd.DataFrame()

    print(f"generated ratings: {summary['generated_from_behavior_rows']}")
    print(f"high-priority generated ratings: {_team_list(high_priority)}")
    print(f"top teams needing manual review: {_team_list(_top_review_rows(generated))}")
    print(f"teams already manually reviewed: {_team_list(reviewed)}")
    print(f"teams with behavior/manual disagreement: {_team_list(disagreement)}")

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"rating_review_teams_{today}.csv"
    report_path = reports / f"rating_review_audit_{today}.md"
    audit.to_csv(csv_path, index=False)
    report_path.write_text(
        "\n\n".join(
            [
                f"# Rating Review Audit - {today}",
                "",
                "This audit identifies generated team-rating placeholders that should be manually reviewed before serious model interpretation.",
                "",
                "## Summary",
                "",
                _to_markdown(pd.DataFrame([summary])),
                "",
                "## High-Priority Generated Ratings",
                "",
                _to_markdown(_printable(high_priority)),
                "",
                "## Behavior/Manual Disagreements",
                "",
                _to_markdown(_printable(disagreement)),
                "",
                "Generated ratings are conservative placeholders. They are not final expert priors until reviewed.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nteams CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _top_review_rows(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return df
    priority_order = {"high": 0, "medium": 1, "low": 2, "reviewed": 3}
    out = df.copy()
    out["_priority_order"] = out["priority"].map(priority_order).fillna(9)
    return out.sort_values(
        ["_priority_order", "backtest_match_count", "fixture_match_count", "team"],
        ascending=[True, False, False, True],
    ).head(n)


def _team_list(df: pd.DataFrame) -> str:
    if df.empty or "team" not in df.columns:
        return "None"
    return ", ".join(df["team"].dropna().astype(str).tolist()) or "None"


def _printable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "_priority_order" in out.columns:
        out = out.drop(columns=["_priority_order"])
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
