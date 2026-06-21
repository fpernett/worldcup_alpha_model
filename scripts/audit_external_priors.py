from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.external_priors import (  # noqa: E402
    compare_ratings_to_external_priors,
    external_prior_review_summary,
    load_external_priors,
    load_review_proposals,
)
from src.rating_coverage import load_team_ratings_csv  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    ratings = load_team_ratings_csv()
    proposals = load_review_proposals(DATA_DIR / "team_ratings_review_proposals.csv")
    priors = load_external_priors()
    comparison = compare_ratings_to_external_priors(ratings, proposals, priors)
    summary = external_prior_review_summary(comparison)

    warning_text = comparison["overall_warning"].astype(str) if not comparison.empty else pd.Series(dtype="object")
    missing = comparison.loc[warning_text.str.contains("missing external prior", case=False, na=False)] if not comparison.empty else pd.DataFrame()
    large = comparison.loc[warning_text.str.contains("large ", case=False, na=False)] if not comparison.empty else pd.DataFrame()
    generated_without_check = comparison.loc[
        warning_text.str.contains("generated rating without external check", case=False, na=False)
    ] if not comparison.empty else pd.DataFrame()

    print(f"teams with external priors: {summary['teams_with_external_priors']}")
    print(f"teams missing external priors: {_team_list(missing)}")
    print(f"large disagreements: {_team_list(large)}")
    print(f"generated ratings without external check: {_team_list(generated_without_check)}")
    print(f"recommended review priorities: {_team_list(_priority_rows(comparison))}")

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"external_prior_comparison_{today}.csv"
    report_path = reports / f"external_prior_audit_{today}.md"
    comparison.to_csv(csv_path, index=False)
    report_path.write_text(
        "\n\n".join(
            [
                f"# External Prior Audit - {today}",
                "",
                "This audit compares current/generated ratings and review proposals against independent reference priors when available.",
                "",
                "## Summary",
                "",
                _to_markdown(pd.DataFrame([summary])),
                "",
                "## Missing External Priors",
                "",
                _to_markdown(_printable(missing)),
                "",
                "## Large Disagreements",
                "",
                _to_markdown(_printable(large)),
                "",
                "External priors inform manual review. They do not silently replace model ratings.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\ncomparison CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _priority_rows(comparison: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    warnings = comparison["overall_warning"].astype(str)
    return comparison.loc[warnings != ""].head(n)


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
