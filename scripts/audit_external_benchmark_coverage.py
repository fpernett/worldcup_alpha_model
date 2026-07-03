from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import load_completed_matches_for_backtest  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.external_benchmark_calibration import (  # noqa: E402
    audit_external_benchmark_coverage,
    external_benchmark_coverage_summary,
)
from src.external_priors import load_external_priors  # noqa: E402
from src.historical_data import load_historical_matches  # noqa: E402
from src.rating_coverage import collect_required_teams, load_team_ratings_csv  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit required-team external benchmark coverage.")
    parser.add_argument("--start-date", default="2026-06-11")
    parser.add_argument("--end-date", default="2026-06-21")
    parser.add_argument("--competition", default="World Cup")
    args = parser.parse_args()

    required = _load_required_teams(args)
    priors = load_external_priors()
    ratings = load_team_ratings_csv()
    coverage = audit_external_benchmark_coverage(required, priors, ratings)
    summary = external_benchmark_coverage_summary(coverage)
    missing = coverage.loc[~coverage["has_external_benchmark"].astype(bool)].copy() if not coverage.empty else pd.DataFrame()
    high_missing = missing.loc[missing["priority"].astype(str) == "high"].copy() if not missing.empty else pd.DataFrame()

    print(f"required teams: {summary['required_teams']}")
    print(f"teams with external benchmark: {summary['teams_with_external_benchmark']}")
    print(f"teams missing external benchmark: {summary['teams_missing_external_benchmark']}")
    print(f"high-priority missing teams: {_team_list(high_missing)}")
    print(f"last external benchmark update: {summary['last_external_benchmark_update'] or 'Unavailable'}")

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"external_benchmark_coverage_{today}.csv"
    report_path = reports / f"external_benchmark_coverage_audit_{today}.md"
    coverage.to_csv(csv_path, index=False)
    report_path.write_text(
        "\n\n".join(
            [
                f"# External Benchmark Coverage Audit - {today}",
                "",
                "This audit checks whether required teams have independent FIFA/Elo-style benchmark strength inputs. It does not modify model ratings.",
                "",
                "## Summary",
                "",
                _to_markdown(pd.DataFrame([summary])),
                "",
                "## Missing Benchmarks",
                "",
                _to_markdown(_printable(missing)),
                "",
                "## High-Priority Missing Teams",
                "",
                _to_markdown(_printable(high_missing)),
                "",
                "Use `scripts/build_external_team_strength_template.py` to create a CSV template for missing benchmark snapshots.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\ncoverage CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _load_required_teams(args: argparse.Namespace) -> pd.DataFrame:
    fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", ["home", "away"])
    historical = load_historical_matches(use_cache=False)
    behavior = load_team_behavior()
    backtest = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date)
    if args.competition and not backtest.empty and "competition" in backtest.columns:
        needle = str(args.competition).strip().lower()
        backtest = backtest.loc[backtest["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    return collect_required_teams(fixtures, historical, behavior, backtest)


def _team_list(df: pd.DataFrame) -> str:
    if df is None or df.empty or "team" not in df.columns:
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
