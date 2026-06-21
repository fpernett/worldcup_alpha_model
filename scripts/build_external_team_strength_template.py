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
    build_external_strength_template,
    external_benchmark_coverage_summary,
)
from src.external_priors import load_external_priors  # noqa: E402
from src.historical_data import load_historical_matches  # noqa: E402
from src.rating_coverage import collect_required_teams, load_team_ratings_csv  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.utils import read_csv_with_columns  # noqa: E402


DEFAULT_OUTPUT = DATA_DIR / "raw" / "external_team_strength_missing_template.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manual external-team-strength CSV template.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--all-required-teams", action="store_true", help="Include every required team, not only missing benchmarks.")
    parser.add_argument("--start-date", default="2026-06-11")
    parser.add_argument("--end-date", default="2026-06-21")
    parser.add_argument("--competition", default="World Cup")
    args = parser.parse_args()

    coverage = audit_external_benchmark_coverage(
        _load_required_teams(args),
        load_external_priors(),
        load_team_ratings_csv(),
    )
    template = build_external_strength_template(coverage, all_required_teams=args.all_required_teams)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)

    summary = external_benchmark_coverage_summary(coverage)
    high_missing = coverage.loc[
        (~coverage["has_external_benchmark"].astype(bool)) & (coverage["priority"].astype(str) == "high")
    ] if not coverage.empty else pd.DataFrame()

    print(f"required teams: {summary['required_teams']}")
    print(f"teams with external benchmark: {summary['teams_with_external_benchmark']}")
    print(f"teams missing external benchmark: {summary['teams_missing_external_benchmark']}")
    print(f"template rows written: {len(template)}")
    print(f"high-priority missing teams: {_team_list(high_missing)}")
    print(f"template path: {output_path.relative_to(PROJECT_ROOT)}")


def _load_required_teams(args: argparse.Namespace) -> pd.DataFrame:
    fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", ["home", "away"])
    historical = load_historical_matches(use_cache=False)
    behavior = load_team_behavior()
    backtest = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date)
    if args.competition and not backtest.empty and "competition" in backtest.columns:
        needle = str(args.competition).strip().lower()
        backtest = backtest.loc[backtest["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    return collect_required_teams(fixtures, historical, behavior, backtest)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _team_list(df: pd.DataFrame) -> str:
    if df is None or df.empty or "team" not in df.columns:
        return "None"
    return ", ".join(df["team"].dropna().astype(str).tolist()) or "None"


if __name__ == "__main__":
    main()
