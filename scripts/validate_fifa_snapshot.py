from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import load_completed_matches_for_backtest  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.fifa_ranking_import import load_fifa_ranking_snapshot  # noqa: E402
from src.fifa_snapshot_validation import (  # noqa: E402
    build_missing_fifa_snapshot_template,
    validate_fifa_snapshot,
)
from src.historical_data import load_historical_matches  # noqa: E402
from src.rating_coverage import collect_required_teams  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.team_names import load_team_name_aliases_df  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


DEFAULT_INPUT = DATA_DIR / "raw" / "fifa_rankings_snapshot.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local FIFA ranking snapshot before import.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--start-date", default="2026-06-11")
    parser.add_argument("--end-date", default="2026-06-21")
    parser.add_argument("--competition", default="World Cup")
    parser.add_argument("--missing-output", default="", help="Optional CSV template for missing or incomplete teams.")
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    snapshot = load_fifa_ranking_snapshot(input_path)
    required = _load_required_teams(args)
    aliases = load_team_name_aliases_df(include_defaults=True)
    validation, summary = validate_fifa_snapshot(snapshot, required, aliases)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports_dir / f"fifa_snapshot_validation_{today}.csv"
    report_path = reports_dir / f"fifa_snapshot_validation_{today}.md"
    validation.to_csv(csv_path, index=False)
    report_path.write_text(_markdown_report(today, args, summary, validation, csv_path), encoding="utf-8")

    if args.missing_output:
        missing_path = _resolve_path(args.missing_output)
        missing_template = build_missing_fifa_snapshot_template(validation)
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_template.to_csv(missing_path, index=False)

    print("FIFA snapshot validation")
    print(f"Required teams: {summary['required_teams']}")
    print(f"Snapshot rows: {summary['snapshot_rows']}")
    print(f"Matched required teams: {summary['matched_required_teams']}")
    print(f"Missing required teams: {summary['missing_required_teams']}")
    print(f"Rows missing rank: {summary['matched_missing_rank']}")
    print(f"Rows missing points: {summary['matched_missing_points']}")
    print(f"Ambiguous matches: {summary['ambiguous_matches']}")
    print(f"Unused FIFA rows: {summary['unused_fifa_rows']}")
    print(f"Ready for import: {'yes' if summary['ready_for_import'] else 'no'}")
    print(f"validation CSV: {_display_path(csv_path)}")
    print(f"validation report: {_display_path(report_path)}")
    if args.missing_output:
        print(f"missing/incomplete template: {_display_path(_resolve_path(args.missing_output))}")


def _load_required_teams(args: argparse.Namespace) -> pd.DataFrame:
    fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", ["home", "away"])
    historical = load_historical_matches(use_cache=False)
    behavior = load_team_behavior()
    backtest = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date)
    if args.competition and not backtest.empty and "competition" in backtest.columns:
        needle = str(args.competition).strip().lower()
        backtest = backtest.loc[backtest["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    return collect_required_teams(fixtures, historical, behavior, backtest)


def _markdown_report(
    today: str,
    args: argparse.Namespace,
    summary: dict[str, object],
    validation: pd.DataFrame,
    csv_path: Path,
) -> str:
    missing = _status_rows(validation, "missing_from_snapshot")
    missing_rank = validation.loc[validation["match_status"].isin(["matched_missing_rank", "matched_missing_rank_and_points"])].copy()
    missing_points = validation.loc[validation["match_status"].isin(["matched_missing_points", "matched_missing_rank_and_points"])].copy()
    ambiguous = _status_rows(validation, "ambiguous_match")
    unused = _status_rows(validation, "unused_fifa_row")
    return "\n".join(
        [
            f"# FIFA Snapshot Validation - {today}",
            "",
            "This report validates a local FIFA ranking snapshot before importing it into external priors. It does not modify model ratings or external priors.",
            "",
            "## Summary",
            "",
            _to_markdown(pd.DataFrame([summary])),
            "",
            "## Files",
            "",
            f"- Input snapshot: `{args.input}`",
            f"- Validation CSV: `{_display_path(csv_path)}`",
            "",
            "## Missing Required Teams",
            "",
            _to_markdown(_printable(missing)),
            "",
            "## Teams Missing Rank",
            "",
            _to_markdown(_printable(missing_rank)),
            "",
            "## Teams Missing Points",
            "",
            _to_markdown(_printable(missing_points)),
            "",
            "## Ambiguous Matches",
            "",
            _to_markdown(_printable(ambiguous)),
            "",
            "## Unused FIFA Rows",
            "",
            _to_markdown(_printable(unused.head(75))),
        ]
    )


def _status_rows(validation: pd.DataFrame, status: str) -> pd.DataFrame:
    if validation.empty or "match_status" not in validation.columns:
        return pd.DataFrame()
    return validation.loc[validation["match_status"].astype(str) == status].copy()


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


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
