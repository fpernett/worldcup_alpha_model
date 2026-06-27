from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_sources import get_upcoming_fixtures  # noqa: E402
from src.model import ModelConfig  # noqa: E402
from src.odds import load_market_odds  # noqa: E402
from src.prediction_ledger import PREDICTION_LEDGER_PATH, snapshot_predictions_for_fixtures_with_diagnostics  # noqa: E402
from src.ratings import get_team_ratings  # noqa: E402
from src.utils import today_iso  # noqa: E402
from src.weather import load_venues  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prediction ledger snapshot behavior.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--competition", default="")
    parser.add_argument("--write", action="store_true", help="Append eligible snapshots to data/prediction_ledger.csv.")
    parser.add_argument("--include-past", action="store_true", help="Allow post-kickoff snapshots and mark them as not pre-kickoff.")
    parser.add_argument("--selected-match-id", action="append", default=None)
    args = parser.parse_args()

    fixtures = get_upcoming_fixtures(start_date=args.start_date, end_date=args.end_date)
    if args.competition and not fixtures.empty and "competition" in fixtures.columns:
        needle = str(args.competition).strip().lower()
        fixtures = fixtures.loc[fixtures["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()

    snapshots, diagnostics = snapshot_predictions_for_fixtures_with_diagnostics(
        fixtures,
        team_ratings_df=get_team_ratings(model_mode="baseline_manual"),
        venues_df=load_venues(),
        market_odds_df=load_market_odds(),
        cfg=ModelConfig(),
        model_version="baseline_external_calibrated_v1",
        parameter_set_id="baseline_current",
        primary_model_mode="baseline_manual",
        notes="Prediction snapshot audit" + ("" if args.write else " dry-run"),
        path=PREDICTION_LEDGER_PATH,
        append=args.write,
        include_past=args.include_past,
        selected_match_ids=args.selected_match_id,
    )
    eligible_pre_kickoff = int(len(snapshots.loc[snapshots["prediction_before_kickoff"].astype(str).str.lower().isin(["true", "1"])])) if not snapshots.empty else 0
    csv_path, md_path = _write_reports(args, snapshots, diagnostics, eligible_pre_kickoff)

    print(f"fixtures_available: {diagnostics.get('fixtures_available', 0)}")
    print(f"eligible_pre_kickoff_fixtures: {eligible_pre_kickoff}")
    print(f"predictions_that_would_be_written: {len(snapshots)}")
    print(f"predictions_written: {diagnostics.get('predictions_written', 0)}")
    print(f"output_path: {diagnostics.get('output_path', '')}")
    print("skip_reasons:")
    skip_reasons = diagnostics.get("skip_reasons", [])
    if skip_reasons:
        print(pd.DataFrame(skip_reasons).to_string(index=False))
    else:
        print("None")
    print(f"report CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {md_path.relative_to(PROJECT_ROOT)}")


def _write_reports(args: argparse.Namespace, snapshots: pd.DataFrame, diagnostics: dict, eligible_pre_kickoff: int) -> tuple[Path, Path]:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"prediction_snapshot_audit_{today}.csv"
    md_path = reports / f"prediction_snapshot_audit_{today}.md"
    snapshots.to_csv(csv_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "competition": args.competition,
                "write": bool(args.write),
                "include_past": bool(args.include_past),
                "fixtures_available": diagnostics.get("fixtures_available", 0),
                "fixtures_selected": diagnostics.get("fixtures_selected", 0),
                "eligible_pre_kickoff_fixtures": eligible_pre_kickoff,
                "predictions_that_would_be_written": len(snapshots),
                "predictions_written": diagnostics.get("predictions_written", 0),
                "predictions_skipped": diagnostics.get("predictions_skipped", 0),
                "output_path": diagnostics.get("output_path", ""),
                "warnings": "; ".join(diagnostics.get("warnings", [])),
            }
        ]
    )
    md_path.write_text(
        "\n".join(
            [
                f"# Prediction Snapshot Audit - {today}",
                "",
                "Dry-run by default. Use `--write` to append eligible rows to `data/prediction_ledger.csv`.",
                "",
                "## Summary",
                "",
                _to_markdown(summary),
                "",
                "## Skip Reasons",
                "",
                _to_markdown(pd.DataFrame(diagnostics.get("skip_reasons", []))),
                "",
                "## Snapshot Rows",
                "",
                _to_markdown(snapshots),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


def _to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    display = df.copy()
    display.attrs = {}
    display = display.fillna("").astype(str)
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
