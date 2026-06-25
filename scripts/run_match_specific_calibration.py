from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.historical_data import load_historical_matches  # noqa: E402
from src.match_specific_calibration import build_match_specific_calibration_set, recommend_match_specific_parameter_set  # noqa: E402
from src.tournament_learning import load_tournament_learning_ledger  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a match-specific pre-match calibration evidence set.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--prediction-date", required=True)
    parser.add_argument("--kickoff-utc", default=None)
    parser.add_argument("--match-id", default=None)
    parser.add_argument("--competition", default="World Cup")
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()

    target_kickoff = args.kickoff_utc or f"{args.prediction_date}T00:00:00+00:00"
    historical = load_historical_matches(use_cache=False)
    learning = load_tournament_learning_ledger()
    calibration, diagnostics = build_match_specific_calibration_set(
        historical,
        learning,
        args.home,
        args.away,
        target_kickoff,
        target_match_id=args.match_id,
        competition=args.competition,
    )
    recommended, candidates = recommend_match_specific_parameter_set(calibration)

    print(f"match: {args.home} vs {args.away}")
    print(f"target kickoff: {target_kickoff}")
    print(f"calibration rows: {len(calibration):,}")
    print(f"calibration matches: {diagnostics.get('calibration_matches', 0):,}")
    print(f"completed World Cup matches used: {diagnostics.get('completed_world_cup_matches_used', 0):,}")
    print(f"lookahead safe: {diagnostics.get('lookahead_safe')}")
    print(f"recommended parameter set (diagnostic only): {recommended.get('parameter_set_id', 'n/a') if not recommended.empty else 'n/a'}")

    if args.save_report:
        reports = PROJECT_ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        today = today_iso()
        csv_path = reports / f"match_specific_calibration_{today}.csv"
        report_path = reports / f"match_specific_calibration_{today}.md"
        calibration.to_csv(csv_path, index=False)
        report_path.write_text(
            "\n\n".join(
                [
                    f"# Match-Specific Calibration - {today}",
                    "",
                    f"Match: `{args.home} vs {args.away}`",
                    f"Target kickoff: `{target_kickoff}`",
                    f"Recommended parameter set: `{recommended.get('parameter_set_id', 'n/a') if not recommended.empty else 'n/a'}`",
                    "",
                    "The recommendation is diagnostic-only and does not change the global primary model.",
                    "",
                    "## Diagnostics",
                    "",
                    _to_markdown(pd.DataFrame([diagnostics])),
                    "",
                    "## Calibration Evidence",
                    "",
                    _to_markdown(calibration.head(200)),
                    "",
                    "This report uses only evidence available before kickoff and does not provide staking, trade execution, Kelly sizing, or betting advice.",
                ]
            ),
            encoding="utf-8",
        )
        print(f"calibration csv: {csv_path.relative_to(PROJECT_ROOT)}")
        print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


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
