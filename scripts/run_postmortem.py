from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.postmortem import build_postmortem_report, calculate_prediction_errors, join_predictions_to_results  # noqa: E402
from src.prediction_ledger import load_prediction_ledger, load_results_ledger  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-mortem analysis by joining saved predictions to final results.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()

    predictions = load_prediction_ledger()
    results = load_results_ledger()
    predictions = _filter_predictions(predictions, args.start_date, args.end_date, args.competition)
    results = _filter_results(results, args.start_date, args.end_date, args.competition)
    joined = join_predictions_to_results(predictions, results)
    errors = calculate_prediction_errors(joined)
    report = build_postmortem_report(errors, joined)

    print(f"prediction ledger rows: {len(predictions):,}")
    print(f"results ledger rows: {len(results):,}")
    print(f"joined valid predictions: {len(errors):,}")
    summary = report.get("summary_metrics", pd.DataFrame())
    print("\nSummary metrics")
    print(_printable(summary).to_string(index=False) if not summary.empty else "No scored pre-kickoff predictions available.")
    print("\nWorst misses")
    print(_printable(report.get("worst_misses", pd.DataFrame()).head(5)).to_string(index=False))
    print("\nBest calls")
    print(_printable(report.get("best_calls", pd.DataFrame()).head(5)).to_string(index=False))

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    predictions_path = reports / f"postmortem_predictions_{today}.csv"
    errors.to_csv(predictions_path, index=False)
    report_path = reports / f"postmortem_report_{today}.md"
    if args.save_report:
        report_path.write_text(
            "\n\n".join(
                [
                    f"# Post-Mortem Model Report - {today}",
                    "",
                    f"Prediction ledger rows: `{len(predictions):,}`",
                    f"Results ledger rows: `{len(results):,}`",
                    f"Scored pre-kickoff predictions: `{len(errors):,}`",
                    "",
                    "## Summary Metrics",
                    "",
                    _to_markdown(_printable(summary)),
                    "",
                    "## Worst Misses",
                    "",
                    _to_markdown(_printable(report.get("worst_misses", pd.DataFrame()))),
                    "",
                    "## Best Calls",
                    "",
                    _to_markdown(_printable(report.get("best_calls", pd.DataFrame()))),
                    "",
                    "## Team-Level Errors",
                    "",
                    _to_markdown(_printable(report.get("team_level_error_table", pd.DataFrame()))),
                    "",
                    "This report is evaluation-only. It does not provide staking, wallet, trade execution, or betting advice.",
                ]
            ),
            encoding="utf-8",
        )
    print(f"\npostmortem predictions: {predictions_path.relative_to(PROJECT_ROOT)}")
    if args.save_report:
        print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _filter_predictions(df: pd.DataFrame, start: str | None, end: str | None, competition: str | None) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    dates = pd.to_datetime(out["kickoff_utc"], errors="coerce", utc=True)
    if start:
        out = out.loc[dates >= pd.to_datetime(start, errors="coerce", utc=True)]
        dates = pd.to_datetime(out["kickoff_utc"], errors="coerce", utc=True)
    if end:
        out = out.loc[dates <= pd.to_datetime(end, errors="coerce", utc=True)]
    if competition and "competition" in out.columns:
        out = out.loc[out["competition"].astype(str).str.lower().str.contains(str(competition).lower(), na=False)]
    return out.copy()


def _filter_results(df: pd.DataFrame, start: str | None, end: str | None, competition: str | None) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    dates = pd.to_datetime(out["date_utc"], errors="coerce")
    if start:
        out = out.loc[dates >= pd.to_datetime(start, errors="coerce")]
        dates = pd.to_datetime(out["date_utc"], errors="coerce")
    if end:
        out = out.loc[dates <= pd.to_datetime(end, errors="coerce")]
    if competition and "competition" in out.columns:
        out = out.loc[out["competition"].astype(str).str.lower().str.contains(str(competition).lower(), na=False)]
    return out.copy()


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
