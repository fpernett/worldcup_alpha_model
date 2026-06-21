from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behavior-adjusted backtesting against completed matches.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--save-report", action="store_true")
    parser.add_argument("--strict-as-of-date", action="store_true")
    args = parser.parse_args()

    result = run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        teams=args.teams,
        competition=args.competition,
        strict_as_of_date=args.strict_as_of_date,
    )
    matches = _frame(result.get("matches"))
    predictions = _frame(result.get("predictions"))
    metrics = _frame(result.get("metrics"))
    comparison = _frame(result.get("comparison"))
    calibration = _frame(result.get("calibration"))
    warning = str(result.get("warning", ""))

    print(f"completed matches: {len(matches):,}")
    if warning:
        print(f"warning: {warning}")
    print("\nSummary metrics")
    print(_printable(metrics).to_string(index=False) if not metrics.empty else "No metrics available.")

    if not comparison.empty:
        helped = comparison.loc[comparison["actual_prob_delta"] > 0].sort_values("actual_prob_delta", ascending=False).head(5)
        hurt = comparison.loc[comparison["actual_prob_delta"] < 0].sort_values("actual_prob_delta", ascending=True).head(5)
        print("\nLargest behavior improvements")
        print(_printable(helped).to_string(index=False) if not helped.empty else "No behavior improvements in this sample.")
        print("\nLargest behavior declines")
        print(_printable(hurt).to_string(index=False) if not hurt.empty else "No behavior declines in this sample.")

    if args.save_report:
        report_dir = PROJECT_ROOT / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        today = today_iso()
        predictions_path = report_dir / f"backtest_predictions_{today}.csv"
        report_path = report_dir / f"backtest_report_{today}.md"
        predictions.to_csv(predictions_path, index=False)
        report_path.write_text(
            "\n\n".join(
                [
                    f"# Behavior-Adjusted Backtest - {today}",
                    "",
                    f"Completed matches used: `{len(matches):,}`",
                    "",
                    f"Warning: {warning}",
                    "",
                    "## Summary Metrics",
                    "",
                    _to_markdown(_printable(metrics)),
                    "",
                    "## Per-Match Comparison",
                    "",
                    _to_markdown(_printable(comparison)),
                    "",
                    "## Calibration",
                    "",
                    _to_markdown(_printable(calibration)),
                    "",
                    "Lower Brier score and log loss are better. Higher probability assigned to the actual result is better.",
                    "This report is evaluation-only and does not provide staking, trade execution, or betting advice.",
                ]
            ),
            encoding="utf-8",
        )
        print(f"\npredictions: {predictions_path.relative_to(PROJECT_ROOT)}")
        print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _frame(value: object) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


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
