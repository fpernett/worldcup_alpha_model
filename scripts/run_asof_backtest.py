from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.asof_backtest import (  # noqa: E402
    ASOF_BACKTEST_WARNING,
    calculate_asof_backtest_metrics,
    compare_asof_backtest_predictions,
    run_asof_backtest,
)
from src.backtest import calculate_backtest_metrics, load_completed_matches_for_backtest  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.historical_data import load_historical_matches  # noqa: E402
from src.model_policy import get_current_model_policy  # noqa: E402
from src.ratings import TEAM_RATING_COLUMNS  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict as-of-date backtesting without current-behavior look-ahead.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()

    matches = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date, teams=args.teams)
    if args.competition and not matches.empty and "competition" in matches.columns:
        needle = str(args.competition).strip().lower()
        matches = matches.loc[matches["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()

    ratings = read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)
    historical = load_historical_matches(use_cache=False)
    predictions = run_asof_backtest(matches, ratings, historical, mode="both")
    metrics = calculate_asof_backtest_metrics(predictions)
    comparison = compare_asof_backtest_predictions(predictions)
    v1_metrics = _load_v1_metrics(args)

    scored_matches = predictions["match_id"].nunique() if not predictions.empty and "match_id" in predictions.columns else 0
    skipped = max(len(matches) - scored_matches, 0)
    lookahead_violations = _lookahead_violations(predictions)
    insufficient = _insufficient_predictions(predictions)
    policy = get_current_model_policy()

    print(f"completed matches used: {len(matches):,}")
    print(f"matches successfully scored: {scored_matches:,}")
    print(f"matches skipped: {skipped:,}")
    print(f"lookahead violations: {lookahead_violations:,}")
    print(f"matches with insufficient as-of behavior: {insufficient:,}")
    print(f"warning: {ASOF_BACKTEST_WARNING}")
    print("\nStrict v2 metrics")
    print(_printable(metrics).to_string(index=False) if not metrics.empty else "No metrics available.")
    print("\nDelta vs baseline")
    print(_printable(_delta_vs_baseline(metrics)).to_string(index=False) if not metrics.empty else "No metrics available.")
    if not v1_metrics.empty:
        print("\nApproximate v1 metrics for same filter")
        print(_printable(v1_metrics).to_string(index=False))

    helped = comparison.loc[comparison["actual_prob_delta"] > 0].sort_values("actual_prob_delta", ascending=False).head(5) if not comparison.empty else pd.DataFrame()
    hurt = comparison.loc[comparison["actual_prob_delta"] < 0].sort_values("actual_prob_delta", ascending=True).head(5) if not comparison.empty else pd.DataFrame()
    print("\nLargest strict as-of behavior improvements")
    print(_printable(helped).to_string(index=False) if not helped.empty else "No strict as-of behavior improvements in this sample.")
    print("\nLargest strict as-of behavior declines")
    print(_printable(hurt).to_string(index=False) if not hurt.empty else "No strict as-of behavior declines in this sample.")

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    predictions_path = reports / f"asof_backtest_predictions_{today}.csv"
    predictions.to_csv(predictions_path, index=False)
    report_path = reports / f"asof_backtest_report_{today}.md"
    if args.save_report:
        report_path.write_text(
            "\n\n".join(
                [
                    f"# Strict As-Of-Date Backtest - {today}",
                    "",
                    f"Completed matches used: `{len(matches):,}`",
                    f"Matches successfully scored: `{scored_matches:,}`",
                    f"Matches skipped: `{skipped:,}`",
                    f"Lookahead violations: `{lookahead_violations:,}`",
                    "",
                    ASOF_BACKTEST_WARNING,
                    "",
                    "## Model Policy",
                    "",
                    f"Primary model mode: `{policy['primary_model_mode']}`",
                    f"Behavior status: `{policy['behavior_status']}`",
                    f"Behavior default blend: `{float(policy['behavior_default_blend']):.2f}`",
                    f"Strict validation summary: {policy['reason']}",
                    "",
                    "## Strict V2 Metrics",
                    "",
                    _to_markdown(_printable(metrics)),
                    "",
                    "## Delta Vs Baseline",
                    "",
                    _to_markdown(_printable(_delta_vs_baseline(metrics))),
                    "",
                    "## Approximate V1 Metrics",
                    "",
                    _to_markdown(_printable(v1_metrics)),
                    "",
                    "## Per-Match Comparison",
                    "",
                    _to_markdown(_printable(comparison)),
                    "",
                    "## Per-Match Diagnostics",
                    "",
                    _to_markdown(_printable(predictions)),
                    "",
                    "This report is evaluation-only and does not provide staking, trade execution, or betting advice.",
                ]
            ),
            encoding="utf-8",
        )

    print(f"\npredictions: {predictions_path.relative_to(PROJECT_ROOT)}")
    if args.save_report:
        print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _load_v1_metrics(args: argparse.Namespace) -> pd.DataFrame:
    reports = PROJECT_ROOT / "reports"
    candidates = sorted(reports.glob("backtest_predictions_*.csv")) if reports.exists() else []
    if not candidates:
        return pd.DataFrame()
    try:
        predictions = pd.read_csv(candidates[-1])
    except Exception:
        return pd.DataFrame()
    if predictions.empty:
        return pd.DataFrame()
    if args.start_date and "date_utc" in predictions.columns:
        predictions = predictions.loc[pd.to_datetime(predictions["date_utc"], errors="coerce") >= pd.to_datetime(args.start_date, errors="coerce")]
    if args.end_date and "date_utc" in predictions.columns:
        predictions = predictions.loc[pd.to_datetime(predictions["date_utc"], errors="coerce") <= pd.to_datetime(args.end_date, errors="coerce")]
    if args.competition and "competition" in predictions.columns:
        needle = str(args.competition).strip().lower()
        predictions = predictions.loc[predictions["competition"].astype(str).str.lower().str.contains(needle, na=False)]
    if predictions.empty:
        return pd.DataFrame()
    return calculate_backtest_metrics(predictions)


def _delta_vs_baseline(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "model_mode" not in metrics.columns:
        return pd.DataFrame()
    baseline = metrics.loc[metrics["model_mode"] == "baseline_manual"]
    behavior = metrics.loc[metrics["model_mode"] == "behavior_adjusted_asof"]
    if baseline.empty or behavior.empty:
        return pd.DataFrame()
    b = baseline.iloc[0]
    a = behavior.iloc[0]
    rows = []
    for metric in ["brier_score_1x2", "log_loss_1x2", "most_likely_result_accuracy", "mean_probability_assigned_to_actual_result", "total_goals_mae"]:
        rows.append({"metric": metric, "baseline_manual": b.get(metric), "behavior_adjusted_asof": a.get(metric), "delta": a.get(metric) - b.get(metric)})
    return pd.DataFrame(rows)


def _lookahead_violations(predictions: pd.DataFrame) -> int:
    if predictions.empty:
        return 0
    unsafe = ~predictions["lookahead_safe"].astype(bool) if "lookahead_safe" in predictions.columns else pd.Series(False, index=predictions.index)
    warning = predictions["warning"].astype(str).str.contains("lookahead_violation", case=False, na=False) if "warning" in predictions.columns else pd.Series(False, index=predictions.index)
    return int((unsafe | warning).sum())


def _insufficient_predictions(predictions: pd.DataFrame) -> int:
    if predictions.empty or "warning" not in predictions.columns:
        return 0
    return int(predictions["warning"].astype(str).str.contains("insufficient_asof_behavior", case=False, na=False).sum())


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
