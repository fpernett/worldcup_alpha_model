from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.blend_sensitivity import (  # noqa: E402
    DEFAULT_BLEND_MULTIPLIERS,
    blend_sensitivity_recommendation,
    build_team_sensitivity_table,
    normalise_blend_multipliers,
    run_asof_blend_sensitivity,
)
from src.backtest import load_completed_matches_for_backtest  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.historical_data import load_historical_matches  # noqa: E402
from src.model_policy import get_current_model_policy  # noqa: E402
from src.ratings import TEAM_RATING_COLUMNS  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict as-of behavior blend sensitivity backtests.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--blend-multipliers", nargs="*", type=float, default=DEFAULT_BLEND_MULTIPLIERS)
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()

    multipliers = normalise_blend_multipliers(args.blend_multipliers)
    matches = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date, teams=args.teams)
    if args.competition and not matches.empty and "competition" in matches.columns:
        needle = str(args.competition).strip().lower()
        matches = matches.loc[matches["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    ratings = read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)
    historical = load_historical_matches(use_cache=False)

    predictions, metrics = run_asof_blend_sensitivity(matches, ratings, historical, multipliers)
    team_table = build_team_sensitivity_table(predictions)
    recommendation = blend_sensitivity_recommendation(metrics)
    best_brier = _best_metric(metrics, "brier_score_1x2")
    best_log_loss = _best_metric(metrics, "log_loss_1x2")
    beats_baseline = _beats_baseline(metrics)
    policy = get_current_model_policy()

    print(f"completed matches: {len(matches):,}")
    print(f"blend multipliers tested: {', '.join(f'{value:.2f}' for value in multipliers)}")
    print(f"best blend by Brier: {_blend_label(best_brier)}")
    print(f"best blend by log loss: {_blend_label(best_log_loss)}")
    print(f"any blend beats baseline on both Brier and log loss: {'yes' if beats_baseline else 'no'}")
    print(f"recommendation: {recommendation}")
    print("\nMetrics by blend")
    print(_printable(metrics).to_string(index=False) if not metrics.empty else "No metrics available.")
    print("\nTeams where behavior helped")
    helped = team_table.loc[team_table["actual_prob_delta"] > 0.0005].sort_values("actual_prob_delta", ascending=False).head(15) if not team_table.empty else pd.DataFrame()
    print(_printable(helped).to_string(index=False) if not helped.empty else "None")
    print("\nTeams where default behavior hurt")
    hurt = team_table.loc[team_table["behavior_hurt_count"] > team_table["behavior_helped_count"]].sort_values("behavior_hurt_count", ascending=False).head(15) if not team_table.empty else pd.DataFrame()
    print(_printable(hurt).to_string(index=False) if not hurt.empty else "None")

    if args.save_report:
        reports = PROJECT_ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        today = today_iso()
        predictions_path = reports / f"blend_sensitivity_predictions_{today}.csv"
        metrics_path = reports / f"blend_sensitivity_metrics_{today}.csv"
        team_path = reports / f"blend_sensitivity_team_{today}.csv"
        report_path = reports / f"blend_sensitivity_report_{today}.md"
        predictions.to_csv(predictions_path, index=False)
        metrics.to_csv(metrics_path, index=False)
        team_table.to_csv(team_path, index=False)
        report_path.write_text(
            "\n\n".join(
                [
                    f"# Behavior Blend Sensitivity - {today}",
                    "",
                    f"Completed matches: `{len(matches):,}`",
                    f"Blend multipliers tested: `{', '.join(f'{value:.2f}' for value in multipliers)}`",
                    f"Best blend by Brier: `{_blend_label(best_brier)}`",
                    f"Best blend by log loss: `{_blend_label(best_log_loss)}`",
                    f"Any blend beats baseline on both Brier and log loss: `{'yes' if beats_baseline else 'no'}`",
                    "",
                    "## Model Policy",
                    "",
                    f"Primary model mode: `{policy['primary_model_mode']}`",
                    f"Behavior status: `{policy['behavior_status']}`",
                    f"Behavior default blend: `{float(policy['behavior_default_blend']):.2f}`",
                    f"Strict validation summary: {policy['reason']}",
                    "",
                    f"Recommendation: {recommendation}",
                    "",
                    "## Metrics By Blend",
                    "",
                    _to_markdown(_printable(metrics)),
                    "",
                    "## Team Sensitivity",
                    "",
                    _to_markdown(_printable(team_table)),
                    "",
                    "This report is evaluation-only and does not provide staking, trade execution, Kelly sizing, or betting advice.",
                ]
            ),
            encoding="utf-8",
        )
        print(f"\npredictions: {predictions_path.relative_to(PROJECT_ROOT)}")
        print(f"metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
        print(f"team sensitivity: {team_path.relative_to(PROJECT_ROOT)}")
        print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _best_metric(metrics: pd.DataFrame, metric: str) -> pd.Series:
    if metrics.empty or metric not in metrics.columns:
        return pd.Series(dtype="object")
    return metrics.sort_values(metric, ascending=True).iloc[0]


def _beats_baseline(metrics: pd.DataFrame) -> bool:
    if metrics.empty:
        return False
    rows = metrics.loc[(metrics["delta_brier_vs_baseline"] < 0) & (metrics["delta_log_loss_vs_baseline"] < 0)]
    return not rows.empty


def _blend_label(row: pd.Series) -> str:
    if row is None or row.empty:
        return "n/a"
    return f"{float(row.get('blend_multiplier', 0.0)):.2f}"


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
