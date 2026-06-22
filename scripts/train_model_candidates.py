from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import load_completed_matches_for_backtest  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.historical_data import load_historical_matches  # noqa: E402
from src.model_training import (  # noqa: E402
    MODEL_PARAMETER_SETS_PATH,
    generate_candidate_parameter_sets,
    run_walk_forward_training,
)
from src.ratings import TEAM_RATING_COLUMNS  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward candidate parameter training.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()

    matches = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date, teams=args.teams)
    historical = load_historical_matches(use_cache=False)
    ratings = read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)
    candidates = generate_candidate_parameter_sets()
    MODEL_PARAMETER_SETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(MODEL_PARAMETER_SETS_PATH, index=False)
    predictions, metrics = run_walk_forward_training(
        historical,
        matches,
        candidates,
        args.start_date,
        args.end_date,
        team_ratings_df=ratings,
        target_competition=args.competition or "World Cup",
    )

    print(f"completed matches used: {len(matches):,}")
    print(f"candidate parameter sets tested: {len(candidates):,}")
    print("\nCandidate leaderboard")
    print(_printable(metrics.head(15)).to_string(index=False) if not metrics.empty else "No candidate metrics available.")
    promoted = metrics.loc[metrics["promotion_status"] == "promotion_candidate"] if not metrics.empty else pd.DataFrame()
    if promoted.empty:
        print("\npromotion: no candidate promoted; current baseline remains primary.")
    else:
        best = promoted.sort_values(["brier_1x2", "log_loss_1x2"]).iloc[0]
        print(f"\npromotion candidate: {best['parameter_set_id']}")

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    candidates_path = reports / f"model_training_candidates_{today}.csv"
    predictions_path = reports / f"model_training_predictions_{today}.csv"
    report_path = reports / f"model_training_report_{today}.md"
    metrics.to_csv(candidates_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    if args.save_report:
        report_path.write_text(
            "\n\n".join(
                [
                    f"# Model Candidate Training - {today}",
                    "",
                    f"Completed matches used: `{len(matches):,}`",
                    f"Target competition for relevance weighting: `{args.competition or 'World Cup'}`",
                    f"Candidate parameter sets tested: `{len(candidates):,}`",
                    "",
                    "Walk-forward scoring uses all senior national-team matches in the requested date window, not only World Cup rows. "
                    "Matches are weighted by relevance to the target competition. Friendlies are included with lower relevance weight rather than excluded.",
                    "",
                    "## Leaderboard",
                    "",
                    _to_markdown(_printable(metrics)),
                    "",
                    "## Prediction Diagnostics",
                    "",
                    _to_markdown(_printable(predictions.head(100))),
                    "",
                    "Promotion requires better Brier and log loss, no material goal-market regression, at least 30 evaluated matches, and no look-ahead leakage.",
                    "",
                    "This report is evaluation-only and does not provide staking, trade execution, Kelly sizing, or betting advice.",
                ]
            ),
            encoding="utf-8",
        )
    print(f"\nparameter sets: {MODEL_PARAMETER_SETS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"candidate metrics: {candidates_path.relative_to(PROJECT_ROOT)}")
    print(f"training predictions: {predictions_path.relative_to(PROJECT_ROOT)}")
    if args.save_report:
        print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


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
