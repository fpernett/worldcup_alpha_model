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
from src.model_training import generate_candidate_parameter_sets, run_walk_forward_training  # noqa: E402
from src.postmortem import build_postmortem_report, calculate_prediction_errors, join_predictions_to_results  # noqa: E402
from src.prediction_ledger import (  # noqa: E402
    RESULTS_LEDGER_PATH,
    import_completed_results,
    load_prediction_ledger,
    load_results_ledger,
)
from src.ratings import TEAM_RATING_COLUMNS  # noqa: E402
from src.tournament_learning import (  # noqa: E402
    TOURNAMENT_LEARNING_LEDGER_PATH,
    filter_tournament_learning_asof,
    load_tournament_learning_ledger,
    save_tournament_learning_ledger,
    update_tournament_learning_ledger,
)
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Update post-match learning ledgers after completed World Cup games.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default="World Cup")
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--force-refresh", action="store_true", help="Refresh completed-result provider/cache before post-match updates.")
    args = parser.parse_args()

    if args.force_refresh:
        from src.completed_results import load_completed_results  # noqa: WPS433

        load_completed_results(args.start_date, args.end_date, teams=args.teams, force_refresh=True, persist=True)
    matches = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date, teams=args.teams)
    results = import_completed_results(matches, competition=args.competition, path=RESULTS_LEDGER_PATH)
    existing_learning = load_tournament_learning_ledger()
    before_ids = set(existing_learning["match_id"].dropna().astype(str)) if not existing_learning.empty else set()
    learning = update_tournament_learning_ledger(results, existing_learning)
    learning = save_tournament_learning_ledger(learning, TOURNAMENT_LEARNING_LEDGER_PATH)
    after_ids = set(learning["match_id"].dropna().astype(str)) if not learning.empty else set()
    added_ids = sorted(after_ids - before_ids)

    predictions = load_prediction_ledger()
    joined = join_predictions_to_results(predictions, results)
    errors = calculate_prediction_errors(joined)
    postmortem_report = build_postmortem_report(errors, joined)

    historical = load_historical_matches(use_cache=False)
    ratings = read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)
    candidates = generate_candidate_parameter_sets()
    training_predictions, training_metrics = run_walk_forward_training(
        historical,
        matches,
        candidates,
        args.start_date,
        args.end_date,
        team_ratings_df=ratings,
        target_competition=args.competition or "World Cup",
    )

    target_for_summary = pd.to_datetime(args.end_date, errors="coerce", utc=True)
    if pd.isna(target_for_summary):
        target_for_summary = pd.Timestamp.now(tz="UTC")
    eligible, eligible_diag = filter_tournament_learning_asof(learning, target_for_summary + pd.Timedelta(days=1))

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"tournament_learning_update_{today}.csv"
    report_path = reports / f"tournament_learning_update_{today}.md"
    learning.to_csv(csv_path, index=False)
    report_path.write_text(
        "\n\n".join(
            [
                f"# Tournament Learning Update - {today}",
                "",
                f"Completed matches read: `{len(matches):,}`",
                f"Results ledger rows: `{len(results):,}`",
                f"Tournament learning rows: `{len(learning):,}`",
                f"New learning rows added: `{len(added_ids):,}`",
                f"Eligible for future calibration by summary cutoff: `{len(eligible):,}`",
                f"Lookahead safe: `{eligible_diag.get('lookahead_safe')}`",
                "",
                "## Newly Available Matches",
                "",
                _to_markdown(learning.loc[learning["match_id"].astype(str).isin(added_ids)].copy()),
                "",
                "## Post-Mortem Summary",
                "",
                _to_markdown(_printable(postmortem_report.get("summary_metrics", pd.DataFrame()))),
                "",
                "## Candidate Leaderboard",
                "",
                _to_markdown(_printable(training_metrics)),
                "",
                "This update is evaluation-only. It does not promote candidates automatically and does not provide staking, trade execution, Kelly sizing, or betting advice.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"completed matches read: {len(matches):,}")
    print(f"results ledger rows: {len(results):,}")
    print(f"tournament learning rows: {len(learning):,}")
    print(f"new learning rows added: {len(added_ids):,}")
    print(f"eligible for future calibration: {len(eligible):,}")
    print(f"lookahead safe: {eligible_diag.get('lookahead_safe')}")
    print("completed matches now available for future calibration:")
    display_cols = ["match_id", "date_utc", "home", "away", "actual_result", "eligible_after_utc"]
    available = learning[[col for col in display_cols if col in learning.columns]].tail(20)
    print(available.to_string(index=False) if not available.empty else "None")
    print(f"\nledger: {TOURNAMENT_LEARNING_LEDGER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"csv report: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"markdown report: {report_path.relative_to(PROJECT_ROOT)}")


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
