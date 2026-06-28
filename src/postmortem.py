from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.prediction_ledger import PREDICTION_LEDGER_COLUMNS, RESULTS_LEDGER_COLUMNS
from src.utils import coerce_float


POSTMORTEM_ERROR_COLUMNS = [
    "prediction_id",
    "match_id",
    "home",
    "away",
    "model_version",
    "parameter_set_id",
    "actual_result",
    "actual_result_prob",
    "brier_1x2",
    "log_loss_1x2",
    "result_correct",
    "over_2_5_brier",
    "btts_brier",
    "xg_total_error",
    "favorite_prob",
    "favorite_won",
    "upset_flag",
    "home_weighted_training_matches",
    "away_weighted_training_matches",
    "home_recent_friendlies_used",
    "away_recent_friendlies_used",
    "home_qualifiers_used",
    "away_qualifiers_used",
    "home_mean_relevance_weight",
    "away_mean_relevance_weight",
    "error_notes",
]

POSTMORTEM_ACTION_COLUMNS = [
    "area",
    "status",
    "what_it_means",
    "next_action",
    "command_or_file",
]


def join_predictions_to_results(prediction_ledger_df: pd.DataFrame | None, results_ledger_df: pd.DataFrame | None) -> pd.DataFrame:
    predictions = _ensure_columns(prediction_ledger_df, PREDICTION_LEDGER_COLUMNS)
    results = _ensure_columns(results_ledger_df, RESULTS_LEDGER_COLUMNS)
    if predictions.empty or results.empty:
        return pd.DataFrame(columns=[*PREDICTION_LEDGER_COLUMNS, *[f"result_{col}" for col in RESULTS_LEDGER_COLUMNS]])
    joined = predictions.merge(results, on="match_id", how="inner", suffixes=("", "_result"))
    kickoff = pd.to_datetime(joined["kickoff_utc"], errors="coerce", utc=True)
    snapshot = pd.to_datetime(joined["snapshot_utc"], errors="coerce", utc=True)
    joined["prediction_before_kickoff"] = snapshot.notna() & kickoff.notna() & (snapshot < kickoff)
    return joined.reset_index(drop=True)


def calculate_prediction_errors(joined_df: pd.DataFrame | None, only_valid: bool = True) -> pd.DataFrame:
    joined = joined_df.copy() if joined_df is not None else pd.DataFrame()
    if joined.empty:
        return pd.DataFrame(columns=POSTMORTEM_ERROR_COLUMNS)
    if only_valid and "prediction_before_kickoff" in joined.columns:
        joined = joined.loc[joined["prediction_before_kickoff"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        actual = str(row.get("actual_result", "") or row.get("actual_result_result", ""))
        home_prob = coerce_float(row.get("home_win_prob"), 0.0)
        draw_prob = coerce_float(row.get("draw_prob"), 0.0)
        away_prob = coerce_float(row.get("away_win_prob"), 0.0)
        actual_prob = _actual_probability(row, actual)
        brier = _brier_1x2(home_prob, draw_prob, away_prob, actual)
        log_loss = -math.log(max(min(actual_prob, 0.999), 0.001))
        most_likely = _most_likely_result(home_prob, draw_prob, away_prob)
        total_xg = coerce_float(row.get("home_xg"), 0.0) + coerce_float(row.get("away_xg"), 0.0)
        total_goals = coerce_float(row.get("total_goals"), coerce_float(row.get("home_goals"), 0.0) + coerce_float(row.get("away_goals"), 0.0))
        favorite_side, favorite_prob = _favorite_side(home_prob, away_prob)
        favorite_won = actual == favorite_side
        notes = []
        if row.get("prediction_before_kickoff") is False:
            notes.append("snapshot after kickoff")
        if actual_prob < 0.15:
            notes.append("low probability assigned to actual result")
        if favorite_prob >= 0.60 and not favorite_won:
            notes.append("favorite miss")
        rows.append(
            {
                "prediction_id": row.get("prediction_id", ""),
                "match_id": row.get("match_id", ""),
                "home": row.get("home", ""),
                "away": row.get("away", ""),
                "model_version": row.get("model_version", ""),
                "parameter_set_id": row.get("parameter_set_id", ""),
                "actual_result": actual,
                "actual_result_prob": actual_prob,
                "brier_1x2": brier,
                "log_loss_1x2": log_loss,
                "result_correct": bool(most_likely == actual),
                "over_2_5_brier": _binary_brier(row.get("over_2_5_prob"), row.get("over_2_5_actual")),
                "btts_brier": _binary_brier(row.get("btts_yes_prob"), row.get("btts_actual")),
                "xg_total_error": abs(total_xg - total_goals),
                "favorite_prob": favorite_prob,
                "favorite_won": bool(favorite_won),
                "upset_flag": bool(favorite_prob >= 0.55 and not favorite_won),
                "home_weighted_training_matches": row.get("home_weighted_training_matches", pd.NA),
                "away_weighted_training_matches": row.get("away_weighted_training_matches", pd.NA),
                "home_recent_friendlies_used": row.get("home_recent_friendlies_used", pd.NA),
                "away_recent_friendlies_used": row.get("away_recent_friendlies_used", pd.NA),
                "home_qualifiers_used": row.get("home_qualifiers_used", pd.NA),
                "away_qualifiers_used": row.get("away_qualifiers_used", pd.NA),
                "home_mean_relevance_weight": row.get("home_mean_relevance_weight", pd.NA),
                "away_mean_relevance_weight": row.get("away_mean_relevance_weight", pd.NA),
                "error_notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows, columns=POSTMORTEM_ERROR_COLUMNS)


def build_postmortem_report(errors_df: pd.DataFrame | None, joined_df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    errors = errors_df.copy() if errors_df is not None else pd.DataFrame(columns=POSTMORTEM_ERROR_COLUMNS)
    if errors.empty:
        return {
            "summary_metrics": pd.DataFrame(),
            "worst_misses": pd.DataFrame(),
            "best_calls": pd.DataFrame(),
            "favorites_overestimated": pd.DataFrame(),
            "underdogs_underestimated": pd.DataFrame(),
            "draw_calibration": pd.DataFrame(),
            "totals_calibration": pd.DataFrame(),
            "btts_calibration": pd.DataFrame(),
            "team_level_error_table": pd.DataFrame(),
            "model_version_comparison": pd.DataFrame(),
        }
    summary = _summary_metrics(errors)
    joined = joined_df.copy() if joined_df is not None else pd.DataFrame()
    return {
        "summary_metrics": summary,
        "worst_misses": errors.sort_values("log_loss_1x2", ascending=False).head(10).reset_index(drop=True),
        "best_calls": errors.sort_values("actual_result_prob", ascending=False).head(10).reset_index(drop=True),
        "favorites_overestimated": errors.loc[errors["favorite_prob"] >= 0.55].sort_values("favorite_won").head(10).reset_index(drop=True),
        "underdogs_underestimated": errors.loc[errors["upset_flag"].astype(bool)].sort_values("favorite_prob", ascending=False).head(10).reset_index(drop=True),
        "draw_calibration": _draw_calibration(joined),
        "totals_calibration": _binary_calibration(joined, "over_2_5_prob", "over_2_5_actual"),
        "btts_calibration": _binary_calibration(joined, "btts_yes_prob", "btts_actual"),
        "team_level_error_table": _team_level_errors(errors),
        "model_version_comparison": summary.copy(),
    }


def build_postmortem_action_plan(
    prediction_rows: int,
    result_rows: int,
    scored_prediction_rows: int,
    leaderboard_df: pd.DataFrame | None = None,
    minimum_training_sample: int = 30,
) -> pd.DataFrame:
    """Translate post-mortem diagnostics into evaluation-only next steps."""
    leaderboard = leaderboard_df.copy() if leaderboard_df is not None else pd.DataFrame()
    rows: list[dict[str, str]] = []

    if prediction_rows <= 0:
        rows.append(
            _action_row(
                "Prediction snapshots",
                "Missing",
                "There are no saved pre-match model snapshots to score later.",
                "Snapshot upcoming predictions before kickoff.",
                ".venv/bin/python scripts/snapshot_upcoming_predictions.py",
            )
        )
    else:
        rows.append(
            _action_row(
                "Prediction snapshots",
                "Ready",
                f"{prediction_rows:,} prediction ledger row(s) are available.",
                "Keep snapshotting before each match so post-mortems use pre-kickoff evidence.",
                "data/prediction_ledger.csv",
            )
        )

    if result_rows <= 0:
        rows.append(
            _action_row(
                "Completed results",
                "Missing",
                "No completed result rows are available to score predictions.",
                "Import or enter completed results after matches finish.",
                ".venv/bin/python scripts/import_completed_results.py",
            )
        )
    else:
        rows.append(
            _action_row(
                "Completed results",
                "Ready",
                f"{result_rows:,} result ledger row(s) are available.",
                "Keep results current after completed matches.",
                "data/results_ledger.csv",
            )
        )

    if scored_prediction_rows <= 0:
        rows.append(
            _action_row(
                "Scored predictions",
                "Blocked",
                "No prediction can be matched to a completed result before kickoff.",
                "Check match IDs, kickoff timestamps, and whether snapshots were saved before kickoff.",
                "Post-mortem Training > Worst misses / Best calls",
            )
        )
    elif scored_prediction_rows < minimum_training_sample:
        rows.append(
            _action_row(
                "Scored predictions",
                "Directional",
                f"{scored_prediction_rows:,} scored prediction(s) are available; this is below the {minimum_training_sample:,}-match promotion threshold.",
                "Use misses for diagnosis, but do not promote model settings yet.",
                "Collect more completed pre-kickoff predictions.",
            )
        )
    else:
        rows.append(
            _action_row(
                "Scored predictions",
                "Usable",
                f"{scored_prediction_rows:,} scored prediction(s) meet the minimum candidate-training sample.",
                "Review Brier, log loss, calibration, and miss tables before changing model policy.",
                "Post-mortem Training > Post-mortem metrics",
            )
        )

    promotion_status = _promotion_status(leaderboard)
    if leaderboard.empty:
        rows.append(
            _action_row(
                "Candidate training",
                "Not run",
                "No saved candidate leaderboard is available.",
                "Run candidate training after enough scored results exist.",
                ".venv/bin/python scripts/train_model_candidates.py --save-report",
            )
        )
    elif promotion_status == "promotion_candidate":
        rows.append(
            _action_row(
                "Candidate training",
                "Review needed",
                "At least one candidate beat the baseline under the promotion rule.",
                "Manually inspect the report and validation caveats before any model-policy change.",
                "reports/model_training_candidates_*.csv",
            )
        )
    else:
        rows.append(
            _action_row(
                "Candidate training",
                "Keep baseline",
                "No saved candidate currently meets the promotion rule.",
                "Keep the baseline external-calibrated model as primary and use diagnostics to guide future review.",
                "reports/model_training_candidates_*.csv",
            )
        )

    return pd.DataFrame(rows, columns=POSTMORTEM_ACTION_COLUMNS)


def _action_row(area: str, status: str, what_it_means: str, next_action: str, command_or_file: str) -> dict[str, str]:
    return {
        "area": area,
        "status": status,
        "what_it_means": what_it_means,
        "next_action": next_action,
        "command_or_file": command_or_file,
    }


def _promotion_status(leaderboard: pd.DataFrame) -> str:
    if leaderboard.empty or "promotion_status" not in leaderboard.columns:
        return ""
    statuses = leaderboard["promotion_status"].fillna("").astype(str)
    return "promotion_candidate" if statuses.eq("promotion_candidate").any() else ""


def _summary_metrics(errors: pd.DataFrame) -> pd.DataFrame:
    grouped = errors.groupby(["model_version", "parameter_set_id"], dropna=False)
    rows = []
    for (model_version, parameter_set_id), group in grouped:
        rows.append(
            {
                "model_version": model_version,
                "parameter_set_id": parameter_set_id,
                "n_predictions": int(len(group)),
                "brier_1x2": float(pd.to_numeric(group["brier_1x2"], errors="coerce").mean()),
                "log_loss_1x2": float(pd.to_numeric(group["log_loss_1x2"], errors="coerce").mean()),
                "accuracy": float(group["result_correct"].astype(bool).mean()),
                "mean_actual_result_prob": float(pd.to_numeric(group["actual_result_prob"], errors="coerce").mean()),
                "over_2_5_brier": float(pd.to_numeric(group["over_2_5_brier"], errors="coerce").mean()),
                "btts_brier": float(pd.to_numeric(group["btts_brier"], errors="coerce").mean()),
                "total_goals_mae": float(pd.to_numeric(group["xg_total_error"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _team_level_errors(errors: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in errors.iterrows():
        for side in ["home", "away"]:
            records.append(
                {
                    "team": row.get(side, ""),
                    "brier_1x2": row.get("brier_1x2"),
                    "log_loss_1x2": row.get("log_loss_1x2"),
                    "actual_result_prob": row.get("actual_result_prob"),
                    "result_correct": row.get("result_correct"),
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return (
        frame.groupby("team", dropna=False)
        .agg(
            matches=("team", "size"),
            mean_brier_1x2=("brier_1x2", "mean"),
            mean_log_loss_1x2=("log_loss_1x2", "mean"),
            mean_actual_result_prob=("actual_result_prob", "mean"),
            accuracy=("result_correct", "mean"),
        )
        .reset_index()
        .sort_values(["mean_log_loss_1x2", "team"], ascending=[False, True])
    )


def _draw_calibration(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty or "draw_prob" not in joined.columns:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "prob": pd.to_numeric(joined["draw_prob"], errors="coerce"),
            "actual": (joined["actual_result"].astype(str) == "draw").astype(float),
        }
    )
    return _calibration_table(frame)


def _binary_calibration(joined: pd.DataFrame, prob_col: str, actual_col: str) -> pd.DataFrame:
    if joined.empty or prob_col not in joined.columns or actual_col not in joined.columns:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "prob": pd.to_numeric(joined[prob_col], errors="coerce"),
            "actual": pd.to_numeric(joined[actual_col], errors="coerce"),
        }
    )
    return _calibration_table(frame)


def _calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(subset=["prob", "actual"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "n", "mean_predicted_probability", "observed_hit_rate", "calibration_error"])
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    frame["bucket"] = pd.cut(frame["prob"].clip(0.0, 1.0), bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=labels, include_lowest=True)
    out = frame.groupby("bucket", observed=False).agg(
        n=("actual", "size"),
        mean_predicted_probability=("prob", "mean"),
        observed_hit_rate=("actual", "mean"),
    )
    out["calibration_error"] = (out["mean_predicted_probability"] - out["observed_hit_rate"]).abs()
    return out.reset_index()


def _actual_probability(row: pd.Series, actual_result: str) -> float:
    if actual_result == "home_win":
        return coerce_float(row.get("home_win_prob"), 0.0)
    if actual_result == "draw":
        return coerce_float(row.get("draw_prob"), 0.0)
    if actual_result == "away_win":
        return coerce_float(row.get("away_win_prob"), 0.0)
    return 0.0


def _brier_1x2(home_prob: float, draw_prob: float, away_prob: float, actual: str) -> float:
    return float((home_prob - float(actual == "home_win")) ** 2 + (draw_prob - float(actual == "draw")) ** 2 + (away_prob - float(actual == "away_win")) ** 2)


def _binary_brier(probability: Any, actual: Any) -> float:
    return float((coerce_float(probability, 0.0) - coerce_float(actual, 0.0)) ** 2)


def _most_likely_result(home_prob: float, draw_prob: float, away_prob: float) -> str:
    return max({"home_win": home_prob, "draw": draw_prob, "away_win": away_prob}.items(), key=lambda item: item[1])[0]


def _favorite_side(home_prob: float, away_prob: float) -> tuple[str, float]:
    if home_prob >= away_prob:
        return "home_win", float(home_prob)
    return "away_win", float(away_prob)


def _ensure_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame(columns=columns)
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out.copy()
