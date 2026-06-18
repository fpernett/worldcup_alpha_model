from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from src.cache import utc_now_iso
from src.config import DATA_DIR
from src.storage import append_csv, load_csv


PREDICTION_LOG_COLUMNS = [
    "timestamp_utc",
    "match_id",
    "home",
    "away",
    "kickoff_utc",
    "market_id",
    "question",
    "market_type",
    "model_side",
    "polymarket_side",
    "model_probability",
    "fair_price_cents",
    "polymarket_price_cents",
    "alpha_gap_cents",
    "alpha_ev",
    "signal_strength",
    "home_xg",
    "away_xg",
    "model_config_name",
    "mapping_confidence",
    "liquidity",
    "volume",
]

RESULTS_LOG_COLUMNS = [
    "match_id",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "result_home_win",
    "result_draw",
    "result_away_win",
    "over_2_5",
    "under_2_5",
    "btts_yes",
    "btts_no",
    "completed",
    "result_source",
    "last_updated",
]


def save_prediction_snapshot(match_row, model_result: dict, polymarket_alpha_df: pd.DataFrame) -> None:
    if polymarket_alpha_df is None or polymarket_alpha_df.empty:
        return

    kickoff = f"{match_row.get('date_utc', '')} {match_row.get('time_utc', '')} UTC"
    rows = []
    for _, alpha in polymarket_alpha_df.iterrows():
        rows.append(
            {
                "timestamp_utc": utc_now_iso(),
                "match_id": str(match_row.get("match_id", "")),
                "home": match_row.get("home", model_result.get("home", "")),
                "away": match_row.get("away", model_result.get("away", "")),
                "kickoff_utc": kickoff,
                "market_id": alpha.get("market_id", ""),
                "question": alpha.get("question", ""),
                "market_type": alpha.get("market_type", ""),
                "model_side": alpha.get("model_side", ""),
                "polymarket_side": alpha.get("polymarket_side", ""),
                "model_probability": alpha.get("model_probability", pd.NA),
                "fair_price_cents": alpha.get("fair_price_cents", pd.NA),
                "polymarket_price_cents": alpha.get("polymarket_price_cents", pd.NA),
                "alpha_gap_cents": alpha.get("alpha_gap_cents", pd.NA),
                "alpha_ev": alpha.get("alpha_ev", pd.NA),
                "signal_strength": alpha.get("signal_strength", ""),
                "home_xg": model_result.get("hxg", pd.NA),
                "away_xg": model_result.get("axg", pd.NA),
                "model_config_name": "Dashboard",
                "mapping_confidence": alpha.get("mapping_confidence", ""),
                "liquidity": alpha.get("liquidity", pd.NA),
                "volume": alpha.get("volume", pd.NA),
            }
        )
    append_csv("prediction_log.csv", pd.DataFrame(rows), PREDICTION_LOG_COLUMNS)


def load_prediction_log() -> pd.DataFrame:
    return load_csv("prediction_log.csv", PREDICTION_LOG_COLUMNS)


def load_results_log() -> pd.DataFrame:
    return load_csv("results_log.csv", RESULTS_LOG_COLUMNS)


def evaluate_predictions(prediction_log: pd.DataFrame, results_log: pd.DataFrame) -> pd.DataFrame:
    if prediction_log is None or prediction_log.empty:
        return _empty_summary("No saved predictions.")
    if results_log is None or results_log.empty:
        return _empty_summary("No completed results available.", n_predictions=len(prediction_log))

    preds = prediction_log.copy()
    results = results_log.copy()
    results = results.loc[results["completed"].astype(str).str.lower().isin(["1", "true", "yes"])]
    if results.empty:
        return _empty_summary("Results log has no completed matches.", n_predictions=len(prediction_log))

    merged = preds.merge(results, on="match_id", how="inner", suffixes=("", "_result"))
    if merged.empty:
        return _empty_summary("No prediction rows match completed results.", n_predictions=len(prediction_log))

    outcomes = []
    for _, row in merged.iterrows():
        actual = _actual_for_prediction(row)
        prob = _safe_probability(row.get("model_probability"))
        if actual is None or prob is None:
            continue
        outcomes.append(
            {
                "prob": prob,
                "actual": actual,
                "signal_strength": row.get("signal_strength", ""),
                "alpha_gap_cents": row.get("alpha_gap_cents", pd.NA),
            }
        )

    scored = pd.DataFrame(outcomes)
    if scored.empty:
        return _empty_summary("Completed results exist, but no prediction market types could be evaluated.", n_predictions=len(prediction_log))

    brier = float(((scored["prob"] - scored["actual"]) ** 2).mean())
    clipped = scored["prob"].clip(0.001, 0.999)
    log_loss = float((-(scored["actual"] * clipped.apply(math.log) + (1 - scored["actual"]) * (1 - clipped).apply(math.log))).mean())
    mean_alpha_gap = float(pd.to_numeric(scored["alpha_gap_cents"], errors="coerce").mean())
    hit_rate = scored.groupby("signal_strength")["actual"].mean().round(3).to_dict()
    calibration = _calibration_summary(scored)

    return pd.DataFrame(
        [
            {
                "n_predictions": int(len(scored)),
                "brier_score": brier,
                "log_loss": log_loss,
                "mean_alpha_gap": mean_alpha_gap,
                "hit_rate_by_signal_strength": json.dumps(hit_rate, sort_keys=True),
                "calibration_bucket": json.dumps(calibration, sort_keys=True),
                "notes": "Evaluated only prediction rows with completed results and supported market types.",
            }
        ]
    )


def _actual_for_prediction(row: pd.Series) -> int | None:
    side = str(row.get("model_side", "")).lower()
    market_type = str(row.get("market_type", "")).lower()
    if side in {"home_win", "match_winner_home"} or market_type == "match_winner_home":
        actual = _bool_result(row.get("result_home_win"))
    elif side in {"away_win", "match_winner_away"} or market_type == "match_winner_away":
        actual = _bool_result(row.get("result_away_win"))
    elif side == "draw" or market_type == "draw":
        actual = _bool_result(row.get("result_draw"))
    elif side in {"over_2_5"} or market_type == "over_2_5":
        actual = _bool_result(row.get("over_2_5"))
    elif side in {"under_2_5"} or market_type == "under_2_5":
        actual = _bool_result(row.get("under_2_5"))
    elif side in {"btts_yes"} or market_type == "btts_yes":
        actual = _bool_result(row.get("btts_yes"))
    elif side in {"btts_no"} or market_type == "btts_no":
        actual = _bool_result(row.get("btts_no"))
    elif side in {"home_not_win", "away_or_draw"}:
        home_win = _bool_result(row.get("result_home_win"))
        actual = None if home_win is None else 1 - home_win
    elif side in {"away_not_win", "home_or_draw"}:
        away_win = _bool_result(row.get("result_away_win"))
        actual = None if away_win is None else 1 - away_win
    else:
        return None

    if actual is None:
        return None
    if str(row.get("polymarket_side", "YES")).upper() == "NO":
        return 1 - actual
    return actual


def _safe_probability(value: Any) -> float | None:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(prob) or prob < 0 or prob > 1:
        return None
    return prob


def _bool_result(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no"}:
        return 0
    try:
        return 1 if float(value) > 0 else 0
    except (TypeError, ValueError):
        return None


def _calibration_summary(scored: pd.DataFrame) -> list[dict[str, float]]:
    if scored.empty:
        return []
    buckets = pd.cut(scored["prob"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    table = scored.groupby(buckets, observed=False).agg(n=("actual", "size"), avg_pred=("prob", "mean"), actual_rate=("actual", "mean"))
    out = []
    for bucket, row in table.dropna(how="all").iterrows():
        if int(row["n"]) == 0:
            continue
        out.append(
            {
                "bucket": str(bucket),
                "n": int(row["n"]),
                "avg_pred": round(float(row["avg_pred"]), 3),
                "actual_rate": round(float(row["actual_rate"]), 3),
            }
        )
    return out


def _empty_summary(notes: str, n_predictions: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_predictions": int(n_predictions),
                "brier_score": pd.NA,
                "log_loss": pd.NA,
                "mean_alpha_gap": pd.NA,
                "hit_rate_by_signal_strength": "{}",
                "calibration_bucket": "[]",
                "notes": notes,
            }
        ]
    )
