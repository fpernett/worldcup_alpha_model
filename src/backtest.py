from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.historical_data import HISTORICAL_MATCH_COLUMNS, load_historical_matches
from src.model import ModelConfig, run_match_model
from src.ratings import TEAM_RATING_COLUMNS, get_team_ratings, _normalise_ratings
from src.utils import coerce_bool, coerce_float, read_csv_with_columns
from src.weather import load_venues


COMPLETED_MATCH_COLUMNS = [
    "match_id",
    "date_utc",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "competition",
    "group",
    "venue",
    "city",
    "country",
    "neutral_site",
]

BACKTEST_PREDICTION_COLUMNS = [
    "model_mode",
    "match_id",
    "date_utc",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "competition",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "home_xg",
    "away_xg",
    "over_2_5_prob",
    "under_2_5_prob",
    "btts_yes_prob",
    "btts_no_prob",
    "most_likely_result",
    "actual_result",
    "actual_result_prob",
    "baseline_warning",
    "log_loss_1x2",
]

BACKTEST_METRIC_COLUMNS = [
    "model_mode",
    "n_matches",
    "brier_score_1x2",
    "log_loss_1x2",
    "most_likely_result_accuracy",
    "mean_probability_assigned_to_actual_result",
    "over_2_5_brier",
    "btts_brier",
    "home_xg_mae",
    "away_xg_mae",
    "total_goals_mae",
]

CALIBRATION_COLUMNS = [
    "bucket",
    "n",
    "mean_predicted_probability",
    "observed_hit_rate",
    "calibration_error",
]

COMPARISON_COLUMNS = [
    "match_id",
    "date_utc",
    "home",
    "away",
    "score",
    "actual_result",
    "baseline_home",
    "baseline_draw",
    "baseline_away",
    "behavior_home",
    "behavior_draw",
    "behavior_away",
    "baseline_actual_prob",
    "behavior_actual_prob",
    "actual_prob_delta",
    "baseline_log_loss",
    "behavior_log_loss",
    "improved_by_behavior",
    "notes",
]

BACKTEST_LOOKAHEAD_WARNING = (
    "Backtest may have look-ahead bias unless behavior metrics are rebuilt as of each match date. "
    "Behavior-Adjusted Backtesting v1 uses the current behavior table, so treat results as approximate."
)


def load_completed_matches_for_backtest(
    start_date: str | None = None,
    end_date: str | None = None,
    teams: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load completed matches as one row per match.

    `fixtures.csv` remains supported if actual goal columns are added later.
    The current historical layer stores national-team results in long format,
    so those rows are deduplicated into home/away match rows here.
    """
    frames = [
        _completed_matches_from_fixtures(),
        _completed_matches_from_historical(load_historical_matches(use_cache=False)),
    ]
    completed = pd.concat([df for df in frames if df is not None and not df.empty], ignore_index=True)
    if completed.empty:
        return pd.DataFrame(columns=COMPLETED_MATCH_COLUMNS)

    completed = _normalise_completed_matches(completed)
    completed = _filter_completed_matches(completed, start_date, end_date, teams)
    completed = _deduplicate_completed_matches(completed)
    return completed[COMPLETED_MATCH_COLUMNS].reset_index(drop=True)


def run_model_for_backtest_match(
    match_row: pd.Series | dict[str, Any],
    mode: str = "baseline_manual",
    teams_df: pd.DataFrame | None = None,
    venues_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    cfg: ModelConfig | None = None,
) -> dict[str, Any]:
    """Run the existing model for a completed match in one rating mode."""
    row = pd.Series(match_row).copy()
    if "time_utc" not in row or pd.isna(row.get("time_utc")):
        row["time_utc"] = "12:00"
    if "match_id" not in row or pd.isna(row.get("match_id")) or str(row.get("match_id")).strip() == "":
        row["match_id"] = _fallback_match_id(row)

    teams = teams_df if teams_df is not None else _ratings_for_mode(mode)
    venues = venues_df if venues_df is not None else load_venues()
    odds = market_odds_df if market_odds_df is not None else pd.DataFrame()

    result = run_match_model(row, teams, venues, odds, cfg or ModelConfig())
    probs = result.get("probs", {})
    actual_result = classify_actual_result(row.get("home_goals"), row.get("away_goals"))
    actual_prob = _actual_result_probability(probs, actual_result)
    most_likely = _most_likely_result(probs)
    home_goals = coerce_float(row.get("home_goals"), 0.0)
    away_goals = coerce_float(row.get("away_goals"), 0.0)

    return {
        "model_mode": mode,
        "match_id": str(row.get("match_id", "")),
        "date_utc": row.get("date_utc", ""),
        "home": row.get("home", ""),
        "away": row.get("away", ""),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "competition": row.get("competition", ""),
        "home_win_prob": float(probs.get("home_win", 0.0)),
        "draw_prob": float(probs.get("draw", 0.0)),
        "away_win_prob": float(probs.get("away_win", 0.0)),
        "home_xg": float(result.get("hxg", 0.0)),
        "away_xg": float(result.get("axg", 0.0)),
        "over_2_5_prob": float(probs.get("over_2_5", 0.0)),
        "under_2_5_prob": float(probs.get("under_2_5", 0.0)),
        "btts_yes_prob": float(probs.get("btts_yes", 0.0)),
        "btts_no_prob": float(probs.get("btts_no", 0.0)),
        "most_likely_result": most_likely,
        "actual_result": actual_result,
        "actual_result_prob": actual_prob,
        "baseline_warning": BACKTEST_LOOKAHEAD_WARNING if mode == "behavior_adjusted" else "",
        "log_loss_1x2": _safe_log_loss(actual_prob),
    }


def run_backtest(
    start_date: str | None = None,
    end_date: str | None = None,
    teams: list[str] | tuple[str, ...] | None = None,
    competition: str | None = None,
    strict_as_of_date: bool = False,
) -> dict[str, pd.DataFrame | str]:
    """Run baseline-manual and behavior-adjusted model modes on completed matches."""
    matches = load_completed_matches_for_backtest(start_date=start_date, end_date=end_date, teams=teams)
    if competition and not matches.empty and "competition" in matches.columns:
        needle = str(competition).strip().lower()
        matches = matches.loc[matches["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()

    if matches.empty:
        empty_predictions = pd.DataFrame(columns=BACKTEST_PREDICTION_COLUMNS)
        return {
            "matches": pd.DataFrame(columns=COMPLETED_MATCH_COLUMNS),
            "predictions": empty_predictions,
            "metrics": pd.DataFrame(columns=BACKTEST_METRIC_COLUMNS),
            "comparison": pd.DataFrame(columns=COMPARISON_COLUMNS),
            "calibration": pd.DataFrame(columns=["model_mode", *CALIBRATION_COLUMNS]),
            "warning": BACKTEST_LOOKAHEAD_WARNING,
        }

    if strict_as_of_date:
        warning = (
            "strict_as_of_date=True was requested, but v1 does not rebuild behavior as of each match date. "
            f"{BACKTEST_LOOKAHEAD_WARNING}"
        )
    else:
        warning = BACKTEST_LOOKAHEAD_WARNING

    baseline_teams = _manual_team_ratings()
    behavior_teams = get_team_ratings()
    venues = load_venues()
    odds = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, match in matches.iterrows():
        rows.append(run_model_for_backtest_match(match, "baseline_manual", baseline_teams, venues, odds))
        rows.append(run_model_for_backtest_match(match, "behavior_adjusted", behavior_teams, venues, odds))

    predictions = pd.DataFrame(rows)
    metrics = calculate_backtest_metrics(predictions)
    comparison = compare_backtest_predictions(predictions)
    calibration_frames = []
    for mode in ["baseline_manual", "behavior_adjusted"]:
        table = build_calibration_table(predictions, mode)
        if not table.empty:
            table.insert(0, "model_mode", mode)
        calibration_frames.append(table)
    calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame(columns=["model_mode", *CALIBRATION_COLUMNS])

    return {
        "matches": matches.reset_index(drop=True),
        "predictions": predictions[BACKTEST_PREDICTION_COLUMNS].reset_index(drop=True),
        "metrics": metrics,
        "comparison": comparison,
        "calibration": calibration,
        "warning": warning,
    }


def calculate_backtest_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=BACKTEST_METRIC_COLUMNS)

    rows: list[dict[str, Any]] = []
    for mode, group in predictions_df.groupby("model_mode", dropna=False):
        scored = group.copy()
        actual = scored["actual_result"].astype(str)
        y_home = (actual == "home_win").astype(float)
        y_draw = (actual == "draw").astype(float)
        y_away = (actual == "away_win").astype(float)
        home_prob = pd.to_numeric(scored["home_win_prob"], errors="coerce").fillna(0.0)
        draw_prob = pd.to_numeric(scored["draw_prob"], errors="coerce").fillna(0.0)
        away_prob = pd.to_numeric(scored["away_win_prob"], errors="coerce").fillna(0.0)
        brier = ((home_prob - y_home) ** 2 + (draw_prob - y_draw) ** 2 + (away_prob - y_away) ** 2).mean()

        actual_prob = pd.to_numeric(scored["actual_result_prob"], errors="coerce").clip(0.001, 0.999)
        log_loss = (-(actual_prob.apply(math.log))).mean()
        accuracy = (scored["most_likely_result"].astype(str) == actual).mean()
        over_actual = ((pd.to_numeric(scored["home_goals"], errors="coerce") + pd.to_numeric(scored["away_goals"], errors="coerce")) > 2.5).astype(float)
        btts_actual = ((pd.to_numeric(scored["home_goals"], errors="coerce") > 0) & (pd.to_numeric(scored["away_goals"], errors="coerce") > 0)).astype(float)
        over_brier = ((pd.to_numeric(scored["over_2_5_prob"], errors="coerce").fillna(0.0) - over_actual) ** 2).mean()
        btts_brier = ((pd.to_numeric(scored["btts_yes_prob"], errors="coerce").fillna(0.0) - btts_actual) ** 2).mean()
        home_xg_mae = (pd.to_numeric(scored["home_xg"], errors="coerce") - pd.to_numeric(scored["home_goals"], errors="coerce")).abs().mean()
        away_xg_mae = (pd.to_numeric(scored["away_xg"], errors="coerce") - pd.to_numeric(scored["away_goals"], errors="coerce")).abs().mean()
        total_goals_mae = (
            pd.to_numeric(scored["home_xg"], errors="coerce")
            + pd.to_numeric(scored["away_xg"], errors="coerce")
            - pd.to_numeric(scored["home_goals"], errors="coerce")
            - pd.to_numeric(scored["away_goals"], errors="coerce")
        ).abs().mean()

        rows.append(
            {
                "model_mode": mode,
                "n_matches": int(len(scored)),
                "brier_score_1x2": float(brier),
                "log_loss_1x2": float(log_loss),
                "most_likely_result_accuracy": float(accuracy),
                "mean_probability_assigned_to_actual_result": float(actual_prob.mean()),
                "over_2_5_brier": float(over_brier),
                "btts_brier": float(btts_brier),
                "home_xg_mae": float(home_xg_mae),
                "away_xg_mae": float(away_xg_mae),
                "total_goals_mae": float(total_goals_mae),
            }
        )
    return pd.DataFrame(rows, columns=BACKTEST_METRIC_COLUMNS)


def build_calibration_table(predictions_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Build one-vs-rest 1X2 calibration buckets for a model mode."""
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)
    rows = predictions_df.loc[predictions_df["model_mode"].astype(str) == str(mode)].copy()
    if rows.empty:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)

    records: list[dict[str, float]] = []
    for _, row in rows.iterrows():
        actual = str(row.get("actual_result", ""))
        for outcome, prob_col in [
            ("home_win", "home_win_prob"),
            ("draw", "draw_prob"),
            ("away_win", "away_win_prob"),
        ]:
            records.append(
                {
                    "prob": coerce_float(row.get(prob_col), 0.0),
                    "actual": 1.0 if actual == outcome else 0.0,
                }
            )
    scored = pd.DataFrame(records)
    if scored.empty:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)

    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    scored["bucket"] = pd.cut(scored["prob"].clip(0.0, 1.0), bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=labels, include_lowest=True)
    grouped = scored.groupby("bucket", observed=False).agg(
        n=("actual", "size"),
        mean_predicted_probability=("prob", "mean"),
        observed_hit_rate=("actual", "mean"),
    )
    grouped["calibration_error"] = (grouped["mean_predicted_probability"] - grouped["observed_hit_rate"]).abs()
    return grouped.reset_index()[CALIBRATION_COLUMNS]


def compare_backtest_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    baseline = predictions_df.loc[predictions_df["model_mode"] == "baseline_manual"].copy()
    behavior = predictions_df.loc[predictions_df["model_mode"] == "behavior_adjusted"].copy()
    if baseline.empty or behavior.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    merged = baseline.merge(behavior, on="match_id", suffixes=("_baseline", "_behavior"), how="inner")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        baseline_prob = coerce_float(row.get("actual_result_prob_baseline"), 0.0)
        behavior_prob = coerce_float(row.get("actual_result_prob_behavior"), 0.0)
        delta = behavior_prob - baseline_prob
        if delta > 0.0005:
            notes = "behavior improved actual-result probability"
        elif delta < -0.0005:
            notes = "behavior lowered actual-result probability"
        else:
            notes = "no material difference"
        rows.append(
            {
                "match_id": row.get("match_id", ""),
                "date_utc": row.get("date_utc_baseline", row.get("date_utc_behavior", "")),
                "home": row.get("home_baseline", row.get("home_behavior", "")),
                "away": row.get("away_baseline", row.get("away_behavior", "")),
                "score": f"{int(coerce_float(row.get('home_goals_baseline'), 0))}-{int(coerce_float(row.get('away_goals_baseline'), 0))}",
                "actual_result": row.get("actual_result_baseline", ""),
                "baseline_home": row.get("home_win_prob_baseline", pd.NA),
                "baseline_draw": row.get("draw_prob_baseline", pd.NA),
                "baseline_away": row.get("away_win_prob_baseline", pd.NA),
                "behavior_home": row.get("home_win_prob_behavior", pd.NA),
                "behavior_draw": row.get("draw_prob_behavior", pd.NA),
                "behavior_away": row.get("away_win_prob_behavior", pd.NA),
                "baseline_actual_prob": baseline_prob,
                "behavior_actual_prob": behavior_prob,
                "actual_prob_delta": delta,
                "baseline_log_loss": row.get("log_loss_1x2_baseline", pd.NA),
                "behavior_log_loss": row.get("log_loss_1x2_behavior", pd.NA),
                "improved_by_behavior": bool(delta > 0),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS).sort_values(["date_utc", "match_id"]).reset_index(drop=True)


def classify_actual_result(home_goals: Any, away_goals: Any) -> str:
    home = coerce_float(home_goals, 0.0)
    away = coerce_float(away_goals, 0.0)
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def _completed_matches_from_fixtures() -> pd.DataFrame:
    columns = [
        "match_id",
        "date_utc",
        "time_utc",
        "competition",
        "group",
        "home",
        "away",
        "venue",
        "city",
        "country",
        "home_goals",
        "away_goals",
        "neutral_site",
    ]
    fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", columns)
    if fixtures.empty or "home_goals" not in fixtures.columns or "away_goals" not in fixtures.columns:
        return pd.DataFrame(columns=COMPLETED_MATCH_COLUMNS)
    fixtures["home_goals"] = pd.to_numeric(fixtures["home_goals"], errors="coerce")
    fixtures["away_goals"] = pd.to_numeric(fixtures["away_goals"], errors="coerce")
    fixtures = fixtures.dropna(subset=["home", "away", "home_goals", "away_goals"])
    if fixtures.empty:
        return pd.DataFrame(columns=COMPLETED_MATCH_COLUMNS)
    if "neutral_site" not in fixtures.columns:
        fixtures["neutral_site"] = 1
    return fixtures[COMPLETED_MATCH_COLUMNS].copy()


def _completed_matches_from_historical(historical: pd.DataFrame) -> pd.DataFrame:
    if historical is None or historical.empty:
        return pd.DataFrame(columns=COMPLETED_MATCH_COLUMNS)
    df = historical.copy()
    for col in HISTORICAL_MATCH_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["team_goals"] = pd.to_numeric(df["team_goals"], errors="coerce")
    df["opponent_goals"] = pd.to_numeric(df["opponent_goals"], errors="coerce")
    df = df.dropna(subset=["date_utc", "team", "opponent", "team_goals", "opponent_goals"]).copy()
    if df.empty:
        return pd.DataFrame(columns=COMPLETED_MATCH_COLUMNS)

    df["_match_key"] = df.apply(_historical_match_key, axis=1)
    rows: list[dict[str, Any]] = []
    for _, group in df.groupby("_match_key", dropna=False):
        selected = _select_home_row(group)
        if selected is None:
            continue
        match_id = selected.get("match_id", "")
        if pd.isna(match_id) or str(match_id).strip() == "":
            match_id = _fallback_match_id(
                {
                    "date_utc": selected.get("date_utc"),
                    "home": selected.get("team"),
                    "away": selected.get("opponent"),
                }
            )
        rows.append(
            {
                "match_id": match_id,
                "date_utc": selected.get("date_utc", ""),
                "home": selected.get("team", ""),
                "away": selected.get("opponent", ""),
                "home_goals": selected.get("team_goals", pd.NA),
                "away_goals": selected.get("opponent_goals", pd.NA),
                "competition": selected.get("competition", ""),
                "group": "",
                "venue": selected.get("venue", ""),
                "city": selected.get("city", ""),
                "country": selected.get("country", ""),
                "neutral_site": int(coerce_bool(selected.get("is_neutral", True))),
            }
        )
    if not rows:
        return pd.DataFrame(columns=COMPLETED_MATCH_COLUMNS)
    return pd.DataFrame(rows, columns=COMPLETED_MATCH_COLUMNS)


def _normalise_completed_matches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in COMPLETED_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    out = out.dropna(subset=["date_utc", "home", "away", "home_goals", "away_goals"]).copy()
    out["home"] = out["home"].astype(str).str.strip()
    out["away"] = out["away"].astype(str).str.strip()
    out = out.loc[(out["home"] != "") & (out["away"] != "")]
    for idx, row in out.iterrows():
        if pd.isna(row.get("match_id")) or str(row.get("match_id")).strip() == "":
            out.at[idx, "match_id"] = _fallback_match_id(row)
        out.at[idx, "neutral_site"] = int(coerce_bool(row.get("neutral_site", True)))
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce").dt.date.astype(str)
    return out[COMPLETED_MATCH_COLUMNS]


def _filter_completed_matches(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    teams: list[str] | tuple[str, ...] | None,
) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["date_utc"], errors="coerce")
    if start_date:
        out = out.loc[dates >= pd.to_datetime(start_date, errors="coerce")]
        dates = pd.to_datetime(out["date_utc"], errors="coerce")
    if end_date:
        out = out.loc[dates <= pd.to_datetime(end_date, errors="coerce")]
    if teams:
        wanted = {str(team).strip().lower() for team in teams if str(team).strip()}
        out = out.loc[
            out["home"].astype(str).str.lower().isin(wanted)
            | out["away"].astype(str).str.lower().isin(wanted)
        ]
    return out.copy()


def _deduplicate_completed_matches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_dedupe_key"] = out.apply(
        lambda row: "|".join(
            [
                str(row.get("date_utc", "")).strip().lower(),
                str(row.get("home", "")).strip().lower(),
                str(row.get("away", "")).strip().lower(),
                str(int(coerce_float(row.get("home_goals"), 0))),
                str(int(coerce_float(row.get("away_goals"), 0))),
            ]
        ),
        axis=1,
    )
    out = out.drop_duplicates(subset=["_dedupe_key"], keep="first")
    out = out.sort_values(["date_utc", "home", "away"]).reset_index(drop=True)
    return out.drop(columns=["_dedupe_key"])


def _historical_match_key(row: pd.Series) -> str:
    match_id = str(row.get("match_id", "") or "").strip()
    if match_id:
        return match_id
    teams = sorted([str(row.get("team", "")).strip().lower(), str(row.get("opponent", "")).strip().lower()])
    goals = sorted([int(coerce_float(row.get("team_goals"), 0)), int(coerce_float(row.get("opponent_goals"), 0))])
    return "|".join([str(row.get("date_utc", "")), *teams, str(goals[0]), str(goals[1])])


def _select_home_row(group: pd.DataFrame) -> pd.Series | None:
    if group.empty:
        return None
    is_home = group["is_home"].map(coerce_bool) if "is_home" in group.columns else pd.Series([False] * len(group), index=group.index)
    home_rows = group.loc[is_home]
    if not home_rows.empty:
        return home_rows.sort_values("date_utc").iloc[0]
    return group.sort_values(["team", "opponent"]).iloc[0]


def _ratings_for_mode(mode: str) -> pd.DataFrame:
    if mode == "baseline_manual":
        return _manual_team_ratings()
    if mode == "behavior_adjusted":
        return get_team_ratings()
    raise ValueError(f"Unknown backtest mode: {mode}")


def _manual_team_ratings() -> pd.DataFrame:
    return _normalise_ratings(read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS), "manual_csv")


def _fallback_match_id(row: pd.Series | dict[str, Any]) -> str:
    date = str(row.get("date_utc", "")).replace("/", "-")
    home = str(row.get("home", "")).strip().lower().replace(" ", "_")
    away = str(row.get("away", "")).strip().lower().replace(" ", "_")
    return f"backtest_{date}_{home}_{away}"


def _actual_result_probability(probs: dict[str, Any], actual_result: str) -> float:
    mapping = {"home_win": "home_win", "draw": "draw", "away_win": "away_win"}
    return float(coerce_float(probs.get(mapping.get(actual_result, ""), 0.0), 0.0))


def _safe_log_loss(probability: Any) -> float:
    prob = min(0.999, max(0.001, coerce_float(probability, 0.001)))
    return float(-math.log(prob))


def _most_likely_result(probs: dict[str, Any]) -> str:
    outcomes = {
        "home_win": coerce_float(probs.get("home_win"), 0.0),
        "draw": coerce_float(probs.get("draw"), 0.0),
        "away_win": coerce_float(probs.get("away_win"), 0.0),
    }
    return max(outcomes, key=outcomes.get)
