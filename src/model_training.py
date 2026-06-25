from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.asof_backtest import (
    _as_naive_timestamp,
    _date_label,
    _elo_asof_from_rolling_rows,
    _has_elo_inputs,
    _normalise_completed_matches,
    _rating_frame,
    get_team_ratings_asof,
)
from src.backtest import run_model_for_backtest_match
from src.config import DATA_DIR
from src.elo import ELO_ASOF_COLUMNS, add_elo_to_historical_matches, calculate_rolling_elo
from src.historical_data import HISTORICAL_MATCH_COLUMNS
from src.historical_ingestion import classify_competition_type
from src.model import ModelConfig
from src.utils import clamp, coerce_float
from src.weather import load_venues


MODEL_PARAMETER_SETS_PATH = DATA_DIR / "model_parameter_sets.csv"

MODEL_PARAMETER_COLUMNS = [
    "parameter_set_id",
    "base_total_goals",
    "elo_xg_weight",
    "attack_weight",
    "defense_weight",
    "draw_inflation_factor",
    "favorite_strength_scale",
    "underdog_resistance_scale",
    "external_prior_weight",
    "rating_gap_to_xg_scale",
    "goal_correlation_adjustment",
    "notes",
]

TRAINING_PREDICTION_COLUMNS = [
    "parameter_set_id",
    "evaluation_relevance_weight",
    "match_id",
    "date_utc",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "competition",
    "actual_result",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "actual_result_prob",
    "home_xg",
    "away_xg",
    "over_2_5_prob",
    "btts_yes_prob",
    "most_likely_result",
    "log_loss_1x2",
    "lookahead_safe",
    "matches_used_total",
    "world_cup_matches_used",
    "qualifier_matches_used",
    "friendly_matches_used",
    "weighted_match_count",
    "mean_relevance_weight",
    "latest_match_used",
    "oldest_match_used",
    "home_weighted_training_matches",
    "away_weighted_training_matches",
    "home_recent_friendlies_used",
    "away_recent_friendlies_used",
    "home_qualifiers_used",
    "away_qualifiers_used",
    "home_mean_relevance_weight",
    "away_mean_relevance_weight",
    "warning",
]

TRAINING_METRIC_COLUMNS = [
    "parameter_set_id",
    "n_matches",
    "brier_1x2",
    "log_loss_1x2",
    "accuracy",
    "mean_actual_result_prob",
    "over_2_5_brier",
    "btts_brier",
    "total_goals_mae",
    "calibration_error",
    "matches_used_total",
    "world_cup_matches_used",
    "qualifier_matches_used",
    "friendly_matches_used",
    "weighted_match_count",
    "mean_relevance_weight",
    "latest_match_used",
    "oldest_match_used",
    "beats_current_baseline",
    "promotion_status",
]


def calculate_match_relevance_weight(
    historical_match_row: pd.Series | dict[str, Any],
    prediction_date: Any,
    target_competition: str = "World Cup",
) -> float:
    """Estimate how informative a historical national-team match is for a target fixture."""
    row = pd.Series(historical_match_row)
    competition_type = str(row.get("competition_type", "") or "").strip().lower()
    match_date = _as_naive_datetime(row.get("date_utc"))
    prediction_ts = _as_naive_datetime(prediction_date)
    if pd.isna(match_date) or match_date >= prediction_ts:
        return 0.0
    days = max((prediction_ts - match_date).days, 0)
    competition_weight = _competition_relevance_weight(competition_type, days)
    recency_multiplier = _recency_multiplier(days)
    cycle_multiplier = _tournament_cycle_multiplier(match_date, prediction_ts, target_competition)
    return clamp(competition_weight * recency_multiplier * cycle_multiplier, 0.10, 2.00)


def build_training_relevance_diagnostics(
    historical_matches_df: pd.DataFrame | None,
    prediction_date: Any,
    teams: list[str] | tuple[str, ...] | None = None,
    target_competition: str = "World Cup",
) -> dict[str, Any]:
    """Summarize all pre-match historical rows and their relevance weights."""
    rows = _historical_before_prediction(historical_matches_df, prediction_date)
    if teams:
        wanted = {str(team).strip().lower() for team in teams}
        rows = rows.loc[rows["team"].astype(str).str.lower().isin(wanted)].copy()
    if rows.empty:
        return _empty_relevance_diagnostics()
    rows = _add_relevance_weights(rows, prediction_date, target_competition)
    rows = rows.loc[rows["relevance_weight"] > 0].copy()
    if rows.empty:
        return _empty_relevance_diagnostics()
    dates = pd.to_datetime(rows["date_utc"], errors="coerce")
    comp = rows["competition_type"].astype(str).str.lower()
    return {
        "matches_used_total": int(rows["match_id"].astype(str).replace("", pd.NA).nunique() or len(rows) / 2),
        "world_cup_matches_used": int(rows.loc[comp.eq("world_cup"), "match_id"].astype(str).nunique()),
        "qualifier_matches_used": int(rows.loc[comp.str.contains("qualifier", na=False), "match_id"].astype(str).nunique()),
        "friendly_matches_used": int(rows.loc[comp.eq("friendly"), "match_id"].astype(str).nunique()),
        "weighted_match_count": float(rows["relevance_weight"].sum() / 2.0),
        "mean_relevance_weight": float(rows["relevance_weight"].mean()),
        "latest_match_used": "" if dates.dropna().empty else dates.max().date().isoformat(),
        "oldest_match_used": "" if dates.dropna().empty else dates.min().date().isoformat(),
    }


def build_team_training_relevance_diagnostics(
    historical_matches_df: pd.DataFrame | None,
    team: str,
    prediction_date: Any,
    target_competition: str = "World Cup",
) -> dict[str, Any]:
    rows = _historical_before_prediction(historical_matches_df, prediction_date)
    if rows.empty:
        return {
            "weighted_training_matches": 0.0,
            "recent_friendlies_used": 0,
            "qualifiers_used": 0,
            "mean_relevance_weight": 0.0,
        }
    team_rows = rows.loc[rows["team"].astype(str).str.lower() == str(team).strip().lower()].copy()
    if team_rows.empty:
        return {
            "weighted_training_matches": 0.0,
            "recent_friendlies_used": 0,
            "qualifiers_used": 0,
            "mean_relevance_weight": 0.0,
        }
    team_rows = _add_relevance_weights(team_rows, prediction_date, target_competition)
    team_rows = team_rows.loc[team_rows["relevance_weight"] > 0].copy()
    comp = team_rows["competition_type"].astype(str).str.lower()
    dates = pd.to_datetime(team_rows["date_utc"], errors="coerce")
    recent_friendly = comp.eq("friendly") & ((pd.Timestamp(prediction_date) - dates).dt.days <= 365)
    return {
        "weighted_training_matches": float(team_rows["relevance_weight"].sum()),
        "recent_friendlies_used": int(recent_friendly.sum()),
        "qualifiers_used": int(comp.str.contains("qualifier", na=False).sum()),
        "mean_relevance_weight": float(team_rows["relevance_weight"].mean()) if not team_rows.empty else 0.0,
    }


def generate_candidate_parameter_sets() -> pd.DataFrame:
    baseline = ModelConfig()
    rows: list[dict[str, Any]] = [
        _candidate_row(
            "baseline_current",
            baseline.base_total_goals,
            baseline.elo_xg_weight,
            baseline.attack_weight,
            baseline.defense_weight,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            "Current transparent baseline.",
        )
    ]
    for value in [2.2, 2.4, 2.6, 2.8]:
        if value != baseline.base_total_goals:
            rows.append(_candidate_row(f"base_goals_{value:.1f}", value, baseline.elo_xg_weight, baseline.attack_weight, baseline.defense_weight, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, "One-axis base total goals sensitivity."))
    for mult in [0.75, 1.25]:
        rows.append(_candidate_row(f"elo_weight_{mult:.2f}", baseline.base_total_goals, baseline.elo_xg_weight * mult, baseline.attack_weight, baseline.defense_weight, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, "One-axis Elo xG sensitivity."))
    for attack, defense in [(0.45, 0.55), (0.65, 0.35)]:
        rows.append(_candidate_row(f"atk_{attack:.2f}_def_{defense:.2f}", baseline.base_total_goals, baseline.elo_xg_weight, attack, defense, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, "Attack/defense balance sensitivity."))
    for draw in [0.90, 1.10, 1.20]:
        rows.append(_candidate_row(f"draw_{draw:.2f}", baseline.base_total_goals, baseline.elo_xg_weight, baseline.attack_weight, baseline.defense_weight, draw, 1.0, 1.0, 0.0, 1.0, 0.0, "Transparent draw probability post-calibration."))
    for favorite in [0.85, 1.15]:
        rows.append(_candidate_row(f"favorite_{favorite:.2f}", baseline.base_total_goals, baseline.elo_xg_weight, baseline.attack_weight, baseline.defense_weight, 1.0, favorite, 1.0, 0.0, 1.0, 0.0, "Favorite probability scaling sensitivity."))
    for underdog in [0.90, 1.10]:
        rows.append(_candidate_row(f"underdog_{underdog:.2f}", baseline.base_total_goals, baseline.elo_xg_weight, baseline.attack_weight, baseline.defense_weight, 1.0, 1.0, underdog, 0.0, 1.0, 0.0, "Underdog resistance sensitivity."))
    for corr in [-0.05, 0.05]:
        rows.append(_candidate_row(f"goal_corr_{corr:+.2f}".replace("+", "p").replace("-", "m"), baseline.base_total_goals, baseline.elo_xg_weight, baseline.attack_weight, baseline.defense_weight, 1.0, 1.0, 1.0, 0.0, 1.0, corr, "Goal-market correlation sensitivity."))
    return pd.DataFrame(rows, columns=MODEL_PARAMETER_COLUMNS)


def run_walk_forward_training(
    historical_matches_df: pd.DataFrame | None,
    fixtures_or_completed_matches_df: pd.DataFrame | None,
    candidate_params_df: pd.DataFrame | None,
    start_date: str | None,
    end_date: str | None,
    team_ratings_df: pd.DataFrame | None = None,
    target_competition: str = "World Cup",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = _normalise_completed_matches(fixtures_or_completed_matches_df)
    if start_date and not matches.empty:
        matches = matches.loc[pd.to_datetime(matches["date_utc"], errors="coerce") >= pd.to_datetime(start_date, errors="coerce")]
    if end_date and not matches.empty:
        matches = matches.loc[pd.to_datetime(matches["date_utc"], errors="coerce") <= pd.to_datetime(end_date, errors="coerce")]
    candidates = _ensure_candidate_params(candidate_params_df)
    if matches.empty or candidates.empty:
        return pd.DataFrame(columns=TRAINING_PREDICTION_COLUMNS), pd.DataFrame(columns=TRAINING_METRIC_COLUMNS)

    historical = historical_matches_df.copy() if historical_matches_df is not None else pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
    historical_with_elo = historical if _has_elo_inputs(historical) else add_elo_to_historical_matches(historical) if not historical.empty else historical
    rolling_elo = pd.DataFrame()
    if not historical.empty and not _has_elo_inputs(historical):
        rolling_elo = calculate_rolling_elo(historical)
    ratings = _rating_frame(team_ratings_df)
    venues = load_venues()
    rows: list[dict[str, Any]] = []
    relevance_cache: dict[tuple[str, str], dict[str, Any]] = {}
    team_relevance_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    elo_cache: dict[str, pd.DataFrame] = {}

    for _, match in matches.sort_values("date_utc").iterrows():
        asof_label = _date_label(match.get("date_utc"))
        if asof_label not in elo_cache:
            if _has_elo_inputs(historical_with_elo):
                elo_cache[asof_label] = _elo_asof_from_historical_pre_rows(historical_with_elo, asof_label)
            else:
                elo_cache[asof_label] = _elo_asof_from_rolling_rows(rolling_elo, asof_label)
        elo_asof = elo_cache[asof_label]
        teams_asof = get_team_ratings_asof(
            ratings,
            historical_with_elo,
            asof_label,
            teams=[str(match.get("home", "")), str(match.get("away", ""))],
            mode="baseline_manual",
            elo_asof_df=elo_asof,
        )
        competition = str(target_competition or "World Cup")
        rel_key = (asof_label, competition)
        if rel_key not in relevance_cache:
            relevance_cache[rel_key] = build_training_relevance_diagnostics(historical_with_elo, asof_label, target_competition=competition)
        relevance = relevance_cache[rel_key]
        home_team = str(match.get("home", ""))
        away_team = str(match.get("away", ""))
        home_key = (asof_label, competition, home_team)
        away_key = (asof_label, competition, away_team)
        if home_key not in team_relevance_cache:
            team_relevance_cache[home_key] = build_team_training_relevance_diagnostics(historical_with_elo, home_team, asof_label, competition)
        if away_key not in team_relevance_cache:
            team_relevance_cache[away_key] = build_team_training_relevance_diagnostics(historical_with_elo, away_team, asof_label, competition)
        home_rel = team_relevance_cache[home_key]
        away_rel = team_relevance_cache[away_key]
        for _, params in candidates.iterrows():
            cfg = model_config_from_parameter_row(params)
            prediction = run_model_for_backtest_match(match, "baseline_manual", teams_asof, venues, pd.DataFrame(), cfg=cfg)
            adjusted = apply_candidate_probability_adjustments(prediction, params)
            rows.append(
                _training_prediction_row(
                    adjusted,
                    match,
                    params,
                    relevance,
                    home_rel,
                    away_rel,
                    _evaluation_relevance_weight(match, end_date or matches["date_utc"].max(), target_competition=competition),
                )
            )

    predictions = pd.DataFrame(rows, columns=TRAINING_PREDICTION_COLUMNS)
    metrics = calculate_training_metrics(predictions)
    return predictions, metrics


def model_config_from_parameter_row(row: pd.Series | dict[str, Any]) -> ModelConfig:
    series = pd.Series(row)
    return ModelConfig(
        base_total_goals=coerce_float(series.get("base_total_goals"), ModelConfig().base_total_goals),
        elo_xg_weight=coerce_float(series.get("elo_xg_weight"), ModelConfig().elo_xg_weight),
        attack_weight=coerce_float(series.get("attack_weight"), ModelConfig().attack_weight),
        defense_weight=coerce_float(series.get("defense_weight"), ModelConfig().defense_weight),
        draw_inflation_factor=coerce_float(series.get("draw_inflation_factor"), 1.0),
        favorite_strength_scale=coerce_float(series.get("favorite_strength_scale"), 1.0),
        underdog_resistance_scale=coerce_float(series.get("underdog_resistance_scale"), 1.0),
        external_prior_weight=coerce_float(series.get("external_prior_weight"), 0.0),
        rating_gap_to_xg_scale=coerce_float(series.get("rating_gap_to_xg_scale"), 1.0),
        goal_correlation_adjustment=coerce_float(series.get("goal_correlation_adjustment"), 0.0),
    )


def apply_candidate_probability_adjustments(prediction: dict[str, Any], params: pd.Series | dict[str, Any]) -> dict[str, Any]:
    out = dict(prediction)
    series = pd.Series(params)
    home = coerce_float(out.get("home_win_prob"), 0.0)
    draw = coerce_float(out.get("draw_prob"), 0.0)
    away = coerce_float(out.get("away_win_prob"), 0.0)
    draw *= coerce_float(series.get("draw_inflation_factor"), 1.0)
    home, draw, away = _renormalise_1x2(home, draw, away)

    favorite_scale = coerce_float(series.get("favorite_strength_scale"), 1.0)
    underdog_scale = coerce_float(series.get("underdog_resistance_scale"), 1.0)
    if favorite_scale != 1.0:
        home, draw, away = _scale_favorite(home, draw, away, favorite_scale)
    if underdog_scale != 1.0:
        home, draw, away = _scale_underdog(home, draw, away, underdog_scale)
    home, draw, away = _renormalise_1x2(home, draw, away)

    corr = coerce_float(series.get("goal_correlation_adjustment"), 0.0)
    over = clamp(coerce_float(out.get("over_2_5_prob"), 0.0) + corr, 0.01, 0.99)
    btts = clamp(coerce_float(out.get("btts_yes_prob"), 0.0) + corr * 0.5, 0.01, 0.99)
    actual = str(out.get("actual_result", ""))
    out.update(
        {
            "home_win_prob": home,
            "draw_prob": draw,
            "away_win_prob": away,
            "actual_result_prob": _actual_prob_from_values(home, draw, away, actual),
            "most_likely_result": max({"home_win": home, "draw": draw, "away_win": away}.items(), key=lambda item: item[1])[0],
            "over_2_5_prob": over,
            "under_2_5_prob": 1.0 - over,
            "btts_yes_prob": btts,
            "btts_no_prob": 1.0 - btts,
        }
    )
    out["log_loss_1x2"] = -math.log(max(min(coerce_float(out["actual_result_prob"], 0.0), 0.999), 0.001))
    return out


def calculate_training_metrics(predictions_df: pd.DataFrame | None) -> pd.DataFrame:
    predictions = predictions_df.copy() if predictions_df is not None else pd.DataFrame(columns=TRAINING_PREDICTION_COLUMNS)
    if predictions.empty:
        return pd.DataFrame(columns=TRAINING_METRIC_COLUMNS)
    rows = []
    for parameter_set_id, group in predictions.groupby("parameter_set_id", dropna=False):
        group = predictions.loc[predictions["parameter_set_id"].astype(str) == parameter_set_id].copy()
        weights = pd.to_numeric(group.get("evaluation_relevance_weight", 1.0), errors="coerce").fillna(1.0).clip(lower=0.10)
        actual = group["actual_result"].astype(str)
        home_prob = pd.to_numeric(group["home_win_prob"], errors="coerce").fillna(0.0)
        draw_prob = pd.to_numeric(group["draw_prob"], errors="coerce").fillna(0.0)
        away_prob = pd.to_numeric(group["away_win_prob"], errors="coerce").fillna(0.0)
        brier = (
            (home_prob - (actual == "home_win").astype(float)) ** 2
            + (draw_prob - (actual == "draw").astype(float)) ** 2
            + (away_prob - (actual == "away_win").astype(float)) ** 2
        )
        log_loss = pd.to_numeric(group["log_loss_1x2"], errors="coerce")
        accuracy = (group["most_likely_result"].astype(str) == actual).astype(float)
        actual_prob = pd.to_numeric(group["actual_result_prob"], errors="coerce")
        home_goals = pd.to_numeric(group["home_goals"], errors="coerce").fillna(0.0)
        away_goals = pd.to_numeric(group["away_goals"], errors="coerce").fillna(0.0)
        total_goals = home_goals + away_goals
        over_actual = (total_goals > 2.5).astype(float)
        btts_actual = ((home_goals > 0) & (away_goals > 0)).astype(float)
        over_brier = (pd.to_numeric(group["over_2_5_prob"], errors="coerce").fillna(0.0) - over_actual) ** 2
        btts_brier = (pd.to_numeric(group["btts_yes_prob"], errors="coerce").fillna(0.0) - btts_actual) ** 2
        total_goals_mae = (
            pd.to_numeric(group["home_xg"], errors="coerce").fillna(0.0)
            + pd.to_numeric(group["away_xg"], errors="coerce").fillna(0.0)
            - total_goals
        ).abs()
        rows.append(
            {
                "parameter_set_id": parameter_set_id,
                "n_matches": int(group["match_id"].nunique()),
                "brier_1x2": _weighted_mean(brier, weights),
                "log_loss_1x2": _weighted_mean(log_loss, weights),
                "accuracy": _weighted_mean(accuracy, weights),
                "mean_actual_result_prob": _weighted_mean(actual_prob, weights),
                "over_2_5_brier": _weighted_mean(over_brier, weights),
                "btts_brier": _weighted_mean(btts_brier, weights),
                "total_goals_mae": _weighted_mean(total_goals_mae, weights),
                "calibration_error": _actual_probability_calibration_error(group),
                "matches_used_total": float(pd.to_numeric(group["matches_used_total"], errors="coerce").mean()),
                "world_cup_matches_used": float(pd.to_numeric(group["world_cup_matches_used"], errors="coerce").mean()),
                "qualifier_matches_used": float(pd.to_numeric(group["qualifier_matches_used"], errors="coerce").mean()),
                "friendly_matches_used": float(pd.to_numeric(group["friendly_matches_used"], errors="coerce").mean()),
                "weighted_match_count": float(pd.to_numeric(group["weighted_match_count"], errors="coerce").mean()),
                "mean_relevance_weight": float(pd.to_numeric(group["mean_relevance_weight"], errors="coerce").mean()),
                "latest_match_used": _max_date_label(group["latest_match_used"]),
                "oldest_match_used": _min_date_label(group["oldest_match_used"]),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics = evaluate_candidate_promotion(metrics)
    return metrics[TRAINING_METRIC_COLUMNS].sort_values(["promotion_status", "brier_1x2", "log_loss_1x2"]).reset_index(drop=True)


def evaluate_candidate_promotion(metrics_df: pd.DataFrame | None, baseline_parameter_set_id: str = "baseline_current") -> pd.DataFrame:
    metrics = metrics_df.copy() if metrics_df is not None else pd.DataFrame(columns=TRAINING_METRIC_COLUMNS)
    if metrics.empty:
        return pd.DataFrame(columns=TRAINING_METRIC_COLUMNS)
    baseline_rows = metrics.loc[metrics["parameter_set_id"].astype(str) == baseline_parameter_set_id]
    if baseline_rows.empty:
        metrics["beats_current_baseline"] = False
        metrics["promotion_status"] = "no_baseline_comparison"
        return metrics
    baseline = baseline_rows.iloc[0]
    statuses = []
    beats = []
    for _, row in metrics.iterrows():
        is_baseline = str(row["parameter_set_id"]) == baseline_parameter_set_id
        improves = (
            not is_baseline
            and coerce_float(row.get("brier_1x2"), 999.0) < coerce_float(baseline.get("brier_1x2"), 999.0)
            and coerce_float(row.get("log_loss_1x2"), 999.0) < coerce_float(baseline.get("log_loss_1x2"), 999.0)
        )
        over_ok = coerce_float(row.get("over_2_5_brier"), 999.0) <= coerce_float(baseline.get("over_2_5_brier"), 999.0) * 1.05
        sample_ok = int(coerce_float(row.get("n_matches"), 0.0)) >= 30
        beats.append(bool(improves))
        if is_baseline:
            statuses.append("current_primary_baseline")
        elif improves and over_ok and sample_ok:
            statuses.append("promotion_candidate")
        elif improves and not sample_ok:
            statuses.append("not_promoted_sample_too_small")
        elif improves and not over_ok:
            statuses.append("not_promoted_goal_market_worse")
        else:
            statuses.append("not_promoted_did_not_beat_baseline")
    metrics["beats_current_baseline"] = beats
    metrics["promotion_status"] = statuses
    return metrics


def _competition_relevance_weight(competition_type: str, days_since_match: int) -> float:
    if competition_type == "world_cup":
        return 1.50
    if competition_type == "world_cup_qualifier":
        return 1.25
    if competition_type == "continental_tournament":
        return 1.20
    if competition_type == "continental_qualifier":
        return 1.10
    if competition_type == "nations_league":
        return 1.00
    if competition_type == "friendly":
        return 0.70 if days_since_match <= 365 else 0.35
    return 0.50


def _recency_multiplier(days: int) -> float:
    if days <= 90:
        return 1.25
    if days <= 365:
        return 1.00
    if days <= 365 * 2:
        return 0.70
    if days <= 365 * 4:
        return 0.45
    return 0.20


def _tournament_cycle_multiplier(match_date: pd.Timestamp, prediction_ts: pd.Timestamp, target_competition: str) -> float:
    if "world cup" not in str(target_competition).lower() and "world_cup" not in str(target_competition).lower():
        return 1.0
    prediction_cycle_start = prediction_ts.year - ((prediction_ts.year - 2022) % 4)
    match_cycle_start = match_date.year - ((match_date.year - 2022) % 4)
    if match_cycle_start == prediction_cycle_start:
        return 1.15
    if match_cycle_start == prediction_cycle_start - 4:
        return 0.70
    return 0.40


def _historical_before_prediction(historical_matches_df: pd.DataFrame | None, prediction_date: Any) -> pd.DataFrame:
    if historical_matches_df is None or historical_matches_df.empty:
        return pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
    rows = historical_matches_df.copy()
    for col in HISTORICAL_MATCH_COLUMNS:
        if col not in rows.columns:
            rows[col] = pd.NA
    dates = pd.to_datetime(rows["date_utc"], errors="coerce")
    prediction_ts = _as_naive_timestamp(prediction_date)
    return rows.loc[dates.notna() & (dates < prediction_ts)].copy()


def _add_relevance_weights(rows: pd.DataFrame, prediction_date: Any, target_competition: str) -> pd.DataFrame:
    out = rows.copy()
    prediction_ts = _as_naive_timestamp(prediction_date)
    dates = pd.to_datetime(out["date_utc"], errors="coerce")
    days = (prediction_ts - dates).dt.days.clip(lower=0)
    comp = out["competition_type"].astype(str).str.lower().fillna("")

    competition_weight = pd.Series(0.50, index=out.index, dtype=float)
    competition_weight.loc[comp.eq("world_cup")] = 1.50
    competition_weight.loc[comp.eq("world_cup_qualifier")] = 1.25
    competition_weight.loc[comp.eq("continental_tournament")] = 1.20
    competition_weight.loc[comp.eq("continental_qualifier")] = 1.10
    competition_weight.loc[comp.eq("nations_league")] = 1.00
    recent_friendly = comp.eq("friendly") & (days <= 365)
    old_friendly = comp.eq("friendly") & (days > 365)
    competition_weight.loc[recent_friendly] = 0.70
    competition_weight.loc[old_friendly] = 0.35

    recency = pd.Series(0.20, index=out.index, dtype=float)
    recency.loc[days <= 365 * 4] = 0.45
    recency.loc[days <= 365 * 2] = 0.70
    recency.loc[days <= 365] = 1.00
    recency.loc[days <= 90] = 1.25

    cycle = pd.Series(1.0, index=out.index, dtype=float)
    if "world cup" in str(target_competition).lower() or "world_cup" in str(target_competition).lower():
        prediction_cycle_start = prediction_ts.year - ((prediction_ts.year - 2022) % 4)
        match_cycle_start = dates.dt.year - ((dates.dt.year - 2022) % 4)
        cycle[:] = 0.40
        cycle.loc[match_cycle_start == prediction_cycle_start - 4] = 0.70
        cycle.loc[match_cycle_start == prediction_cycle_start] = 1.15

    out["relevance_weight"] = (competition_weight * recency * cycle).clip(lower=0.10, upper=2.00)
    return out


def _elo_asof_from_historical_pre_rows(historical_matches_df: pd.DataFrame | None, as_of_date: Any) -> pd.DataFrame:
    if historical_matches_df is None or historical_matches_df.empty:
        return pd.DataFrame(columns=ELO_ASOF_COLUMNS)
    rows = historical_matches_df.copy()
    for col in ["team", "date_utc", "team_elo_pre"]:
        if col not in rows.columns:
            return pd.DataFrame(columns=ELO_ASOF_COLUMNS)
    rows["date_utc"] = pd.to_datetime(rows["date_utc"], errors="coerce")
    rows["team_elo_pre"] = pd.to_numeric(rows["team_elo_pre"], errors="coerce")
    asof_ts = _as_naive_timestamp(as_of_date)
    rows = rows.loc[rows["date_utc"].notna() & (rows["date_utc"] < asof_ts) & rows["team_elo_pre"].notna()].copy()
    if rows.empty:
        return pd.DataFrame(columns=ELO_ASOF_COLUMNS)
    rows = rows.sort_values(["team", "date_utc", "match_id"])
    out = []
    for team, group in rows.groupby("team", dropna=False):
        latest = group.iloc[-1]
        out.append(
            {
                "team": str(team),
                "elo_asof": coerce_float(latest.get("team_elo_pre"), 1500.0),
                "matches_used_for_elo": int(len(group)),
                "latest_match_used_for_elo": latest["date_utc"].date().isoformat(),
                "as_of_date": asof_ts.date().isoformat(),
            }
        )
    return pd.DataFrame(out, columns=ELO_ASOF_COLUMNS)


def _empty_relevance_diagnostics() -> dict[str, Any]:
    return {
        "matches_used_total": 0,
        "world_cup_matches_used": 0,
        "qualifier_matches_used": 0,
        "friendly_matches_used": 0,
        "weighted_match_count": 0.0,
        "mean_relevance_weight": 0.0,
        "latest_match_used": "",
        "oldest_match_used": "",
    }


def _candidate_row(*values: Any) -> dict[str, Any]:
    return dict(zip(MODEL_PARAMETER_COLUMNS, values, strict=False))


def _ensure_candidate_params(candidate_params_df: pd.DataFrame | None) -> pd.DataFrame:
    out = candidate_params_df.copy() if candidate_params_df is not None else generate_candidate_parameter_sets()
    for col in MODEL_PARAMETER_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[MODEL_PARAMETER_COLUMNS].copy()


def _training_prediction_row(
    prediction: dict[str, Any],
    match: pd.Series,
    params: pd.Series,
    relevance: dict[str, Any],
    home_rel: dict[str, Any],
    away_rel: dict[str, Any],
    evaluation_relevance_weight: float,
) -> dict[str, Any]:
    return {
        "parameter_set_id": str(params.get("parameter_set_id", "")),
        "evaluation_relevance_weight": float(evaluation_relevance_weight),
        "match_id": prediction.get("match_id", match.get("match_id", "")),
        "date_utc": prediction.get("date_utc", match.get("date_utc", "")),
        "home": prediction.get("home", match.get("home", "")),
        "away": prediction.get("away", match.get("away", "")),
        "home_goals": prediction.get("home_goals", match.get("home_goals", pd.NA)),
        "away_goals": prediction.get("away_goals", match.get("away_goals", pd.NA)),
        "competition": prediction.get("competition", match.get("competition", "")),
        "actual_result": prediction.get("actual_result", ""),
        "home_win_prob": prediction.get("home_win_prob", pd.NA),
        "draw_prob": prediction.get("draw_prob", pd.NA),
        "away_win_prob": prediction.get("away_win_prob", pd.NA),
        "actual_result_prob": prediction.get("actual_result_prob", pd.NA),
        "home_xg": prediction.get("home_xg", pd.NA),
        "away_xg": prediction.get("away_xg", pd.NA),
        "over_2_5_prob": prediction.get("over_2_5_prob", pd.NA),
        "btts_yes_prob": prediction.get("btts_yes_prob", pd.NA),
        "most_likely_result": prediction.get("most_likely_result", ""),
        "log_loss_1x2": prediction.get("log_loss_1x2", pd.NA),
        "lookahead_safe": True,
        **relevance,
        "home_weighted_training_matches": home_rel["weighted_training_matches"],
        "away_weighted_training_matches": away_rel["weighted_training_matches"],
        "home_recent_friendlies_used": home_rel["recent_friendlies_used"],
        "away_recent_friendlies_used": away_rel["recent_friendlies_used"],
        "home_qualifiers_used": home_rel["qualifiers_used"],
        "away_qualifiers_used": away_rel["qualifiers_used"],
        "home_mean_relevance_weight": home_rel["mean_relevance_weight"],
        "away_mean_relevance_weight": away_rel["mean_relevance_weight"],
        "warning": "",
    }


def _renormalise_1x2(home: float, draw: float, away: float) -> tuple[float, float, float]:
    home = max(home, 0.001)
    draw = max(draw, 0.001)
    away = max(away, 0.001)
    total = home + draw + away
    return home / total, draw / total, away / total


def _scale_favorite(home: float, draw: float, away: float, scale: float) -> tuple[float, float, float]:
    if home >= away:
        home *= scale
    else:
        away *= scale
    return _renormalise_1x2(home, draw, away)


def _scale_underdog(home: float, draw: float, away: float, scale: float) -> tuple[float, float, float]:
    if home < away:
        home *= scale
    else:
        away *= scale
    return _renormalise_1x2(home, draw, away)


def _actual_prob_from_values(home: float, draw: float, away: float, actual: str) -> float:
    if actual == "home_win":
        return home
    if actual == "draw":
        return draw
    if actual == "away_win":
        return away
    return 0.0


def _actual_probability_calibration_error(group: pd.DataFrame) -> float:
    if group.empty:
        return 0.0
    probs = pd.to_numeric(group["actual_result_prob"], errors="coerce").dropna()
    if probs.empty:
        return 0.0
    return float((1.0 - probs).mean())


def _evaluation_relevance_weight(match: pd.Series, reference_date: Any, target_competition: str = "World Cup") -> float:
    ref = pd.to_datetime(reference_date, errors="coerce")
    if pd.isna(ref):
        ref = pd.to_datetime(match.get("date_utc"), errors="coerce")
    if pd.isna(ref):
        return 1.0
    prediction_date = (ref + pd.Timedelta(days=1)).date().isoformat()
    competition_type = str(match.get("competition_type", "") or "")
    if not competition_type:
        competition_type = classify_competition_type(str(match.get("competition", "")))
    return calculate_match_relevance_weight(
        {
            "date_utc": match.get("date_utc", ""),
            "competition_type": competition_type,
        },
        prediction_date,
        target_competition=target_competition,
    )


def _weighted_mean(values: Any, weights: pd.Series) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    aligned_weights = pd.to_numeric(pd.Series(weights), errors="coerce").reindex(series.index).fillna(1.0)
    valid = series.notna() & aligned_weights.notna() & (aligned_weights > 0)
    if not valid.any():
        return 0.0
    return float((series.loc[valid] * aligned_weights.loc[valid]).sum() / aligned_weights.loc[valid].sum())


def _max_date_label(values: pd.Series) -> str:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    return "" if dates.empty else dates.max().date().isoformat()


def _min_date_label(values: pd.Series) -> str:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    return "" if dates.empty else dates.min().date().isoformat()


def _as_naive_datetime(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert("UTC").tz_localize(None)
