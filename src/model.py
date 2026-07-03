from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.climate import get_venue_environment, summarize_environment_adjustment
from src.model_policy import get_current_model_policy, model_policy_label
from src.odds import GENERATED_ODDS_SOURCE_ALIASES, decimal_odds_or_nan
from src.ratings import neutral_team_rating, rating_row_for_team
from src.utils import clamp, coerce_bool, coerce_float
from src.venue_features import altitude_log_penalty, venue_log_adjustments


MAX_GOALS = 7


def fair_odds(prob: float) -> float:
    if prob <= 0:
        return float("inf")
    return 1.0 / prob


@dataclass
class ModelConfig:
    base_total_goals: float = 2.50
    global_xg_calibration_multiplier: float = 1.00
    elo_xg_weight: float = 0.0032
    attack_weight: float = 0.55
    defense_weight: float = 0.45
    form_weight: float = 0.20
    environment_weight: float = 0.010
    neutral_home_bias: float = 0.00
    draw_inflation_factor: float = 1.00
    favorite_strength_scale: float = 1.00
    underdog_resistance_scale: float = 1.00
    external_prior_weight: float = 0.00
    rating_gap_to_xg_scale: float = 1.00
    goal_correlation_adjustment: float = 0.00
    max_goals: int = MAX_GOALS


def _team_row(teams: pd.DataFrame, team_name: str) -> pd.Series:
    row = rating_row_for_team(teams, team_name)
    if row.empty:
        return neutral_team_rating(team_name)
    return row


def _venue_environment(match_row: pd.Series, venues: pd.DataFrame | None) -> dict[str, Any]:
    venue_name = match_row.get("venue", "")
    date_utc = match_row.get("date_utc")
    time_utc = match_row.get("time_utc")
    return get_venue_environment(venue_name, date_utc, time_utc)


def environmental_adjustments(
    home: pd.Series,
    away: pd.Series,
    env: dict[str, Any],
    cfg: ModelConfig,
) -> dict[str, float | str]:
    """Conservative weather, altitude, and training-climate adjustment.

    Team effects change each side's log-xG slightly. Wind and precipitation
    mainly reduce total-goals quality. Roof-closed venues reduce weather impact
    by 85%; altitude is still present because the stadium elevation remains.
    """
    roof_closed = coerce_bool(env.get("roof_expected_closed", 0))
    weather_multiplier = 0.15 if roof_closed else 1.0

    effective_temp = coerce_float(env.get("effective_temp_c"), coerce_float(env.get("temp_c"), 22.0))
    effective_humidity = coerce_float(env.get("effective_humidity_pct"), coerce_float(env.get("humidity_pct"), 55.0))
    wind_kmh = coerce_float(env.get("effective_wind_kmh"), coerce_float(env.get("wind_kmh"), 0.0) * weather_multiplier)
    precipitation_mm = coerce_float(
        env.get("effective_precipitation_mm"),
        coerce_float(env.get("precipitation_mm"), 0.0) * weather_multiplier,
    )
    altitude_m = coerce_float(env.get("altitude_m"), 0.0)

    def team_penalty(team: pd.Series) -> float:
        training_temp = coerce_float(team.get("training_temp_c"), 20.0)
        training_humidity = coerce_float(team.get("training_humidity_pct"), 60.0)
        temp_gap = abs(effective_temp - training_temp)
        humidity_gap = abs(effective_humidity - training_humidity) / 10.0
        heat_stress = max(effective_temp - 28.0, 0.0) * 0.35
        humid_heat = max(effective_humidity - 70.0, 0.0) / 10.0 * max(effective_temp - 25.0, 0.0) * 0.08
        penalty = -cfg.environment_weight * weather_multiplier * (
            0.45 * temp_gap + 0.18 * humidity_gap + heat_stress + humid_heat
        )

        # Altitude mainly matters above roughly 1000-1200 m. Teams with known
        # high-altitude familiarity receive a smaller penalty, while venue-host
        # advantage is handled separately in expected_goals.
        penalty += altitude_log_penalty(team, altitude_m, team.get("team", ""))
        return clamp(penalty, -0.14, 0.04)

    total_drag = 0.0
    total_drag += -cfg.environment_weight * weather_multiplier * max(wind_kmh - 12.0, 0.0) * 0.55
    total_drag += -cfg.environment_weight * weather_multiplier * max(precipitation_mm, 0.0) * 0.80
    total_drag += -cfg.environment_weight * weather_multiplier * max(effective_temp - 30.0, 0.0) * 0.35
    if altitude_m > 1200.0:
        total_drag += -0.000025 * (altitude_m - 1200.0)
    total_drag = clamp(total_drag, -0.16, 0.04)

    return {
        "home_environment_log_adj": float(team_penalty(home)),
        "away_environment_log_adj": float(team_penalty(away)),
        "total_environment_log_adj": float(total_drag),
        "weather_impact_multiplier": float(weather_multiplier),
        "roof_expected_closed": float(roof_closed),
        "venue_temp_c": coerce_float(env.get("temp_c"), 22.0),
        "effective_temp_c": effective_temp,
        "venue_humidity_pct": coerce_float(env.get("humidity_pct"), 55.0),
        "effective_humidity_pct": effective_humidity,
        "altitude_m": altitude_m,
        "wind_kmh": coerce_float(env.get("wind_kmh"), 0.0),
        "effective_wind_kmh": wind_kmh,
        "precipitation_mm": coerce_float(env.get("precipitation_mm"), 0.0),
        "effective_precipitation_mm": precipitation_mm,
        "environment_note": summarize_environment_adjustment(env),
    }


def expected_goals(
    home: pd.Series,
    away: pd.Series,
    env: dict[str, Any],
    cfg: ModelConfig,
) -> Tuple[float, float, Dict[str, Any]]:
    elo_diff = coerce_float(home.get("elo"), 1700.0) - coerce_float(away.get("elo"), 1700.0)

    home_attack = coerce_float(home.get("attack"), 0.55) - 0.60
    away_attack = coerce_float(away.get("attack"), 0.55) - 0.60
    home_def = coerce_float(home.get("defense"), 0.55) - 0.60
    away_def = coerce_float(away.get("defense"), 0.55) - 0.60
    home_form = coerce_float(home.get("recent_form"), 0.55) - 0.60
    away_form = coerce_float(away.get("recent_form"), 0.55) - 0.60

    env_adj = environmental_adjustments(home, away, env, cfg)
    venue_adj = venue_log_adjustments(home.get("team", ""), away.get("team", ""), env)

    h_neutral_log = cfg.neutral_home_bias
    a_neutral_log = -cfg.neutral_home_bias
    h_rating_log = (
        h_neutral_log
        + cfg.elo_xg_weight * cfg.rating_gap_to_xg_scale * elo_diff
        + cfg.attack_weight * home_attack
        - cfg.defense_weight * away_def
    )
    a_rating_log = (
        a_neutral_log
        - cfg.elo_xg_weight * cfg.rating_gap_to_xg_scale * elo_diff
        + cfg.attack_weight * away_attack
        - cfg.defense_weight * home_def
    )
    h_base_log = h_rating_log + cfg.form_weight * home_form
    a_base_log = a_rating_log + cfg.form_weight * away_form

    h_log = h_base_log + float(env_adj["home_environment_log_adj"]) + float(venue_adj["home_venue_log_adj"])
    a_log = a_base_log + float(env_adj["away_environment_log_adj"]) + float(venue_adj["away_venue_log_adj"])

    base_home_xg = math.exp(h_neutral_log)
    base_away_xg = math.exp(a_neutral_log)
    rating_adjusted_home_xg = math.exp(h_rating_log)
    rating_adjusted_away_xg = math.exp(a_rating_log)
    base_raw_h = math.exp(h_base_log)
    base_raw_a = math.exp(a_base_log)
    raw_h = math.exp(h_log)
    raw_a = math.exp(a_log)
    weather_xg_multiplier = math.exp(float(env_adj["total_environment_log_adj"]))
    venue_adjusted_home_xg = raw_h * weather_xg_multiplier
    venue_adjusted_away_xg = raw_a * weather_xg_multiplier
    global_xg_multiplier = max(coerce_float(cfg.global_xg_calibration_multiplier, 1.0), 0.0)
    unclamped_hxg = raw_h * weather_xg_multiplier * global_xg_multiplier
    unclamped_axg = raw_a * weather_xg_multiplier * global_xg_multiplier
    hxg = clamp(unclamped_hxg, 0.15, 3.80)
    axg = clamp(unclamped_axg, 0.15, 3.80)

    old_adjusted_total = cfg.base_total_goals * weather_xg_multiplier
    old_total_scale = old_adjusted_total / (raw_h + raw_a)
    old_anchored_hxg = clamp(raw_h * old_total_scale, 0.15, 3.80)
    old_anchored_axg = clamp(raw_a * old_total_scale, 0.15, 3.80)

    components: Dict[str, Any] = {
        "elo_diff": elo_diff,
        "elo_xg_weight": cfg.elo_xg_weight,
        "rating_gap_to_xg_scale": cfg.rating_gap_to_xg_scale,
        "base_total_goals": cfg.base_total_goals,
        "global_xg_calibration_multiplier": global_xg_multiplier,
        "draw_inflation_factor": cfg.draw_inflation_factor,
        "favorite_strength_scale": cfg.favorite_strength_scale,
        "underdog_resistance_scale": cfg.underdog_resistance_scale,
        "external_prior_weight": cfg.external_prior_weight,
        "goal_correlation_adjustment": cfg.goal_correlation_adjustment,
        "home_base_log_strength": h_base_log,
        "away_base_log_strength": a_base_log,
        "home_final_log_strength": h_log,
        "away_final_log_strength": a_log,
        "base_home_xg": base_home_xg,
        "base_away_xg": base_away_xg,
        "rating_adjusted_home_xg": rating_adjusted_home_xg,
        "rating_adjusted_away_xg": rating_adjusted_away_xg,
        "form_adjusted_home_xg": base_raw_h,
        "form_adjusted_away_xg": base_raw_a,
        "venue_adjusted_home_xg": venue_adjusted_home_xg,
        "venue_adjusted_away_xg": venue_adjusted_away_xg,
        "final_home_xg": hxg,
        "final_away_xg": axg,
        "home_raw_xg_before_adjustments": base_raw_h,
        "away_raw_xg_before_adjustments": base_raw_a,
        "home_raw_xg": raw_h,
        "away_raw_xg": raw_a,
        "weather_xg_multiplier": weather_xg_multiplier,
        "home_unclamped_xg": unclamped_hxg,
        "away_unclamped_xg": unclamped_axg,
        "home_final_xg": hxg,
        "away_final_xg": axg,
        "adjusted_total_goals": hxg + axg,
        "old_base_total_goals_adjusted_total": old_adjusted_total,
        "old_base_total_goals_total_scale": old_total_scale,
        "old_base_total_goals_anchored_home_xg": old_anchored_hxg,
        "old_base_total_goals_anchored_away_xg": old_anchored_axg,
        "old_base_total_goals_anchored_total_xg": old_anchored_hxg + old_anchored_axg,
        "home_attack_input": coerce_float(home.get("attack"), 0.55),
        "away_attack_input": coerce_float(away.get("attack"), 0.55),
        "home_defense_input": coerce_float(home.get("defense"), 0.55),
        "away_defense_input": coerce_float(away.get("defense"), 0.55),
        **env_adj,
        **venue_adj,
    }
    return hxg, axg, components


def score_matrix(hxg: float, axg: float, max_goals: int = MAX_GOALS) -> pd.DataFrame:
    rows = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
            rows.append({"home_goals": h, "away_goals": a, "prob": p})
    mat = pd.DataFrame(rows)
    covered = mat["prob"].sum()
    if covered > 0:
        mat["prob"] = mat["prob"] / covered
    return mat


def outcome_probs(mat: pd.DataFrame) -> Dict[str, float]:
    h = mat["home_goals"]
    a = mat["away_goals"]
    probs = {
        "home_win": float(mat.loc[h > a, "prob"].sum()),
        "draw": float(mat.loc[h == a, "prob"].sum()),
        "away_win": float(mat.loc[h < a, "prob"].sum()),
        "over_2_5": float(mat.loc[(h + a) >= 3, "prob"].sum()),
        "under_2_5": float(mat.loc[(h + a) <= 2, "prob"].sum()),
        "over_3_5": float(mat.loc[(h + a) >= 4, "prob"].sum()),
        "under_3_5": float(mat.loc[(h + a) <= 3, "prob"].sum()),
        "btts_yes": float(mat.loc[(h > 0) & (a > 0), "prob"].sum()),
        "btts_no": float(mat.loc[(h == 0) | (a == 0), "prob"].sum()),
        "home_minus_1_5": float(mat.loc[(h - a) >= 2, "prob"].sum()),
        "away_plus_1_5": float(mat.loc[(h - a) < 2, "prob"].sum()),
        "away_minus_1_5": float(mat.loc[(a - h) >= 2, "prob"].sum()),
        "home_plus_1_5": float(mat.loc[(a - h) < 2, "prob"].sum()),
        "home_or_draw": float(mat.loc[h >= a, "prob"].sum()),
        "draw_or_away": float(mat.loc[h <= a, "prob"].sum()),
    }
    return probs


def market_probability_map(home_team: str, away_team: str, probs: Dict[str, float]) -> Dict[Tuple[str, str], float]:
    return {
        ("1X2", home_team): probs["home_win"],
        ("1X2", "Draw"): probs["draw"],
        ("1X2", away_team): probs["away_win"],
        ("Total", "Under 2.5"): probs["under_2_5"],
        ("Total", "Over 2.5"): probs["over_2_5"],
        ("Total", "Under 3.5"): probs["under_3_5"],
        ("Total", "Over 3.5"): probs["over_3_5"],
        ("BTTS", "Yes"): probs["btts_yes"],
        ("BTTS", "No"): probs["btts_no"],
        ("Handicap", f"{home_team} -1.5"): probs["home_minus_1_5"],
        ("Handicap", f"{away_team} +1.5"): probs["away_plus_1_5"],
        ("Handicap", f"{away_team} -1.5"): probs["away_minus_1_5"],
        ("Handicap", f"{home_team} +1.5"): probs["home_plus_1_5"],
        ("Double Chance", f"{home_team}/Draw"): probs["home_or_draw"],
        ("Double Chance", f"Draw/{away_team}"): probs["draw_or_away"],
    }


def alpha_table(
    match_id: str,
    home_team: str,
    away_team: str,
    probs: Dict[str, float],
    market_odds: pd.DataFrame,
    model_mode: str | None = None,
) -> pd.DataFrame:
    policy = get_current_model_policy()
    active_mode = model_mode or str(policy["primary_model_mode"])
    pmap = market_probability_map(home_team, away_team, probs)
    odds_df = market_odds.copy() if market_odds is not None else pd.DataFrame()
    for col in ["match_id", "market", "selection", "odds", "source", "last_updated"]:
        if col not in odds_df.columns:
            odds_df[col] = pd.NA
    odds_df["match_id"] = odds_df["match_id"].astype(str)

    rows = []
    for (market, selection), p in pmap.items():
        fair = fair_odds(p)
        m = odds_df[
            (odds_df["match_id"].astype(str) == str(match_id))
            & (odds_df["market"] == market)
            & (odds_df["selection"] == selection)
        ]
        odds = decimal_odds_or_nan(m["odds"].iloc[0]) if len(m) else np.nan
        ev = (p * odds - 1.0) if not np.isnan(odds) else np.nan
        rows.append(
            {
                "market": market,
                "selection": selection,
                "model_prob": p,
                "primary_model_probability": p,
                "behavior_diagnostic_probability": pd.NA,
                "behavior_probability_delta": pd.NA,
                "model_policy": model_policy_label(policy),
                "edge_source": "primary_model" if active_mode == policy["primary_model_mode"] else "diagnostic_behavior_view",
                "fair_odds": fair,
                "market_odds": odds,
                "alpha_ev": ev,
                "odds_source": m["source"].iloc[0] if len(m) else "",
                "odds_last_updated": m["last_updated"].iloc[0] if len(m) else "",
            }
        )
    out = pd.DataFrame(rows)
    out["model_prob_pct"] = 100 * out["model_prob"]
    out["alpha_ev_pct"] = 100 * out["alpha_ev"]
    return out.sort_values(["alpha_ev"], ascending=False, na_position="last").reset_index(drop=True)


def top_scorelines(mat: pd.DataFrame, home_team: str, away_team: str, n: int = 8) -> pd.DataFrame:
    out = mat.sort_values("prob", ascending=False).head(n).copy()
    out["score"] = out["home_goals"].astype(str) + "-" + out["away_goals"].astype(str)
    out["label"] = home_team + " " + out["score"] + " " + away_team
    out["prob_pct"] = 100 * out["prob"]
    return out[["label", "score", "prob", "prob_pct"]]


def model_confidence(
    home: pd.Series,
    away: pd.Series,
    env: dict[str, Any],
    match_odds: pd.DataFrame,
) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []

    qualities = [str(home.get("data_quality", "")), str(away.get("data_quality", ""))]
    if all("neutral_fallback" not in q for q in qualities):
        score += 2
        reasons.append("team ratings available")
    else:
        reasons.append("one or more teams used neutral rating fallback")

    if any("recent_match_history" in q for q in qualities):
        score += 2
        reasons.append("recent match history used for team features")
    elif all("manual" in q or "api" in q for q in qualities):
        score += 1
        reasons.append("recent form available as manual/API prior")
    else:
        reasons.append("recent form partly inferred")

    source = str(env.get("environment_source", ""))
    source_label = str(env.get("source_label", ""))
    if source_label == "API" or "Open-Meteo" in source:
        score += 2
        reasons.append("venue weather API available")
    elif source_label == "cache":
        score += 1
        reasons.append("venue weather available from cache")
    elif source_label == "local CSV" or source in {"venues_csv", "manual_csv", "data/venues.csv"} or "csv" in source:
        score += 1
        reasons.append("venue environment available from local CSV")
    else:
        reasons.append("venue environment used neutral fallback")

    if match_odds is not None and not match_odds.empty:
        generated_mask = _generated_benchmark_odds_mask(match_odds)
        if bool(generated_mask.all()):
            reasons.append("local benchmark odds available; alpha EV is benchmark-only")
        elif bool(generated_mask.any()):
            score += 1
            reasons.append("market odds available; local benchmark filled missing selections")
        else:
            score += 1
            reasons.append("market odds available")
    else:
        reasons.append("market odds missing; alpha EV left blank")

    fallback_count = sum("fallback" in q for q in qualities) + int("fallback" in source)
    if fallback_count:
        score -= fallback_count
        reasons.append(f"{fallback_count} fallback input group(s) used")

    if score >= 6:
        label = "High"
    elif score >= 3:
        label = "Moderate"
    else:
        label = "Low"

    return {"label": label, "score": score, "reasons": reasons}


def _generated_benchmark_odds_mask(match_odds: pd.DataFrame) -> pd.Series:
    if match_odds is None or match_odds.empty or "source" not in match_odds.columns:
        index = match_odds.index if match_odds is not None else pd.RangeIndex(0)
        return pd.Series(False, index=index)
    aliases = {source.lower() for source in GENERATED_ODDS_SOURCE_ALIASES}
    sources = match_odds["source"].fillna("").astype(str).str.strip().str.lower()
    return sources.isin(aliases)


def run_match_model(
    match_row: pd.Series,
    teams: pd.DataFrame,
    venues: pd.DataFrame,
    market_odds: pd.DataFrame,
    cfg: ModelConfig | None = None,
    model_mode: str | None = None,
) -> Dict[str, Any]:
    cfg = cfg or ModelConfig()
    policy = get_current_model_policy()
    active_mode = model_mode or str(getattr(teams, "attrs", {}).get("model_mode", policy["primary_model_mode"]))
    home_name = str(match_row["home"])
    away_name = str(match_row["away"])
    home = _team_row(teams, home_name)
    away = _team_row(teams, away_name)
    env = _venue_environment(match_row, venues)

    odds_df = market_odds if market_odds is not None else pd.DataFrame()
    if not odds_df.empty and "match_id" in odds_df.columns:
        match_odds = odds_df.loc[odds_df["match_id"].astype(str) == str(match_row["match_id"])].copy()
    else:
        match_odds = pd.DataFrame()

    hxg, axg, components = expected_goals(home, away, env, cfg)
    mat = score_matrix(hxg, axg, cfg.max_goals)
    probs = outcome_probs(mat)
    alpha = alpha_table(str(match_row["match_id"]), home_name, away_name, probs, odds_df, model_mode=active_mode)
    confidence = model_confidence(home, away, env, match_odds)

    return {
        "match_id": str(match_row["match_id"]),
        "home": home_name,
        "away": away_name,
        "hxg": hxg,
        "axg": axg,
        "environment": env,
        "components": components,
        "score_matrix": mat,
        "probs": probs,
        "alpha": alpha,
        "top_scores": top_scorelines(mat, home_name, away_name),
        "home_inputs": home,
        "away_inputs": away,
        "confidence": confidence,
        "model_mode": active_mode,
        "model_policy": policy,
    }
