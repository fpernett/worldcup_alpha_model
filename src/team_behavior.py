from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.behavior_calibration import (
    DEFAULT_BEHAVIOR_CONFIG,
    BehaviorConfig,
    build_opponent_quality_map,
    calculate_opponent_quality_modifier,
    calculate_schedule_strength_metrics,
    filter_recent_team_matches,
    reliability_grade,
    warning_text,
)
from src.cache import set_source_attrs, utc_now_iso
from src.config import DATA_DIR, SOURCE_LOCAL
from src.environment_response import calculate_environment_response
from src.historical_data import HISTORICAL_MATCH_COLUMNS, load_historical_matches
from src.recency import calculate_match_weight
from src.schema_registry import validate_team_behavior_schema
from src.utils import clamp, read_csv_with_columns


TEAM_BEHAVIOR_COLUMNS = [
    "team",
    "reference_date",
    "behavior_window_start",
    "behavior_window_end",
    "matches_available_all_time",
    "matches_used_recent",
    "oldest_match_used",
    "latest_match_used",
    "behavior_config_name",
    "n_matches",
    "weighted_goals_for",
    "weighted_goals_for_raw",
    "weighted_goals_for_adjusted",
    "weighted_goals_against",
    "weighted_goals_against_raw",
    "weighted_goals_against_adjusted",
    "weighted_goal_difference",
    "all_time_goals_for",
    "all_time_goals_against",
    "attack_index",
    "attack_index_raw",
    "attack_index_adjusted",
    "defense_index",
    "defense_index_raw",
    "defense_index_adjusted",
    "recent_form_index",
    "weighted_btts_rate",
    "weighted_over_2_5_rate",
    "clean_sheet_rate",
    "failed_to_score_rate",
    "environment_response_index",
    "environment_sample_size",
    "attack_data_quality",
    "defense_data_quality",
    "form_data_quality",
    "environment_data_quality",
    "overall_data_quality",
    "behavior_warning",
    "sample_size_warning",
    "staleness_warning",
    "opponent_quality_warning",
    "mean_opponent_elo_recent",
    "median_opponent_elo_recent",
    "min_opponent_elo_recent",
    "max_opponent_elo_recent",
    "opponent_elo_coverage_recent",
    "strong_opponent_match_count",
    "weak_opponent_match_count",
    "schedule_strength_label",
    "schedule_strength_warning",
    "opponent_adjustment_warning",
    "last_updated",
]


def load_team_behavior(path=None) -> pd.DataFrame:
    local_path = DATA_DIR / "team_behavior.csv" if path is None else path
    try:
        df = read_csv_with_columns(local_path, TEAM_BEHAVIOR_COLUMNS)
        out = _normalise_team_behavior(df)
        warnings = validate_team_behavior_schema(out)
        return set_source_attrs(
            out,
            SOURCE_LOCAL,
            _relative_data_path(local_path),
            _csv_last_modified(local_path),
            warning="; ".join(warnings) if warnings else None,
        )
    except Exception as exc:
        out = pd.DataFrame(columns=TEAM_BEHAVIOR_COLUMNS)
        return set_source_attrs(out, SOURCE_LOCAL, _relative_data_path(local_path), warning=f"team_behavior.csv read failed: {exc}")


def calculate_attack_behavior(
    matches_df: pd.DataFrame,
    team: str,
    reference_date: Any,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
    opponent_quality_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    view = _weighted_team_view(matches_df, team, reference_date, config, opponent_quality_map)
    if view.empty:
        return _empty_attack(team)

    raw_goals = _capped_goal_contribution(view["team_goals"], 1.0, config.goal_contribution_cap)
    adjusted_goals = _capped_goal_contribution(view["team_goals"], view["attack_quality_modifier"], config.goal_contribution_cap)
    weighted_goals_raw = _weighted_series(view, raw_goals)
    weighted_goals_adjusted = _weighted_series(view, adjusted_goals)
    weighted_xg = _weighted_series(view, view["team_xg"] * view["attack_quality_modifier"])
    weighted_shots = _weighted_mean(view, "shots_for")
    weighted_sot = _weighted_mean(view, "shots_on_target_for")
    scoring_rate_raw = _weighted_rate(view, view["team_goals"] > 0)
    multi_goal_rate_raw = _weighted_rate(view, view["team_goals"] >= 2)
    scoring_rate_adjusted = _weighted_series(view, (view["team_goals"] > 0).astype(float) * view["attack_quality_modifier"])
    multi_goal_rate_adjusted = _weighted_series(view, (view["team_goals"] >= 2).astype(float) * view["attack_quality_modifier"])
    scoring_rate_adjusted = clamp(_nan_to_default(scoring_rate_adjusted, 0.0), 0.0, 1.0)
    multi_goal_rate_adjusted = clamp(_nan_to_default(multi_goal_rate_adjusted, 0.0), 0.0, 1.0)
    scoring_rate = scoring_rate_adjusted
    multi_goal_rate = multi_goal_rate_adjusted
    first_goal_proxy = clamp(0.65 * _nan_to_default(scoring_rate, 0.0) + 0.35 * _nan_to_default(multi_goal_rate, 0.0), 0.0, 1.0)

    attack_index_raw = _weighted_composite(
        [
            (clamp(_nan_to_default(weighted_goals_raw, 1.15) / 2.65, 0.0, 1.0), 0.36, not np.isnan(weighted_goals_raw)),
            (_nan_to_default(scoring_rate_raw, 0.0), 0.26, not np.isnan(scoring_rate_raw)),
            (_nan_to_default(multi_goal_rate_raw, 0.0), 0.20, not np.isnan(multi_goal_rate_raw)),
            (clamp(_nan_to_default(weighted_xg, 1.15) / 2.50, 0.0, 1.0), 0.10, not np.isnan(weighted_xg)),
            (clamp(_nan_to_default(weighted_shots, 9.0) / 18.0, 0.0, 1.0), 0.04, not np.isnan(weighted_shots)),
            (clamp(_nan_to_default(weighted_sot, 3.5) / 7.0, 0.0, 1.0), 0.04, not np.isnan(weighted_sot)),
        ],
        default=0.5,
    )
    attack_index_adjusted = _weighted_composite(
        [
            (clamp(_nan_to_default(weighted_goals_adjusted, 1.15) / 2.65, 0.0, 1.0), 0.36, not np.isnan(weighted_goals_adjusted)),
            (_nan_to_default(scoring_rate, 0.0), 0.26, not np.isnan(scoring_rate)),
            (_nan_to_default(multi_goal_rate, 0.0), 0.20, not np.isnan(multi_goal_rate)),
            (clamp(_nan_to_default(weighted_xg, 1.15) / 2.50, 0.0, 1.0), 0.10, not np.isnan(weighted_xg)),
            (clamp(_nan_to_default(weighted_shots, 9.0) / 18.0, 0.0, 1.0), 0.04, not np.isnan(weighted_shots)),
            (clamp(_nan_to_default(weighted_sot, 3.5) / 7.0, 0.0, 1.0), 0.04, not np.isnan(weighted_sot)),
        ],
        default=0.5,
    )

    return {
        "team": team,
        "n_matches_attack": int(len(view)),
        "weighted_goals_for": _round_or_nan(weighted_goals_adjusted),
        "weighted_goals_for_raw": _round_or_nan(weighted_goals_raw),
        "weighted_goals_for_adjusted": _round_or_nan(weighted_goals_adjusted),
        "weighted_xg_for": _round_or_nan(weighted_xg),
        "weighted_shots_for": _round_or_nan(weighted_shots),
        "weighted_shots_on_target_for": _round_or_nan(weighted_sot),
        "scoring_rate": _round_or_nan(scoring_rate),
        "multi_goal_rate": _round_or_nan(multi_goal_rate),
        "first_goal_proxy": round(first_goal_proxy, 3),
        "attack_index": round(attack_index_adjusted, 3),
        "attack_index_raw": round(attack_index_raw, 3),
        "attack_index_adjusted": round(attack_index_adjusted, 3),
        "attack_data_quality": _recent_quality(view, reference_date),
    }


def calculate_defense_behavior(
    matches_df: pd.DataFrame,
    team: str,
    reference_date: Any,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
    opponent_quality_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    view = _weighted_team_view(matches_df, team, reference_date, config, opponent_quality_map)
    if view.empty:
        return _empty_defense(team)

    raw_against = _capped_goal_contribution(view["opponent_goals"], 1.0, config.goal_contribution_cap)
    adjusted_against = _capped_goal_contribution(view["opponent_goals"], view["defense_quality_modifier"], config.goal_contribution_cap)
    weighted_goals_against_raw = _weighted_series(view, raw_against)
    weighted_goals_against_adjusted = _weighted_series(view, adjusted_against)
    weighted_xg_against = _weighted_series(view, view["opponent_xg"] * view["defense_quality_modifier"])
    weighted_shots_against = _weighted_mean(view, "shots_against")
    weighted_sot_against = _weighted_mean(view, "shots_on_target_against")
    clean_sheet_rate_raw = _weighted_rate(view, view["opponent_goals"] == 0)
    conceded_rate = _weighted_rate(view, view["opponent_goals"] > 0)
    multi_conceded_rate_raw = _weighted_rate(view, view["opponent_goals"] >= 2)
    clean_sheet_rate_adjusted = _weighted_series(view, (view["opponent_goals"] == 0).astype(float) * view["attack_quality_modifier"])
    multi_conceded_rate_adjusted = _weighted_series(view, (view["opponent_goals"] >= 2).astype(float) * view["defense_quality_modifier"])
    clean_sheet_rate_adjusted = clamp(_nan_to_default(clean_sheet_rate_adjusted, 0.0), 0.0, 1.0)
    multi_conceded_rate_adjusted = clamp(_nan_to_default(multi_conceded_rate_adjusted, 0.0), 0.0, 1.0)
    clean_sheet_rate = clean_sheet_rate_adjusted
    multi_conceded_rate = multi_conceded_rate_adjusted

    defense_index_raw = _weighted_composite(
        [
            (1.0 - clamp(_nan_to_default(weighted_goals_against_raw, 1.10) / 2.55, 0.0, 1.0), 0.38, not np.isnan(weighted_goals_against_raw)),
            (_nan_to_default(clean_sheet_rate_raw, 0.0), 0.24, not np.isnan(clean_sheet_rate_raw)),
            (1.0 - _nan_to_default(multi_conceded_rate_raw, 0.0), 0.20, not np.isnan(multi_conceded_rate_raw)),
            (1.0 - clamp(_nan_to_default(weighted_xg_against, 1.10) / 2.45, 0.0, 1.0), 0.10, not np.isnan(weighted_xg_against)),
            (1.0 - clamp(_nan_to_default(weighted_shots_against, 9.0) / 18.0, 0.0, 1.0), 0.04, not np.isnan(weighted_shots_against)),
            (1.0 - clamp(_nan_to_default(weighted_sot_against, 3.5) / 7.0, 0.0, 1.0), 0.04, not np.isnan(weighted_sot_against)),
        ],
        default=0.5,
    )
    defense_index_adjusted = _weighted_composite(
        [
            (1.0 - clamp(_nan_to_default(weighted_goals_against_adjusted, 1.10) / 2.55, 0.0, 1.0), 0.38, not np.isnan(weighted_goals_against_adjusted)),
            (_nan_to_default(clean_sheet_rate, 0.0), 0.24, not np.isnan(clean_sheet_rate)),
            (1.0 - _nan_to_default(multi_conceded_rate, 0.0), 0.20, not np.isnan(multi_conceded_rate)),
            (1.0 - clamp(_nan_to_default(weighted_xg_against, 1.10) / 2.45, 0.0, 1.0), 0.10, not np.isnan(weighted_xg_against)),
            (1.0 - clamp(_nan_to_default(weighted_shots_against, 9.0) / 18.0, 0.0, 1.0), 0.04, not np.isnan(weighted_shots_against)),
            (1.0 - clamp(_nan_to_default(weighted_sot_against, 3.5) / 7.0, 0.0, 1.0), 0.04, not np.isnan(weighted_sot_against)),
        ],
        default=0.5,
    )

    return {
        "team": team,
        "n_matches_defense": int(len(view)),
        "weighted_goals_against": _round_or_nan(weighted_goals_against_adjusted),
        "weighted_goals_against_raw": _round_or_nan(weighted_goals_against_raw),
        "weighted_goals_against_adjusted": _round_or_nan(weighted_goals_against_adjusted),
        "weighted_xg_against": _round_or_nan(weighted_xg_against),
        "weighted_shots_against": _round_or_nan(weighted_shots_against),
        "weighted_shots_on_target_against": _round_or_nan(weighted_sot_against),
        "clean_sheet_rate": _round_or_nan(clean_sheet_rate),
        "conceded_rate": _round_or_nan(conceded_rate),
        "multi_conceded_rate": _round_or_nan(multi_conceded_rate),
        "defense_index": round(defense_index_adjusted, 3),
        "defense_index_raw": round(defense_index_raw, 3),
        "defense_index_adjusted": round(defense_index_adjusted, 3),
        "defense_data_quality": _recent_quality(view, reference_date),
    }


def calculate_recent_form_behavior(
    matches_df: pd.DataFrame,
    team: str,
    reference_date: Any,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
    opponent_quality_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    view = _weighted_team_view(matches_df, team, reference_date, config, opponent_quality_map)
    if view.empty:
        return {
            "team": team,
            "weighted_points_per_match": np.nan,
            "weighted_win_rate": np.nan,
            "weighted_draw_rate": np.nan,
            "weighted_loss_rate": np.nan,
            "weighted_goal_difference": np.nan,
            "recent_form_index": 0.5,
            "form_data_quality": "insufficient",
        }

    goals_for = view["team_goals"]
    goals_against = view["opponent_goals"]
    win_mask = goals_for > goals_against
    draw_mask = goals_for == goals_against
    loss_mask = goals_for < goals_against
    points = pd.Series(np.select([win_mask, draw_mask, loss_mask], [3.0, 1.0, 0.0], default=np.nan), index=view.index)
    adjusted_points = (points * view["attack_quality_modifier"]).clip(lower=0.0, upper=3.0)
    weighted_ppm = _weighted_series(view, adjusted_points)
    weighted_win_rate = clamp(_nan_to_default(_weighted_series(view, win_mask.astype(float) * view["attack_quality_modifier"]), 0.0), 0.0, 1.0)
    weighted_draw_rate = _weighted_rate(view, draw_mask)
    weighted_loss_rate = _weighted_rate(view, loss_mask)
    adjusted_gd = _capped_goal_contribution(goals_for, view["attack_quality_modifier"], config.goal_contribution_cap) - _capped_goal_contribution(
        goals_against,
        view["defense_quality_modifier"],
        config.goal_contribution_cap,
    )
    weighted_goal_difference = _weighted_series(view, adjusted_gd)
    gd_score = clamp((_nan_to_default(weighted_goal_difference, 0.0) + 2.0) / 4.0, 0.0, 1.0)
    recent_form_index = clamp(
        0.55 * clamp(_nan_to_default(weighted_ppm, 1.5) / 3.0, 0.0, 1.0)
        + 0.25 * _nan_to_default(weighted_win_rate, 0.0)
        + 0.20 * gd_score,
        0.0,
        1.0,
    )
    return {
        "team": team,
        "weighted_points_per_match": _round_or_nan(weighted_ppm),
        "weighted_win_rate": _round_or_nan(weighted_win_rate),
        "weighted_draw_rate": _round_or_nan(weighted_draw_rate),
        "weighted_loss_rate": _round_or_nan(weighted_loss_rate),
        "weighted_goal_difference": _round_or_nan(weighted_goal_difference),
        "recent_form_index": round(recent_form_index, 3),
        "form_data_quality": _recent_quality(view, reference_date),
    }


def calculate_goal_market_behavior(
    matches_df: pd.DataFrame,
    team: str,
    reference_date: Any,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
    opponent_quality_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    view = _weighted_team_view(matches_df, team, reference_date, config, opponent_quality_map)
    if view.empty:
        return {
            "team": team,
            "weighted_btts_rate": np.nan,
            "weighted_over_1_5_rate": np.nan,
            "weighted_over_2_5_rate": np.nan,
            "weighted_under_2_5_rate": np.nan,
            "weighted_clean_sheet_rate": np.nan,
            "weighted_failed_to_score_rate": np.nan,
            "goal_market_data_quality": "insufficient",
        }

    total_goals = view["team_goals"] + view["opponent_goals"]
    return {
        "team": team,
        "weighted_btts_rate": _round_or_nan(_weighted_rate(view, (view["team_goals"] > 0) & (view["opponent_goals"] > 0))),
        "weighted_over_1_5_rate": _round_or_nan(_weighted_rate(view, total_goals >= 2)),
        "weighted_over_2_5_rate": _round_or_nan(_weighted_rate(view, total_goals >= 3)),
        "weighted_under_2_5_rate": _round_or_nan(_weighted_rate(view, total_goals <= 2)),
        "weighted_clean_sheet_rate": _round_or_nan(_weighted_rate(view, view["opponent_goals"] == 0)),
        "weighted_failed_to_score_rate": _round_or_nan(_weighted_rate(view, view["team_goals"] == 0)),
        "goal_market_data_quality": _recent_quality(view, reference_date),
    }


def build_team_behavior_table(
    matches_df: pd.DataFrame,
    teams_df: pd.DataFrame | None,
    reference_date: Any,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> pd.DataFrame:
    matches = _normalise_historical_for_behavior(matches_df)
    teams = _team_list(matches, teams_df)
    if not teams:
        return pd.DataFrame(columns=TEAM_BEHAVIOR_COLUMNS)

    opponent_quality_map = build_opponent_quality_map(matches, teams_df)
    rows = []
    now = utc_now_iso()
    reference_label = pd.Timestamp(reference_date).date().isoformat()
    window_start = (pd.Timestamp(reference_date).normalize() - pd.DateOffset(years=config.lookback_years)).date().isoformat()

    for team in teams:
        all_time = _team_all_time_view(matches, team)
        recent = _weighted_team_view(matches, team, reference_date, config, opponent_quality_map)
        attack = calculate_attack_behavior(matches, team, reference_date, config, opponent_quality_map)
        defense = calculate_defense_behavior(matches, team, reference_date, config, opponent_quality_map)
        form = calculate_recent_form_behavior(matches, team, reference_date, config, opponent_quality_map)
        goals = calculate_goal_market_behavior(matches, team, reference_date, config, opponent_quality_map)
        env = calculate_environment_response(recent, team, reference_date)
        schedule = calculate_schedule_strength_metrics(recent, config)
        n_matches = int(len(recent))
        latest = _date_label(recent["date_utc"].max()) if not recent.empty else ""
        oldest = _date_label(recent["date_utc"].min()) if not recent.empty else ""
        quality, sample_warning, staleness_warning = reliability_grade(n_matches, latest, reference_date)
        opponent_warning = _opponent_quality_warning(recent)
        all_time_goals_for = _round_or_nan(pd.to_numeric(all_time.get("team_goals"), errors="coerce").mean()) if not all_time.empty else np.nan
        all_time_goals_against = _round_or_nan(pd.to_numeric(all_time.get("opponent_goals"), errors="coerce").mean()) if not all_time.empty else np.nan
        stability_warning = _behavior_stability_warning(
            attack["weighted_goals_for"],
            defense["weighted_goals_against"],
            all_time_goals_for,
            all_time_goals_against,
        )
        opponent_adjustment_warning = _opponent_adjustment_warning(attack, defense)
        behavior_warning = warning_text(
            sample_warning,
            staleness_warning,
            opponent_warning,
            stability_warning,
            schedule["schedule_strength_warning"],
            opponent_adjustment_warning,
        )

        rows.append(
            {
                "team": team,
                "reference_date": reference_label,
                "behavior_window_start": window_start,
                "behavior_window_end": reference_label,
                "matches_available_all_time": int(len(all_time)),
                "matches_used_recent": n_matches,
                "oldest_match_used": oldest,
                "latest_match_used": latest,
                "behavior_config_name": config.name,
                "n_matches": n_matches,
                "weighted_goals_for": attack["weighted_goals_for"],
                "weighted_goals_for_raw": attack["weighted_goals_for_raw"],
                "weighted_goals_for_adjusted": attack["weighted_goals_for_adjusted"],
                "weighted_goals_against": defense["weighted_goals_against"],
                "weighted_goals_against_raw": defense["weighted_goals_against_raw"],
                "weighted_goals_against_adjusted": defense["weighted_goals_against_adjusted"],
                "weighted_goal_difference": form["weighted_goal_difference"],
                "all_time_goals_for": all_time_goals_for,
                "all_time_goals_against": all_time_goals_against,
                "attack_index": attack["attack_index"],
                "attack_index_raw": attack["attack_index_raw"],
                "attack_index_adjusted": attack["attack_index_adjusted"],
                "defense_index": defense["defense_index"],
                "defense_index_raw": defense["defense_index_raw"],
                "defense_index_adjusted": defense["defense_index_adjusted"],
                "recent_form_index": form["recent_form_index"],
                "weighted_btts_rate": goals["weighted_btts_rate"],
                "weighted_over_2_5_rate": goals["weighted_over_2_5_rate"],
                "clean_sheet_rate": defense["clean_sheet_rate"],
                "failed_to_score_rate": goals["weighted_failed_to_score_rate"],
                "environment_response_index": env["environment_response_index"],
                "environment_sample_size": env["environment_sample_size"],
                "attack_data_quality": quality,
                "defense_data_quality": quality,
                "form_data_quality": quality,
                "environment_data_quality": env["environment_data_quality"],
                "overall_data_quality": quality,
                "behavior_warning": behavior_warning,
                "sample_size_warning": sample_warning,
                "staleness_warning": staleness_warning,
                "opponent_quality_warning": opponent_warning,
                "mean_opponent_elo_recent": schedule["mean_opponent_elo_recent"],
                "median_opponent_elo_recent": schedule["median_opponent_elo_recent"],
                "min_opponent_elo_recent": schedule["min_opponent_elo_recent"],
                "max_opponent_elo_recent": schedule["max_opponent_elo_recent"],
                "opponent_elo_coverage_recent": schedule["opponent_elo_coverage_recent"],
                "strong_opponent_match_count": schedule["strong_opponent_match_count"],
                "weak_opponent_match_count": schedule["weak_opponent_match_count"],
                "schedule_strength_label": schedule["schedule_strength_label"],
                "schedule_strength_warning": schedule["schedule_strength_warning"],
                "opponent_adjustment_warning": opponent_adjustment_warning,
                "last_updated": now,
            }
        )

    return _normalise_team_behavior(pd.DataFrame(rows, columns=TEAM_BEHAVIOR_COLUMNS))


def rebuild_team_behavior_csv(
    matches_df: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
    reference_date: Any | None = None,
    path=None,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> pd.DataFrame:
    matches = load_historical_matches() if matches_df is None else matches_df
    reference = reference_date or pd.Timestamp.now(tz="UTC")
    behavior = build_team_behavior_table(matches, teams_df, reference, config)
    local_path = DATA_DIR / "team_behavior.csv" if path is None else path
    behavior.to_csv(local_path, index=False)
    return set_source_attrs(behavior, SOURCE_LOCAL, _relative_data_path(local_path), _csv_last_modified(local_path))


def _weighted_team_view(
    matches_df: pd.DataFrame | None,
    team: str,
    reference_date: Any,
    config: BehaviorConfig,
    opponent_quality_map: dict[str, float] | None,
) -> pd.DataFrame:
    matches = _normalise_historical_for_behavior(matches_df)
    if matches.empty:
        return matches
    view = filter_recent_team_matches(matches, team, reference_date, config)
    if view.empty:
        return view
    quality_map = opponent_quality_map or {}
    view["opponent_elo_resolved"] = view.apply(lambda row: _resolved_opponent_elo(row, quality_map), axis=1)
    view["opponent_quality_missing"] = view.apply(lambda row: _opponent_quality_missing(row, quality_map), axis=1)
    view["match_weight"] = view.apply(lambda row: calculate_match_weight(row, reference_date, config), axis=1)
    view["match_weight"] = pd.to_numeric(view["match_weight"], errors="coerce").fillna(0.0)
    if float(view["match_weight"].sum()) <= 0:
        view["match_weight"] = 1.0
    view["attack_quality_modifier"] = view["opponent_elo_resolved"].map(
        lambda elo: calculate_opponent_quality_modifier(elo, strength=config.opponent_adjustment_strength)
    )
    view["defense_quality_modifier"] = (1.0 / view["attack_quality_modifier"]).clip(0.75, 1.25)
    return view.sort_values("date_utc", ascending=False).reset_index(drop=True)


def _normalise_historical_for_behavior(matches_df: pd.DataFrame | None) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        return pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
    if matches_df.attrs.get("behavior_normalised"):
        return matches_df
    out = matches_df.copy()
    for col in HISTORICAL_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce")
    for col in [
        "team_goals",
        "opponent_goals",
        "team_xg",
        "opponent_xg",
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
        "opponent_elo",
        "team_elo_pre",
        "temperature_c",
        "humidity_pct",
        "altitude_m",
        "wind_kmh",
        "precipitation_mm",
        "roof_closed",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date_utc", "team", "opponent", "team_goals", "opponent_goals"]).copy()
    out = out.sort_values("date_utc", ascending=False).reset_index(drop=True)
    out.attrs["behavior_normalised"] = True
    return out


def _normalise_team_behavior(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=TEAM_BEHAVIOR_COLUMNS)
    out = df.copy()
    for col in TEAM_BEHAVIOR_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[TEAM_BEHAVIOR_COLUMNS].copy()
    out = out.dropna(subset=["team"]).copy()
    numeric_cols = [
        "matches_available_all_time",
        "matches_used_recent",
        "n_matches",
        "weighted_goals_for",
        "weighted_goals_for_raw",
        "weighted_goals_for_adjusted",
        "weighted_goals_against",
        "weighted_goals_against_raw",
        "weighted_goals_against_adjusted",
        "weighted_goal_difference",
        "all_time_goals_for",
        "all_time_goals_against",
        "attack_index",
        "attack_index_raw",
        "attack_index_adjusted",
        "defense_index",
        "defense_index_raw",
        "defense_index_adjusted",
        "recent_form_index",
        "weighted_btts_rate",
        "weighted_over_2_5_rate",
        "clean_sheet_rate",
        "failed_to_score_rate",
        "environment_response_index",
        "environment_sample_size",
        "mean_opponent_elo_recent",
        "median_opponent_elo_recent",
        "min_opponent_elo_recent",
        "max_opponent_elo_recent",
        "opponent_elo_coverage_recent",
        "strong_opponent_match_count",
        "weak_opponent_match_count",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    text_cols = [col for col in TEAM_BEHAVIOR_COLUMNS if col not in numeric_cols]
    for col in text_cols:
        out[col] = out[col].fillna("").astype(str)
    return out.sort_values("team").reset_index(drop=True)


def _team_all_time_view(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    if matches.empty or "team" not in matches.columns:
        return pd.DataFrame(columns=matches.columns)
    return matches.loc[matches["team"].astype(str).str.lower() == str(team).lower()].copy()


def _team_list(matches: pd.DataFrame, teams_df: pd.DataFrame | None) -> list[str]:
    teams: list[str] = []
    if teams_df is not None and not teams_df.empty and "team" in teams_df.columns:
        teams.extend(teams_df["team"].dropna().astype(str).str.strip().tolist())
    if not matches.empty and "team" in matches.columns:
        teams.extend(matches["team"].dropna().astype(str).str.strip().tolist())
    return sorted({team for team in teams if team})


def _weighted_mean(view: pd.DataFrame, column: str) -> float:
    return _weighted_series(view, pd.to_numeric(view[column], errors="coerce"))


def _weighted_rate(view: pd.DataFrame, mask: pd.Series) -> float:
    return _weighted_series(view, mask.astype(float))


def _weighted_series(view: pd.DataFrame, values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(view["match_weight"], errors="coerce").fillna(0.0)
    mask = numeric.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(numeric.loc[mask], weights=weights.loc[mask]))


def _capped_goal_contribution(goals: pd.Series, modifier: pd.Series, cap: float) -> pd.Series:
    adjusted = pd.to_numeric(goals, errors="coerce") * pd.to_numeric(modifier, errors="coerce")
    return adjusted.clip(lower=0.0, upper=float(cap))


def _weighted_composite(components: list[tuple[float, float, bool]], default: float) -> float:
    usable = [(value, weight) for value, weight, present in components if present and not np.isnan(value)]
    if not usable:
        return default
    total_weight = sum(weight for _, weight in usable)
    if total_weight <= 0:
        return default
    return clamp(sum(value * weight for value, weight in usable) / total_weight, 0.0, 1.0)


def _recent_quality(view: pd.DataFrame, reference_date: Any) -> str:
    latest = view["date_utc"].max() if view is not None and not view.empty and "date_utc" in view.columns else ""
    quality, _, _ = reliability_grade(len(view) if view is not None else 0, latest, reference_date)
    return quality


def _opponent_quality_warning(view: pd.DataFrame) -> str:
    if view is None or view.empty or "opponent_quality_missing" not in view.columns:
        return ""
    missing_rate = float(view["opponent_quality_missing"].mean())
    if missing_rate > 0.35:
        return f"Opponent quality fallback used for {missing_rate:.0%} of recent matches."
    return ""


def _behavior_stability_warning(
    weighted_goals_for: float,
    weighted_goals_against: float,
    all_time_goals_for: float,
    all_time_goals_against: float,
) -> str:
    warnings = []
    if not any(pd.isna(value) for value in [weighted_goals_for, all_time_goals_for]):
        diff_for = float(weighted_goals_for) - float(all_time_goals_for)
        if abs(diff_for) >= 0.90:
            warnings.append(f"Recent goals-for differs from all-time by {diff_for:+.2f}.")
    if not any(pd.isna(value) for value in [weighted_goals_against, all_time_goals_against]):
        diff_against = float(weighted_goals_against) - float(all_time_goals_against)
        if abs(diff_against) >= 0.80:
            warnings.append(f"Recent goals-against differs from all-time by {diff_against:+.2f}.")
    return " ".join(warnings)


def _opponent_adjustment_warning(attack: dict[str, Any], defense: dict[str, Any]) -> str:
    warnings = []
    attack_raw = _numeric_or_nan(attack.get("attack_index_raw", attack.get("attack_index", np.nan)))
    attack_adjusted = _numeric_or_nan(attack.get("attack_index_adjusted", attack.get("attack_index", np.nan)))
    defense_raw = _numeric_or_nan(defense.get("defense_index_raw", defense.get("defense_index", np.nan)))
    defense_adjusted = _numeric_or_nan(defense.get("defense_index_adjusted", defense.get("defense_index", np.nan)))
    if not np.isnan(attack_raw) and not np.isnan(attack_adjusted):
        delta = attack_adjusted - attack_raw
        if delta <= -0.03:
            warnings.append("Attack index reduced after opponent-quality adjustment.")
        elif delta >= 0.03:
            warnings.append("Attack index increased after opponent-quality adjustment.")
    if not np.isnan(defense_raw) and not np.isnan(defense_adjusted):
        delta = defense_adjusted - defense_raw
        if delta <= -0.03:
            warnings.append("Defense index reduced after opponent-quality adjustment.")
        elif delta >= 0.03:
            warnings.append("Defense index increased after opponent-quality adjustment.")
    return " ".join(warnings)


def _numeric_or_nan(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _resolved_opponent_elo(row: pd.Series, quality_map: dict[str, float]) -> float:
    raw = pd.to_numeric(pd.Series([row.get("opponent_elo")]), errors="coerce").iloc[0]
    if not pd.isna(raw):
        return float(raw)
    opponent = str(row.get("opponent", "") or "").lower()
    return float(quality_map.get(opponent, 1500.0))


def _opponent_quality_missing(row: pd.Series, quality_map: dict[str, float]) -> bool:
    raw = pd.to_numeric(pd.Series([row.get("opponent_elo")]), errors="coerce").iloc[0]
    if not pd.isna(raw):
        return False
    opponent = str(row.get("opponent", "") or "").lower()
    return opponent not in quality_map


def _nan_to_default(value: float, default: float) -> float:
    return default if np.isnan(value) else float(value)


def _round_or_nan(value: float) -> float:
    return float("nan") if pd.isna(value) or np.isnan(value) else round(float(value), 3)


def _date_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def _empty_attack(team: str) -> dict[str, Any]:
    return {
        "team": team,
        "n_matches_attack": 0,
        "weighted_goals_for": np.nan,
        "weighted_goals_for_raw": np.nan,
        "weighted_goals_for_adjusted": np.nan,
        "weighted_xg_for": np.nan,
        "weighted_shots_for": np.nan,
        "weighted_shots_on_target_for": np.nan,
        "scoring_rate": np.nan,
        "multi_goal_rate": np.nan,
        "first_goal_proxy": np.nan,
        "attack_index": 0.5,
        "attack_index_raw": 0.5,
        "attack_index_adjusted": 0.5,
        "attack_data_quality": "insufficient",
    }


def _empty_defense(team: str) -> dict[str, Any]:
    return {
        "team": team,
        "n_matches_defense": 0,
        "weighted_goals_against": np.nan,
        "weighted_goals_against_raw": np.nan,
        "weighted_goals_against_adjusted": np.nan,
        "weighted_xg_against": np.nan,
        "weighted_shots_against": np.nan,
        "weighted_shots_on_target_against": np.nan,
        "clean_sheet_rate": np.nan,
        "conceded_rate": np.nan,
        "multi_conceded_rate": np.nan,
        "defense_index": 0.5,
        "defense_index_raw": 0.5,
        "defense_index_adjusted": 0.5,
        "defense_data_quality": "insufficient",
    }


def _csv_last_modified(path) -> str:
    if path is None or not path.exists():
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()


def _relative_data_path(path) -> str:
    try:
        return str(path.relative_to(DATA_DIR.parent))
    except Exception:
        return str(path)
