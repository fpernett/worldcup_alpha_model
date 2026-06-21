from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.behavior_calibration import DEFAULT_BEHAVIOR_CONFIG, BehaviorConfig, filter_recent_team_matches
from src.performance_residuals import add_performance_residuals
from src.recency import calculate_match_weight
from src.utils import clamp, coerce_bool, coerce_float


DRIVER_COLUMNS = [
    "date_utc",
    "competition",
    "competition_type",
    "opponent",
    "team_goals",
    "opponent_goals",
    "team_elo_pre",
    "opponent_elo",
    "match_weight",
    "blowout_weight",
    "expected_goals_for",
    "expected_goals_against",
    "goals_for_residual_raw",
    "goals_for_residual_robust",
    "goals_against_residual_raw",
    "goals_against_residual_robust",
    "attack_contribution",
    "defense_contribution",
]

BREAKDOWN_COLUMNS = [
    "matches",
    "weighted_goals_for",
    "weighted_goals_against",
    "weighted_goal_for_residual_robust",
    "weighted_goal_against_residual_robust",
    "attack_contribution",
    "defense_contribution",
]

MODEL_INPUT_AUDIT_COLUMNS = [
    "team",
    "manual_attack",
    "behavior_attack_final",
    "adjusted_attack_used_by_model",
    "attack_delta",
    "attack_delta_cap_applied",
    "manual_defense",
    "behavior_defense_final",
    "adjusted_defense_used_by_model",
    "defense_delta",
    "defense_delta_cap_applied",
    "manual_recent_form",
    "behavior_recent_form",
    "adjusted_recent_form_used_by_model",
    "recent_form_delta",
    "form_delta_cap_applied",
    "behavior_blend_used",
    "warning",
]


def get_behavior_driver_matches(
    matches_df: pd.DataFrame | None,
    team: str,
    reference_date: Any,
    n: int = 10,
    sort_by: str = "attack",
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> pd.DataFrame:
    view = _prepared_team_matches(matches_df, team, reference_date, config)
    if view.empty:
        return pd.DataFrame(columns=DRIVER_COLUMNS)
    sort_col = "defense_contribution" if str(sort_by).lower().startswith("def") else "attack_contribution"
    return _driver_display(view.sort_values(sort_col, ascending=False).head(int(n)))


def calculate_opponent_tier_breakdown(
    matches_df: pd.DataFrame | None,
    team: str,
    reference_date: Any,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> pd.DataFrame:
    view = _prepared_team_matches(matches_df, team, reference_date, config)
    if view.empty:
        return pd.DataFrame(columns=["opponent_tier", *BREAKDOWN_COLUMNS])
    view["segment"] = view["opponent_elo"].map(classify_opponent_tier)
    order = ["elite", "strong", "average", "weak", "unknown"]
    return _breakdown(view, "segment", order).rename(columns={"segment": "opponent_tier"})


def calculate_competition_breakdown(
    matches_df: pd.DataFrame | None,
    team: str,
    reference_date: Any,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> pd.DataFrame:
    view = _prepared_team_matches(matches_df, team, reference_date, config)
    if view.empty:
        return pd.DataFrame(columns=["competition_type", *BREAKDOWN_COLUMNS])
    view["segment"] = view["competition_type"].fillna("other").astype(str).str.strip().replace("", "other")
    order = ["world_cup", "world_cup_qualifier", "continental_tournament", "nations_league", "friendly", "other"]
    return _breakdown(view, "segment", order).rename(columns={"segment": "competition_type"})


def audit_final_model_inputs(
    team_ratings_raw_df: pd.DataFrame | None,
    team_ratings_adjusted_df: pd.DataFrame | None,
    team_behavior_df: pd.DataFrame | None,
) -> pd.DataFrame:
    adjusted = team_ratings_adjusted_df.copy() if team_ratings_adjusted_df is not None else pd.DataFrame()
    raw = team_ratings_raw_df.copy() if team_ratings_raw_df is not None else pd.DataFrame()
    behavior = team_behavior_df.copy() if team_behavior_df is not None else pd.DataFrame()
    if adjusted.empty or "team" not in adjusted.columns:
        return pd.DataFrame(columns=MODEL_INPUT_AUDIT_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, adjusted_row in adjusted.sort_values("team").iterrows():
        team = str(adjusted_row.get("team", "") or "")
        raw_row = _matching_row(raw, team)
        behavior_row = _matching_row(behavior, team)

        manual_attack = _value_from_adjusted_or_raw(adjusted_row, raw_row, "manual_attack", "attack")
        manual_defense = _value_from_adjusted_or_raw(adjusted_row, raw_row, "manual_defense", "defense")
        manual_form = _value_from_adjusted_or_raw(adjusted_row, raw_row, "manual_recent_form", "recent_form")
        adjusted_attack = coerce_float(adjusted_row.get("attack"), manual_attack)
        adjusted_defense = coerce_float(adjusted_row.get("defense"), manual_defense)
        adjusted_form = coerce_float(adjusted_row.get("recent_form"), manual_form)
        behavior_attack = _first_numeric(
            behavior_row,
            adjusted_row,
            ["attack_index", "attack_index_final", "behavior_attack_index"],
        )
        behavior_defense = _first_numeric(
            behavior_row,
            adjusted_row,
            ["defense_index", "defense_index_final", "behavior_defense_index"],
        )
        behavior_form = _first_numeric(behavior_row, adjusted_row, ["recent_form_index", "behavior_recent_form_index"])

        rows.append(
            {
                "team": team,
                "manual_attack": _round_or_nan(manual_attack),
                "behavior_attack_final": _round_or_nan(behavior_attack),
                "adjusted_attack_used_by_model": _round_or_nan(adjusted_attack),
                "attack_delta": _round_or_nan(adjusted_attack - manual_attack),
                "attack_delta_cap_applied": coerce_bool(adjusted_row.get("behavior_attack_cap_hit", False)),
                "manual_defense": _round_or_nan(manual_defense),
                "behavior_defense_final": _round_or_nan(behavior_defense),
                "adjusted_defense_used_by_model": _round_or_nan(adjusted_defense),
                "defense_delta": _round_or_nan(adjusted_defense - manual_defense),
                "defense_delta_cap_applied": coerce_bool(adjusted_row.get("behavior_defense_cap_hit", False)),
                "manual_recent_form": _round_or_nan(manual_form),
                "behavior_recent_form": _round_or_nan(behavior_form),
                "adjusted_recent_form_used_by_model": _round_or_nan(adjusted_form),
                "recent_form_delta": _round_or_nan(adjusted_form - manual_form),
                "form_delta_cap_applied": coerce_bool(adjusted_row.get("behavior_recent_form_cap_hit", False)),
                "behavior_blend_used": coerce_bool(adjusted_row.get("behavior_blend_used", False)),
                "warning": behavior_quality_warnings(behavior_row, raw_row, adjusted_row),
            }
        )
    return pd.DataFrame(rows, columns=MODEL_INPUT_AUDIT_COLUMNS)


def classify_opponent_tier(opponent_elo: Any) -> str:
    elo = coerce_float(opponent_elo, float("nan"))
    if pd.isna(elo):
        return "unknown"
    if elo >= 1850:
        return "elite"
    if elo >= 1700:
        return "strong"
    if elo >= 1500:
        return "average"
    return "weak"


def behavior_quality_warnings(
    behavior_row: pd.Series | None,
    manual_row: pd.Series | None = None,
    adjusted_row: pd.Series | None = None,
    opponent_breakdown: pd.DataFrame | None = None,
    competition_breakdown: pd.DataFrame | None = None,
) -> str:
    behavior = behavior_row if isinstance(behavior_row, pd.Series) else pd.Series(dtype="object")
    manual = manual_row if isinstance(manual_row, pd.Series) else pd.Series(dtype="object")
    adjusted = adjusted_row if isinstance(adjusted_row, pd.Series) else pd.Series(dtype="object")
    warnings: list[str] = []

    manual_attack = coerce_float(adjusted.get("manual_attack", manual.get("attack")), float("nan"))
    manual_defense = coerce_float(adjusted.get("manual_defense", manual.get("defense")), float("nan"))
    behavior_attack = _first_numeric(behavior, adjusted, ["attack_index", "attack_index_final", "behavior_attack_index"])
    behavior_defense = _first_numeric(behavior, adjusted, ["defense_index", "defense_index_final", "behavior_defense_index"])
    attack_raw = coerce_float(behavior.get("attack_index_raw"), float("nan"))
    attack_robust = coerce_float(behavior.get("attack_index_residual_robust", behavior.get("attack_index_residual")), float("nan"))
    attack_top3 = coerce_float(behavior.get("top_3_attack_residual_share"), float("nan"))
    defense_top3 = coerce_float(behavior.get("top_3_defense_residual_share"), float("nan"))

    if (not pd.isna(behavior_attack) and not pd.isna(manual_attack) and behavior_attack - manual_attack >= 0.25) or (
        not pd.isna(behavior_defense) and not pd.isna(manual_defense) and behavior_defense - manual_defense >= 0.25
    ):
        warnings.append("high_behavior_low_manual_prior")
    if not pd.isna(behavior_attack) and not pd.isna(manual_attack) and behavior_attack > 0.85 and manual_attack < 0.65:
        warnings.append("behavior_final_above_0_85_but_manual_below_0_65")
    if any(
        coerce_bool(adjusted.get(col, False))
        for col in ["behavior_blend_cap_hit", "behavior_attack_cap_hit", "behavior_defense_cap_hit", "behavior_recent_form_cap_hit"]
    ):
        warnings.append("behavior_delta_capped")
    if not pd.isna(attack_raw) and not pd.isna(attack_robust) and attack_raw >= 0.75 and attack_raw - attack_robust >= 0.15:
        warnings.append("attack_raw_high_but_residual_robust_much_lower")
    if (not pd.isna(attack_top3) and attack_top3 >= 0.60) or (not pd.isna(defense_top3) and defense_top3 >= 0.60):
        warnings.append("top3_residual_share_high")
    if _segment_contribution_share(competition_breakdown, "friendly", "attack_contribution") >= 0.50:
        warnings.append("behavior_driven_by_friendlies")
    if _segment_contribution_share(opponent_breakdown, "weak", "attack_contribution") >= 0.50:
        warnings.append("behavior_driven_by_weak_opponents")

    existing = str(behavior.get("behavior_warning", "") or "").strip()
    if existing:
        warnings.append(existing)
    return "; ".join(dict.fromkeys([warning for warning in warnings if warning]))


def write_markdown_report(title: str, sections: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join([f"# {title}", *sections]), encoding="utf-8")
    return path


def _prepared_team_matches(
    matches_df: pd.DataFrame | None,
    team: str,
    reference_date: Any,
    config: BehaviorConfig,
) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        return pd.DataFrame()
    matches = matches_df.copy()
    for col in [
        "date_utc",
        "competition",
        "competition_type",
        "team",
        "opponent",
        "team_goals",
        "opponent_goals",
        "team_elo_pre",
        "opponent_elo",
        "is_home",
        "is_neutral",
    ]:
        if col not in matches.columns:
            matches[col] = pd.NA
    recent = filter_recent_team_matches(matches, team, reference_date, config)
    if recent.empty:
        return recent
    recent["match_weight"] = recent.apply(lambda row: calculate_match_weight(row, reference_date, config), axis=1)
    recent["match_weight"] = pd.to_numeric(recent["match_weight"], errors="coerce").fillna(0.0)
    if float(recent["match_weight"].sum()) <= 0:
        recent["match_weight"] = 1.0
    recent = add_performance_residuals(recent, residual_cap=config.residual_cap)
    recent = _add_driver_columns(recent, config)
    return recent.sort_values("date_utc", ascending=False).reset_index(drop=True)


def _add_driver_columns(view: pd.DataFrame, config: BehaviorConfig) -> pd.DataFrame:
    out = view.copy()
    for col in ["team_goals", "opponent_goals", "match_weight", "blowout_weight"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out["goals_for_residual_raw"] = pd.to_numeric(out.get("goals_for_residual"), errors="coerce")
    out["goals_for_residual_robust"] = pd.to_numeric(out.get("goals_for_residual_capped"), errors="coerce")
    out["goals_against_residual_raw"] = pd.to_numeric(out.get("goals_against_residual"), errors="coerce")
    out["goals_against_residual_robust"] = pd.to_numeric(out.get("goals_against_residual_capped"), errors="coerce")
    blowout = out["blowout_weight"].fillna(1.0).clip(lower=0.0)
    match_weight = out["match_weight"].fillna(0.0).clip(lower=0.0)
    attack_goal_signal = (out["team_goals"].fillna(0.0) / float(config.goal_contribution_cap)).clip(0.0, 1.0)
    attack_residual_signal = (out["goals_for_residual_robust"].fillna(0.0).clip(lower=0.0) / float(config.residual_cap)).clip(0.0, 1.0)
    defense_goal_signal = ((float(config.goal_contribution_cap) - out["opponent_goals"].fillna(config.goal_contribution_cap)) / float(config.goal_contribution_cap)).clip(0.0, 1.0)
    defense_residual_signal = ((-out["goals_against_residual_robust"].fillna(0.0)).clip(lower=0.0) / float(config.residual_cap)).clip(0.0, 1.0)
    out["attack_contribution"] = match_weight * blowout * (0.70 * attack_goal_signal + 0.30 * attack_residual_signal)
    out["defense_contribution"] = match_weight * blowout * (0.70 * defense_goal_signal + 0.30 * defense_residual_signal)
    return out


def _driver_display(view: pd.DataFrame) -> pd.DataFrame:
    if view.empty:
        return pd.DataFrame(columns=DRIVER_COLUMNS)
    out = view.copy()
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce").dt.date.astype(str)
    for col in DRIVER_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[DRIVER_COLUMNS].reset_index(drop=True)


def _breakdown(view: pd.DataFrame, group_col: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for segment, group in view.groupby(group_col, dropna=False):
        weights = pd.to_numeric(group["match_weight"], errors="coerce").fillna(0.0)
        robust_weights = weights * pd.to_numeric(group.get("blowout_weight"), errors="coerce").fillna(1.0)
        rows.append(
            {
                "segment": str(segment or "unknown"),
                "matches": int(len(group)),
                "weighted_goals_for": _weighted_average(group["team_goals"], weights),
                "weighted_goals_against": _weighted_average(group["opponent_goals"], weights),
                "weighted_goal_for_residual_robust": _weighted_average(group["goals_for_residual_robust"], robust_weights),
                "weighted_goal_against_residual_robust": _weighted_average(group["goals_against_residual_robust"], robust_weights),
                "attack_contribution": float(pd.to_numeric(group["attack_contribution"], errors="coerce").fillna(0.0).sum()),
                "defense_contribution": float(pd.to_numeric(group["defense_contribution"], errors="coerce").fillna(0.0).sum()),
            }
        )
    out = pd.DataFrame(rows, columns=["segment", *BREAKDOWN_COLUMNS])
    if out.empty:
        return out
    order_map = {label: idx for idx, label in enumerate(order)}
    out["_order"] = out["segment"].map(lambda value: order_map.get(value, len(order_map)))
    return out.sort_values(["_order", "segment"]).drop(columns=["_order"]).reset_index(drop=True)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = numeric.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return round(float(np.average(numeric.loc[mask], weights=weights.loc[mask])), 3)


def _segment_contribution_share(df: pd.DataFrame | None, segment: str, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    label_col = next((candidate for candidate in ["segment", "opponent_tier", "competition_type"] if candidate in df.columns), "")
    if not label_col:
        return 0.0
    total = pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum()
    if total <= 0:
        return 0.0
    selected = df.loc[df[label_col].astype(str).str.lower() == segment.lower()]
    if selected.empty:
        return 0.0
    return float(pd.to_numeric(selected[col], errors="coerce").fillna(0.0).sum() / total)


def _matching_row(df: pd.DataFrame | None, team: str) -> pd.Series:
    if df is None or df.empty or "team" not in df.columns:
        return pd.Series(dtype="object")
    rows = df.loc[df["team"].astype(str).str.lower() == str(team).lower()]
    if rows.empty:
        return pd.Series(dtype="object")
    return rows.iloc[0]


def _value_from_adjusted_or_raw(adjusted_row: pd.Series, raw_row: pd.Series, adjusted_col: str, raw_col: str) -> float:
    adjusted_value = coerce_float(adjusted_row.get(adjusted_col), float("nan"))
    if not pd.isna(adjusted_value):
        return adjusted_value
    raw_value = coerce_float(raw_row.get(raw_col), float("nan"))
    if not pd.isna(raw_value):
        return raw_value
    return coerce_float(adjusted_row.get(raw_col), 0.55)


def _first_numeric(primary: pd.Series, secondary: pd.Series, columns: list[str]) -> float:
    for col in columns:
        value = coerce_float(primary.get(col), float("nan"))
        if not pd.isna(value):
            return value
        value = coerce_float(secondary.get(col), float("nan"))
        if not pd.isna(value):
            return value
    return float("nan")


def _round_or_nan(value: Any) -> float:
    numeric = coerce_float(value, float("nan"))
    return float("nan") if pd.isna(numeric) else round(float(numeric), 3)
