from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.external_prior_import import (
    scale_external_elo_to_strength as _scale_external_elo_to_strength,
    scale_fifa_points_to_strength as _scale_fifa_points_to_strength,
    scale_fifa_rank_to_strength as _scale_fifa_rank_to_strength,
)
from src.external_priors import EXTERNAL_PRIOR_COLUMNS, load_external_priors
from src.ratings import TEAM_RATING_COLUMNS, classify_rating_status, rating_row_for_team
from src.team_names import normalize_team_name
from src.utils import clamp, coerce_float, read_csv_with_columns, today_iso


CALIBRATED_PROPOSAL_COLUMNS = [
    "team",
    "current_attack",
    "current_defense",
    "current_recent_form",
    "behavior_attack_final",
    "behavior_defense_final",
    "behavior_recent_form",
    "external_overall_strength",
    "calibrated_attack",
    "calibrated_defense",
    "calibrated_recent_form",
    "attack_delta",
    "defense_delta",
    "recent_form_delta",
    "data_quality_current",
    "data_quality_proposed",
    "source",
    "warning",
    "notes",
]

DEFAULT_PROPOSAL_PATH = DATA_DIR / "team_ratings_external_calibrated_proposed.csv"
CALIBRATED_DATA_QUALITY = "external_benchmark_calibrated"
CALIBRATION_NOTE = "External benchmark calibrated from FIFA/Elo prior; formula documented; review after backtesting."
ATTACK_DELTA_CAP = 0.08
DEFENSE_DELTA_CAP = 0.08
RECENT_FORM_DELTA_CAP = 0.10
DISAGREEMENT_THRESHOLD = 0.12
BEHAVIOR_DISAGREEMENT_THRESHOLD = 0.15


def scale_fifa_rank_to_strength(rank: Any, max_rank: int = 210) -> float | pd._libs.missing.NAType:
    return _scale_fifa_rank_to_strength(rank, max_rank=max_rank)


def scale_fifa_points_to_strength(points: Any, min_points: float = 900, max_points: float = 1900) -> float | pd._libs.missing.NAType:
    return _scale_fifa_points_to_strength(points, min_points=min_points, max_points=max_points)


def scale_external_elo_to_strength(elo: Any, min_elo: float = 1200, max_elo: float = 2200) -> float | pd._libs.missing.NAType:
    return _scale_external_elo_to_strength(elo, min_elo=min_elo, max_elo=max_elo)


def combine_external_strengths(
    rank_strength: Any = None,
    points_strength: Any = None,
    elo_strength: Any = None,
) -> float | pd._libs.missing.NAType:
    """Average already-scaled external benchmark strengths."""
    values = [coerce_float(value, float("nan")) for value in [rank_strength, points_strength, elo_strength]]
    numeric = [value for value in values if not pd.isna(value)]
    if not numeric:
        return pd.NA
    return round(clamp(sum(numeric) / len(numeric), 0.35, 0.90), 3)


def calibrate_rating_against_external(
    team: str,
    current_rating: pd.Series | dict[str, Any] | None,
    behavior_row: pd.Series | dict[str, Any] | None,
    external_prior: pd.Series | dict[str, Any] | None,
    include_reviewed: bool = False,
) -> dict[str, Any]:
    """Create a transparent external-benchmark calibration proposal for one team."""
    rating = _as_series(current_rating)
    behavior = _as_series(behavior_row)
    prior = _as_series(external_prior)
    canonical_team = normalize_team_name(team)

    current_attack = _rating_value(rating, "attack")
    current_defense = _rating_value(rating, "defense")
    current_form = _rating_value(rating, "recent_form")
    behavior_attack = _first_numeric(behavior, ["attack_index_final", "attack_index", "attack_index_residual_robust"])
    behavior_defense = _first_numeric(behavior, ["defense_index_final", "defense_index", "defense_index_residual_robust"])
    behavior_form = _first_numeric(behavior, ["recent_form_index"])
    external_overall = _first_numeric(prior, ["reference_overall_strength"])
    current_quality = _text_value(rating.get("data_quality", "")) if not rating.empty else "missing"
    status = classify_rating_status(rating)

    warnings: list[str] = []
    if pd.isna(external_overall):
        warnings.append("missing_external_benchmark")

    if not pd.isna(external_overall):
        internal_values = [current_attack, current_defense, current_form]
        internal_strength = sum(internal_values) / len(internal_values)
        if internal_strength - external_overall > DISAGREEMENT_THRESHOLD:
            warnings.append("internal_above_external_by_more_than_0_12")
        if external_overall - internal_strength > DISAGREEMENT_THRESHOLD:
            warnings.append("internal_below_external_by_more_than_0_12")

        behavior_values = [value for value in [behavior_attack, behavior_defense, behavior_form] if not pd.isna(value)]
        if behavior_values and max(behavior_values) - external_overall > BEHAVIOR_DISAGREEMENT_THRESHOLD:
            warnings.append("behavior_above_external_by_more_than_0_15")

    if status == "manual_reviewed" and not include_reviewed:
        warnings.append("manual_reviewed_preserved")
        return _proposal_row(
            canonical_team,
            current_attack,
            current_defense,
            current_form,
            behavior_attack,
            behavior_defense,
            behavior_form,
            external_overall,
            current_attack,
            current_defense,
            current_form,
            current_quality,
            current_quality,
            _text_value(prior.get("source", "")),
            warnings,
            "Manual reviewed rating preserved; external benchmark reported for audit only.",
        )

    if status == "external_benchmark_calibrated":
        return _proposal_row(
            canonical_team,
            current_attack,
            current_defense,
            current_form,
            behavior_attack,
            behavior_defense,
            behavior_form,
            external_overall,
            current_attack,
            current_defense,
            current_form,
            current_quality,
            current_quality,
            _text_value(prior.get("source", "")),
            warnings,
            "Existing external benchmark calibrated rating preserved; rerunning calibration is idempotent.",
        )

    if pd.isna(external_overall):
        return _proposal_row(
            canonical_team,
            current_attack,
            current_defense,
            current_form,
            behavior_attack,
            behavior_defense,
            behavior_form,
            external_overall,
            current_attack,
            current_defense,
            current_form,
            current_quality,
            current_quality,
            _text_value(prior.get("source", "")),
            warnings,
            "External benchmark missing; rating unchanged.",
        )

    if status in {"generated_from_behavior", "neutral_placeholder"}:
        desired_attack = 0.45 * current_attack + 0.25 * _fallback(behavior_attack, current_attack) + 0.30 * external_overall
        desired_defense = 0.45 * current_defense + 0.25 * _fallback(behavior_defense, current_defense) + 0.30 * external_overall
        desired_form = 0.50 * current_form + 0.30 * _fallback(behavior_form, current_form) + 0.20 * external_overall
        warnings.append("generated_rating_calibrated")
    else:
        desired_attack = 0.70 * current_attack + 0.30 * external_overall
        desired_defense = 0.70 * current_defense + 0.30 * external_overall
        desired_form = 0.80 * current_form + 0.20 * external_overall
        warnings.append("manual_existing_calibration_suggested")

    calibrated_attack, attack_capped = _capped_calibrated_value(current_attack, desired_attack, ATTACK_DELTA_CAP)
    calibrated_defense, defense_capped = _capped_calibrated_value(current_defense, desired_defense, DEFENSE_DELTA_CAP)
    calibrated_form, form_capped = _capped_calibrated_value(current_form, desired_form, RECENT_FORM_DELTA_CAP)
    if attack_capped or defense_capped or form_capped:
        warnings.append("large_delta_capped")

    return _proposal_row(
        canonical_team,
        current_attack,
        current_defense,
        current_form,
        behavior_attack,
        behavior_defense,
        behavior_form,
        external_overall,
        calibrated_attack,
        calibrated_defense,
        calibrated_form,
        current_quality,
        CALIBRATED_DATA_QUALITY,
        _text_value(prior.get("source", "")),
        warnings,
        CALIBRATION_NOTE,
    )


def build_external_calibration_proposals(
    team_ratings_df: pd.DataFrame | None,
    team_behavior_df: pd.DataFrame | None,
    external_priors_df: pd.DataFrame | None,
    include_reviewed: bool = False,
) -> pd.DataFrame:
    ratings = _normalise_rating_frame(team_ratings_df)
    behavior = team_behavior_df.copy() if team_behavior_df is not None else pd.DataFrame()
    priors = external_priors_df.copy() if external_priors_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, rating in ratings.iterrows():
        team = normalize_team_name(rating.get("team", ""))
        if not team:
            continue
        behavior_row = rating_row_for_team(behavior, team)
        prior_row = rating_row_for_team(priors, team)
        rows.append(calibrate_rating_against_external(team, rating, behavior_row, prior_row, include_reviewed=include_reviewed))
    return pd.DataFrame(rows, columns=CALIBRATED_PROPOSAL_COLUMNS)


def load_external_calibration_proposals(path: str | Path = DEFAULT_PROPOSAL_PATH) -> pd.DataFrame:
    return read_csv_with_columns(_resolve_path(path), CALIBRATED_PROPOSAL_COLUMNS)


def apply_external_calibration_proposals(
    team_ratings_df: pd.DataFrame | None,
    proposals_df: pd.DataFrame | None,
    include_manual_existing: bool = False,
    include_reviewed: bool = False,
) -> pd.DataFrame:
    ratings = _normalise_rating_frame(team_ratings_df)
    proposals = proposals_df.copy() if proposals_df is not None else pd.DataFrame(columns=CALIBRATED_PROPOSAL_COLUMNS)
    for col in CALIBRATED_PROPOSAL_COLUMNS:
        if col not in proposals.columns:
            proposals[col] = pd.NA
    if ratings.empty or proposals.empty:
        return ratings[TEAM_RATING_COLUMNS].reset_index(drop=True)

    out = ratings.copy()
    for _, proposal in proposals.iterrows():
        team = normalize_team_name(_text_value(proposal.get("team", "")))
        if not team:
            continue
        external_strength = coerce_float(proposal.get("external_overall_strength"), float("nan"))
        if pd.isna(external_strength):
            continue
        mask = out["team"].astype(str).map(lambda value: normalize_team_name(value).lower()) == team.lower()
        if not mask.any():
            continue
        idx = out.loc[mask].index[0]
        status = classify_rating_status(out.loc[idx])
        if not _write_allowed(status, include_manual_existing, include_reviewed):
            continue

        out.at[idx, "attack"] = round(clamp(coerce_float(proposal.get("calibrated_attack"), out.at[idx, "attack"]), 0.35, 0.90), 3)
        out.at[idx, "defense"] = round(clamp(coerce_float(proposal.get("calibrated_defense"), out.at[idx, "defense"]), 0.35, 0.90), 3)
        out.at[idx, "recent_form"] = round(clamp(coerce_float(proposal.get("calibrated_recent_form"), out.at[idx, "recent_form"]), 0.35, 0.90), 3)
        out.at[idx, "data_quality"] = CALIBRATED_DATA_QUALITY
        out.at[idx, "last_updated"] = today_iso()
        out.at[idx, "notes"] = _append_note(out.at[idx, "notes"], CALIBRATION_NOTE)
    return out[TEAM_RATING_COLUMNS].reset_index(drop=True)


def external_benchmark_calibration_summary(
    proposals_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None = None,
) -> dict[str, int]:
    proposals = proposals_df.copy() if proposals_df is not None else pd.DataFrame(columns=CALIBRATED_PROPOSAL_COLUMNS)
    ratings = _normalise_rating_frame(team_ratings_df)
    warnings = proposals["warning"].astype(str) if "warning" in proposals.columns else pd.Series("", index=proposals.index)
    quality = ratings["data_quality"].astype(str).str.lower() if "data_quality" in ratings.columns else pd.Series("", index=ratings.index)
    return {
        "teams_with_external_benchmark": int(proposals["external_overall_strength"].notna().sum()) if "external_overall_strength" in proposals.columns else 0,
        "teams_missing_external_benchmark": int(warnings.str.contains("missing_external_benchmark", case=False, na=False).sum()),
        "generated_rows_calibrated": int(warnings.str.contains("generated_rating_calibrated", case=False, na=False).sum()),
        "manual_rows_preserved_or_suggested": int(warnings.str.contains("manual_existing_calibration_suggested|manual_reviewed_preserved", case=False, na=False).sum()),
        "large_disagreement_warnings": int(
            warnings.str.contains("internal_.*external|behavior_above_external|large_delta_capped", case=False, na=False, regex=True).sum()
        ),
        "current_external_benchmark_calibrated_rows": int(quality.str.contains(CALIBRATED_DATA_QUALITY, case=False, na=False).sum()),
        "current_generated_from_behavior_rows": int(quality.str.contains("generated_from_behavior", case=False, na=False).sum()),
    }


def load_current_external_priors() -> pd.DataFrame:
    return load_external_priors()


def _proposal_row(
    team: str,
    current_attack: float,
    current_defense: float,
    current_form: float,
    behavior_attack: Any,
    behavior_defense: Any,
    behavior_form: Any,
    external_overall: Any,
    calibrated_attack: float,
    calibrated_defense: float,
    calibrated_form: float,
    data_quality_current: str,
    data_quality_proposed: str,
    source: str,
    warnings: list[str],
    notes: str,
) -> dict[str, Any]:
    return {
        "team": team,
        "current_attack": round(current_attack, 3),
        "current_defense": round(current_defense, 3),
        "current_recent_form": round(current_form, 3),
        "behavior_attack_final": _round_or_na(behavior_attack),
        "behavior_defense_final": _round_or_na(behavior_defense),
        "behavior_recent_form": _round_or_na(behavior_form),
        "external_overall_strength": _round_or_na(external_overall),
        "calibrated_attack": round(calibrated_attack, 3),
        "calibrated_defense": round(calibrated_defense, 3),
        "calibrated_recent_form": round(calibrated_form, 3),
        "attack_delta": round(calibrated_attack - current_attack, 3),
        "defense_delta": round(calibrated_defense - current_defense, 3),
        "recent_form_delta": round(calibrated_form - current_form, 3),
        "data_quality_current": data_quality_current,
        "data_quality_proposed": data_quality_proposed,
        "source": source,
        "warning": "; ".join(dict.fromkeys([warning for warning in warnings if warning])),
        "notes": notes,
    }


def _normalise_rating_frame(team_ratings_df: pd.DataFrame | None) -> pd.DataFrame:
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    for col in TEAM_RATING_COLUMNS:
        if col not in ratings.columns:
            ratings[col] = pd.NA
    if ratings.empty:
        return ratings[TEAM_RATING_COLUMNS].copy()
    out = ratings[TEAM_RATING_COLUMNS].copy()
    out["team"] = out["team"].astype(str).str.strip()
    out = out.loc[out["team"] != ""].copy()
    return out.reset_index(drop=True)


def _as_series(row: pd.Series | dict[str, Any] | None) -> pd.Series:
    if row is None:
        return pd.Series(dtype="object")
    if isinstance(row, pd.Series):
        return row
    if isinstance(row, dict):
        return pd.Series(row)
    return pd.Series(dtype="object")


def _rating_value(row: pd.Series, col: str, default: float = 0.55) -> float:
    return round(clamp(coerce_float(row.get(col), default), 0.35, 0.90), 3)


def _first_numeric(row: pd.Series, cols: list[str]) -> float:
    if row.empty:
        return float("nan")
    for col in cols:
        value = coerce_float(row.get(col), float("nan"))
        if not pd.isna(value):
            return round(clamp(value, 0.35, 0.90), 3)
    return float("nan")


def _fallback(value: Any, fallback: float) -> float:
    numeric = coerce_float(value, float("nan"))
    if pd.isna(numeric):
        return fallback
    return clamp(numeric, 0.35, 0.90)


def _capped_calibrated_value(current: float, desired: float, cap: float) -> tuple[float, bool]:
    desired = clamp(desired, 0.35, 0.90)
    delta = desired - current
    capped_delta = clamp(delta, -cap, cap)
    return round(clamp(current + capped_delta, 0.35, 0.90), 3), abs(delta) > cap + 1e-12


def _write_allowed(status: str, include_manual_existing: bool, include_reviewed: bool) -> bool:
    if status == "manual_reviewed":
        return bool(include_reviewed)
    if status in {"generated_from_behavior", "neutral_placeholder"}:
        return True
    if status == "manual_existing":
        return bool(include_manual_existing)
    if status == "external_benchmark_calibrated":
        return False
    return False


def _append_note(existing: Any, note: str) -> str:
    current = _text_value(existing)
    if note in current:
        return current
    return " ".join(part for part in [current, note] if part).strip()


def _round_or_na(value: Any) -> Any:
    numeric = coerce_float(value, float("nan"))
    if pd.isna(numeric):
        return pd.NA
    return round(clamp(numeric, 0.35, 0.90), 3)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return DATA_DIR.parent / resolved


def _text_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
