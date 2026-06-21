from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.external_priors import EXTERNAL_PRIOR_COLUMNS
from src.ratings import TEAM_RATING_COLUMNS, rating_row_for_team
from src.team_names import normalize_team_name
from src.utils import clamp, coerce_float, today_iso


EXTERNAL_STRENGTH_INPUT_COLUMNS = [
    "team",
    "fifa_rank",
    "fifa_points",
    "external_elo",
    "source",
    "last_updated",
    "notes",
]

OPTIONAL_REFERENCE_COLUMNS = [
    "reference_attack",
    "reference_defense",
    "reference_recent_form",
]

EXTERNAL_PRIOR_IMPORT_DIAGNOSTIC_COLUMNS = [
    "team",
    "canonical_team",
    "matched_external_prior",
    "has_strength_input",
    "has_optional_reference_input",
    "update_ready",
    "warning",
]

EXTERNAL_PRIOR_TEXT_COLUMNS = [
    "team",
    "source",
    "last_updated",
    "notes",
    "warning",
    "status",
    "review_status",
]

EXTERNAL_PRIOR_NUMERIC_COLUMNS = [
    "reference_attack",
    "reference_defense",
    "reference_recent_form",
    "reference_overall_strength",
]

GENERATED_DISAGREEMENT_COLUMNS = [
    "team",
    "generated_rating_strength",
    "reference_overall_strength",
    "strength_disagreement",
    "warning",
]


def scale_fifa_rank_to_strength(rank: Any, max_rank: int = 210) -> float | pd._libs.missing.NAType:
    value = coerce_float(rank, float("nan"))
    if pd.isna(value):
        return pd.NA
    clipped_rank = clamp(value, 1.0, float(max_rank))
    if max_rank <= 1:
        return 0.90
    strength = 0.90 - ((clipped_rank - 1.0) / (float(max_rank) - 1.0)) * 0.55
    return round(clamp(strength, 0.35, 0.90), 3)


def scale_fifa_points_to_strength(points: Any, min_points: float = 900, max_points: float = 1900) -> float | pd._libs.missing.NAType:
    value = coerce_float(points, float("nan"))
    if pd.isna(value):
        return pd.NA
    if max_points <= min_points:
        return 0.55
    strength = 0.35 + ((value - float(min_points)) / (float(max_points) - float(min_points))) * 0.55
    return round(clamp(strength, 0.35, 0.90), 3)


def scale_external_elo_to_strength(elo: Any, min_elo: float = 1200, max_elo: float = 2200) -> float | pd._libs.missing.NAType:
    value = coerce_float(elo, float("nan"))
    if pd.isna(value):
        return pd.NA
    if max_elo <= min_elo:
        return 0.55
    strength = 0.35 + ((value - float(min_elo)) / (float(max_elo) - float(min_elo))) * 0.55
    return round(clamp(strength, 0.35, 0.90), 3)


def combine_external_strengths(
    fifa_rank: Any = None,
    fifa_points: Any = None,
    external_elo: Any = None,
) -> float | pd._libs.missing.NAType:
    values = [
        scale_fifa_rank_to_strength(fifa_rank),
        scale_fifa_points_to_strength(fifa_points),
        scale_external_elo_to_strength(external_elo),
    ]
    numeric_values = [float(value) for value in values if _is_number(value)]
    if not numeric_values:
        return pd.NA
    return round(clamp(sum(numeric_values) / len(numeric_values), 0.35, 0.90), 3)


def load_external_strength_input(path: str | Path = DATA_DIR / "raw" / "external_team_strength.csv") -> pd.DataFrame:
    resolved = _resolve_path(path)
    if not resolved.exists():
        return pd.DataFrame(columns=EXTERNAL_STRENGTH_INPUT_COLUMNS + OPTIONAL_REFERENCE_COLUMNS)
    try:
        df = pd.read_csv(resolved)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=EXTERNAL_STRENGTH_INPUT_COLUMNS + OPTIONAL_REFERENCE_COLUMNS)
    for col in EXTERNAL_STRENGTH_INPUT_COLUMNS + OPTIONAL_REFERENCE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["team"] = df["team"].astype(str).str.strip()
    df = df.loc[df["team"] != ""].copy()
    return df.reset_index(drop=True)


def build_external_prior_updates(
    input_df: pd.DataFrame | None,
    existing_priors_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = input_df.copy() if input_df is not None else pd.DataFrame(columns=EXTERNAL_STRENGTH_INPUT_COLUMNS)
    existing = existing_priors_df.copy() if existing_priors_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_COLUMNS)
    for col in EXTERNAL_STRENGTH_INPUT_COLUMNS + OPTIONAL_REFERENCE_COLUMNS:
        if col not in source_df.columns:
            source_df[col] = pd.NA
    for col in EXTERNAL_PRIOR_COLUMNS:
        if col not in existing.columns:
            existing[col] = pd.NA

    update_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for _, row in source_df.iterrows():
        raw_team = _text_value(row.get("team", ""))
        canonical_team = normalize_team_name(raw_team)
        existing_row = rating_row_for_team(existing, canonical_team)
        matched = not existing_row.empty
        has_strength_input = any(_is_number(row.get(col)) for col in ["fifa_rank", "fifa_points", "external_elo"])
        optional_values = {
            "reference_attack": _optional_reference(row.get("reference_attack")),
            "reference_defense": _optional_reference(row.get("reference_defense")),
            "reference_recent_form": _optional_reference(row.get("reference_recent_form")),
        }
        has_optional_reference = any(_is_number(value) for value in optional_values.values())
        overall = combine_external_strengths(row.get("fifa_rank"), row.get("fifa_points"), row.get("external_elo"))
        update_ready = bool(matched and (has_strength_input or has_optional_reference))
        warnings: list[str] = []
        if not matched:
            warnings.append("team not found in external prior table")
        if not has_strength_input:
            warnings.append("missing rank/Elo data")
        if not has_strength_input and not has_optional_reference:
            warnings.append("no reference values supplied")

        update_rows.append(
            {
                "team": _text_value(existing_row.get("team", "")) if matched else canonical_team,
                "reference_attack": optional_values["reference_attack"],
                "reference_defense": optional_values["reference_defense"],
                "reference_recent_form": optional_values["reference_recent_form"],
                "reference_overall_strength": overall,
                "source": _text_value(row.get("source", "")),
                "last_updated": _text_value(row.get("last_updated", "")) or today_iso(),
                "notes": _proposed_notes(row, has_strength_input, has_optional_reference),
            }
        )
        diagnostic_rows.append(
            {
                "team": raw_team,
                "canonical_team": canonical_team,
                "matched_external_prior": bool(matched),
                "has_strength_input": bool(has_strength_input),
                "has_optional_reference_input": bool(has_optional_reference),
                "update_ready": bool(update_ready),
                "warning": "; ".join(warnings),
            }
        )

    return (
        pd.DataFrame(update_rows, columns=EXTERNAL_PRIOR_COLUMNS),
        pd.DataFrame(diagnostic_rows, columns=EXTERNAL_PRIOR_IMPORT_DIAGNOSTIC_COLUMNS),
    )


def merge_external_prior_updates(
    existing_priors_df: pd.DataFrame | None,
    proposed_updates_df: pd.DataFrame | None,
    diagnostics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    existing = existing_priors_df.copy() if existing_priors_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_COLUMNS)
    proposed = proposed_updates_df.copy() if proposed_updates_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_COLUMNS)
    diagnostics = diagnostics_df.copy() if diagnostics_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_IMPORT_DIAGNOSTIC_COLUMNS)
    for col in EXTERNAL_PRIOR_COLUMNS:
        if col not in existing.columns:
            existing[col] = pd.NA
        if col not in proposed.columns:
            proposed[col] = pd.NA
    existing = _normalise_external_prior_dtypes(existing)
    proposed = _normalise_external_prior_dtypes(proposed)
    if existing.empty or proposed.empty:
        return existing[EXTERNAL_PRIOR_COLUMNS].reset_index(drop=True)

    ready_teams = _ready_team_set(proposed, diagnostics)
    out = _normalise_external_prior_dtypes(existing.copy())
    for _, row in proposed.iterrows():
        team = normalize_team_name(_text_value(row.get("team", "")))
        if not team or team.lower() not in ready_teams:
            continue
        mask = out["team"].astype(str).map(lambda value: normalize_team_name(value).lower()) == team.lower()
        if not mask.any():
            continue
        idx = out.loc[mask].index[0]
        changed = False
        for col in ["reference_attack", "reference_defense", "reference_recent_form", "reference_overall_strength"]:
            value = row.get(col)
            if _is_number(value):
                out.at[idx, col] = round(float(value), 3)
                changed = True
        if changed:
            for col in ["source", "last_updated", "notes"]:
                value = _text_value(row.get(col, ""))
                if value:
                    out.at[idx, col] = value
    return out[EXTERNAL_PRIOR_COLUMNS].reset_index(drop=True)


def calculate_generated_rating_disagreements(
    proposed_updates_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None,
    threshold: float = 0.12,
) -> pd.DataFrame:
    proposed = proposed_updates_df.copy() if proposed_updates_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_COLUMNS)
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in proposed.iterrows():
        team = normalize_team_name(_text_value(row.get("team", "")))
        reference_overall = coerce_float(row.get("reference_overall_strength"), float("nan"))
        rating = rating_row_for_team(ratings, team)
        if rating.empty or pd.isna(reference_overall):
            continue
        values = [coerce_float(rating.get(col), float("nan")) for col in ["attack", "defense", "recent_form"]]
        values = [value for value in values if not pd.isna(value)]
        if not values:
            continue
        generated_strength = sum(values) / len(values)
        diff = round(generated_strength - reference_overall, 3)
        warning = "large disagreement with generated ratings" if abs(diff) >= float(threshold) else ""
        rows.append(
            {
                "team": team,
                "generated_rating_strength": round(generated_strength, 3),
                "reference_overall_strength": round(reference_overall, 3),
                "strength_disagreement": diff,
                "warning": warning,
            }
        )
    return pd.DataFrame(rows, columns=GENERATED_DISAGREEMENT_COLUMNS)


def import_diagnostics_summary(
    diagnostics_df: pd.DataFrame | None,
    updated_count: int = 0,
) -> dict[str, int]:
    diagnostics = diagnostics_df.copy() if diagnostics_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_IMPORT_DIAGNOSTIC_COLUMNS)
    if diagnostics.empty:
        return {
            "input_teams_read": 0,
            "teams_matched_to_external_prior_table": 0,
            "teams_not_found": 0,
            "teams_updated": int(updated_count),
            "teams_skipped": 0,
            "missing_rank_elo_data": 0,
        }
    return {
        "input_teams_read": int(len(diagnostics)),
        "teams_matched_to_external_prior_table": int(diagnostics["matched_external_prior"].astype(bool).sum()),
        "teams_not_found": int((~diagnostics["matched_external_prior"].astype(bool)).sum()),
        "teams_updated": int(updated_count),
        "teams_skipped": int((~diagnostics["update_ready"].astype(bool)).sum()),
        "missing_rank_elo_data": int(diagnostics["warning"].astype(str).str.contains("missing rank/Elo data", case=False, na=False).sum()),
    }


def _ready_team_set(proposed: pd.DataFrame, diagnostics: pd.DataFrame) -> set[str]:
    if diagnostics is None or diagnostics.empty or "canonical_team" not in diagnostics.columns or "update_ready" not in diagnostics.columns:
        return {
            normalize_team_name(row.get("team", "")).lower()
            for _, row in proposed.iterrows()
            if any(_is_number(row.get(col)) for col in ["reference_attack", "reference_defense", "reference_recent_form", "reference_overall_strength"])
        }
    return {
        normalize_team_name(row.get("canonical_team", "")).lower()
        for _, row in diagnostics.iterrows()
        if bool(row.get("update_ready", False))
    }


def _normalise_external_prior_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EXTERNAL_PRIOR_TEXT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("object")
    for col in EXTERNAL_PRIOR_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _optional_reference(value: Any) -> float | pd._libs.missing.NAType:
    numeric = coerce_float(value, float("nan"))
    if pd.isna(numeric):
        return pd.NA
    return round(clamp(numeric, 0.35, 0.90), 3)


def _proposed_notes(row: pd.Series, has_strength_input: bool, has_optional_reference: bool) -> str:
    notes = _text_value(row.get("notes", ""))
    generated_notes: list[str] = []
    if has_strength_input:
        generated_notes.append("Overall strength imported from manual rank/points/Elo input.")
    else:
        generated_notes.append("No rank/points/Elo supplied; reference_overall_strength left blank.")
    if has_optional_reference:
        generated_notes.append("Attack/defense/form imported only because explicit reference columns were supplied.")
    else:
        generated_notes.append("Attack/defense/form left blank; not inferred from overall strength.")
    return " ".join(part for part in [notes, " ".join(generated_notes)] if part).strip()


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return DATA_DIR.parent / resolved


def _is_number(value: Any) -> bool:
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return False


def _text_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
