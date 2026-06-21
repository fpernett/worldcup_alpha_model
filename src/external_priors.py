from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.ratings import TEAM_RATING_COLUMNS, classify_rating_status, rating_row_for_team
from src.rating_review import RATING_REVIEW_AUDIT_COLUMNS, REVIEW_PROPOSAL_COLUMNS
from src.team_names import normalize_team_name
from src.utils import clamp, coerce_float, read_csv_with_columns, today_iso


EXTERNAL_PRIOR_COLUMNS = [
    "team",
    "reference_attack",
    "reference_defense",
    "reference_recent_form",
    "reference_overall_strength",
    "source",
    "last_updated",
    "notes",
]

EXTERNAL_PRIOR_COMPARISON_COLUMNS = [
    "team",
    "current_attack",
    "current_defense",
    "current_recent_form",
    "proposal_attack",
    "proposal_defense",
    "proposal_recent_form",
    "reference_attack",
    "reference_defense",
    "reference_recent_form",
    "reference_overall_strength",
    "attack_disagreement",
    "defense_disagreement",
    "form_disagreement",
    "overall_warning",
    "recommended_action",
]

MANUAL_REVIEW_SHEET_COLUMNS = [
    "team",
    "rating_status",
    "priority",
    "current_attack",
    "current_defense",
    "current_recent_form",
    "proposal_attack",
    "proposal_defense",
    "proposal_recent_form",
    "reference_attack",
    "reference_defense",
    "reference_recent_form",
    "recommended_attack",
    "recommended_defense",
    "recommended_recent_form",
    "human_decision",
    "human_notes",
    "review_status",
]

DEFAULT_EXTERNAL_PRIOR_PATH = DATA_DIR / "team_rating_external_priors.csv"
DEFAULT_REVIEW_SHEET_PATH = DATA_DIR / "manual_rating_review_sheet.csv"
DISAGREEMENT_THRESHOLD = 0.12
PROPOSAL_DIRECTION_THRESHOLD = 0.10


def load_external_priors(path: str | Path = DEFAULT_EXTERNAL_PRIOR_PATH) -> pd.DataFrame:
    priors = read_csv_with_columns(_resolve_path(path), EXTERNAL_PRIOR_COLUMNS)
    if priors.empty:
        return priors
    priors = priors[EXTERNAL_PRIOR_COLUMNS].copy()
    priors["team"] = priors["team"].astype(str).str.strip()
    priors = priors.loc[priors["team"] != ""].copy()
    for col in ["reference_attack", "reference_defense", "reference_recent_form", "reference_overall_strength"]:
        priors[col] = pd.to_numeric(priors[col], errors="coerce")
    return priors.reset_index(drop=True)


def load_review_proposals(path: str | Path = DATA_DIR / "team_ratings_review_proposals.csv") -> pd.DataFrame:
    return read_csv_with_columns(_resolve_path(path), REVIEW_PROPOSAL_COLUMNS)


def load_manual_rating_review_sheet(path: str | Path = DEFAULT_REVIEW_SHEET_PATH) -> pd.DataFrame:
    return read_csv_with_columns(_resolve_path(path), MANUAL_REVIEW_SHEET_COLUMNS)


def compare_ratings_to_external_priors(
    team_ratings_df: pd.DataFrame | None,
    review_proposals_df: pd.DataFrame | None,
    external_priors_df: pd.DataFrame | None,
) -> pd.DataFrame:
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    proposals = review_proposals_df.copy() if review_proposals_df is not None else pd.DataFrame(columns=REVIEW_PROPOSAL_COLUMNS)
    priors = external_priors_df.copy() if external_priors_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_COLUMNS)

    teams = _canonical_team_union(ratings, proposals, priors)
    rows: list[dict[str, Any]] = []
    for team in teams:
        rating = rating_row_for_team(ratings, team)
        proposal = rating_row_for_team(proposals, team)
        prior = rating_row_for_team(priors, team)
        status = classify_rating_status(rating)

        current_attack = _value_or_na(rating, "attack")
        current_defense = _value_or_na(rating, "defense")
        current_form = _value_or_na(rating, "recent_form")
        proposal_attack = _first_numeric(proposal, ["reviewed_attack", "proposal_attack"])
        proposal_defense = _first_numeric(proposal, ["reviewed_defense", "proposal_defense"])
        proposal_form = _first_numeric(proposal, ["reviewed_recent_form", "proposal_recent_form"])
        reference_attack = _value_or_na(prior, "reference_attack")
        reference_defense = _value_or_na(prior, "reference_defense")
        reference_form = _value_or_na(prior, "reference_recent_form")
        reference_overall = _value_or_na(prior, "reference_overall_strength")

        attack_basis = _numeric_or_fallback(proposal_attack, current_attack)
        defense_basis = _numeric_or_fallback(proposal_defense, current_defense)
        form_basis = _numeric_or_fallback(proposal_form, current_form)
        attack_diff = _difference(attack_basis, reference_attack)
        defense_diff = _difference(defense_basis, reference_defense)
        form_diff = _difference(form_basis, reference_form)

        warnings = _comparison_warnings(
            status,
            reference_attack,
            reference_defense,
            reference_form,
            reference_overall,
            attack_diff,
            defense_diff,
            form_diff,
        )
        rows.append(
            {
                "team": team,
                "current_attack": current_attack,
                "current_defense": current_defense,
                "current_recent_form": current_form,
                "proposal_attack": proposal_attack,
                "proposal_defense": proposal_defense,
                "proposal_recent_form": proposal_form,
                "reference_attack": reference_attack,
                "reference_defense": reference_defense,
                "reference_recent_form": reference_form,
                "reference_overall_strength": reference_overall,
                "attack_disagreement": attack_diff,
                "defense_disagreement": defense_diff,
                "form_disagreement": form_diff,
                "overall_warning": "; ".join(warnings),
                "recommended_action": _recommended_action(warnings, status),
            }
        )
    return pd.DataFrame(rows, columns=EXTERNAL_PRIOR_COMPARISON_COLUMNS)


def build_manual_rating_review_sheet(
    team_ratings_df: pd.DataFrame | None,
    rating_review_audit_df: pd.DataFrame | None,
    review_proposals_df: pd.DataFrame | None,
    external_priors_df: pd.DataFrame | None,
    teams: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    audit = rating_review_audit_df.copy() if rating_review_audit_df is not None else pd.DataFrame(columns=RATING_REVIEW_AUDIT_COLUMNS)
    proposals = review_proposals_df.copy() if review_proposals_df is not None else pd.DataFrame(columns=REVIEW_PROPOSAL_COLUMNS)
    priors = external_priors_df.copy() if external_priors_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_COLUMNS)
    comparison = compare_ratings_to_external_priors(ratings, proposals, priors)

    selected = _selected_teams(teams, audit, comparison)
    rows: list[dict[str, Any]] = []
    for team in selected:
        comparison_row = rating_row_for_team(comparison, team)
        audit_row = rating_row_for_team(audit, team)
        rating = rating_row_for_team(ratings, team)
        status = _text_value(audit_row.get("rating_status", "")) if not audit_row.empty else classify_rating_status(rating)
        priority = _text_value(audit_row.get("priority", "")) if not audit_row.empty else "reviewed"
        current_attack = _first_existing(comparison_row, "current_attack", rating, "attack")
        current_defense = _first_existing(comparison_row, "current_defense", rating, "defense")
        current_form = _first_existing(comparison_row, "current_recent_form", rating, "recent_form")
        proposal_attack = _numeric_or_fallback(_value_or_na(comparison_row, "proposal_attack"), current_attack)
        proposal_defense = _numeric_or_fallback(_value_or_na(comparison_row, "proposal_defense"), current_defense)
        proposal_form = _numeric_or_fallback(_value_or_na(comparison_row, "proposal_recent_form"), current_form)
        reference_attack = _value_or_na(comparison_row, "reference_attack")
        reference_defense = _value_or_na(comparison_row, "reference_defense")
        reference_form = _value_or_na(comparison_row, "reference_recent_form")

        recommended_attack = _recommended_metric(proposal_attack, reference_attack, proposal_weight=0.50)
        recommended_defense = _recommended_metric(proposal_defense, reference_defense, proposal_weight=0.50)
        recommended_form = _recommended_metric(proposal_form, reference_form, proposal_weight=0.60)
        warnings = _text_value(comparison_row.get("overall_warning", ""))

        rows.append(
            {
                "team": normalize_team_name(team),
                "rating_status": status,
                "priority": priority,
                "current_attack": current_attack,
                "current_defense": current_defense,
                "current_recent_form": current_form,
                "proposal_attack": proposal_attack,
                "proposal_defense": proposal_defense,
                "proposal_recent_form": proposal_form,
                "reference_attack": reference_attack,
                "reference_defense": reference_defense,
                "reference_recent_form": reference_form,
                "recommended_attack": recommended_attack,
                "recommended_defense": recommended_defense,
                "recommended_recent_form": recommended_form,
                "human_decision": "",
                "human_notes": "",
                "review_status": _review_status(status, warnings),
            }
        )
    return pd.DataFrame(rows, columns=MANUAL_REVIEW_SHEET_COLUMNS)


def apply_approved_review_sheet(team_ratings_df: pd.DataFrame | None, review_sheet_df: pd.DataFrame | None) -> pd.DataFrame:
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    sheet = review_sheet_df.copy() if review_sheet_df is not None else pd.DataFrame(columns=MANUAL_REVIEW_SHEET_COLUMNS)
    for col in TEAM_RATING_COLUMNS:
        if col not in ratings.columns:
            ratings[col] = pd.NA
    for col in MANUAL_REVIEW_SHEET_COLUMNS:
        if col not in sheet.columns:
            sheet[col] = pd.NA
    if ratings.empty or sheet.empty:
        return ratings[TEAM_RATING_COLUMNS].reset_index(drop=True)

    out = ratings.copy()
    for _, row in sheet.iterrows():
        if _text_value(row.get("review_status", "")).lower() != "approved":
            continue
        team = normalize_team_name(_text_value(row.get("team", "")))
        if not team:
            continue
        mask = out["team"].astype(str).map(lambda value: normalize_team_name(value).lower()) == team.lower()
        if not mask.any():
            continue
        idx = out.loc[mask].index[0]
        quality = _text_value(out.at[idx, "data_quality"])
        if quality not in {"generated_from_behavior", "manual_review_candidate"}:
            continue
        out.at[idx, "attack"] = round(clamp(coerce_float(row.get("recommended_attack"), out.at[idx, "attack"]), 0.35, 0.90), 3)
        out.at[idx, "defense"] = round(clamp(coerce_float(row.get("recommended_defense"), out.at[idx, "defense"]), 0.35, 0.90), 3)
        out.at[idx, "recent_form"] = round(
            clamp(coerce_float(row.get("recommended_recent_form"), out.at[idx, "recent_form"]), 0.35, 0.90),
            3,
        )
        out.at[idx, "data_quality"] = "manual_reviewed"
        existing_notes = _text_value(out.at[idx, "notes"])
        human_notes = _text_value(row.get("human_notes", ""))
        review_note = "External prior review approved; manual reviewed prior."
        out.at[idx, "notes"] = " ".join(part for part in [existing_notes, review_note, human_notes] if part).strip()
        out.at[idx, "last_updated"] = today_iso()
    return out[TEAM_RATING_COLUMNS].reset_index(drop=True)


def external_prior_review_summary(
    comparison_df: pd.DataFrame | None,
    review_sheet_df: pd.DataFrame | None = None,
) -> dict[str, int]:
    comparison = comparison_df.copy() if comparison_df is not None else pd.DataFrame(columns=EXTERNAL_PRIOR_COMPARISON_COLUMNS)
    sheet = review_sheet_df.copy() if review_sheet_df is not None else pd.DataFrame(columns=MANUAL_REVIEW_SHEET_COLUMNS)
    if comparison.empty:
        return {
            "teams_with_external_priors": 0,
            "teams_missing_external_priors": 0,
            "generated_without_external_check": 0,
            "large_disagreement_warnings": 0,
            "worksheet_rows": int(len(sheet)),
            "approved_review_rows": _approved_count(sheet),
        }
    warnings = comparison["overall_warning"].astype(str) if "overall_warning" in comparison.columns else pd.Series("", index=comparison.index)
    return {
        "teams_with_external_priors": int((~warnings.str.contains("missing external prior", case=False, na=False)).sum()),
        "teams_missing_external_priors": int(warnings.str.contains("missing external prior", case=False, na=False).sum()),
        "generated_without_external_check": int(warnings.str.contains("generated rating without external check", case=False, na=False).sum()),
        "large_disagreement_warnings": int(warnings.str.contains("large ", case=False, na=False).sum()),
        "worksheet_rows": int(len(sheet)),
        "approved_review_rows": _approved_count(sheet),
    }


def _comparison_warnings(
    rating_status: str,
    reference_attack: Any,
    reference_defense: Any,
    reference_form: Any,
    reference_overall: Any,
    attack_diff: Any,
    defense_diff: Any,
    form_diff: Any,
) -> list[str]:
    warnings: list[str] = []
    has_external = any(_is_number(value) for value in [reference_attack, reference_defense, reference_form, reference_overall])
    if not has_external:
        warnings.append("missing external prior")
    if rating_status in {"generated_from_behavior", "neutral_placeholder"} and not has_external:
        warnings.append("generated rating without external check")
    for label, diff in [("attack", attack_diff), ("defense", defense_diff), ("form", form_diff)]:
        if not _is_number(diff):
            continue
        if abs(float(diff)) >= DISAGREEMENT_THRESHOLD:
            warnings.append(f"large {label} disagreement")
    directional = [value for value in [attack_diff, defense_diff, form_diff] if _is_number(value)]
    if directional and any(float(value) >= PROPOSAL_DIRECTION_THRESHOLD for value in directional):
        warnings.append("proposal above external prior")
    if directional and any(float(value) <= -PROPOSAL_DIRECTION_THRESHOLD for value in directional):
        warnings.append("proposal below external prior")
    return list(dict.fromkeys(warnings))


def _recommended_action(warnings: list[str], rating_status: str) -> str:
    warning_text = "; ".join(warnings)
    if "missing external prior" in warning_text:
        return "fill external prior before approval"
    if "large " in warning_text or "proposal above external prior" in warning_text or "proposal below external prior" in warning_text:
        return "review disagreement before approval"
    if rating_status == "manual_reviewed":
        return "preserve manual reviewed rating"
    return "ready for human review"


def _review_status(rating_status: str, warnings: str) -> str:
    if rating_status == "manual_reviewed":
        return "preserve_manual_reviewed"
    if rating_status == "manual_existing":
        return "preserve_manual_existing"
    if "missing external prior" in warnings:
        return "needs_external_prior"
    if "large " in warnings or "proposal above external prior" in warnings or "proposal below external prior" in warnings:
        return "needs_review"
    return "pending_human_approval"


def _selected_teams(teams: list[str] | tuple[str, ...] | None, audit: pd.DataFrame, comparison: pd.DataFrame) -> list[str]:
    if teams:
        return [normalize_team_name(team) for team in teams if _text_value(team)]
    if audit is not None and not audit.empty and {"priority", "team"}.issubset(audit.columns):
        selected = audit.loc[audit["priority"].isin(["high", "medium"]), "team"].dropna().astype(str).tolist()
        if selected:
            return [normalize_team_name(team) for team in selected]
    if comparison is not None and not comparison.empty and "team" in comparison.columns:
        return comparison["team"].dropna().astype(str).tolist()
    return []


def _canonical_team_union(*frames: pd.DataFrame) -> list[str]:
    teams: set[str] = set()
    for df in frames:
        if df is None or df.empty or "team" not in df.columns:
            continue
        teams.update(normalize_team_name(team) for team in df["team"].dropna().astype(str).str.strip() if team)
    return sorted(teams)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return DATA_DIR.parent / resolved


def _value_or_na(row: pd.Series, column: str) -> float | pd._libs.missing.NAType:
    if row is None or row.empty or column not in row.index:
        return pd.NA
    value = coerce_float(row.get(column), float("nan"))
    if pd.isna(value):
        return pd.NA
    return round(float(value), 3)


def _first_numeric(row: pd.Series, columns: list[str]) -> float | pd._libs.missing.NAType:
    if row is None or row.empty:
        return pd.NA
    for col in columns:
        value = _value_or_na(row, col)
        if _is_number(value):
            return value
    return pd.NA


def _first_existing(primary_row: pd.Series, primary_col: str, fallback_row: pd.Series, fallback_col: str) -> float | pd._libs.missing.NAType:
    primary = _value_or_na(primary_row, primary_col)
    if _is_number(primary):
        return primary
    return _value_or_na(fallback_row, fallback_col)


def _difference(value: Any, reference: Any) -> float | pd._libs.missing.NAType:
    if not _is_number(value) or not _is_number(reference):
        return pd.NA
    return round(float(value) - float(reference), 3)


def _numeric_or_fallback(value: Any, fallback: Any) -> float | pd._libs.missing.NAType:
    return value if _is_number(value) else fallback


def _recommended_metric(proposal: Any, reference: Any, proposal_weight: float) -> float | pd._libs.missing.NAType:
    if not _is_number(proposal):
        return pd.NA
    if not _is_number(reference):
        return round(clamp(float(proposal), 0.35, 0.90), 3)
    value = float(proposal_weight) * float(proposal) + (1.0 - float(proposal_weight)) * float(reference)
    return round(clamp(value, 0.35, 0.90), 3)


def _is_number(value: Any) -> bool:
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return False


def _approved_count(sheet: pd.DataFrame) -> int:
    if sheet.empty or "review_status" not in sheet.columns:
        return 0
    return int(sheet["review_status"].astype(str).str.lower().eq("approved").sum())


def _text_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
