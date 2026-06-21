from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.ratings import TEAM_RATING_COLUMNS, classify_rating_status, exact_rating_row_for_team, rating_row_for_team
from src.team_names import normalize_team_name
from src.utils import clamp, coerce_float, read_csv_with_columns, today_iso


MAJOR_REVIEW_TEAMS = {
    "Argentina",
    "Brazil",
    "France",
    "Germany",
    "Spain",
    "Netherlands",
    "Portugal",
    "England",
    "Uruguay",
    "Morocco",
    "United States",
    "Norway",
    "Belgium",
    "Croatia",
    "Mexico",
    "Colombia",
    "Switzerland",
}

RATING_REVIEW_AUDIT_COLUMNS = [
    "team",
    "rating_status",
    "data_quality",
    "attack",
    "defense",
    "recent_form",
    "elo",
    "behavior_attack_final",
    "behavior_defense_final",
    "behavior_recent_form",
    "matches_used_recent",
    "mean_opponent_elo_recent",
    "backtest_match_count",
    "fixture_match_count",
    "priority",
    "warning",
]

REVIEW_PROPOSAL_COLUMNS = [
    "team",
    "rating_status",
    "current_attack",
    "current_defense",
    "current_recent_form",
    "current_elo",
    "behavior_attack_final",
    "behavior_defense_final",
    "behavior_recent_form",
    "elo_scaled_attack",
    "elo_scaled_defense",
    "reviewed_attack",
    "reviewed_defense",
    "reviewed_recent_form",
    "current_data_quality",
    "recommended_data_quality",
    "review_action",
    "notes",
]


def audit_generated_ratings(
    team_ratings_df: pd.DataFrame | None,
    team_behavior_df: pd.DataFrame | None,
    historical_matches_df: pd.DataFrame | None,
    backtest_matches_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame()
    behavior = team_behavior_df.copy() if team_behavior_df is not None else pd.DataFrame()
    backtest = backtest_matches_df.copy() if backtest_matches_df is not None else pd.DataFrame()
    fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", ["home", "away"])

    raw_teams = (
        set(_team_names(ratings))
        | set(_team_names(behavior))
        | _teams_from_matches(backtest)
        | _teams_from_matches(fixtures)
        | _teams_from_historical(historical_matches_df)
    )
    teams = {normalize_team_name(team) for team in raw_teams if str(team).strip()}
    if not teams:
        return pd.DataFrame(columns=RATING_REVIEW_AUDIT_COLUMNS)

    backtest_counts = _team_counts(backtest)
    fixture_counts = _team_counts(fixtures)
    rows: list[dict[str, Any]] = []
    for team in sorted(teams):
        rating_row = rating_row_for_team(ratings, team)
        behavior_row = rating_row_for_team(behavior, team)
        status = classify_rating_status(rating_row)
        data_quality = _text_value(rating_row.get("data_quality", "")) if not rating_row.empty else ""
        attack = _value_or_na(rating_row, "attack")
        defense = _value_or_na(rating_row, "defense")
        form = _value_or_na(rating_row, "recent_form")
        behavior_attack = _behavior_value(behavior_row, ["attack_index_final", "attack_index", "attack_index_residual_robust"])
        behavior_defense = _behavior_value(behavior_row, ["defense_index_final", "defense_index", "defense_index_residual_robust"])
        behavior_form = _behavior_value(behavior_row, ["recent_form_index"])
        backtest_count = int(backtest_counts.get(team, 0))
        fixture_count = int(fixture_counts.get(team, 0))
        priority = classify_review_priority(team, status, backtest_count, fixture_count)
        warnings = _review_warnings(status, attack, defense, form, behavior_attack, behavior_defense, behavior_form)

        rows.append(
            {
                "team": team,
                "rating_status": status,
                "data_quality": data_quality,
                "attack": attack,
                "defense": defense,
                "recent_form": form,
                "elo": _value_or_na(rating_row, "elo"),
                "behavior_attack_final": behavior_attack,
                "behavior_defense_final": behavior_defense,
                "behavior_recent_form": behavior_form,
                "matches_used_recent": _value_or_na(behavior_row, "matches_used_recent"),
                "mean_opponent_elo_recent": _value_or_na(behavior_row, "mean_opponent_elo_recent"),
                "backtest_match_count": backtest_count,
                "fixture_match_count": fixture_count,
                "priority": priority,
                "warning": warnings,
            }
        )
    return pd.DataFrame(rows, columns=RATING_REVIEW_AUDIT_COLUMNS).sort_values(
        ["priority", "backtest_match_count", "fixture_match_count", "team"],
        ascending=[True, False, False, True],
        key=lambda col: col.map(_priority_sort) if col.name == "priority" else col,
    ).reset_index(drop=True)


def classify_review_priority(team: str, rating_status: str, backtest_match_count: int = 0, fixture_match_count: int = 0) -> str:
    canonical = normalize_team_name(team)
    needs_review = rating_status in {"generated_from_behavior", "neutral_placeholder", "missing"}
    if needs_review and (canonical in MAJOR_REVIEW_TEAMS or backtest_match_count > 0):
        return "high"
    if needs_review and fixture_match_count > 0:
        return "medium"
    if needs_review:
        return "low"
    return "reviewed"


def propose_reviewed_rating(
    team: str,
    team_ratings_df: pd.DataFrame | None,
    team_behavior_df: pd.DataFrame | None,
) -> dict[str, Any]:
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame()
    behavior = team_behavior_df.copy() if team_behavior_df is not None else pd.DataFrame()
    rating_row = rating_row_for_team(ratings, team)
    if rating_row.empty:
        return _empty_proposal(team, "missing", "No rating row exists; add a generated row before manual review.")

    status = classify_rating_status(rating_row)
    behavior_row = rating_row_for_team(behavior, team)
    current_attack = coerce_float(rating_row.get("attack"), 0.55)
    current_defense = coerce_float(rating_row.get("defense"), 0.55)
    current_form = coerce_float(rating_row.get("recent_form"), 0.55)
    current_elo = coerce_float(rating_row.get("elo"), 1700.0)
    behavior_attack = _behavior_value(behavior_row, ["attack_index_final", "attack_index", "attack_index_residual_robust"], current_attack)
    behavior_defense = _behavior_value(behavior_row, ["defense_index_final", "defense_index", "defense_index_residual_robust"], current_defense)
    behavior_form = _behavior_value(behavior_row, ["recent_form_index"], current_form)
    elo_scaled = _elo_scaled_value(current_elo, ratings)

    if status == "manual_reviewed":
        action = "skip_manual_reviewed"
        recommended_quality = _text_value(rating_row.get("data_quality", "")) or "manual_reviewed"
        notes = "Manual reviewed row preserved; no generated review proposal."
        reviewed_attack, reviewed_defense, reviewed_form = current_attack, current_defense, current_form
    elif status == "manual_existing":
        action = "preserve_existing_manual"
        recommended_quality = _text_value(rating_row.get("data_quality", "")) or "manual_csv"
        notes = "Existing manual rating preserved; no generated review proposal."
        reviewed_attack, reviewed_defense, reviewed_form = current_attack, current_defense, current_form
    else:
        action = "review_candidate"
        recommended_quality = "manual_review_candidate"
        reviewed_attack = clamp(0.60 * current_attack + 0.25 * behavior_attack + 0.15 * elo_scaled, 0.35, 0.90)
        reviewed_defense = clamp(0.60 * current_defense + 0.25 * behavior_defense + 0.15 * elo_scaled, 0.35, 0.90)
        reviewed_form = clamp(0.60 * current_form + 0.40 * behavior_form, 0.35, 0.90)
        notes = "Generated review suggestion; requires human confirmation. Source row was generated and should remain marked until reviewed."

    return {
        "team": _text_value(rating_row.get("team", team)) or team,
        "rating_status": status,
        "current_attack": round(current_attack, 3),
        "current_defense": round(current_defense, 3),
        "current_recent_form": round(current_form, 3),
        "current_elo": round(current_elo, 1),
        "behavior_attack_final": round(behavior_attack, 3),
        "behavior_defense_final": round(behavior_defense, 3),
        "behavior_recent_form": round(behavior_form, 3),
        "elo_scaled_attack": round(elo_scaled, 3),
        "elo_scaled_defense": round(elo_scaled, 3),
        "reviewed_attack": round(float(reviewed_attack), 3),
        "reviewed_defense": round(float(reviewed_defense), 3),
        "reviewed_recent_form": round(float(reviewed_form), 3),
        "current_data_quality": _text_value(rating_row.get("data_quality", "")),
        "recommended_data_quality": recommended_quality,
        "review_action": action,
        "notes": notes,
    }


def build_review_proposals(
    teams: list[str] | tuple[str, ...] | None,
    team_ratings_df: pd.DataFrame,
    team_behavior_df: pd.DataFrame,
    audit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if teams:
        selected = [str(team) for team in teams if str(team).strip()]
    else:
        audit = audit_df.copy() if audit_df is not None else audit_generated_ratings(team_ratings_df, team_behavior_df, pd.DataFrame())
        selected = audit.loc[audit["priority"].isin(["high", "medium"]), "team"].dropna().astype(str).tolist()
    rows = [propose_reviewed_rating(team, team_ratings_df, team_behavior_df) for team in selected]
    rows = [row for row in rows if row]
    return pd.DataFrame(rows, columns=REVIEW_PROPOSAL_COLUMNS)


def apply_review_proposals(team_ratings_df: pd.DataFrame, proposals_df: pd.DataFrame) -> pd.DataFrame:
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    proposals = proposals_df.copy() if proposals_df is not None else pd.DataFrame(columns=REVIEW_PROPOSAL_COLUMNS)
    for col in TEAM_RATING_COLUMNS:
        if col not in ratings.columns:
            ratings[col] = pd.NA
    if ratings.empty or proposals.empty:
        return ratings[TEAM_RATING_COLUMNS].reset_index(drop=True)

    out = ratings.copy()
    for _, proposal in proposals.iterrows():
        team = _text_value(proposal.get("team", ""))
        if not team:
            continue
        rows = out.loc[out["team"].astype(str).str.lower() == team.lower()]
        if rows.empty:
            continue
        idx = rows.index[0]
        quality_raw = out.at[idx, "data_quality"]
        quality = "" if pd.isna(quality_raw) else str(quality_raw)
        if "manual_reviewed" in quality.lower() or quality != "generated_from_behavior":
            continue
        out.at[idx, "attack"] = round(coerce_float(proposal.get("reviewed_attack"), out.at[idx, "attack"]), 3)
        out.at[idx, "defense"] = round(coerce_float(proposal.get("reviewed_defense"), out.at[idx, "defense"]), 3)
        out.at[idx, "recent_form"] = round(coerce_float(proposal.get("reviewed_recent_form"), out.at[idx, "recent_form"]), 3)
        out.at[idx, "data_quality"] = "manual_review_candidate"
        notes_raw = out.at[idx, "notes"]
        existing_notes = "" if pd.isna(notes_raw) else str(notes_raw)
        review_note = "Manual review candidate generated from behavior/Elo; requires human confirmation."
        out.at[idx, "notes"] = f"{existing_notes} {review_note}".strip()
        out.at[idx, "last_updated"] = today_iso()
    return out[TEAM_RATING_COLUMNS].reset_index(drop=True)


def rating_review_summary(audit_df: pd.DataFrame | None) -> dict[str, Any]:
    audit = audit_df.copy() if audit_df is not None else pd.DataFrame()
    if audit.empty or "rating_status" not in audit.columns:
        return {
            "manual_reviewed_rows": 0,
            "manual_existing_rows": 0,
            "generated_from_behavior_rows": 0,
            "neutral_placeholder_rows": 0,
            "high_priority_needing_review": 0,
        }
    return {
        "manual_reviewed_rows": int((audit["rating_status"] == "manual_reviewed").sum()),
        "manual_existing_rows": int((audit["rating_status"] == "manual_existing").sum()),
        "generated_from_behavior_rows": int((audit["rating_status"] == "generated_from_behavior").sum()),
        "neutral_placeholder_rows": int((audit["rating_status"] == "neutral_placeholder").sum()),
        "high_priority_needing_review": int(
            ((audit["priority"] == "high") & audit["rating_status"].isin(["generated_from_behavior", "neutral_placeholder", "missing"])).sum()
        )
        if "priority" in audit.columns
        else 0,
    }


def _review_warnings(
    status: str,
    attack: Any,
    defense: Any,
    form: Any,
    behavior_attack: Any,
    behavior_defense: Any,
    behavior_form: Any,
) -> str:
    warnings: list[str] = []
    if status == "generated_from_behavior":
        warnings.append("generated rating requires review")
    if status == "neutral_placeholder":
        warnings.append("neutral placeholder requires review")
    if status == "missing":
        warnings.append("missing rating row")
    for label, current, behavior in [
        ("attack", attack, behavior_attack),
        ("defense", defense, behavior_defense),
        ("recent_form", form, behavior_form),
    ]:
        current_value = coerce_float(current, float("nan"))
        behavior_value = coerce_float(behavior, float("nan"))
        if not pd.isna(current_value) and not pd.isna(behavior_value) and abs(current_value - behavior_value) >= 0.15:
            warnings.append(f"{label} rating differs from behavior")
    return "; ".join(dict.fromkeys(warnings))


def _empty_proposal(team: str, status: str, notes: str) -> dict[str, Any]:
    return {
        "team": team,
        "rating_status": status,
        "current_attack": pd.NA,
        "current_defense": pd.NA,
        "current_recent_form": pd.NA,
        "current_elo": pd.NA,
        "behavior_attack_final": pd.NA,
        "behavior_defense_final": pd.NA,
        "behavior_recent_form": pd.NA,
        "elo_scaled_attack": pd.NA,
        "elo_scaled_defense": pd.NA,
        "reviewed_attack": pd.NA,
        "reviewed_defense": pd.NA,
        "reviewed_recent_form": pd.NA,
        "current_data_quality": "",
        "recommended_data_quality": "",
        "review_action": "missing",
        "notes": notes,
    }


def _elo_scaled_value(elo: float, ratings: pd.DataFrame) -> float:
    if ratings is None or ratings.empty or "elo" not in ratings.columns:
        return 0.55
    values = pd.to_numeric(ratings["elo"], errors="coerce").dropna()
    if values.empty:
        return 0.55
    percentile = float((values <= float(elo)).mean())
    return float(clamp(0.35 + percentile * 0.55, 0.35, 0.90))


def _team_names(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "team" not in df.columns:
        return []
    return sorted({str(team).strip() for team in df["team"].dropna() if str(team).strip()})


def _teams_from_matches(df: pd.DataFrame) -> set[str]:
    teams: set[str] = set()
    if df is None or df.empty:
        return teams
    for col in ["home", "away"]:
        if col in df.columns:
            teams.update(team for team in df[col].dropna().astype(str).str.strip() if team)
    return teams


def _teams_from_historical(df: pd.DataFrame | None) -> set[str]:
    teams: set[str] = set()
    if df is None or df.empty:
        return teams
    for col in ["team", "opponent"]:
        if col in df.columns:
            teams.update(team for team in df[col].dropna().astype(str).str.strip() if team)
    return teams


def _team_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if df is None or df.empty:
        return counts
    for col in ["home", "away"]:
        if col not in df.columns:
            continue
        for raw_team in df[col].dropna().astype(str).str.strip():
            if not raw_team:
                continue
            canonical = normalize_team_name(raw_team)
            counts[raw_team] = counts.get(raw_team, 0) + 1
            if canonical != raw_team:
                counts[canonical] = counts.get(canonical, 0) + 1
    return counts


def _behavior_value(row: pd.Series, columns: list[str], default: float | None = None) -> float | pd._libs.missing.NAType:
    if row is None or row.empty:
        return pd.NA if default is None else default
    for col in columns:
        value = coerce_float(row.get(col), float("nan"))
        if not pd.isna(value):
            return round(float(value), 4)
    return pd.NA if default is None else default


def _value_or_na(row: pd.Series, column: str) -> float | pd._libs.missing.NAType:
    if row is None or row.empty or column not in row.index or pd.isna(row.get(column)):
        return pd.NA
    return round(coerce_float(row.get(column), float("nan")), 4)


def _priority_sort(value: Any) -> int:
    return {"high": 0, "medium": 1, "low": 2, "reviewed": 3}.get(str(value), 4)


def _text_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)
