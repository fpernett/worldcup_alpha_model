from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.ratings import TEAM_RATING_COLUMNS, exact_rating_row_for_team, rating_row_for_team
from src.team_names import is_unresolved_team_slot, load_team_name_aliases_df, normalize_team_name, team_name_key
from src.utils import clamp, coerce_float, read_csv_with_columns, today_iso


REQUIRED_TEAM_COLUMNS = [
    "team",
    "source_fixtures",
    "source_historical",
    "source_behavior",
    "source_backtest",
    "required_for_model",
    "in_team_ratings",
    "in_team_behavior",
    "possible_alias",
    "canonical_team",
    "warning",
]

RATING_COVERAGE_COLUMNS = [
    "team",
    "canonical_team",
    "in_team_ratings",
    "rating_source",
    "attack",
    "defense",
    "recent_form",
    "elo",
    "data_quality",
    "possible_alias_match",
    "recommended_action",
    "warning",
]

PROPOSED_ALIAS_COLUMNS = ["alias", "canonical", "reason"]


def collect_required_teams(
    fixtures_df: pd.DataFrame | None,
    historical_matches_df: pd.DataFrame | None,
    team_behavior_df: pd.DataFrame | None,
    backtest_matches_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collect teams that can affect the current model/backtest workflow."""
    fixtures = fixtures_df.copy() if fixtures_df is not None else pd.DataFrame()
    historical = historical_matches_df.copy() if historical_matches_df is not None else pd.DataFrame()
    behavior = team_behavior_df.copy() if team_behavior_df is not None else pd.DataFrame()
    backtest = backtest_matches_df.copy() if backtest_matches_df is not None else pd.DataFrame()
    ratings = read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)
    aliases = load_team_name_aliases_df()

    fixture_teams = _teams_from_columns(fixtures, ["home", "away"])
    historical_teams = _teams_from_columns(historical, ["team", "opponent", "home", "away"])
    behavior_teams = _teams_from_columns(behavior, ["team"])
    backtest_teams = _teams_from_columns(backtest, ["home", "away"])
    all_teams = sorted(fixture_teams | historical_teams | behavior_teams | backtest_teams)

    rows: list[dict[str, Any]] = []
    for team in all_teams:
        source_fixtures = team in fixture_teams
        source_backtest = team in backtest_teams
        source_behavior = team in behavior_teams
        source_historical = team in historical_teams
        required = bool(source_fixtures or source_backtest or (source_behavior and (source_fixtures or source_backtest)))
        in_ratings = not rating_row_for_team(ratings, team).empty
        in_behavior = team in behavior_teams
        canonical = _canonical_team(team, aliases)
        possible_alias = _possible_alias_match(team, ratings, aliases)
        warnings: list[str] = []
        if required and not in_ratings:
            warnings.append("missing rating row")
        if possible_alias and not in_ratings:
            warnings.append("possible alias mismatch")
        if in_behavior and required and not in_ratings:
            warnings.append("behavior exists but manual rating missing")
        if source_fixtures and not in_ratings:
            warnings.append("fixture team missing rating")

        rows.append(
            {
                "team": team,
                "source_fixtures": bool(source_fixtures),
                "source_historical": bool(source_historical),
                "source_behavior": bool(source_behavior),
                "source_backtest": bool(source_backtest),
                "required_for_model": bool(required),
                "in_team_ratings": bool(in_ratings),
                "in_team_behavior": bool(in_behavior),
                "possible_alias": possible_alias,
                "canonical_team": canonical,
                "warning": "; ".join(dict.fromkeys(warnings)),
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_TEAM_COLUMNS)


def audit_rating_coverage(
    required_teams_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None,
    aliases_df: pd.DataFrame | None,
) -> pd.DataFrame:
    required = required_teams_df.copy() if required_teams_df is not None else pd.DataFrame()
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame()
    aliases = aliases_df.copy() if aliases_df is not None else load_team_name_aliases_df()
    if required.empty:
        return pd.DataFrame(columns=RATING_COVERAGE_COLUMNS)

    rows: list[dict[str, Any]] = []
    model_required = required.loc[_required_mask(required)].copy()
    canonical_counts = model_required["canonical_team"].astype(str).value_counts().to_dict() if "canonical_team" in model_required.columns else {}
    for _, team_row in model_required.sort_values("team").iterrows():
        team = str(team_row.get("team", "") or "")
        canonical = str(team_row.get("canonical_team", "") or normalize_team_name(team))
        rating = rating_row_for_team(ratings, team)
        in_ratings = not rating.empty
        possible_alias = _possible_alias_match(team, ratings, aliases)
        warnings: list[str] = []
        if not in_ratings:
            warnings.append("missing rating row")
        if possible_alias and not in_ratings:
            warnings.append("possible alias mismatch")
        if int(canonical_counts.get(canonical, 0)) > 1:
            warnings.append("duplicate canonical names")
        if bool(team_row.get("source_behavior", False)) and not in_ratings:
            warnings.append("behavior exists but manual rating missing")
        if bool(team_row.get("source_fixtures", False)) and not in_ratings:
            warnings.append("fixture team missing rating")

        if not in_ratings:
            action = "add generated manual rating row for review"
        elif possible_alias:
            action = "review alias mapping"
        else:
            action = "none"

        rows.append(
            {
                "team": team,
                "canonical_team": canonical,
                "in_team_ratings": bool(in_ratings),
                "rating_source": str(rating.get("data_quality", "") or "") if in_ratings else "",
                "attack": _value_or_na(rating, "attack"),
                "defense": _value_or_na(rating, "defense"),
                "recent_form": _value_or_na(rating, "recent_form"),
                "elo": _value_or_na(rating, "elo"),
                "data_quality": str(rating.get("data_quality", "") or "") if in_ratings else "",
                "possible_alias_match": possible_alias,
                "recommended_action": action,
                "warning": "; ".join(dict.fromkeys(warnings)),
            }
        )
    return pd.DataFrame(rows, columns=RATING_COVERAGE_COLUMNS)


def propose_missing_team_rating(
    team: str,
    team_behavior_df: pd.DataFrame | None,
    historical_matches_df: pd.DataFrame | None,
    existing_team_ratings_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Return one conservative generated rating row for a missing team."""
    ratings = existing_team_ratings_df.copy() if existing_team_ratings_df is not None else pd.DataFrame()
    if not rating_row_for_team(ratings, team).empty:
        return {}

    canonical_team = normalize_team_name(team)
    behavior = _matching_team_row(team_behavior_df, team)
    if behavior.empty and canonical_team != team:
        behavior = _matching_team_row(team_behavior_df, canonical_team)
    neutral_base = 0.55
    has_behavior = not behavior.empty
    if has_behavior:
        attack_index = _first_numeric(behavior, ["attack_index_final", "attack_index", "attack_index_residual_robust"], neutral_base)
        defense_index = _first_numeric(behavior, ["defense_index_final", "defense_index", "defense_index_residual_robust"], neutral_base)
        form_index = _first_numeric(behavior, ["recent_form_index"], neutral_base)
        attack = clamp(0.70 * neutral_base + 0.30 * attack_index, 0.35, 0.85)
        defense = clamp(0.70 * neutral_base + 0.30 * defense_index, 0.35, 0.85)
        recent_form = clamp(0.70 * neutral_base + 0.30 * form_index, 0.35, 0.85)
        data_quality = "generated_from_behavior"
        notes = "Generated from historical behavior; manual review recommended."
    else:
        attack = defense = recent_form = neutral_base
        data_quality = "generated_neutral_placeholder_low"
        notes = "Generated neutral placeholder; requires manual review."

    elo = _latest_team_elo(team, historical_matches_df)
    if pd.isna(elo) and canonical_team != team:
        elo = _latest_team_elo(canonical_team, historical_matches_df)
    if pd.isna(elo):
        elo = 1700.0
    return {
        "team": canonical_team or team,
        "elo": round(float(clamp(elo, 1200.0, 2200.0)), 1),
        "attack": round(float(attack), 3),
        "defense": round(float(defense), 3),
        "recent_form": round(float(recent_form), 3),
        "fifa_rank_proxy": _rank_from_elo(elo),
        "training_temp_c": 20.0,
        "training_humidity_pct": 60.0,
        "data_quality": data_quality,
        "last_updated": today_iso(),
        "notes": notes,
    }


def build_missing_rating_proposals(
    coverage_audit_df: pd.DataFrame,
    team_behavior_df: pd.DataFrame,
    historical_matches_df: pd.DataFrame,
    existing_team_ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    if coverage_audit_df is None or coverage_audit_df.empty:
        return pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    missing = coverage_audit_df.loc[~coverage_audit_df["in_team_ratings"].astype(bool)].copy()
    rows = []
    proposed_names: set[str] = set()
    for _, row in missing.iterrows():
        team = str(row.get("canonical_team", row.get("team", "")) or row.get("team", ""))
        if not team or team.lower() in proposed_names:
            continue
        proposal = propose_missing_team_rating(team, team_behavior_df, historical_matches_df, existing_team_ratings_df)
        if proposal:
            rows.append(proposal)
            proposed_names.add(str(proposal["team"]).lower())
    if not rows:
        return pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    return pd.DataFrame(rows, columns=TEAM_RATING_COLUMNS).sort_values("team").reset_index(drop=True)


def append_missing_team_ratings(existing_team_ratings_df: pd.DataFrame, proposals_df: pd.DataFrame) -> pd.DataFrame:
    existing = existing_team_ratings_df.copy() if existing_team_ratings_df is not None else pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    proposals = proposals_df.copy() if proposals_df is not None else pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    for col in TEAM_RATING_COLUMNS:
        if col not in existing.columns:
            existing[col] = pd.NA
        if col not in proposals.columns:
            proposals[col] = pd.NA
    existing_names = {str(team).strip().lower() for team in existing["team"].dropna()}
    additions = proposals.loc[~proposals["team"].astype(str).str.lower().isin(existing_names)].copy()
    if additions.empty:
        return existing[TEAM_RATING_COLUMNS].reset_index(drop=True)
    out = pd.concat([existing[TEAM_RATING_COLUMNS], additions[TEAM_RATING_COLUMNS]], ignore_index=True)
    return out[TEAM_RATING_COLUMNS].reset_index(drop=True)


def propose_team_aliases(
    required_teams_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None,
    aliases_df: pd.DataFrame | None,
) -> pd.DataFrame:
    required = required_teams_df.copy() if required_teams_df is not None else pd.DataFrame()
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame()
    aliases = aliases_df.copy() if aliases_df is not None else load_team_name_aliases_df()
    if required.empty:
        return pd.DataFrame(columns=PROPOSED_ALIAS_COLUMNS)

    existing_alias_keys = set(aliases["alias"].astype(str).map(team_name_key)) if not aliases.empty and "alias" in aliases.columns else set()
    rating_names = _rating_names(ratings)
    rows: list[dict[str, str]] = []
    for _, row in required.loc[_required_mask(required)].iterrows():
        team = str(row.get("team", "") or "")
        canonical = str(row.get("canonical_team", "") or normalize_team_name(team))
        exact_rating = not exact_rating_row_for_team(ratings, team).empty
        alias_target = ""
        reason = ""
        if team_name_key(team) in existing_alias_keys:
            continue
        if team_name_key(team) not in existing_alias_keys and canonical != team:
            alias_target = canonical
            reason = "known canonical alias"
        elif not exact_rating:
            possible = _possible_alias_match(team, ratings, aliases)
            if possible:
                alias_target = possible
                reason = "similar or canonical rating name"
        if alias_target:
            rows.append({"alias": team, "canonical": alias_target, "reason": reason})

    # Useful bidirectional accent alias for manual CSV editors.
    names = set(required["team"].dropna().astype(str)) | set(rating_names)
    curacao_key = team_name_key("Curacao")
    if "Curaçao" in names and curacao_key not in existing_alias_keys:
        rows.append({"alias": "Curacao", "canonical": "Curaçao", "reason": "accent-insensitive alias"})
    if "Curacao" in names and curacao_key not in existing_alias_keys:
        rows.append({"alias": "Curaçao", "canonical": "Curacao", "reason": "accent-insensitive alias"})

    out = pd.DataFrame(rows, columns=PROPOSED_ALIAS_COLUMNS)
    if out.empty:
        return out
    out["_key"] = out["alias"].map(team_name_key)
    return out.drop_duplicates(subset=["_key"], keep="first").drop(columns=["_key"]).reset_index(drop=True)


def append_missing_aliases(existing_aliases_df: pd.DataFrame, proposals_df: pd.DataFrame) -> pd.DataFrame:
    existing = existing_aliases_df.copy() if existing_aliases_df is not None else pd.DataFrame(columns=["alias", "canonical"])
    proposals = proposals_df.copy() if proposals_df is not None else pd.DataFrame(columns=PROPOSED_ALIAS_COLUMNS)
    for col in ["alias", "canonical"]:
        if col not in existing.columns:
            existing[col] = pd.NA
        if col not in proposals.columns:
            proposals[col] = pd.NA
    existing_keys = set(existing["alias"].dropna().astype(str).map(team_name_key))
    additions = proposals.loc[~proposals["alias"].astype(str).map(team_name_key).isin(existing_keys), ["alias", "canonical"]].copy()
    if additions.empty:
        return existing[["alias", "canonical"]].reset_index(drop=True)
    return pd.concat([existing[["alias", "canonical"]], additions], ignore_index=True).reset_index(drop=True)


def rating_coverage_summary(coverage_audit_df: pd.DataFrame, alias_proposals_df: pd.DataFrame | None = None) -> dict[str, Any]:
    coverage = coverage_audit_df.copy() if coverage_audit_df is not None else pd.DataFrame()
    aliases = alias_proposals_df.copy() if alias_proposals_df is not None else pd.DataFrame()
    if coverage.empty:
        return {
            "required_teams": 0,
            "teams_in_team_ratings": 0,
            "missing_manual_ratings": 0,
            "generated_rating_rows": 0,
            "neutral_fallback_risk_count": 0,
            "alias_warning_count": 0,
            "recommended_fixes": "No required teams found.",
        }
    missing = coverage.loc[~coverage["in_team_ratings"].astype(bool)]
    generated = coverage.loc[coverage["data_quality"].astype(str).str.contains("generated", case=False, na=False)] if "data_quality" in coverage.columns else pd.DataFrame()
    if len(missing) or len(aliases):
        recommended = "Add missing generated rows and review aliases."
    elif len(generated):
        recommended = "Coverage is clean; review generated rows and replace with manual priors when available."
    else:
        recommended = "Rating coverage is clean."
    return {
        "required_teams": int(len(coverage)),
        "teams_in_team_ratings": int(coverage["in_team_ratings"].astype(bool).sum()),
        "missing_manual_ratings": int(len(missing)),
        "generated_rating_rows": int(len(generated)),
        "neutral_fallback_risk_count": int(len(missing)),
        "alias_warning_count": int(len(aliases)),
        "recommended_fixes": recommended,
    }


def load_team_ratings_csv(path: Path | None = None) -> pd.DataFrame:
    return read_csv_with_columns(path or (DATA_DIR / "team_ratings.csv"), TEAM_RATING_COLUMNS)


def _teams_from_columns(df: pd.DataFrame, columns: list[str]) -> set[str]:
    teams: set[str] = set()
    if df is None or df.empty:
        return teams
    for col in columns:
        if col not in df.columns:
            continue
        teams.update(
            team
            for team in df[col].dropna().astype(str).str.strip()
            if team and not is_unresolved_team_slot(team)
        )
    return teams


def _required_mask(required: pd.DataFrame) -> pd.Series:
    if required is None:
        return pd.Series(dtype=bool)
    if required.empty or "required_for_model" not in required.columns:
        return pd.Series([False] * len(required), index=required.index)
    return required["required_for_model"].astype(bool)


def _canonical_team(team: str, aliases: pd.DataFrame) -> str:
    key = team_name_key(team)
    if aliases is not None and not aliases.empty:
        rows = aliases.loc[aliases["alias"].astype(str).map(team_name_key) == key]
        if not rows.empty:
            return str(rows.iloc[-1].get("canonical", team))
    return normalize_team_name(team)


def _possible_alias_match(team: str, ratings: pd.DataFrame, aliases: pd.DataFrame) -> str:
    if not exact_rating_row_for_team(ratings, team).empty:
        return ""
    alias_key = team_name_key(team)
    if aliases is not None and not aliases.empty and "alias" in aliases.columns and "canonical" in aliases.columns:
        rows = aliases.loc[aliases["alias"].astype(str).map(team_name_key) == alias_key]
        if not rows.empty:
            canonical_from_csv = str(rows.iloc[-1].get("canonical", "") or "")
            if canonical_from_csv and not exact_rating_row_for_team(ratings, canonical_from_csv).empty:
                return ""
    canonical = _canonical_team(team, aliases)
    if canonical and not rating_row_for_team(ratings, canonical).empty:
        return canonical
    rating_names = _rating_names(ratings)
    rating_keys = {team_name_key(name): name for name in rating_names}
    if team_name_key(team) in rating_keys:
        return rating_keys[team_name_key(team)]
    if team_name_key(canonical) in rating_keys:
        return rating_keys[team_name_key(canonical)]
    best = ""
    score = 0.0
    for name in rating_names:
        candidate = SequenceMatcher(None, team_name_key(team), team_name_key(name)).ratio()
        if candidate > score:
            best = name
            score = candidate
    return best if score >= 0.88 else ""


def _rating_names(ratings: pd.DataFrame) -> list[str]:
    if ratings is None or ratings.empty or "team" not in ratings.columns:
        return []
    return sorted({str(team).strip() for team in ratings["team"].dropna() if str(team).strip()})


def _matching_team_row(df: pd.DataFrame | None, team: str) -> pd.Series:
    return rating_row_for_team(df, team)


def _first_numeric(row: pd.Series, columns: list[str], default: float) -> float:
    for col in columns:
        value = coerce_float(row.get(col), float("nan"))
        if not pd.isna(value):
            return value
    return default


def _latest_team_elo(team: str, historical_matches_df: pd.DataFrame | None) -> float:
    if historical_matches_df is None or historical_matches_df.empty:
        return float("nan")
    df = historical_matches_df.copy()
    if "team" not in df.columns or "team_elo_pre" not in df.columns:
        return float("nan")
    df = df.loc[df["team"].astype(str).str.lower() == str(team).lower()].copy()
    if df.empty:
        return float("nan")
    df["date_utc"] = pd.to_datetime(df.get("date_utc"), errors="coerce")
    df["team_elo_pre"] = pd.to_numeric(df["team_elo_pre"], errors="coerce")
    df = df.dropna(subset=["date_utc", "team_elo_pre"]).sort_values("date_utc")
    if df.empty:
        return float("nan")
    return float(df.iloc[-1]["team_elo_pre"])


def _rank_from_elo(elo: float) -> int:
    rank = round((2110.0 - float(elo)) / 5.3 + 1.0)
    return int(clamp(rank, 1, 150))


def _value_or_na(row: pd.Series, column: str) -> float | pd._libs.missing.NAType:
    if row is None or row.empty or column not in row.index or pd.isna(row.get(column)):
        return pd.NA
    return round(coerce_float(row.get(column), float("nan")), 4)
