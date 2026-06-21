from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from src.ratings import is_neutral_fallback_rating, neutral_team_rating, rating_row_for_team
from src.team_names import load_team_name_aliases_df, normalize_team_name, team_name_key
from src.utils import coerce_bool, coerce_float


RATING_COVERAGE_COLUMNS = [
    "team",
    "canonical_team",
    "appears_in_completed_matches",
    "in_manual_ratings",
    "in_adjusted_ratings",
    "in_team_behavior",
    "manual_attack",
    "manual_defense",
    "manual_recent_form",
    "adjusted_attack",
    "adjusted_defense",
    "adjusted_recent_form",
    "behavior_blend_used",
    "used_neutral_fallback",
    "possible_alias_match",
    "warning",
]

MATCH_INPUT_AUDIT_COLUMNS = [
    "match_id",
    "date_utc",
    "home",
    "away",
    "score",
    "actual_result",
    "home_manual_attack",
    "home_manual_defense",
    "home_manual_recent_form",
    "away_manual_attack",
    "away_manual_defense",
    "away_manual_recent_form",
    "home_adjusted_attack",
    "home_adjusted_defense",
    "home_adjusted_recent_form",
    "away_adjusted_attack",
    "away_adjusted_defense",
    "away_adjusted_recent_form",
    "baseline_home_prob",
    "baseline_draw_prob",
    "baseline_away_prob",
    "behavior_home_prob",
    "behavior_draw_prob",
    "behavior_away_prob",
    "baseline_actual_prob",
    "behavior_actual_prob",
    "rating_coverage_warning",
    "probability_warning",
]

ALIAS_AUDIT_COLUMNS = [
    "team_in_backtest",
    "exact_match_in_ratings",
    "possible_alias",
    "canonical_alias",
    "similarity_score",
    "recommendation",
]


def audit_backtest_rating_coverage(
    completed_matches_df: pd.DataFrame | None,
    manual_ratings_df: pd.DataFrame | None,
    adjusted_ratings_df: pd.DataFrame | None,
    team_behavior_df: pd.DataFrame | None,
    aliases_df: pd.DataFrame | None,
) -> pd.DataFrame:
    completed = completed_matches_df.copy() if completed_matches_df is not None else pd.DataFrame()
    teams = _completed_team_counts(completed)
    if not teams:
        return pd.DataFrame(columns=RATING_COVERAGE_COLUMNS)

    manual = manual_ratings_df.copy() if manual_ratings_df is not None else pd.DataFrame()
    adjusted = adjusted_ratings_df.copy() if adjusted_ratings_df is not None else pd.DataFrame()
    behavior = team_behavior_df.copy() if team_behavior_df is not None else pd.DataFrame()
    aliases = _normalise_alias_df(aliases_df)
    canonical_groups: dict[str, list[str]] = {}
    for team in teams:
        canonical_groups.setdefault(_canonical_team(team, aliases), []).append(team)

    rows: list[dict[str, Any]] = []
    for team, count in sorted(teams.items()):
        canonical = _canonical_team(team, aliases)
        manual_row = rating_row_for_team(manual, team)
        adjusted_row = rating_row_for_team(adjusted, team)
        behavior_row = rating_row_for_team(behavior, team)
        in_manual = not manual_row.empty
        in_adjusted = not adjusted_row.empty
        in_behavior = not behavior_row.empty
        possible_alias = _possible_alias_match(team, [manual, adjusted, behavior], aliases)
        used_fallback = (not in_manual) or (not in_adjusted) or is_neutral_fallback_rating(adjusted_row)

        warnings: list[str] = []
        if not in_manual:
            warnings.append("missing manual rating")
        if not in_adjusted:
            warnings.append("missing adjusted rating")
        if not in_behavior:
            warnings.append("missing behavior row")
        if used_fallback:
            warnings.append("neutral fallback used")
        if possible_alias:
            warnings.append("possible alias mismatch")
        if len(set(canonical_groups.get(canonical, []))) > 1:
            warnings.append("team appears under multiple names")

        rows.append(
            {
                "team": team,
                "canonical_team": canonical,
                "appears_in_completed_matches": int(count),
                "in_manual_ratings": bool(in_manual),
                "in_adjusted_ratings": bool(in_adjusted),
                "in_team_behavior": bool(in_behavior),
                "manual_attack": _rating_value_or_na(manual_row, "attack"),
                "manual_defense": _rating_value_or_na(manual_row, "defense"),
                "manual_recent_form": _rating_value_or_na(manual_row, "recent_form"),
                "adjusted_attack": _rating_value_or_na(adjusted_row, "attack"),
                "adjusted_defense": _rating_value_or_na(adjusted_row, "defense"),
                "adjusted_recent_form": _rating_value_or_na(adjusted_row, "recent_form"),
                "behavior_blend_used": coerce_bool(adjusted_row.get("behavior_blend_used", False)) if not adjusted_row.empty else False,
                "used_neutral_fallback": bool(used_fallback),
                "possible_alias_match": possible_alias,
                "warning": "; ".join(dict.fromkeys(warnings)),
            }
        )
    return pd.DataFrame(rows, columns=RATING_COVERAGE_COLUMNS)


def audit_backtest_match_inputs(
    predictions_df: pd.DataFrame | None,
    completed_matches_df: pd.DataFrame | None,
    manual_ratings_df: pd.DataFrame | None,
    adjusted_ratings_df: pd.DataFrame | None,
) -> pd.DataFrame:
    predictions = predictions_df.copy() if predictions_df is not None else pd.DataFrame()
    completed = completed_matches_df.copy() if completed_matches_df is not None else pd.DataFrame()
    manual = manual_ratings_df.copy() if manual_ratings_df is not None else pd.DataFrame()
    adjusted = adjusted_ratings_df.copy() if adjusted_ratings_df is not None else pd.DataFrame()
    if completed.empty:
        return pd.DataFrame(columns=MATCH_INPUT_AUDIT_COLUMNS)

    baseline = predictions.loc[predictions.get("model_mode", "") == "baseline_manual"].copy() if not predictions.empty else pd.DataFrame()
    behavior = predictions.loc[predictions.get("model_mode", "") == "behavior_adjusted"].copy() if not predictions.empty else pd.DataFrame()
    baseline_by_match = _index_by_match_id(baseline)
    behavior_by_match = _index_by_match_id(behavior)

    rows: list[dict[str, Any]] = []
    for _, match in completed.iterrows():
        match_id = str(match.get("match_id", ""))
        home = str(match.get("home", "") or "")
        away = str(match.get("away", "") or "")
        home_manual = _rating_values_used(manual, home)
        away_manual = _rating_values_used(manual, away)
        home_adjusted = _rating_values_used(adjusted, home)
        away_adjusted = _rating_values_used(adjusted, away)
        baseline_row = baseline_by_match.get(match_id, pd.Series(dtype="object"))
        behavior_row = behavior_by_match.get(match_id, pd.Series(dtype="object"))
        rating_warning = _rating_coverage_warning(home, away, manual, adjusted)
        actual_result = _actual_result(match.get("home_goals"), match.get("away_goals"))

        row = {
            "match_id": match_id,
            "date_utc": match.get("date_utc", ""),
            "home": home,
            "away": away,
            "score": f"{int(coerce_float(match.get('home_goals'), 0))}-{int(coerce_float(match.get('away_goals'), 0))}",
            "actual_result": actual_result,
            "home_manual_attack": home_manual["attack"],
            "home_manual_defense": home_manual["defense"],
            "home_manual_recent_form": home_manual["recent_form"],
            "away_manual_attack": away_manual["attack"],
            "away_manual_defense": away_manual["defense"],
            "away_manual_recent_form": away_manual["recent_form"],
            "home_adjusted_attack": home_adjusted["attack"],
            "home_adjusted_defense": home_adjusted["defense"],
            "home_adjusted_recent_form": home_adjusted["recent_form"],
            "away_adjusted_attack": away_adjusted["attack"],
            "away_adjusted_defense": away_adjusted["defense"],
            "away_adjusted_recent_form": away_adjusted["recent_form"],
            "baseline_home_prob": _rating_value_or_na(baseline_row, "home_win_prob"),
            "baseline_draw_prob": _rating_value_or_na(baseline_row, "draw_prob"),
            "baseline_away_prob": _rating_value_or_na(baseline_row, "away_win_prob"),
            "behavior_home_prob": _rating_value_or_na(behavior_row, "home_win_prob"),
            "behavior_draw_prob": _rating_value_or_na(behavior_row, "draw_prob"),
            "behavior_away_prob": _rating_value_or_na(behavior_row, "away_win_prob"),
            "baseline_actual_prob": _rating_value_or_na(baseline_row, "actual_result_prob"),
            "behavior_actual_prob": _rating_value_or_na(behavior_row, "actual_result_prob"),
            "rating_coverage_warning": rating_warning,
        }
        row["probability_warning"] = _probability_warning(row, home_manual, away_manual, home_adjusted, away_adjusted)
        rows.append(row)
    return pd.DataFrame(rows, columns=MATCH_INPUT_AUDIT_COLUMNS)


def audit_backtest_team_aliases(
    completed_matches_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None,
    aliases_df: pd.DataFrame | None,
) -> pd.DataFrame:
    completed = completed_matches_df.copy() if completed_matches_df is not None else pd.DataFrame()
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame()
    teams = sorted(_completed_team_counts(completed))
    if not teams:
        return pd.DataFrame(columns=ALIAS_AUDIT_COLUMNS)
    aliases = _normalise_alias_df(aliases_df)
    rating_names = _team_names(ratings)
    rating_keys = {team_name_key(name): name for name in rating_names}

    rows: list[dict[str, Any]] = []
    for team in teams:
        exact = _has_exact_team(ratings, team)
        canonical = _canonical_team(team, aliases)
        key_match = rating_keys.get(team_name_key(team), "")
        canonical_match = rating_keys.get(team_name_key(canonical), "")
        possible = ""
        similarity = 0.0
        if not exact:
            possible = canonical_match or key_match
            if possible:
                similarity = 1.0
            else:
                possible, similarity = _best_similarity(team, rating_names)
                if similarity < 0.82:
                    possible = ""
        recommendation = ""
        if not exact and possible:
            recommendation = f"Add alias '{team}' -> '{possible}' or align completed-match team name."
        elif not exact:
            recommendation = f"Add a manual rating row for '{team}' or an alias to the canonical team."

        rows.append(
            {
                "team_in_backtest": team,
                "exact_match_in_ratings": bool(exact),
                "possible_alias": possible,
                "canonical_alias": canonical if canonical != team else "",
                "similarity_score": round(float(similarity), 3),
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows, columns=ALIAS_AUDIT_COLUMNS)


def build_backtest_qa_summary(
    completed_matches_df: pd.DataFrame | None,
    coverage_df: pd.DataFrame | None,
    match_input_audit_df: pd.DataFrame | None,
    alias_audit_df: pd.DataFrame | None,
) -> dict[str, Any]:
    completed = completed_matches_df.copy() if completed_matches_df is not None else pd.DataFrame()
    coverage = coverage_df.copy() if coverage_df is not None else pd.DataFrame()
    match_audit = match_input_audit_df.copy() if match_input_audit_df is not None else pd.DataFrame()
    aliases = alias_audit_df.copy() if alias_audit_df is not None else pd.DataFrame()
    missing_manual = _list_values(coverage, "team", ~coverage.get("in_manual_ratings", pd.Series(dtype=bool)).astype(bool)) if not coverage.empty else []
    missing_behavior = _list_values(coverage, "team", ~coverage.get("in_team_behavior", pd.Series(dtype=bool)).astype(bool)) if not coverage.empty else []
    neutral_fallback_count = int(coverage.get("used_neutral_fallback", pd.Series(dtype=bool)).astype(bool).sum()) if not coverage.empty else 0
    alias_warnings = aliases.loc[aliases["recommendation"].astype(str) != ""] if not aliases.empty else pd.DataFrame()
    suspicious = match_audit.loc[match_audit["probability_warning"].astype(str) != ""] if not match_audit.empty else pd.DataFrame()
    summary = {
        "completed_matches": int(len(completed)),
        "unique_teams": int(len(coverage)),
        "teams_missing_manual_ratings": ", ".join(missing_manual),
        "teams_missing_behavior_rows": ", ".join(missing_behavior),
        "neutral_fallback_count": neutral_fallback_count,
        "alias_warning_count": int(len(alias_warnings)),
        "suspicious_probability_count": int(len(suspicious)),
        "qa_warning": "",
    }
    if neutral_fallback_count:
        summary["qa_warning"] = "Backtest metrics may be unreliable because some teams used neutral fallback ratings."
    elif len(alias_warnings):
        summary["qa_warning"] = "Backtest metrics may be unreliable because some completed-match team names may not match ratings names."
    return summary


def explain_match_probability_issue(match_audit_row: pd.Series | dict[str, Any] | None) -> str:
    row = pd.Series(match_audit_row) if isinstance(match_audit_row, dict) else match_audit_row
    if row is None or row.empty:
        return "Match was not found in the QA audit."
    warnings = " ".join(
        [
            str(row.get("rating_coverage_warning", "") or ""),
            str(row.get("probability_warning", "") or ""),
        ]
    ).lower()
    if "missing" in warnings or "fallback" in warnings:
        return (
            "Low or flat win probabilities are explained by rating coverage. At least one team was missing from the "
            "exact rating table used by the model, so the model used neutral fallback inputs near attack=0.55, "
            "defense=0.55, and recent_form=0.55. With little rating separation, the Poisson 1X2 probabilities become flat."
        )
    if "probabilities_too_flat" in warnings:
        return "The match is flagged because the 1X2 probabilities are clustered near even despite rating separation."
    if "favorite" in warnings:
        return "The match is flagged because a manually stronger team received a low win probability."
    return "No specific QA issue explains this probability; inspect model inputs and source data manually."


def _completed_team_counts(completed: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if completed is None or completed.empty:
        return counts
    for col in ["home", "away"]:
        if col not in completed.columns:
            continue
        for team in completed[col].dropna().astype(str).str.strip():
            if team:
                counts[team] = counts.get(team, 0) + 1
    return counts


def _normalise_alias_df(aliases_df: pd.DataFrame | None) -> pd.DataFrame:
    if aliases_df is None or aliases_df.empty:
        return load_team_name_aliases_df()
    out = aliases_df.copy()
    for col in ["alias", "canonical"]:
        if col not in out.columns:
            out[col] = pd.NA
    return out.dropna(subset=["alias", "canonical"])[["alias", "canonical"]].reset_index(drop=True)


def _canonical_team(team: str, aliases_df: pd.DataFrame) -> str:
    key = team_name_key(team)
    if aliases_df is not None and not aliases_df.empty:
        rows = aliases_df.loc[aliases_df["alias"].astype(str).map(team_name_key) == key]
        if not rows.empty:
            return str(rows.iloc[-1].get("canonical", team))
    return normalize_team_name(team)


def _possible_alias_match(team: str, rating_tables: list[pd.DataFrame], aliases_df: pd.DataFrame) -> str:
    canonical = _canonical_team(team, aliases_df)
    names: list[str] = []
    for table in rating_tables:
        names.extend(_team_names(table))
    if not names:
        return ""
    lower_names = {name.lower(): name for name in names}
    if str(team).lower() in lower_names:
        return ""
    if str(canonical).lower() in lower_names:
        return lower_names[str(canonical).lower()]
    keyed = {team_name_key(name): name for name in names}
    if team_name_key(team) in keyed:
        return keyed[team_name_key(team)]
    if team_name_key(canonical) in keyed:
        return keyed[team_name_key(canonical)]
    best, score = _best_similarity(team, names)
    return best if score >= 0.88 else ""


def _team_names(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty or "team" not in df.columns:
        return []
    return sorted({str(team).strip() for team in df["team"].dropna() if str(team).strip()})


def _has_exact_team(df: pd.DataFrame | None, team: str) -> bool:
    return not rating_row_for_team(df, team).empty


def _best_similarity(team: str, choices: list[str]) -> tuple[str, float]:
    best = ""
    best_score = 0.0
    team_key = team_name_key(team)
    for choice in choices:
        score = SequenceMatcher(None, team_key, team_name_key(choice)).ratio()
        if score > best_score:
            best = choice
            best_score = score
    return best, best_score


def _rating_value_or_na(row: pd.Series, column: str) -> float | pd._libs.missing.NAType:
    if row is None or row.empty or column not in row.index or pd.isna(row.get(column)):
        return pd.NA
    return round(coerce_float(row.get(column), float("nan")), 4)


def _rating_values_used(ratings: pd.DataFrame, team: str) -> dict[str, float]:
    row = rating_row_for_team(ratings, team)
    if row.empty:
        fallback = neutral_team_rating(team)
        row = fallback
    return {
        "attack": round(coerce_float(row.get("attack"), 0.55), 4),
        "defense": round(coerce_float(row.get("defense"), 0.55), 4),
        "recent_form": round(coerce_float(row.get("recent_form"), 0.55), 4),
    }


def _rating_coverage_warning(home: str, away: str, manual: pd.DataFrame, adjusted: pd.DataFrame) -> str:
    warnings: list[str] = []
    for side, team in [("home", home), ("away", away)]:
        manual_row = rating_row_for_team(manual, team)
        adjusted_row = rating_row_for_team(adjusted, team)
        if manual_row.empty:
            warnings.append(f"{side} missing manual rating")
        if adjusted_row.empty:
            warnings.append(f"{side} missing adjusted rating")
        if manual_row.empty or adjusted_row.empty or is_neutral_fallback_rating(adjusted_row):
            warnings.append(f"{side} neutral fallback used")
    return "; ".join(dict.fromkeys(warnings))


def _probability_warning(
    row: dict[str, Any],
    home_manual: dict[str, float],
    away_manual: dict[str, float],
    home_adjusted: dict[str, float],
    away_adjusted: dict[str, float],
) -> str:
    warnings: list[str] = []
    baseline_probs = [
        coerce_float(row.get("baseline_home_prob"), float("nan")),
        coerce_float(row.get("baseline_draw_prob"), float("nan")),
        coerce_float(row.get("baseline_away_prob"), float("nan")),
    ]
    behavior_probs = [
        coerce_float(row.get("behavior_home_prob"), float("nan")),
        coerce_float(row.get("behavior_draw_prob"), float("nan")),
        coerce_float(row.get("behavior_away_prob"), float("nan")),
    ]
    if _probs_flat(baseline_probs) or _probs_flat(behavior_probs):
        warnings.append("probabilities_too_flat")
    if row.get("rating_coverage_warning") and warnings:
        warnings.append("missing_rating_probabilities_flat")

    manual_home = _team_composite(home_manual)
    manual_away = _team_composite(away_manual)
    adjusted_home = _team_composite(home_adjusted)
    adjusted_away = _team_composite(away_adjusted)
    _favorite_warnings(warnings, manual_home, manual_away, baseline_probs, "manual")
    _favorite_warnings(warnings, adjusted_home, adjusted_away, behavior_probs, "adjusted")

    actual_probs = [
        coerce_float(row.get("baseline_actual_prob"), float("nan")),
        coerce_float(row.get("behavior_actual_prob"), float("nan")),
    ]
    if any(not pd.isna(prob) and prob < 0.20 for prob in actual_probs):
        warnings.append("actual_result_probability_unusually_low")
    return "; ".join(dict.fromkeys(warnings))


def _favorite_warnings(warnings: list[str], home_score: float, away_score: float, probs: list[float], label: str) -> None:
    if any(pd.isna(prob) for prob in probs):
        return
    gap = home_score - away_score
    if abs(gap) >= 0.18:
        favorite_prob = probs[0] if gap > 0 else probs[2]
        if favorite_prob < 0.50:
            warnings.append(f"favorite_{label}_rating_gap_high_but_probability_low")
    if max(home_score, away_score) >= 0.72 and min(home_score, away_score) <= 0.56:
        favorite_prob = probs[0] if home_score > away_score else probs[2]
        if favorite_prob < 0.50:
            warnings.append("elite_vs_weak_probability_low")


def _probs_flat(probs: list[float]) -> bool:
    if any(pd.isna(prob) for prob in probs):
        return False
    return all(0.25 <= prob <= 0.45 for prob in probs)


def _team_composite(values: dict[str, float]) -> float:
    return 0.45 * values["attack"] + 0.35 * values["defense"] + 0.20 * values["recent_form"]


def _actual_result(home_goals: Any, away_goals: Any) -> str:
    home = coerce_float(home_goals, 0.0)
    away = coerce_float(away_goals, 0.0)
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def _index_by_match_id(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df is None or df.empty or "match_id" not in df.columns:
        return {}
    return {str(row.get("match_id", "")): row for _, row in df.iterrows()}


def _list_values(df: pd.DataFrame, column: str, mask: pd.Series) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    aligned = mask.reindex(df.index, fill_value=False)
    return sorted(df.loc[aligned, column].dropna().astype(str).tolist())
