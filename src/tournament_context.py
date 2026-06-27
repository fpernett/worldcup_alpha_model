from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.team_names import normalize_team_name, team_name_key
from src.utils import coerce_float


STANDINGS_COLUMNS = [
    "group",
    "team",
    "played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_difference",
    "points",
    "group_position",
]

DEFAULT_QUALIFICATION_RULES = {
    "teams_advance": 2,
    "points_for_win": 3,
    "points_for_draw": 1,
    "group_size": 4,
}


def build_group_standings_asof(
    results_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    group: str,
    as_of_utc: str,
) -> pd.DataFrame:
    fixtures = _normalise_fixtures(fixtures_df)
    results = _normalise_results(results_df)
    group_label = _normalise_group(group)
    if fixtures.empty or not group_label:
        return pd.DataFrame(columns=STANDINGS_COLUMNS)

    group_fixtures = fixtures.loc[fixtures["group_key"] == group_label].copy()
    teams = sorted(set(group_fixtures["home"].dropna().astype(str)) | set(group_fixtures["away"].dropna().astype(str)))
    if not teams:
        return pd.DataFrame(columns=STANDINGS_COLUMNS)

    cutoff = _timestamp(as_of_utc)
    completed = _attach_fixture_context(results, fixtures)
    if completed.empty:
        return _empty_standings(group, teams)
    completed = completed.loc[completed["group_key"] == group_label].copy()
    if cutoff is not None:
        completed = completed.loc[completed["kickoff_ts"].notna() & (completed["kickoff_ts"] < cutoff)].copy()

    table = {team: _blank_team_row(group, team) for team in teams}
    for _, row in completed.iterrows():
        home = str(row.get("home", "") or "")
        away = str(row.get("away", "") or "")
        if home not in table or away not in table:
            continue
        hg = coerce_float(row.get("home_goals"), float("nan"))
        ag = coerce_float(row.get("away_goals"), float("nan"))
        if pd.isna(hg) or pd.isna(ag):
            continue
        _apply_result(table[home], int(hg), int(ag))
        _apply_result(table[away], int(ag), int(hg))

    standings = pd.DataFrame(table.values(), columns=STANDINGS_COLUMNS[:-1])
    if standings.empty:
        return pd.DataFrame(columns=STANDINGS_COLUMNS)
    standings = standings.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    standings["group_position"] = standings.index + 1
    return standings[STANDINGS_COLUMNS]


def remaining_group_fixtures_asof(
    fixtures_df: pd.DataFrame,
    group: str,
    as_of_utc: str,
) -> pd.DataFrame:
    fixtures = _normalise_fixtures(fixtures_df)
    group_key = _normalise_group(group)
    if fixtures.empty or not group_key:
        return pd.DataFrame(columns=fixtures.columns)
    cutoff = _timestamp(as_of_utc)
    out = fixtures.loc[fixtures["group_key"] == group_key].copy()
    if cutoff is not None:
        out = out.loc[out["kickoff_ts"].notna() & (out["kickoff_ts"] >= cutoff)].copy()
    return out.reset_index(drop=True)


def calculate_team_qualification_need(
    team: str,
    group_standings_df: pd.DataFrame,
    remaining_fixtures_df: pd.DataFrame,
    target_match_id: str,
    qualification_rules: dict | None = None,
) -> dict[str, Any]:
    rules = {**DEFAULT_QUALIFICATION_RULES, **(qualification_rules or {})}
    standings = _ensure_standings(group_standings_df)
    canonical = normalize_team_name(team)
    row = _team_standing_row(canonical, standings)
    if row is None:
        return _unknown_need(canonical or team, "Team not found in group standings.")

    current_points = int(row.get("points", 0))
    current_gd = int(row.get("goal_difference", 0))
    position = int(row.get("group_position", 0))
    teams_advance = int(rules.get("teams_advance", 2))
    group_size = int(rules.get("group_size", max(len(standings), 4)))
    remaining_count = _remaining_count_for_team(canonical, remaining_fixtures_df, target_match_id)
    total_played_after_target = int(row.get("played", 0)) + 1
    likely_final_group_match = remaining_count <= 1 or total_played_after_target >= max(group_size - 1, 1)

    sorted_points = standings.sort_values(["points", "goal_difference", "goals_for"], ascending=[False, False, False]).reset_index(drop=True)
    cutoff_points = int(sorted_points.iloc[teams_advance - 1]["points"]) if len(sorted_points) >= teams_advance else 0
    above_cutoff = int(sorted_points.iloc[teams_advance]["points"]) if len(sorted_points) > teams_advance else -99
    margin_to_cutoff = current_points - cutoff_points
    margin_to_chaser = current_points - above_cutoff

    draw_enough = False
    needs_win = False
    must_not_lose = False
    already_qualified = False
    already_eliminated = False
    gd_pressure = False
    label = "balanced"
    score = 0.0
    reasons: list[str] = []

    if position <= teams_advance:
        if likely_final_group_match and position == teams_advance and margin_to_chaser <= 0:
            needs_win = True
            label = "needs_win"
            score = 0.8
            reasons.append("On the qualification cutoff without a points cushion; a draw leaves tiebreaker or parallel-match risk.")
        elif likely_final_group_match and margin_to_chaser >= 1:
            draw_enough = True
            must_not_lose = True
            label = "draw_enough"
            score = 0.65
            reasons.append("Top-two position with enough cushion that a draw is likely sufficient.")
        elif margin_to_chaser >= 4 and remaining_count <= 1:
            already_qualified = True
            label = "already_qualified_likely"
            score = 0.25
            reasons.append("Points cushion makes qualification highly secure.")
        else:
            must_not_lose = True
            label = "must_not_lose"
            score = 0.5
            reasons.append("Currently qualifying but not fully secure.")
    else:
        points_if_draw = current_points + 1
        if likely_final_group_match and points_if_draw <= cutoff_points:
            needs_win = True
            label = "needs_win"
            score = 0.9
            reasons.append("Outside the likely qualifying places and a draw probably is not enough.")
        elif current_points + 3 < cutoff_points:
            already_eliminated = True
            label = "already_eliminated_likely"
            score = 0.15
            reasons.append("Even a win may not close the points gap.")
        else:
            needs_win = True
            label = "needs_win"
            score = 0.75
            reasons.append("Likely needs three points to improve qualification position.")

    if not already_qualified and not already_eliminated:
        peers = standings.loc[standings["points"].astype(int).sub(current_points).abs() <= 1]
        if len(peers) >= 2 and abs(current_gd - int(peers["goal_difference"].median())) <= 2:
            gd_pressure = True
            reasons.append("Nearby teams are close on points; goal difference may matter.")

    return {
        "team": canonical or team,
        "current_points": current_points,
        "current_goal_difference": current_gd,
        "group_position": position,
        "needs_win": bool(needs_win),
        "draw_enough": bool(draw_enough),
        "already_qualified_likely": bool(already_qualified),
        "already_eliminated_likely": bool(already_eliminated),
        "must_not_lose": bool(must_not_lose),
        "goal_difference_pressure": bool(gd_pressure),
        "incentive_label": label,
        "incentive_score": float(score),
        "reason": " ".join(reasons) or "Conservative v1 classification from as-of standings.",
    }


def classify_match_context(
    home: str,
    away: str,
    match_row: pd.Series,
    group_standings_df: pd.DataFrame,
    remaining_fixtures_df: pd.DataFrame,
) -> dict[str, Any]:
    stage = _stage(match_row)
    if stage == "knockout":
        return {
            "stage": "knockout",
            "home_incentive_label": "must_advance",
            "away_incentive_label": "must_advance",
            "home_incentive_score": 1.0,
            "away_incentive_score": 1.0,
            "context_type": "knockout_must_advance",
            "tempo_bias": "neutral_90_minute",
            "draw_bias": "neutral_90_minute",
            "late_open_game_risk": "elevated_extra_time_penalty_context",
            "favorite_risk_bias": "neutral",
            "totals_bias": "neutral",
            "btts_bias": "neutral",
            "home_need": _knockout_need(home),
            "away_need": _knockout_need(away),
            "explanation": "Knockout matches require advancement; group-stage draw-enough incentives are not applied to 90-minute probabilities.",
            "warnings": "Qualification market context can differ from 90-minute 1X2 market context.",
        }

    target_match_id = str(match_row.get("match_id", "") or "")
    home_need = calculate_team_qualification_need(home, group_standings_df, remaining_fixtures_df, target_match_id)
    away_need = calculate_team_qualification_need(away, group_standings_df, remaining_fixtures_df, target_match_id)
    context_type = _context_type(home_need, away_need)
    explanation = _context_explanation(home, away, home_need, away_need, context_type)
    biases = _biases_for_context(context_type)

    return {
        "stage": stage or "unknown",
        "home_incentive_label": home_need.get("incentive_label", "unknown"),
        "away_incentive_label": away_need.get("incentive_label", "unknown"),
        "home_incentive_score": home_need.get("incentive_score", 0.0),
        "away_incentive_score": away_need.get("incentive_score", 0.0),
        "context_type": context_type,
        **biases,
        "home_need": home_need,
        "away_need": away_need,
        "explanation": explanation,
        "warnings": "" if context_type != "unknown" else "Insufficient standings or fixture data for context classification.",
    }


def apply_tournament_context_adjustment(
    baseline_probs: dict,
    baseline_market_families_df: pd.DataFrame,
    match_context: dict,
) -> tuple[dict, pd.DataFrame, dict]:
    probs = {key: coerce_float(value, float("nan")) for key, value in (baseline_probs or {}).items()}
    if not _valid_1x2(probs):
        diag = _adjustment_diag(False, "Baseline 1X2 probabilities missing or invalid.", probs, probs, ["invalid baseline probabilities"])
        return probs, _context_market_rows(baseline_market_families_df, probs, diag["adjustment_reason"]), diag

    context_type = str(match_context.get("context_type", "unknown") or "unknown")
    shifts = _context_shifts(context_type, match_context)
    adjusted = probs.copy()
    adjusted["home_win"] = probs["home_win"] + shifts["home_win"]
    adjusted["draw"] = probs["draw"] + shifts["draw"]
    adjusted["away_win"] = probs["away_win"] + shifts["away_win"]
    adjusted = _cap_and_normalise_1x2(probs, adjusted, max_shift=0.05)

    for key, shift in shifts.items():
        if key in {"home_win", "draw", "away_win"}:
            continue
        if key in probs and pd.notna(probs[key]):
            adjusted[key] = _clamp(float(probs[key]) + float(shift), 0.01, 0.99)
    _fill_market_complements(adjusted)

    reason = _adjustment_reason(context_type)
    warnings = []
    max_shift = max(abs(adjusted[k] - probs[k]) for k in ["home_win", "draw", "away_win"])
    if max_shift > 0.050001:
        warnings.append("1X2 shift cap exceeded before final normalization.")
    if context_type == "unknown":
        warnings.append("No context adjustment applied.")

    diag = _adjustment_diag(
        context_type != "unknown",
        reason,
        probs,
        adjusted,
        warnings,
    )
    context_markets = _context_market_rows(baseline_market_families_df, probs, adjusted, reason)
    return adjusted, context_markets, diag


def evaluate_context_layer_on_completed_matches(
    predictions_df: pd.DataFrame | None,
    results_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compare baseline and context-adjusted probabilities for completed rows.

    This intentionally uses supplied prediction rows only. The context layer is
    diagnostic until strict as-of validation shows improvement.
    """
    preds = predictions_df.copy() if predictions_df is not None else pd.DataFrame()
    results = results_df.copy() if results_df is not None else pd.DataFrame()
    if preds.empty or results.empty:
        return pd.DataFrame(
            [
                {
                    "rows_evaluated": 0,
                    "baseline_brier": pd.NA,
                    "context_brier": pd.NA,
                    "baseline_log_loss": pd.NA,
                    "context_log_loss": pd.NA,
                    "draw_calibration_delta": pd.NA,
                    "totals_calibration_delta": pd.NA,
                    "warning": "Missing predictions or results.",
                }
            ]
        )

    merged = preds.merge(results, on="match_id", how="inner", suffixes=("_pred", "_result"))
    required = [
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "context_home_win_prob",
        "context_draw_prob",
        "context_away_win_prob",
        "actual_result",
    ]
    for col in required:
        if col not in merged.columns:
            return pd.DataFrame([{"rows_evaluated": 0, "warning": f"Missing required column: {col}"}])
    if merged.empty:
        return pd.DataFrame([{"rows_evaluated": 0, "warning": "No completed predictions joined to results."}])

    baseline_brier = []
    context_brier = []
    baseline_log = []
    context_log = []
    for _, row in merged.iterrows():
        actual = str(row.get("actual_result", ""))
        observed = {
            "home_win": 1.0 if actual == "home_win" else 0.0,
            "draw": 1.0 if actual == "draw" else 0.0,
            "away_win": 1.0 if actual == "away_win" else 0.0,
        }
        base = {
            "home_win": coerce_float(row.get("home_win_prob"), 0.0),
            "draw": coerce_float(row.get("draw_prob"), 0.0),
            "away_win": coerce_float(row.get("away_win_prob"), 0.0),
        }
        ctx = {
            "home_win": coerce_float(row.get("context_home_win_prob"), 0.0),
            "draw": coerce_float(row.get("context_draw_prob"), 0.0),
            "away_win": coerce_float(row.get("context_away_win_prob"), 0.0),
        }
        baseline_brier.append(sum((base[k] - observed[k]) ** 2 for k in observed) / 3.0)
        context_brier.append(sum((ctx[k] - observed[k]) ** 2 for k in observed) / 3.0)
        baseline_log.append(-math.log(max(min(base.get(actual, 0.0), 0.999), 0.001)))
        context_log.append(-math.log(max(min(ctx.get(actual, 0.0), 0.999), 0.001)))

    draw_actual = (merged["actual_result"].astype(str) == "draw").astype(float).mean()
    return pd.DataFrame(
        [
            {
                "rows_evaluated": int(len(merged)),
                "baseline_brier": float(pd.Series(baseline_brier).mean()),
                "context_brier": float(pd.Series(context_brier).mean()),
                "baseline_log_loss": float(pd.Series(baseline_log).mean()),
                "context_log_loss": float(pd.Series(context_log).mean()),
                "draw_calibration_delta": float(coerce_float(merged["context_draw_prob"].mean(), 0.0) - draw_actual),
                "totals_calibration_delta": pd.NA,
                "warning": "Context layer remains diagnostic; do not make primary without strict as-of improvement.",
            }
        ]
    )


def context_probabilities_display(baseline_probs: dict, context_probs: dict) -> pd.DataFrame:
    rows = []
    for key, label in [("home_win", "Home win"), ("draw", "Draw"), ("away_win", "Away win"), ("over_2_5", "Over 2.5"), ("btts_yes", "BTTS Yes")]:
        base = coerce_float((baseline_probs or {}).get(key), float("nan"))
        ctx = coerce_float((context_probs or {}).get(key), float("nan"))
        rows.append(
            {
                "market": label,
                "baseline_probability": base,
                "context_probability": ctx,
                "shift_pp": (ctx - base) * 100.0 if pd.notna(base) and pd.notna(ctx) else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _normalise_fixtures(df: pd.DataFrame | None) -> pd.DataFrame:
    fixtures = df.copy() if df is not None else pd.DataFrame()
    for col in ["match_id", "date_utc", "time_utc", "competition", "group", "home", "away"]:
        if col not in fixtures.columns:
            fixtures[col] = pd.NA
    fixtures["home"] = fixtures["home"].map(normalize_team_name)
    fixtures["away"] = fixtures["away"].map(normalize_team_name)
    fixtures["group_key"] = fixtures["group"].map(_normalise_group)
    fixtures["kickoff_ts"] = fixtures.apply(lambda row: _kickoff_ts(row.get("date_utc"), row.get("time_utc")), axis=1)
    return fixtures


def _normalise_results(df: pd.DataFrame | None) -> pd.DataFrame:
    results = df.copy() if df is not None else pd.DataFrame()
    for col in ["match_id", "date_utc", "competition", "home", "away", "home_goals", "away_goals"]:
        if col not in results.columns:
            results[col] = pd.NA
    results["home"] = results["home"].map(normalize_team_name)
    results["away"] = results["away"].map(normalize_team_name)
    return results


def _attach_fixture_context(results: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    team_groups = _team_group_lookup(fixtures)
    fixture_cols = ["match_id", "group", "group_key", "kickoff_ts", "home", "away", "date_utc"]
    merged = results.merge(fixtures[fixture_cols], on="match_id", how="left", suffixes=("", "_fixture"))
    for idx, row in merged.loc[merged["group_key"].isna()].iterrows():
        match = _find_fixture_for_result(row, fixtures)
        if match is not None:
            merged.at[idx, "group"] = match.get("group", "")
            merged.at[idx, "group_key"] = match.get("group_key", "")
            merged.at[idx, "kickoff_ts"] = match.get("kickoff_ts", pd.NaT)
        else:
            home_groups = team_groups.get(team_name_key(str(row.get("home", ""))), set())
            away_groups = team_groups.get(team_name_key(str(row.get("away", ""))), set())
            common = sorted(home_groups & away_groups)
            if common:
                merged.at[idx, "group_key"] = common[0]
                merged.at[idx, "group"] = f"Group {common[0].upper()}"
    merged["kickoff_ts"] = merged["kickoff_ts"].fillna(merged["date_utc"].map(lambda value: _kickoff_ts(value, "00:00")))
    return merged


def _team_group_lookup(fixtures: pd.DataFrame) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    if fixtures.empty:
        return groups
    for _, row in fixtures.iterrows():
        group_key = str(row.get("group_key", "") or "")
        if not group_key:
            continue
        for col in ["home", "away"]:
            key = team_name_key(str(row.get(col, "")))
            if key:
                groups.setdefault(key, set()).add(group_key)
    return groups


def _find_fixture_for_result(result: pd.Series, fixtures: pd.DataFrame) -> pd.Series | None:
    date = _date_only(result.get("date_utc"))
    home_key = team_name_key(str(result.get("home", "")))
    away_key = team_name_key(str(result.get("away", "")))
    if not date or not home_key or not away_key:
        return None
    for _, fixture in fixtures.iterrows():
        if _date_only(fixture.get("date_utc")) != date:
            continue
        f_home = team_name_key(str(fixture.get("home", "")))
        f_away = team_name_key(str(fixture.get("away", "")))
        if {home_key, away_key} == {f_home, f_away}:
            return fixture
    return None


def _normalise_group(group: Any) -> str:
    text = str(group or "").strip().lower()
    if not text:
        return ""
    text = text.replace("group", "").strip()
    return text


def _timestamp(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def _kickoff_ts(date_value: Any, time_value: Any) -> pd.Timestamp:
    date_text = str(date_value or "").strip()
    time_text = str(time_value or "").strip() or "00:00"
    ts = pd.to_datetime(f"{date_text} {time_text}", errors="coerce", utc=True)
    return ts


def _date_only(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else ts.date().isoformat()


def _blank_team_row(group: str, team: str) -> dict[str, Any]:
    return {
        "group": group,
        "team": team,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "points": 0,
    }


def _empty_standings(group: str, teams: list[str]) -> pd.DataFrame:
    out = pd.DataFrame([_blank_team_row(group, team) for team in teams])
    out["group_position"] = range(1, len(out) + 1)
    return out[STANDINGS_COLUMNS]


def _apply_result(row: dict[str, Any], goals_for: int, goals_against: int) -> None:
    row["played"] += 1
    row["goals_for"] += goals_for
    row["goals_against"] += goals_against
    row["goal_difference"] = row["goals_for"] - row["goals_against"]
    if goals_for > goals_against:
        row["wins"] += 1
        row["points"] += 3
    elif goals_for == goals_against:
        row["draws"] += 1
        row["points"] += 1
    else:
        row["losses"] += 1


def _ensure_standings(df: pd.DataFrame | None) -> pd.DataFrame:
    standings = df.copy() if df is not None else pd.DataFrame()
    for col in STANDINGS_COLUMNS:
        if col not in standings.columns:
            standings[col] = pd.NA
    return standings


def _team_standing_row(team: str, standings: pd.DataFrame) -> pd.Series | None:
    key = team_name_key(team)
    if standings.empty or not key:
        return None
    matches = standings.loc[standings["team"].map(lambda value: team_name_key(str(value))) == key]
    return None if matches.empty else matches.iloc[0]


def _remaining_count_for_team(team: str, fixtures: pd.DataFrame | None, target_match_id: str) -> int:
    remaining = fixtures.copy() if fixtures is not None else pd.DataFrame()
    if remaining.empty:
        return 1
    for col in ["match_id", "home", "away"]:
        if col not in remaining.columns:
            remaining[col] = ""
    key = team_name_key(team)
    team_rows = remaining.loc[
        (remaining["home"].map(lambda value: team_name_key(str(value))) == key)
        | (remaining["away"].map(lambda value: team_name_key(str(value))) == key)
    ]
    if target_match_id:
        team_rows = team_rows.loc[team_rows["match_id"].astype(str) != str(target_match_id)]
    return int(len(team_rows) + 1)


def _unknown_need(team: str, reason: str) -> dict[str, Any]:
    return {
        "team": team,
        "current_points": pd.NA,
        "current_goal_difference": pd.NA,
        "group_position": pd.NA,
        "needs_win": False,
        "draw_enough": False,
        "already_qualified_likely": False,
        "already_eliminated_likely": False,
        "must_not_lose": False,
        "goal_difference_pressure": False,
        "incentive_label": "unknown",
        "incentive_score": 0.0,
        "reason": reason,
    }


def _knockout_need(team: str) -> dict[str, Any]:
    return {
        **_unknown_need(normalize_team_name(team) or team, "Knockout match; both teams must advance."),
        "needs_win": True,
        "incentive_label": "must_advance",
        "incentive_score": 1.0,
    }


def _stage(match_row: pd.Series) -> str:
    stage = str(match_row.get("stage", "") or "").lower()
    group = str(match_row.get("group", "") or "").lower()
    competition = str(match_row.get("competition", "") or "").lower()
    if stage:
        return "knockout" if "knockout" in stage or "round" in stage or "final" in stage else "group"
    if "group" in group:
        return "group"
    if any(term in competition for term in ["round of", "quarter", "semi", "final", "knockout"]):
        return "knockout"
    return "unknown"


def _context_type(home_need: dict[str, Any], away_need: dict[str, Any]) -> str:
    labels = {str(home_need.get("incentive_label", "")), str(away_need.get("incentive_label", ""))}
    if "unknown" in labels:
        return "unknown"
    if labels <= {"already_qualified_likely", "already_eliminated_likely"}:
        return "dead_rubber"
    if home_need.get("needs_win") and away_need.get("needs_win"):
        return "both_need_win"
    if (home_need.get("needs_win") and away_need.get("draw_enough")) or (away_need.get("needs_win") and home_need.get("draw_enough")):
        return "one_needs_win_other_draw_enough"
    if (
        home_need.get("needs_win") and away_need.get("already_qualified_likely")
    ) or (
        away_need.get("needs_win") and home_need.get("already_qualified_likely")
    ):
        return "one_safe_other_needs_win"
    if home_need.get("draw_enough") and away_need.get("draw_enough"):
        return "both_draw_ok"
    return "unknown"


def _biases_for_context(context_type: str) -> dict[str, str]:
    mapping = {
        "both_need_win": {
            "tempo_bias": "more_open",
            "draw_bias": "negative",
            "late_open_game_risk": "elevated",
            "favorite_risk_bias": "elevated",
            "totals_bias": "slightly_positive",
            "btts_bias": "slightly_positive",
        },
        "one_needs_win_other_draw_enough": {
            "tempo_bias": "slower_early",
            "draw_bias": "positive",
            "late_open_game_risk": "elevated",
            "favorite_risk_bias": "slightly_elevated",
            "totals_bias": "mixed",
            "btts_bias": "slightly_positive_late",
        },
        "one_safe_other_needs_win": {
            "tempo_bias": "asymmetric",
            "draw_bias": "slightly_negative",
            "late_open_game_risk": "elevated",
            "favorite_risk_bias": "elevated_if_safe_team_favored",
            "totals_bias": "slightly_positive",
            "btts_bias": "slightly_positive",
        },
        "both_draw_ok": {
            "tempo_bias": "slower",
            "draw_bias": "positive",
            "late_open_game_risk": "low",
            "favorite_risk_bias": "lower",
            "totals_bias": "slightly_negative",
            "btts_bias": "slightly_negative",
        },
        "dead_rubber": {
            "tempo_bias": "uncertain",
            "draw_bias": "neutral",
            "late_open_game_risk": "uncertain",
            "favorite_risk_bias": "lineup_rotation",
            "totals_bias": "uncertain",
            "btts_bias": "uncertain",
        },
    }
    return mapping.get(
        context_type,
        {
            "tempo_bias": "unknown",
            "draw_bias": "unknown",
            "late_open_game_risk": "unknown",
            "favorite_risk_bias": "unknown",
            "totals_bias": "unknown",
            "btts_bias": "unknown",
        },
    )


def _context_explanation(home: str, away: str, home_need: dict[str, Any], away_need: dict[str, Any], context_type: str) -> str:
    if context_type == "one_needs_win_other_draw_enough":
        draw_team = home if home_need.get("draw_enough") else away
        win_team = home if home_need.get("needs_win") else away
        return (
            f"{draw_team} can probably accept a draw, while {win_team} likely needs a win. "
            "This can make the early game more cautious for the draw-enough team and more urgent for the chasing team, especially late if tied."
        )
    if context_type == "both_need_win":
        return "Both teams likely need a win, which can reduce draw incentives and raise late open-game risk."
    if context_type == "both_draw_ok":
        return "Both teams appear able to benefit from a draw, which can increase draw bias and lower early tempo."
    if context_type == "one_safe_other_needs_win":
        return "One team appears relatively safe while the other likely needs a win, creating asymmetric urgency."
    if context_type == "dead_rubber":
        return "The match may have reduced qualification stakes; lineup and motivation uncertainty are higher."
    return "Tournament incentive context could not be classified safely from available standings."


def _context_shifts(context_type: str, match_context: dict[str, Any]) -> dict[str, float]:
    shifts = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0, "over_2_5": 0.0, "under_2_5": 0.0, "btts_yes": 0.0, "btts_no": 0.0}
    if context_type == "one_needs_win_other_draw_enough":
        home_label = str(match_context.get("home_incentive_label", ""))
        away_label = str(match_context.get("away_incentive_label", ""))
        shifts["draw"] = 0.03
        if home_label == "draw_enough":
            shifts["home_win"] = -0.02
            shifts["away_win"] = -0.01
        elif away_label == "draw_enough":
            shifts["away_win"] = -0.02
            shifts["home_win"] = -0.01
        shifts["under_2_5"] = 0.015
        shifts["over_2_5"] = -0.015
        shifts["btts_yes"] = 0.01
        shifts["btts_no"] = -0.01
    elif context_type == "both_need_win":
        shifts["draw"] = -0.025
        shifts["home_win"] = 0.0125
        shifts["away_win"] = 0.0125
        shifts["over_2_5"] = 0.02
        shifts["under_2_5"] = -0.02
        shifts["btts_yes"] = 0.02
        shifts["btts_no"] = -0.02
    elif context_type == "both_draw_ok":
        shifts["draw"] = 0.04
        shifts["home_win"] = -0.02
        shifts["away_win"] = -0.02
        shifts["under_2_5"] = 0.02
        shifts["over_2_5"] = -0.02
        shifts["btts_yes"] = -0.01
        shifts["btts_no"] = 0.01
    elif context_type == "one_safe_other_needs_win":
        shifts["draw"] = -0.01
        shifts["over_2_5"] = 0.015
        shifts["under_2_5"] = -0.015
        shifts["btts_yes"] = 0.015
        shifts["btts_no"] = -0.015
    return shifts


def _adjustment_reason(context_type: str) -> str:
    return {
        "one_needs_win_other_draw_enough": "Small group-stage adjustment for asymmetric draw-enough versus must-win incentives.",
        "both_need_win": "Small group-stage adjustment for mutual must-win incentives.",
        "both_draw_ok": "Small group-stage adjustment for mutual draw-friendly incentives.",
        "one_safe_other_needs_win": "Small group-stage adjustment for asymmetric safety versus urgency.",
        "dead_rubber": "No probability adjustment for dead-rubber classification in v1.",
        "knockout_must_advance": "No group-stage draw-enough adjustment for knockout context.",
        "unknown": "No context adjustment applied.",
    }.get(context_type, "No context adjustment applied.")


def _valid_1x2(probs: dict[str, float]) -> bool:
    return all(pd.notna(probs.get(key)) for key in ["home_win", "draw", "away_win"]) and sum(probs[key] for key in ["home_win", "draw", "away_win"]) > 0


def _cap_and_normalise_1x2(baseline: dict[str, float], adjusted: dict[str, float], max_shift: float) -> dict[str, float]:
    out = adjusted.copy()
    for key in ["home_win", "draw", "away_win"]:
        out[key] = _clamp(out[key], baseline[key] - max_shift, baseline[key] + max_shift)
        out[key] = _clamp(out[key], 0.01, 0.98)
    total = sum(out[key] for key in ["home_win", "draw", "away_win"])
    if total <= 0:
        return baseline.copy()
    for key in ["home_win", "draw", "away_win"]:
        out[key] = out[key] / total
    for key in ["home_win", "draw", "away_win"]:
        out[key] = _clamp(out[key], baseline[key] - max_shift, baseline[key] + max_shift)
    total = sum(out[key] for key in ["home_win", "draw", "away_win"])
    for key in ["home_win", "draw", "away_win"]:
        out[key] = out[key] / total
    return out


def _fill_market_complements(probs: dict[str, float]) -> None:
    if "over_2_5" in probs:
        probs["under_2_5"] = 1.0 - probs["over_2_5"]
    elif "under_2_5" in probs:
        probs["over_2_5"] = 1.0 - probs["under_2_5"]
    if "over_3_5" in probs:
        probs["under_3_5"] = 1.0 - probs["over_3_5"]
    elif "under_3_5" in probs:
        probs["over_3_5"] = 1.0 - probs["under_3_5"]
    if "btts_yes" in probs:
        probs["btts_no"] = 1.0 - probs["btts_yes"]
    elif "btts_no" in probs:
        probs["btts_yes"] = 1.0 - probs["btts_no"]


def _context_market_rows(
    baseline_market_families_df: pd.DataFrame,
    baseline_probs: dict[str, float],
    adjusted_probs: dict[str, float],
    reason: str,
) -> pd.DataFrame:
    baseline = baseline_market_families_df.copy() if baseline_market_families_df is not None else pd.DataFrame()
    if baseline.empty:
        return baseline
    rows = []
    for _, row in baseline.iterrows():
        market = str(row.get("market", ""))
        selection = str(row.get("selection", ""))
        prob = _context_probability_for_market_row(row, baseline_probs, adjusted_probs)
        price = float(prob) * 100.0 if pd.notna(prob) else pd.NA
        market_price = row.get("market_price_cents", pd.NA)
        alpha_gap = float(price) - float(market_price) if pd.notna(price) and pd.notna(market_price) else pd.NA
        rows.append(
            {
                **row.to_dict(),
                "context_probability": prob,
                "context_fair_price_cents": price,
                "context_alpha_gap_cents": alpha_gap,
                "context_signal": _context_signal(alpha_gap),
                "context_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _context_probability_for_market_row(row: pd.Series, baseline_probs: dict[str, float], adjusted_probs: dict[str, float]) -> float | pd.NA:
    market = str(row.get("market", "") or "")
    selection = str(row.get("selection", "") or "").lower()
    if market == "1X2":
        if selection == "draw":
            return adjusted_probs.get("draw", pd.NA)
        baseline_prob = coerce_float(row.get("model_prob", row.get("model_probability")), float("nan"))
        candidates = {"home_win": baseline_probs.get("home_win"), "away_win": baseline_probs.get("away_win")}
        side = _closest_probability_key(baseline_prob, candidates)
        return adjusted_probs.get(side, pd.NA)
    if market == "Total":
        if "over 2.5" in selection:
            return adjusted_probs.get("over_2_5", pd.NA)
        if "under 2.5" in selection:
            return adjusted_probs.get("under_2_5", pd.NA)
        if "over 3.5" in selection:
            return adjusted_probs.get("over_3_5", pd.NA)
        if "under 3.5" in selection:
            return adjusted_probs.get("under_3_5", pd.NA)
    if market == "BTTS":
        return adjusted_probs.get("btts_yes" if selection == "yes" else "btts_no", pd.NA)
    baseline_prob = coerce_float(row.get("model_prob", row.get("model_probability")), float("nan"))
    side = _closest_probability_key(
        baseline_prob,
        {
            "home_minus_1_5": baseline_probs.get("home_minus_1_5"),
            "away_plus_1_5": baseline_probs.get("away_plus_1_5"),
            "away_minus_1_5": baseline_probs.get("away_minus_1_5"),
            "home_plus_1_5": baseline_probs.get("home_plus_1_5"),
            "home_or_draw": baseline_probs.get("home_or_draw"),
            "draw_or_away": baseline_probs.get("draw_or_away"),
        },
    )
    return adjusted_probs.get(side, pd.NA) if side else pd.NA


def _closest_probability_key(value: float, candidates: dict[str, Any]) -> str:
    valid = {
        key: coerce_float(prob, float("nan"))
        for key, prob in candidates.items()
        if pd.notna(coerce_float(prob, float("nan"))) and pd.notna(value)
    }
    if not valid:
        return ""
    return min(valid, key=lambda key: abs(valid[key] - value))


def _adjustment_diag(applied: bool, reason: str, baseline: dict[str, float], context: dict[str, float], warnings: list[str]) -> dict[str, Any]:
    shifts = [
        abs(coerce_float(context.get(key), 0.0) - coerce_float(baseline.get(key), 0.0)) * 100.0
        for key in ["home_win", "draw", "away_win"]
        if pd.notna(context.get(key)) and pd.notna(baseline.get(key))
    ]
    return {
        "adjustment_applied": bool(applied),
        "adjustment_reason": reason,
        "baseline_home_win": baseline.get("home_win", pd.NA),
        "baseline_draw": baseline.get("draw", pd.NA),
        "baseline_away_win": baseline.get("away_win", pd.NA),
        "context_home_win": context.get("home_win", pd.NA),
        "context_draw": context.get("draw", pd.NA),
        "context_away_win": context.get("away_win", pd.NA),
        "max_probability_shift_pp": max(shifts) if shifts else pd.NA,
        "warnings": "; ".join(warnings),
    }


def _context_signal(alpha_gap: Any) -> str:
    if pd.isna(alpha_gap):
        return "No signal"
    gap = float(alpha_gap)
    if gap >= 5:
        return "Positive context gap"
    if gap <= -5:
        return "Negative context gap"
    return "Near fair"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
