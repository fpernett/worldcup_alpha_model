from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

import pandas as pd

from src.team_names import normalize_team_name as _canonical_team_name
from src.team_names import team_name_key
from src.utils import coerce_bool, coerce_float


KNOWN_REGULAR_TIME_RESULT_SOURCES = {
    "auto_dashboard_completed_matches",
    "bulk_completed_results_sync",
    "completed_matches",
    "manual_dashboard_completed_results_refresh",
    "results_ledger",
    "test",
}

RESULT_SEMANTICS_COLUMNS = [
    "actual_advancing_team",
    "result_semantics",
    "evaluation_eligible_1x2",
    "had_extra_time",
    "had_penalties",
    "penalties_home",
    "penalties_away",
]

RESULT_FIXTURE_CROSSWALK_COLUMNS = [
    "result_match_id",
    "result_date_utc",
    "result_home",
    "result_away",
    "result_home_goals",
    "result_away_goals",
    "result_actual_result",
    "fixture_match_id",
    "fixture_kickoff_utc",
    "fixture_home",
    "fixture_away",
    "join_method",
    "join_confidence",
    "join_reason",
    "team_order_status",
    "date_delta_hours",
    "canonical_match_key",
    "evaluation_eligible_1x2",
]

RESULT_FIXTURE_SUMMARY_METRICS = [
    "result_rows",
    "fixture_rows",
    "prediction_rows",
    "result_rows_mapped_to_fixtures",
    "mapped_by_exact_match_id",
    "mapped_by_home_away_date",
    "mapped_by_reversed_team_order_date",
    "mapped_by_normalized_team_date",
    "mapped_by_fuzzy_team_date",
    "ambiguous_result_rows",
    "unmapped_result_rows",
]

MAPPED_FIXTURE_JOIN_METHODS = {
    "exact_match_id",
    "exact_home_away_same_utc_date",
    "exact_home_away_within_36h",
    "reversed_home_away_same_utc_date",
    "reversed_home_away_within_36h",
    "normalized_home_away_same_date",
    "normalized_home_away_within_36h",
    "fuzzy_team_match_same_date",
    "fuzzy_team_match_within_36h",
}


def normalize_team_name(name: Any) -> str:
    """Return the canonical local team name used for match identity joins."""
    return _canonical_team_name(str(name or "").strip())


def normalize_competition_name(name: Any) -> str:
    text = _plain_key(name)
    if "world cup" in text:
        return "world cup"
    return text


def normalize_match_date(date_or_kickoff: Any) -> str:
    if date_or_kickoff is None:
        return ""
    text = str(date_or_kickoff or "").strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(text[:10], errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def build_match_key(home: Any, away: Any, date: Any, competition: Any) -> tuple[str, str, str, str]:
    return (
        _team_key(home),
        _team_key(away),
        normalize_match_date(date),
        normalize_competition_name(competition),
    )


def build_symmetric_match_key(team_a: Any, team_b: Any, date: Any, competition: Any) -> tuple[str, str, str, str]:
    first, second = sorted([_team_key(team_a), _team_key(team_b)])
    return (first, second, normalize_match_date(date), normalize_competition_name(competition))


def validate_results_ledger_semantics(results_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return result rows with explicit 90-minute evaluation semantics.

    Existing scores are preserved. Missing semantics are only defaulted to
    regular time for result sources that this project owns and already imports
    from regular-time score fields.
    """
    df = results_df.copy() if results_df is not None else pd.DataFrame()
    for col in RESULT_SEMANTICS_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    if df.empty:
        return df

    statuses: list[str] = []
    eligible: list[bool] = []
    semantics_out: list[str] = []
    actual_out: list[str] = []
    extra_out: list[bool] = []
    penalties_out: list[bool] = []
    penalties_home_out: list[Any] = []
    penalties_away_out: list[Any] = []

    for _, row in df.iterrows():
        home_goals = _goals(row, "home")
        away_goals = _goals(row, "away")
        derived = _actual_result_from_goals(home_goals, away_goals)
        current_actual = _text_value(row.get("actual_result", "")) or _text_value(row.get("actual_result_1x2", ""))
        actual = current_actual or derived
        source = _text_value(row.get("result_source", "")) or _text_value(row.get("source", ""))
        semantics = _text_value(row.get("result_semantics", "")) or _text_value(row.get("score_semantics", ""))
        if not semantics:
            semantics = "90-minute regular time" if _known_regular_time_source(source) else "needs_review"
        semantics_lower = semantics.lower()
        has_valid_goals = pd.notna(home_goals) and pd.notna(away_goals)
        actual_matches = bool(derived and actual == derived)
        is_regular_time = _is_regular_time_semantics(semantics)
        row_eligible = bool(has_valid_goals and actual_matches and is_regular_time)
        if row_eligible:
            status = "ok"
        elif semantics_lower == "needs_review":
            status = "semantics_needs_review"
        elif not actual_matches:
            status = "actual_result_mismatch"
        else:
            status = "not_90_minute_eligible"

        extra = coerce_bool(row.get("had_extra_time"))
        penalties = coerce_bool(row.get("had_penalties"))
        penalties_home = row.get("penalties_home", pd.NA)
        penalties_away = row.get("penalties_away", pd.NA)

        statuses.append(status)
        eligible.append(row_eligible)
        semantics_out.append(semantics)
        actual_out.append(actual)
        extra_out.append(extra)
        penalties_out.append(penalties)
        penalties_home_out.append(penalties_home)
        penalties_away_out.append(penalties_away)

    df["result_semantics"] = semantics_out
    df["actual_result"] = actual_out
    df["evaluation_eligible_1x2"] = eligible
    df["had_extra_time"] = extra_out
    df["had_penalties"] = penalties_out
    df["penalties_home"] = penalties_home_out
    df["penalties_away"] = penalties_away_out
    df["result_semantics_status"] = statuses
    df["actual_advancing_team"] = df["actual_advancing_team"].fillna("")
    return df


def build_result_fixture_crosswalk(results_df: pd.DataFrame | None, fixtures_df: pd.DataFrame | None) -> pd.DataFrame:
    """Map completed-result rows to official fixture rows using schedule identity.

    The crosswalk is intentionally conservative. It records ambiguous and
    unmapped rows instead of silently choosing a low-confidence fixture.
    """
    results = _prepare_crosswalk_results(results_df)
    fixtures = _prepare_crosswalk_fixtures(fixtures_df)
    if results.empty:
        return pd.DataFrame(columns=RESULT_FIXTURE_CROSSWALK_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, result in results.iterrows():
        resolution = _resolve_result_to_fixture(result, fixtures)
        fixture = resolution.get("fixture")
        rows.append(_crosswalk_row(result, fixture, resolution))
    return pd.DataFrame(rows, columns=RESULT_FIXTURE_CROSSWALK_COLUMNS)


def result_fixture_crosswalk_summary(
    crosswalk_df: pd.DataFrame | None,
    *,
    result_rows: int | None = None,
    fixture_rows: int | None = None,
    prediction_rows: int | None = None,
) -> pd.DataFrame:
    """Return a compact count summary for the schedule bridge."""
    crosswalk = _prepare_result_fixture_crosswalk(crosswalk_df)
    mapped = crosswalk["join_method"].isin(MAPPED_FIXTURE_JOIN_METHODS) if not crosswalk.empty else pd.Series(dtype=bool)
    rows = [
        {"metric": "result_rows", "value": int(result_rows if result_rows is not None else len(crosswalk))},
        {"metric": "fixture_rows", "value": int(fixture_rows or 0)},
        {"metric": "prediction_rows", "value": int(prediction_rows or 0)},
        {"metric": "result_rows_mapped_to_fixtures", "value": int(mapped.sum()) if not crosswalk.empty else 0},
        {"metric": "mapped_by_exact_match_id", "value": int((crosswalk["join_method"] == "exact_match_id").sum()) if not crosswalk.empty else 0},
        {
            "metric": "mapped_by_home_away_date",
            "value": int(crosswalk["join_method"].isin(["exact_home_away_same_utc_date", "exact_home_away_within_36h"]).sum()) if not crosswalk.empty else 0,
        },
        {
            "metric": "mapped_by_reversed_team_order_date",
            "value": int(crosswalk["join_method"].isin(["reversed_home_away_same_utc_date", "reversed_home_away_within_36h"]).sum()) if not crosswalk.empty else 0,
        },
        {
            "metric": "mapped_by_normalized_team_date",
            "value": int(crosswalk["join_method"].isin(["normalized_home_away_same_date", "normalized_home_away_within_36h"]).sum()) if not crosswalk.empty else 0,
        },
        {
            "metric": "mapped_by_fuzzy_team_date",
            "value": int(crosswalk["join_method"].isin(["fuzzy_team_match_same_date", "fuzzy_team_match_within_36h"]).sum()) if not crosswalk.empty else 0,
        },
        {"metric": "ambiguous_result_rows", "value": int((crosswalk["join_method"] == "ambiguous_multiple_candidates").sum()) if not crosswalk.empty else 0},
        {"metric": "unmapped_result_rows", "value": int((crosswalk["join_method"] == "no_fixture_match").sum()) if not crosswalk.empty else 0},
    ]
    return pd.DataFrame(rows)


def render_schedule_result_bridge_audit(
    crosswalk_df: pd.DataFrame | None,
    *,
    result_rows: int,
    fixture_rows: int,
    prediction_rows: int,
) -> str:
    crosswalk = _prepare_result_fixture_crosswalk(crosswalk_df)
    summary = result_fixture_crosswalk_summary(
        crosswalk,
        result_rows=result_rows,
        fixture_rows=fixture_rows,
        prediction_rows=prediction_rows,
    )
    unmapped = crosswalk.loc[crosswalk["join_method"].astype(str).eq("no_fixture_match")].copy() if not crosswalk.empty else pd.DataFrame()
    ambiguous = crosswalk.loc[crosswalk["join_method"].astype(str).eq("ambiguous_multiple_candidates")].copy() if not crosswalk.empty else pd.DataFrame()
    lines = [
        "# Schedule Result Bridge Audit",
        "",
        "This audit maps `data/results_ledger.csv` rows to official `data/fixtures.csv` rows before prediction evaluation. It does not overwrite either source ledger.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Unmapped Result Rows",
        "",
        _markdown_table(_audit_display_rows(unmapped)),
        "",
        "## Ambiguous Result Rows",
        "",
        _markdown_table(_audit_display_rows(ambiguous)),
        "",
        "Some result sources use provider IDs while saved prediction snapshots use fixture IDs. The schedule bridge uses date and team names so completed results can be matched to saved predictions without changing the source ledgers.",
    ]
    return "\n".join(lines)


def resolve_prediction_to_result(
    prediction_row: pd.Series | dict[str, Any],
    results_df: pd.DataFrame | None,
    fixtures_df: pd.DataFrame | None = None,
    result_fixture_crosswalk_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Resolve one prediction row to a completed result row with diagnostics."""
    prediction = pd.Series(prediction_row)
    results = _prepare_results(results_df)
    fixtures = _prepare_fixtures(fixtures_df)
    empty = _resolution(None, "no_result_found", 0.0, "No completed result matched this prediction.")
    if results.empty:
        return empty

    match_id = str(prediction.get("match_id", "") or "").strip()
    if match_id:
        exact = results.loc[results["_match_id"] == match_id]
        if len(exact) == 1:
            return _resolution(exact.iloc[0], "exact_match_id", 1.0, "Prediction match_id matched result match_id exactly.")
        if len(exact) > 1:
            return _resolution(None, "ambiguous_exact_match_id", 0.0, f"{len(exact)} results share match_id {match_id}.")

    crosswalk = _prepare_result_fixture_crosswalk(result_fixture_crosswalk_df)
    via_crosswalk = _resolve_prediction_via_crosswalk(prediction, results, crosswalk)
    if via_crosswalk is not None:
        return via_crosswalk

    fixture = _fixture_for_prediction(prediction, fixtures)
    if fixture is not None:
        fixture_match = _matching_results(
            results,
            fixture.get("home", ""),
            fixture.get("away", ""),
            fixture.get("kickoff_utc", fixture.get("date_utc", "")),
            fixture.get("competition", ""),
            allow_reversed=False,
        )
        if len(fixture_match) == 1:
            return _resolution(fixture_match.iloc[0], "fixture_bridge", 0.98, "Prediction match_id matched fixtures.csv, then fixture teams/date matched result.")
        if len(fixture_match) > 1:
            return _resolution(None, "ambiguous_fixture_bridge", 0.0, f"{len(fixture_match)} fixture-bridge result candidates.")

    direct = _matching_results(
        results,
        _prediction_home(prediction),
        _prediction_away(prediction),
        _prediction_date(prediction),
        _prediction_competition(prediction),
        allow_reversed=False,
    )
    if len(direct) == 1:
        return _resolution(direct.iloc[0], "normalized_team_date", 0.95, "Prediction teams/date matched result after normalization.")
    if len(direct) > 1:
        return _resolution(None, "ambiguous_normalized_team_date", 0.0, f"{len(direct)} normalized team/date candidates.")

    reversed_match = _matching_results(
        results,
        _prediction_home(prediction),
        _prediction_away(prediction),
        _prediction_date(prediction),
        _prediction_competition(prediction),
        allow_reversed=True,
    )
    if len(reversed_match) == 1:
        return _resolution(
            reversed_match.iloc[0],
            "symmetric_team_date",
            0.92,
            "Result teams/date matched with home-away order reversed.",
            home_away_order="reversed",
        )
    if len(reversed_match) > 1:
        return _resolution(None, "ambiguous_symmetric_team_date", 0.0, f"{len(reversed_match)} symmetric team/date candidates.")

    fuzzy = _fuzzy_match(results, prediction)
    if fuzzy is not None:
        row, confidence, order = fuzzy
        return _resolution(row, "fuzzy_team_date", confidence, "High-confidence fuzzy team/date match.", home_away_order=order)

    return empty


def align_result_to_prediction(result_row: pd.Series | dict[str, Any], home_away_order: str = "same") -> pd.Series:
    """Return result row values from the prediction's home/away perspective."""
    row = pd.Series(result_row).copy()
    if home_away_order != "reversed":
        return row
    home_goals = row.get("actual_home_goals_90", row.get("home_goals", pd.NA))
    away_goals = row.get("actual_away_goals_90", row.get("away_goals", pd.NA))
    row["actual_home_goals_90"] = away_goals
    row["actual_away_goals_90"] = home_goals
    row["home_goals"] = away_goals
    row["away_goals"] = home_goals
    actual = str(row.get("actual_result_1x2", row.get("actual_result", "")) or "")
    if actual == "home_win":
        actual = "away_win"
    elif actual == "away_win":
        actual = "home_win"
    row["actual_result_1x2"] = actual
    row["actual_result"] = actual
    return row


def _prepare_results(results_df: pd.DataFrame | None) -> pd.DataFrame:
    df = validate_results_ledger_semantics(results_df)
    if df.empty:
        return df
    if "actual_result_1x2" not in df.columns:
        df["actual_result_1x2"] = df["actual_result"]
    if "actual_home_goals_90" not in df.columns:
        df["actual_home_goals_90"] = df.get("home_goals", pd.NA)
    if "actual_away_goals_90" not in df.columns:
        df["actual_away_goals_90"] = df.get("away_goals", pd.NA)
    for col in ["match_id", "home", "away", "home_team", "away_team", "date_utc", "competition"]:
        if col not in df.columns:
            df[col] = ""
    df["_match_id"] = df["match_id"].fillna("").astype(str).str.strip()
    df["_home_key"] = df.apply(lambda row: _team_key(row.get("home") or row.get("home_team")), axis=1)
    df["_away_key"] = df.apply(lambda row: _team_key(row.get("away") or row.get("away_team")), axis=1)
    df["_date_key"] = df["date_utc"].map(normalize_match_date)
    df["_competition_key"] = df["competition"].map(normalize_competition_name)
    return df


def _prepare_fixtures(fixtures_df: pd.DataFrame | None) -> pd.DataFrame:
    df = fixtures_df.copy() if fixtures_df is not None else pd.DataFrame()
    if df.empty:
        return df
    for col in ["match_id", "home", "away", "date_utc", "time_utc", "kickoff_utc", "competition"]:
        if col not in df.columns:
            df[col] = ""
    missing_kickoff = df["kickoff_utc"].fillna("").astype(str).str.strip().eq("")
    df.loc[missing_kickoff, "kickoff_utc"] = (
        df.loc[missing_kickoff, "date_utc"].fillna("").astype(str)
        + "T"
        + df.loc[missing_kickoff, "time_utc"].fillna("00:00").astype(str)
        + ":00+00:00"
    )
    df["_match_id"] = df["match_id"].fillna("").astype(str).str.strip()
    return df


def _prepare_crosswalk_results(results_df: pd.DataFrame | None) -> pd.DataFrame:
    df = validate_results_ledger_semantics(results_df)
    if df.empty:
        return df
    for col in ["match_id", "date_utc", "competition", "home", "away", "home_team", "away_team", "home_goals", "away_goals", "actual_result"]:
        if col not in df.columns:
            df[col] = ""
    df["home"] = df["home"].where(df["home"].fillna("").astype(str).str.strip().ne(""), df["home_team"])
    df["away"] = df["away"].where(df["away"].fillna("").astype(str).str.strip().ne(""), df["away_team"])
    df["_match_id"] = df["match_id"].fillna("").astype(str).str.strip()
    df["_home_exact"] = df["home"].map(_plain_key)
    df["_away_exact"] = df["away"].map(_plain_key)
    df["_home_key"] = df["home"].map(_team_key)
    df["_away_key"] = df["away"].map(_team_key)
    df["_date_key"] = df["date_utc"].map(normalize_match_date)
    df["_competition_key"] = df["competition"].map(normalize_competition_name)
    df["_date_ts"] = pd.to_datetime(df["date_utc"], errors="coerce", utc=True)
    return df


def _prepare_crosswalk_fixtures(fixtures_df: pd.DataFrame | None) -> pd.DataFrame:
    df = _prepare_fixtures(fixtures_df)
    if df.empty:
        return df
    df["_home_exact"] = df["home"].map(_plain_key)
    df["_away_exact"] = df["away"].map(_plain_key)
    df["_home_key"] = df["home"].map(_team_key)
    df["_away_key"] = df["away"].map(_team_key)
    df["_date_key"] = df["kickoff_utc"].map(normalize_match_date)
    df["_competition_key"] = df["competition"].map(normalize_competition_name)
    df["_kickoff_ts"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    return df


def _prepare_result_fixture_crosswalk(crosswalk_df: pd.DataFrame | None) -> pd.DataFrame:
    df = crosswalk_df.copy() if crosswalk_df is not None else pd.DataFrame(columns=RESULT_FIXTURE_CROSSWALK_COLUMNS)
    for col in RESULT_FIXTURE_CROSSWALK_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[RESULT_FIXTURE_CROSSWALK_COLUMNS].copy()


def _resolve_result_to_fixture(result: pd.Series, fixtures: pd.DataFrame) -> dict[str, Any]:
    if fixtures.empty:
        return _fixture_resolution(None, "no_fixture_match", 0.0, "No fixture rows are available for schedule matching.", "unknown")

    match_id = str(result.get("_match_id", "") or "")
    if match_id:
        exact = fixtures.loc[fixtures["_match_id"] == match_id].copy()
        selected = _select_fixture_candidate(
            result,
            exact,
            "exact_match_id",
            1.0,
            "Result match_id matched fixture match_id exactly.",
            "same",
            allow_best_by_delta=False,
        )
        if selected is not None:
            return selected

    steps = [
        ("exact_home_away_same_utc_date", "exact", "same", "same_date", 0.98, "Result teams exactly matched fixture teams on the same UTC date."),
        ("exact_home_away_within_36h", "exact", "same", "within_36h", 0.96, "Result teams exactly matched fixture teams within 36 hours."),
        ("reversed_home_away_same_utc_date", "exact", "reversed", "same_date", 0.93, "Result teams exactly matched fixture teams on the same UTC date with home-away order reversed."),
        ("reversed_home_away_within_36h", "exact", "reversed", "within_36h", 0.90, "Result teams exactly matched fixture teams within 36 hours with home-away order reversed."),
        ("normalized_home_away_same_date", "normalized", "same", "same_date", 0.90, "Normalized result teams matched fixture teams on the same UTC date."),
        ("normalized_home_away_within_36h", "normalized", "same", "within_36h", 0.88, "Normalized result teams matched fixture teams within 36 hours."),
    ]
    for method, name_mode, order, date_mode, confidence, reason in steps:
        candidates = _fixture_candidates(result, fixtures, name_mode=name_mode, order=order, date_mode=date_mode)
        selected = _select_fixture_candidate(
            result,
            candidates,
            method,
            confidence,
            reason,
            order,
            allow_best_by_delta=date_mode == "within_36h",
        )
        if selected is not None:
            return selected

    for method, date_mode, confidence, reason in [
        ("fuzzy_team_match_same_date", "same_date", 0.84, "High-confidence fuzzy team match on the same UTC date."),
        ("fuzzy_team_match_within_36h", "within_36h", 0.82, "High-confidence fuzzy team match within 36 hours."),
    ]:
        selected = _select_fuzzy_fixture_candidate(result, fixtures, method, date_mode, confidence, reason)
        if selected is not None:
            return selected

    return _fixture_resolution(None, "no_fixture_match", 0.0, "No fixture matched this result row by ID, teams, and date.", "unknown")


def _fixture_candidates(
    result: pd.Series,
    fixtures: pd.DataFrame,
    *,
    name_mode: str,
    order: str,
    date_mode: str,
) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures
    if name_mode == "exact":
        home_col, away_col = "_home_exact", "_away_exact"
    else:
        home_col, away_col = "_home_key", "_away_key"
    result_home = str(result.get(home_col, "") or "")
    result_away = str(result.get(away_col, "") or "")
    if not result_home or not result_away:
        return fixtures.iloc[0:0].copy()
    if order == "reversed":
        mask = (fixtures[home_col] == result_away) & (fixtures[away_col] == result_home)
    else:
        mask = (fixtures[home_col] == result_home) & (fixtures[away_col] == result_away)
    mask = mask & _competition_mask(result, fixtures)
    if date_mode == "same_date":
        mask = mask & fixtures["_date_key"].eq(str(result.get("_date_key", "") or ""))
    else:
        deltas = fixtures.apply(lambda fixture: _date_delta_hours(result, fixture), axis=1)
        mask = mask & deltas.le(36.0)
    return fixtures.loc[mask].copy()


def _select_fixture_candidate(
    result: pd.Series,
    candidates: pd.DataFrame,
    method: str,
    confidence: float,
    reason: str,
    order: str,
    *,
    allow_best_by_delta: bool,
) -> dict[str, Any] | None:
    if candidates.empty:
        return None
    candidates = candidates.copy()
    candidates["_date_delta_hours_for_join"] = candidates.apply(lambda fixture: _date_delta_hours(result, fixture), axis=1)
    if len(candidates) == 1:
        return _fixture_resolution(candidates.iloc[0], method, confidence, reason, order)
    if allow_best_by_delta:
        ordered = candidates.sort_values(["_date_delta_hours_for_join", "_match_id"])
        best = ordered.iloc[0]
        second = ordered.iloc[1]
        best_delta = coerce_float(best.get("_date_delta_hours_for_join"), float("inf"))
        second_delta = coerce_float(second.get("_date_delta_hours_for_join"), float("inf"))
        if pd.notna(best_delta) and pd.notna(second_delta) and best_delta + 6.0 < second_delta:
            return _fixture_resolution(best, method, confidence - 0.01, reason + " Closest kickoff was selected.", order)
    return _fixture_resolution(
        None,
        "ambiguous_multiple_candidates",
        0.0,
        f"{len(candidates)} fixture candidates matched by {method}; no row was joined silently.",
        "ambiguous",
    )


def _select_fuzzy_fixture_candidate(
    result: pd.Series,
    fixtures: pd.DataFrame,
    method: str,
    date_mode: str,
    confidence: float,
    reason: str,
) -> dict[str, Any] | None:
    if fixtures.empty:
        return None
    if date_mode == "same_date":
        candidates = fixtures.loc[fixtures["_date_key"].eq(str(result.get("_date_key", "") or ""))].copy()
    else:
        deltas = fixtures.apply(lambda fixture: _date_delta_hours(result, fixture), axis=1)
        candidates = fixtures.loc[deltas.le(36.0)].copy()
    if candidates.empty:
        return None
    candidates = candidates.loc[_competition_mask(result, candidates)].copy()
    if candidates.empty:
        return None

    result_home = str(result.get("_home_key", "") or "")
    result_away = str(result.get("_away_key", "") or "")
    scored: list[tuple[float, str, pd.Series]] = []
    for _, fixture in candidates.iterrows():
        same_score = min(_similarity(result_home, fixture.get("_home_key", "")), _similarity(result_away, fixture.get("_away_key", "")))
        reversed_score = min(_similarity(result_home, fixture.get("_away_key", "")), _similarity(result_away, fixture.get("_home_key", "")))
        if same_score >= reversed_score:
            score, order = same_score, "same"
        else:
            score, order = reversed_score, "reversed"
        if score >= 0.92:
            scored.append((score, order, fixture))
    if not scored:
        return None
    scored = sorted(scored, key=lambda item: (item[0], -_date_delta_hours(result, item[2])), reverse=True)
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 0.03:
        return _fixture_resolution(
            None,
            "ambiguous_multiple_candidates",
            0.0,
            f"{len(scored)} high-confidence fuzzy fixture candidates were too close to choose safely.",
            "ambiguous",
        )
    score, order, fixture = scored[0]
    return _fixture_resolution(fixture, method, min(confidence, score), reason, order)


def _fixture_resolution(
    fixture: pd.Series | None,
    method: str,
    confidence: float,
    reason: str,
    order: str,
) -> dict[str, Any]:
    return {
        "fixture": fixture,
        "join_method": method,
        "join_confidence": float(confidence),
        "join_reason": reason,
        "team_order_status": order,
    }


def _crosswalk_row(result: pd.Series, fixture: pd.Series | None, resolution: dict[str, Any]) -> dict[str, Any]:
    has_fixture = fixture is not None and not fixture.empty
    fixture_kickoff = fixture.get("kickoff_utc", "") if has_fixture else ""
    fixture_home = fixture.get("home", "") if has_fixture else ""
    fixture_away = fixture.get("away", "") if has_fixture else ""
    fixture_match_id = fixture.get("match_id", "") if has_fixture else ""
    fixture_competition = fixture.get("competition", "") if has_fixture else result.get("competition", "")
    key = build_match_key(
        fixture_home if has_fixture else result.get("home", ""),
        fixture_away if has_fixture else result.get("away", ""),
        fixture_kickoff if has_fixture else result.get("date_utc", ""),
        fixture_competition,
    )
    date_delta = _date_delta_hours(result, fixture) if has_fixture else pd.NA
    return {
        "result_match_id": str(result.get("match_id", "") or ""),
        "result_date_utc": result.get("date_utc", ""),
        "result_home": result.get("home", ""),
        "result_away": result.get("away", ""),
        "result_home_goals": result.get("home_goals", result.get("actual_home_goals_90", pd.NA)),
        "result_away_goals": result.get("away_goals", result.get("actual_away_goals_90", pd.NA)),
        "result_actual_result": result.get("actual_result", result.get("actual_result_1x2", "")),
        "fixture_match_id": fixture_match_id,
        "fixture_kickoff_utc": fixture_kickoff,
        "fixture_home": fixture_home,
        "fixture_away": fixture_away,
        "join_method": resolution.get("join_method", ""),
        "join_confidence": resolution.get("join_confidence", 0.0),
        "join_reason": resolution.get("join_reason", ""),
        "team_order_status": resolution.get("team_order_status", "unknown"),
        "date_delta_hours": date_delta,
        "canonical_match_key": _string_match_key(key),
        "evaluation_eligible_1x2": bool(result.get("evaluation_eligible_1x2", False)),
    }


def _resolve_prediction_via_crosswalk(
    prediction: pd.Series,
    results: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> dict[str, Any] | None:
    if crosswalk.empty or results.empty:
        return None
    match_id = str(prediction.get("match_id", "") or "").strip()
    if not match_id:
        return None
    candidates = crosswalk.loc[
        crosswalk["fixture_match_id"].fillna("").astype(str).str.strip().eq(match_id)
        & crosswalk["join_method"].isin(MAPPED_FIXTURE_JOIN_METHODS)
    ].copy()
    if candidates.empty:
        return None
    if len(candidates) > 1:
        exact_fixture_result = candidates.loc[candidates["result_match_id"].fillna("").astype(str).str.strip().eq(match_id)]
        if len(exact_fixture_result) == 1:
            candidates = exact_fixture_result
        else:
            return _resolution(
                None,
                "ambiguous_result_fixture_crosswalk",
                0.0,
                f"{len(candidates)} result rows mapped to fixture {match_id}; no prediction/result join was made silently.",
                fixture_match_id=match_id,
                team_order_status="ambiguous",
            )
    crosswalk_row = candidates.iloc[0]
    result_match_id = str(crosswalk_row.get("result_match_id", "") or "").strip()
    result = results.loc[results["_match_id"].eq(result_match_id)]
    if len(result) != 1:
        return _resolution(
            None,
            "result_fixture_crosswalk_missing_result",
            0.0,
            f"Crosswalk mapped fixture {match_id} to result {result_match_id}, but the result row was not unique in evaluation input.",
            fixture_match_id=match_id,
            team_order_status=str(crosswalk_row.get("team_order_status", "unknown") or "unknown"),
        )
    team_order = str(crosswalk_row.get("team_order_status", "same") or "same")
    return _resolution(
        result.iloc[0],
        "result_fixture_crosswalk",
        coerce_float(crosswalk_row.get("join_confidence"), 0.0),
        f"Prediction fixture_id matched result-fixture crosswalk via {crosswalk_row.get('join_method', '')}: {crosswalk_row.get('join_reason', '')}",
        home_away_order="reversed" if team_order == "reversed" else "same",
        fixture_match_id=match_id,
        team_order_status=team_order,
        crosswalk_join_method=str(crosswalk_row.get("join_method", "") or ""),
    )


def _fixture_for_prediction(prediction: pd.Series, fixtures: pd.DataFrame) -> pd.Series | None:
    if fixtures.empty:
        return None
    match_id = str(prediction.get("match_id", "") or "").strip()
    if match_id:
        found = fixtures.loc[fixtures["_match_id"] == match_id]
        if len(found) == 1:
            return found.iloc[0]
    key = build_match_key(
        _prediction_home(prediction),
        _prediction_away(prediction),
        _prediction_date(prediction),
        _prediction_competition(prediction),
    )
    if not key[2]:
        return None
    fixture_keys = fixtures.apply(
        lambda row: build_match_key(row.get("home"), row.get("away"), row.get("kickoff_utc") or row.get("date_utc"), row.get("competition")),
        axis=1,
    )
    matched = fixtures.loc[fixture_keys == key]
    return matched.iloc[0] if len(matched) == 1 else None


def _matching_results(
    results: pd.DataFrame,
    home: Any,
    away: Any,
    date: Any,
    competition: Any,
    *,
    allow_reversed: bool,
) -> pd.DataFrame:
    home_key = _team_key(home)
    away_key = _team_key(away)
    date_key = normalize_match_date(date)
    competition_key = normalize_competition_name(competition)
    if not home_key or not away_key or not date_key:
        return results.iloc[0:0].copy()
    if allow_reversed:
        mask = (results["_home_key"] == away_key) & (results["_away_key"] == home_key)
    else:
        mask = (results["_home_key"] == home_key) & (results["_away_key"] == away_key)
    mask = mask & (results["_date_key"] == date_key)
    if competition_key:
        comp_mask = results["_competition_key"].eq(competition_key) | results["_competition_key"].eq("")
        mask = mask & comp_mask
    return results.loc[mask].copy()


def _fuzzy_match(results: pd.DataFrame, prediction: pd.Series) -> tuple[pd.Series, float, str] | None:
    date_key = normalize_match_date(_prediction_date(prediction))
    competition_key = normalize_competition_name(_prediction_competition(prediction))
    if not date_key:
        return None
    candidates = results.loc[results["_date_key"] == date_key].copy()
    if competition_key:
        candidates = candidates.loc[candidates["_competition_key"].isin([competition_key, ""])].copy()
    if candidates.empty:
        return None
    pred_home = _team_key(_prediction_home(prediction))
    pred_away = _team_key(_prediction_away(prediction))
    scored: list[tuple[float, str, pd.Series]] = []
    for _, row in candidates.iterrows():
        same = min(_similarity(pred_home, row["_home_key"]), _similarity(pred_away, row["_away_key"]))
        reversed_score = min(_similarity(pred_home, row["_away_key"]), _similarity(pred_away, row["_home_key"]))
        if same >= reversed_score:
            scored.append((same, "same", row))
        else:
            scored.append((reversed_score, "reversed", row))
    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < 0.92:
        return None
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 0.03:
        return None
    return scored[0][2], scored[0][0], scored[0][1]


def _resolution(
    row: pd.Series | None,
    method: str,
    confidence: float,
    reason: str,
    *,
    home_away_order: str = "same",
    fixture_match_id: str = "",
    team_order_status: str | None = None,
    crosswalk_join_method: str = "",
) -> dict[str, Any]:
    return {
        "result": row,
        "resolved_result_match_id": "" if row is None else str(row.get("match_id", "") or ""),
        "result_join_method": method,
        "result_join_confidence": float(confidence),
        "result_join_reason": reason,
        "home_away_order": home_away_order,
        "fixture_match_id": fixture_match_id,
        "team_order_status": team_order_status or home_away_order,
        "crosswalk_join_method": crosswalk_join_method,
    }


def _prediction_home(prediction: pd.Series) -> Any:
    return prediction.get("home_team", prediction.get("home", ""))


def _prediction_away(prediction: pd.Series) -> Any:
    return prediction.get("away_team", prediction.get("away", ""))


def _prediction_date(prediction: pd.Series) -> Any:
    return prediction.get("kickoff_utc", prediction.get("date_utc", ""))


def _prediction_competition(prediction: pd.Series) -> Any:
    return prediction.get("competition", "")


def _team_key(value: Any) -> str:
    return team_name_key(normalize_team_name(value))


def _competition_mask(result: pd.Series, fixtures: pd.DataFrame) -> pd.Series:
    competition_key = str(result.get("_competition_key", "") or "")
    if not competition_key:
        return pd.Series(True, index=fixtures.index)
    return fixtures["_competition_key"].isin([competition_key, ""])


def _date_delta_hours(result: pd.Series, fixture: pd.Series | None) -> Any:
    if fixture is None or fixture.empty:
        return pd.NA
    result_ts = pd.to_datetime(result.get("_date_ts", result.get("date_utc", "")), errors="coerce", utc=True)
    fixture_ts = pd.to_datetime(fixture.get("_kickoff_ts", fixture.get("kickoff_utc", "")), errors="coerce", utc=True)
    if pd.isna(result_ts) or pd.isna(fixture_ts):
        return pd.NA
    return abs((fixture_ts - result_ts).total_seconds()) / 3600.0


def _string_match_key(key: tuple[str, str, str, str]) -> str:
    return "|".join(str(part or "") for part in key)


def _audit_display_rows(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "result_match_id",
        "result_date_utc",
        "result_home",
        "result_away",
        "join_method",
        "join_reason",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df[[col for col in columns if col in df.columns]].copy()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(col, "") if pd.notna(row.get(col, "")) else "").replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _plain_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value or "").strip()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _goals(row: pd.Series, side: str) -> Any:
    candidates = (
        [f"{side}_goals", f"{side}_score_90", f"actual_{side}_goals_90"]
        if side in {"home", "away"}
        else []
    )
    for col in candidates:
        if col in row.index and pd.notna(row.get(col)):
            value = coerce_float(row.get(col), float("nan"))
            if pd.notna(value):
                return value
    return pd.NA


def _actual_result_from_goals(home_goals: Any, away_goals: Any) -> str:
    home = coerce_float(home_goals, float("nan"))
    away = coerce_float(away_goals, float("nan"))
    if pd.isna(home) or pd.isna(away):
        return ""
    if home > away:
        return "home_win"
    if away > home:
        return "away_win"
    return "draw"


def _known_regular_time_source(source: str) -> bool:
    return str(source or "").strip() in KNOWN_REGULAR_TIME_RESULT_SOURCES


def _is_regular_time_semantics(semantics: Any) -> bool:
    text = str(semantics or "").strip().lower()
    if not text or text == "needs_review":
        return False
    if "penalt" in text or "extra time included" in text or "advancement" in text:
        return False
    return "90" in text or "regular" in text
