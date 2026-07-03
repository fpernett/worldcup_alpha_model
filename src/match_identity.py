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


def resolve_prediction_to_result(
    prediction_row: pd.Series | dict[str, Any],
    results_df: pd.DataFrame | None,
    fixtures_df: pd.DataFrame | None = None,
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
) -> dict[str, Any]:
    return {
        "result": row,
        "resolved_result_match_id": "" if row is None else str(row.get("match_id", "") or ""),
        "result_join_method": method,
        "result_join_confidence": float(confidence),
        "result_join_reason": reason,
        "home_away_order": home_away_order,
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
