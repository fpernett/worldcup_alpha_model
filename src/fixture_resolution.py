from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.team_names import is_unresolved_team_slot, normalize_team_name
from src.utils import coerce_float, read_csv_with_columns


FIXTURE_COLUMNS = [
    "match_id",
    "date_utc",
    "time_utc",
    "competition",
    "group",
    "home",
    "away",
    "venue",
    "city",
    "country",
]

RESOLUTION_COLUMNS = [
    "source_home",
    "source_away",
    "home_resolution_status",
    "away_resolution_status",
    "fixture_resolution_status",
    "fixture_resolution_detail",
]

RESULT_COLUMNS = [
    "match_id",
    "date_utc",
    "competition",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "actual_result",
    "actual_advancing_team",
    "result_semantics",
    "result_source",
    "last_updated",
]

COMPLETED_RESULT_COLUMNS = [
    "provider",
    "provider_match_id",
    "provider_kickoff_utc",
    "date_utc",
    "competition",
    "group",
    "home",
    "away",
    "home_score_90",
    "away_score_90",
    "score_source",
    "score_semantics",
    "is_completed",
    "source",
    "last_updated",
]


_WINNER_LOSER_PATTERN = re.compile(r"^\s*(winner|loser)\s+(?:of\s+)?match\s+([A-Za-z0-9_-]+)\s*$", re.IGNORECASE)
_GROUP_PATTERN = re.compile(r"^\s*(winner|runner[-\s]?up)\s+(?:of\s+)?group\s+([A-Za-z])\s*$", re.IGNORECASE)
_THIRD_PATTERN = re.compile(
    r"^\s*(?:best\s+)?(?:3rd|third)(?:\s+place)?\s+(?:of\s+)?groups?\s+([A-Za-z/,\s]+)\s*$",
    re.IGNORECASE,
)


def load_fixture_resolution_results(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load result rows suitable for resolving bracket placeholders."""
    results_ledger = read_csv_with_columns(data_dir / "results_ledger.csv", RESULT_COLUMNS)
    completed = read_csv_with_columns(data_dir / "completed_results.csv", COMPLETED_RESULT_COLUMNS)
    frames = [_normalise_result_rows(results_ledger), _normalise_completed_rows(completed)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["_priority"] = out["match_id"].astype(str).str.startswith("wc2026_").astype(int)
    out["_updated"] = pd.to_datetime(out["last_updated"], errors="coerce", utc=True)
    out = (
        out.sort_values(["_priority", "_updated"], ascending=[False, False])
        .drop_duplicates(subset=["match_id"], keep="first")
        .drop(columns=["_priority", "_updated"])
        .reset_index(drop=True)
    )
    return out[RESULT_COLUMNS].copy()


def resolve_fixture_placeholders(
    fixtures_df: pd.DataFrame | None,
    results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return an in-memory fixture view with bracket placeholders resolved when possible.

    Source fixture rows are not mutated. Rows that cannot be proven from prior
    results keep their original placeholder and receive a pending/manual status.
    """
    fixtures = _normalise_fixtures(fixtures_df)
    if fixtures.empty:
        return _with_resolution_attrs(fixtures)

    results = _normalise_result_rows(results_df) if results_df is not None else load_fixture_resolution_results()
    out = fixtures.copy()
    for col in RESOLUTION_COLUMNS:
        out[col] = ""
    out["source_home"] = out["home"].astype(str)
    out["source_away"] = out["away"].astype(str)

    # Iterate to allow later rounds to resolve after earlier resolved fixture
    # names become available, while still requiring actual result rows for the
    # referenced match.
    for _ in range(max(1, len(out))):
        changed = False
        fixture_lookup = _fixture_lookup(out)
        standings = _group_standings(out, results)
        for idx, row in out.iterrows():
            for side in ["home", "away"]:
                original = str(row.get(side, "") or "").strip()
                resolved, status, detail = _resolve_slot(original, fixture_lookup, standings, results)
                status_col = f"{side}_resolution_status"
                if status == "resolved" and resolved and resolved != original:
                    out.at[idx, side] = resolved
                    out.at[idx, status_col] = "Resolved automatically"
                    changed = True
                elif status == "source":
                    source_value = str(out.at[idx, f"source_{side}"] or "").strip()
                    if is_unresolved_team_slot(source_value) and original and original != source_value:
                        out.at[idx, status_col] = "Resolved automatically"
                    else:
                        out.at[idx, status_col] = "Source team"
                elif status == "pending":
                    out.at[idx, status_col] = "Pending prior result"
                elif status == "manual":
                    out.at[idx, status_col] = "Requires manual mapping"
                if detail:
                    existing = str(out.at[idx, "fixture_resolution_detail"] or "")
                    out.at[idx, "fixture_resolution_detail"] = _join_unique(existing, detail)
        if not changed:
            break

    out["fixture_resolution_status"] = out.apply(_fixture_status, axis=1)
    out.attrs = fixtures.attrs.copy()
    out.attrs.update(_resolution_summary(out))
    return out


def _resolve_slot(
    value: str,
    fixture_lookup: dict[str, pd.Series],
    standings: dict[str, list[dict[str, Any]]],
    results: pd.DataFrame,
) -> tuple[str, str, str]:
    if not is_unresolved_team_slot(value):
        return value, "source", ""

    winner_loser = _WINNER_LOSER_PATTERN.match(value)
    if winner_loser:
        mode = winner_loser.group(1).lower()
        match_ref = winner_loser.group(2)
        referenced = _referenced_fixture(match_ref, fixture_lookup)
        if referenced is None:
            return value, "pending", f"{value}: prior fixture not found"
        prior = _result_for_fixture(referenced, results)
        if prior is None:
            return value, "pending", f"{value}: waiting for Match {match_ref} result"
        winner, loser, reason = _winner_loser_from_result(prior)
        if not winner:
            return value, "manual", f"{value}: {reason}"
        resolved = winner if mode == "winner" else loser
        if not resolved:
            return value, "manual", f"{value}: loser could not be inferred"
        return resolved, "resolved", f"{value}: resolved to {resolved}"

    group_slot = _GROUP_PATTERN.match(value)
    if group_slot:
        mode = group_slot.group(1).lower().replace("-", " ")
        group = group_slot.group(2).upper()
        table = standings.get(group, [])
        if not table:
            return value, "pending", f"{value}: waiting for Group {group} results"
        index = 0 if mode == "winner" else 1
        if len(table) <= index:
            return value, "pending", f"{value}: Group {group} table incomplete"
        return table[index]["team"], "resolved", f"{value}: resolved to {table[index]['team']}"

    third_slot = _THIRD_PATTERN.match(value)
    if third_slot:
        groups = [part.strip().upper() for part in re.split(r"[/,\s]+", third_slot.group(1)) if part.strip()]
        thirds = [standings[group][2] for group in groups if group in standings and len(standings[group]) >= 3]
        if len(thirds) != len(groups):
            return value, "pending", f"{value}: waiting for all candidate third-place groups"
        thirds = sorted(thirds, key=_standing_sort_key)
        return thirds[0]["team"], "resolved", f"{value}: resolved to {thirds[0]['team']}"

    return value, "manual", f"{value}: placeholder format requires manual mapping"


def _winner_loser_from_result(row: pd.Series) -> tuple[str, str, str]:
    home = normalize_team_name(str(row.get("home", "") or ""))
    away = normalize_team_name(str(row.get("away", "") or ""))
    advancing = normalize_team_name(str(row.get("actual_advancing_team", "") or ""))
    home_goals = coerce_float(row.get("home_goals"), float("nan"))
    away_goals = coerce_float(row.get("away_goals"), float("nan"))
    if advancing:
        if _team_key(advancing) == _team_key(home):
            return home, away, "advancing team"
        if _team_key(advancing) == _team_key(away):
            return away, home, "advancing team"
        return "", "", "advancing team does not match fixture teams"
    if pd.isna(home_goals) or pd.isna(away_goals):
        return "", "", "regular-time score missing"
    if home_goals > away_goals:
        return home, away, "90-minute score"
    if away_goals > home_goals:
        return away, home, "90-minute score"
    return "", "", "90-minute draw requires actual_advancing_team"


def _result_for_fixture(fixture: pd.Series, results: pd.DataFrame) -> pd.Series | None:
    if results.empty:
        return None
    match_ids = _match_id_candidates(fixture.get("match_id", ""))
    by_id = results.loc[results["match_id"].astype(str).isin(match_ids)].copy()
    if not by_id.empty:
        return by_id.iloc[0]

    date = pd.to_datetime(fixture.get("date_utc"), errors="coerce")
    home_key = _team_key(fixture.get("home", ""))
    away_key = _team_key(fixture.get("away", ""))
    if pd.isna(date) or not home_key or not away_key:
        return None
    result_dates = pd.to_datetime(results["date_utc"], errors="coerce")
    result_dates = result_dates.dt.normalize()
    date_delta_days = (result_dates - date.normalize()).abs().dt.days.fillna(9999).astype(int)
    same_day = date_delta_days <= 1
    home = results["home"].map(_team_key)
    away = results["away"].map(_team_key)
    same_order = (home == home_key) & (away == away_key)
    reverse_order = (home == away_key) & (away == home_key)
    matched = results.loc[same_day & (same_order | reverse_order)].copy()
    if matched.empty:
        return None
    return matched.iloc[0]


def _group_standings(fixtures: pd.DataFrame, results: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    standings: dict[str, dict[str, dict[str, Any]]] = {}
    group_rows = fixtures.loc[fixtures["group"].astype(str).str.match(r"Group\s+[A-Z]$", case=False, na=False)].copy()
    for _, fixture in group_rows.iterrows():
        result = _result_for_fixture(fixture, results)
        if result is None:
            continue
        group = str(fixture.get("group", "")).split()[-1].upper()
        table = standings.setdefault(group, {})
        home = normalize_team_name(str(fixture.get("home", "") or result.get("home", "")))
        away = normalize_team_name(str(fixture.get("away", "") or result.get("away", "")))
        home_goals = int(coerce_float(result.get("home_goals"), 0.0))
        away_goals = int(coerce_float(result.get("away_goals"), 0.0))
        _add_group_result(table, home, home_goals, away_goals)
        _add_group_result(table, away, away_goals, home_goals)

    out: dict[str, list[dict[str, Any]]] = {}
    for group, table in standings.items():
        out[group] = sorted(table.values(), key=_standing_sort_key)
    return out


def _add_group_result(table: dict[str, dict[str, Any]], team: str, goals_for: int, goals_against: int) -> None:
    if not team:
        return
    row = table.setdefault(team, {"team": team, "points": 0, "goal_diff": 0, "goals_for": 0})
    row["goals_for"] += int(goals_for)
    row["goal_diff"] += int(goals_for - goals_against)
    if goals_for > goals_against:
        row["points"] += 3
    elif goals_for == goals_against:
        row["points"] += 1


def _standing_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (-int(row.get("points", 0)), -int(row.get("goal_diff", 0)), -int(row.get("goals_for", 0)), str(row.get("team", "")))


def _referenced_fixture(match_ref: str, fixture_lookup: dict[str, pd.Series]) -> pd.Series | None:
    for candidate in _match_id_candidates(match_ref):
        if candidate in fixture_lookup:
            return fixture_lookup[candidate]
    return None


def _fixture_lookup(fixtures: pd.DataFrame) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    for _, row in fixtures.iterrows():
        for candidate in _match_id_candidates(row.get("match_id", "")):
            lookup[candidate] = row
    return lookup


def _match_id_candidates(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    candidates = [raw]
    number = raw
    match = re.search(r"(\d+)$", raw)
    if match:
        number = match.group(1)
        candidates.extend([number, f"wc2026_{number}"])
    return list(dict.fromkeys(candidates))


def _normalise_fixtures(fixtures_df: pd.DataFrame | None) -> pd.DataFrame:
    out = fixtures_df.copy() if fixtures_df is not None else pd.DataFrame(columns=FIXTURE_COLUMNS)
    for col in FIXTURE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out.copy()


def _normalise_completed_rows(df: pd.DataFrame | None) -> pd.DataFrame:
    source = df.copy() if df is not None else pd.DataFrame(columns=COMPLETED_RESULT_COLUMNS)
    if source.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    for col in COMPLETED_RESULT_COLUMNS:
        if col not in source.columns:
            source[col] = pd.NA
    out = pd.DataFrame(
        {
            "match_id": source["provider_match_id"],
            "date_utc": source["date_utc"],
            "competition": source["competition"],
            "home": source["home"],
            "away": source["away"],
            "home_goals": source["home_score_90"],
            "away_goals": source["away_score_90"],
            "actual_result": pd.NA,
            "actual_advancing_team": pd.NA,
            "result_semantics": source["score_semantics"],
            "result_source": source["source"],
            "last_updated": source["last_updated"],
        }
    )
    return _normalise_result_rows(out)


def _normalise_result_rows(df: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame(columns=RESULT_COLUMNS)
    for col in RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[RESULT_COLUMNS].copy()
    if out.empty:
        return out
    for col in ["match_id", "date_utc", "competition", "home", "away", "actual_result", "actual_advancing_team", "result_semantics", "result_source", "last_updated"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["home"] = out["home"].map(normalize_team_name)
    out["away"] = out["away"].map(normalize_team_name)
    out["actual_advancing_team"] = out["actual_advancing_team"].map(normalize_team_name)
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    out = out.loc[(out["home"] != "") & (out["away"] != "")].copy()
    return out.reset_index(drop=True)


def _fixture_status(row: pd.Series) -> str:
    statuses = {str(row.get("home_resolution_status", "")), str(row.get("away_resolution_status", ""))}
    if "Requires manual mapping" in statuses:
        return "Requires manual mapping"
    if "Pending prior result" in statuses:
        return "Pending prior result"
    if "Resolved automatically" in statuses:
        return "Resolved automatically"
    return "Source fixture"


def _resolution_summary(df: pd.DataFrame) -> dict[str, Any]:
    status = df.get("fixture_resolution_status", pd.Series(dtype=str)).astype(str)
    source_home = df.get("source_home", df.get("home", pd.Series(dtype=str))).astype(str)
    source_away = df.get("source_away", df.get("away", pd.Series(dtype=str))).astype(str)
    source_unresolved = source_home.map(is_unresolved_team_slot) | source_away.map(is_unresolved_team_slot)
    return {
        "fixtures_resolved_automatically": int((status == "Resolved automatically").sum()),
        "fixtures_pending_prior_result": int((status == "Pending prior result").sum()),
        "fixtures_requiring_manual_mapping": int((status == "Requires manual mapping").sum()),
        "fixtures_source_unresolved": int(source_unresolved.sum()),
    }


def _with_resolution_attrs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.attrs = df.attrs.copy()
    out.attrs.update(_resolution_summary(out))
    return out


def _team_key(value: Any) -> str:
    return " ".join(normalize_team_name(str(value or "")).lower().replace("-", " ").split())


def _join_unique(existing: str, addition: str) -> str:
    parts = [part.strip() for part in f"{existing}; {addition}".split(";") if part.strip()]
    return "; ".join(dict.fromkeys(parts))
