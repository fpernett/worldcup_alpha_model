from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.cache import cache_last_updated, read_dataframe_cache, set_source_attrs, utc_now_iso, write_dataframe_cache
from src.config import DATA_DIR, SOURCE_API, SOURCE_CACHE, SOURCE_FALLBACK, SOURCE_LOCAL
from src.football_api import fetch_finished_matches
from src.team_names import normalize_team_name
from src.utils import coerce_float, parse_date, read_csv_with_columns


COMPLETED_RESULTS_PATH = DATA_DIR / "completed_results.csv"
COMPLETED_RESULTS_CACHE = "completed_results_latest.csv"

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


def load_completed_results(
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    teams: list[str] | tuple[str, ...] | None = None,
    force_refresh: bool = False,
    persist: bool = True,
    path: str | Path = COMPLETED_RESULTS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load completed regular-time results from API/cache/local CSV.

    This is the app's normalized completed-result source. It fetches finished
    matches from the configured football provider when possible, persists them
    locally, and falls back to cache/CSV without crashing when no API is
    configured.
    """
    start, end = _date_window(start_date, end_date)
    diagnostics = _base_diagnostics(start, end, path)
    fetched = pd.DataFrame(columns=COMPLETED_RESULT_COLUMNS)
    warnings: list[str] = []

    raw_matches, source_label, warning = fetch_finished_matches(start, end, force_refresh=force_refresh)
    diagnostics["provider_checked"] = "football-data.org /matches?status=FINISHED"
    diagnostics["raw_rows_loaded"] = len(raw_matches)
    diagnostics["raw_source"] = source_label
    if warning:
        warnings.append(warning)
    if raw_matches:
        fetched = normalise_provider_completed_results(raw_matches, provider="football-data.org")
        diagnostics["normalized_rows_fetched"] = len(fetched)
        if persist and not fetched.empty:
            saved = merge_and_save_completed_results(fetched, path=path)
            write_dataframe_cache(saved, COMPLETED_RESULTS_CACHE, "completed results refresh")
            out = _filter_completed_results(saved, start, end, teams)
            diagnostics.update(
                {
                    "status": "success",
                    "source_label": SOURCE_API if source_label == SOURCE_API else source_label,
                    "source_detail": "football-data.org /matches?status=FINISHED",
                    "completed_rows_loaded": len(out),
                    "completed_rows_available": len(saved),
                    "last_updated": utc_now_iso(),
                }
            )
            return _with_attrs(out, diagnostics, warnings), diagnostics

    local = _load_local_completed_results(path)
    if not local.empty:
        out = _filter_completed_results(local, start, end, teams)
        diagnostics.update(
            {
                "status": "cache_or_local",
                "source_label": local.attrs.get("source_label", SOURCE_LOCAL),
                "source_detail": local.attrs.get("source_detail", _display_path(Path(path))),
                "completed_rows_loaded": len(out),
                "completed_rows_available": len(local),
                "last_updated": local.attrs.get("last_updated", ""),
            }
        )
        return _with_attrs(out, diagnostics, warnings), diagnostics

    cached = read_dataframe_cache(COMPLETED_RESULTS_CACHE, max_age_hours=None)
    if cached is not None and not cached.empty:
        cached = _normalise_completed_results(cached)
        source_frame = merge_and_save_completed_results(cached, path=path) if persist and not cached.empty else cached
        out = _filter_completed_results(source_frame, start, end, teams)
        diagnostics.update(
            {
                "status": "cache_or_local",
                "source_label": SOURCE_CACHE,
                "source_detail": f"data/cache/{COMPLETED_RESULTS_CACHE}",
                "completed_rows_loaded": len(out),
                "completed_rows_available": len(source_frame),
                "last_updated": cache_last_updated(COMPLETED_RESULTS_CACHE),
            }
        )
        return _with_attrs(out, diagnostics, warnings), diagnostics

    diagnostics.update(
        {
            "status": "completed_source_empty",
            "source_label": SOURCE_FALLBACK,
            "source_detail": "no completed-result API/cache/local rows",
            "completed_rows_loaded": 0,
            "completed_rows_available": 0,
            "last_updated": "",
        }
    )
    warnings.append("No completed-result source rows were available for the requested date window.")
    return _with_attrs(pd.DataFrame(columns=COMPLETED_RESULT_COLUMNS), diagnostics, warnings), diagnostics


def refresh_completed_results_for_fixture(
    fixture_row: pd.Series | dict[str, Any],
    window_days: int = 1,
    force_refresh: bool = True,
    path: str | Path = COMPLETED_RESULTS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fixture = pd.Series(fixture_row)
    fixture_date = pd.to_datetime(fixture.get("date_utc"), errors="coerce")
    if pd.isna(fixture_date):
        return (
            pd.DataFrame(columns=COMPLETED_RESULT_COLUMNS),
            {
                "status": "invalid_fixture_date",
                "provider_checked": "football-data.org /matches?status=FINISHED",
                "date_window_checked": "",
                "cache_path_checked": _display_path(Path(path)),
                "completed_rows_loaded": 0,
            },
        )
    start = (fixture_date.date() - timedelta(days=int(window_days))).isoformat()
    end = (fixture_date.date() + timedelta(days=int(window_days))).isoformat()
    teams = [str(fixture.get("home", "") or ""), str(fixture.get("away", "") or "")]
    teams = [team for team in teams if team.strip()]
    return load_completed_results(start, end, teams=teams or None, force_refresh=force_refresh, persist=True, path=path)


def normalise_provider_completed_results(
    matches: list[dict[str, Any]] | pd.DataFrame | None,
    provider: str,
) -> pd.DataFrame:
    records = matches.to_dict("records") if isinstance(matches, pd.DataFrame) else matches or []
    rows: list[dict[str, Any]] = []
    now = utc_now_iso()
    for match in records:
        if not isinstance(match, dict):
            continue
        row = _provider_match_to_completed_row(match, provider=provider, last_updated=now)
        if row:
            rows.append(row)
    return _normalise_completed_results(pd.DataFrame(rows))


def extract_regular_time_score(match: dict[str, Any]) -> tuple[Any, Any, str, str]:
    """Return the provider's 90-minute score fields, never extra time or penalties."""
    return _regular_time_score(match)


def merge_and_save_completed_results(new_df: pd.DataFrame | None, path: str | Path = COMPLETED_RESULTS_PATH) -> pd.DataFrame:
    path = Path(path)
    existing = _load_local_completed_results(path)
    incoming = _normalise_completed_results(new_df)
    if existing.empty and incoming.empty:
        out = pd.DataFrame(columns=COMPLETED_RESULT_COLUMNS)
    elif existing.empty:
        out = incoming
    elif incoming.empty:
        out = existing
    else:
        out = pd.concat([existing, incoming], ignore_index=True)
    if not out.empty:
        out["_key"] = out.apply(_dedupe_key, axis=1)
        out["_updated"] = pd.to_datetime(out["last_updated"], errors="coerce", utc=True).fillna(pd.Timestamp("1970-01-01", tz="UTC"))
        out = out.sort_values(["_key", "_updated"]).drop_duplicates(subset=["_key"], keep="last")
        out = out.drop(columns=["_key", "_updated"]).sort_values(["date_utc", "provider_match_id"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out[COMPLETED_RESULT_COLUMNS].to_csv(path, index=False)
    return set_source_attrs(out[COMPLETED_RESULT_COLUMNS].copy(), SOURCE_LOCAL, _display_path(path), utc_now_iso())


def completed_results_to_backtest_matches(results_df: pd.DataFrame | None) -> pd.DataFrame:
    df = _normalise_completed_results(results_df)
    columns = [
        "match_id",
        "date_utc",
        "home",
        "away",
        "home_goals",
        "away_goals",
        "competition",
        "group",
        "venue",
        "city",
        "country",
        "neutral_site",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(
        {
            "match_id": df["provider_match_id"].astype(str),
            "date_utc": df["date_utc"],
            "home": df["home"],
            "away": df["away"],
            "home_goals": df["home_score_90"],
            "away_goals": df["away_score_90"],
            "competition": df["competition"],
            "group": df["group"],
            "venue": "",
            "city": "",
            "country": "",
            "neutral_site": 1,
        }
    )
    out.attrs = df.attrs.copy()
    return out[columns]


def completed_results_status(path: str | Path = COMPLETED_RESULTS_PATH) -> dict[str, Any]:
    local = _load_local_completed_results(path)
    return {
        "path": _display_path(Path(path)),
        "rows": len(local),
        "source": local.attrs.get("source_label", SOURCE_LOCAL),
        "last_updated": local.attrs.get("last_updated", ""),
        "warning": local.attrs.get("warning", ""),
    }


def _provider_match_to_completed_row(match: dict[str, Any], provider: str, last_updated: str) -> dict[str, Any] | None:
    kickoff = pd.to_datetime(match.get("utcDate") or match.get("provider_kickoff_utc") or match.get("date_utc"), utc=True, errors="coerce")
    if pd.isna(kickoff):
        return None
    home = normalize_team_name(_team_name(match.get("homeTeam")) or match.get("home", ""))
    away = normalize_team_name(_team_name(match.get("awayTeam")) or match.get("away", ""))
    if not home or not away:
        return None
    home_score, away_score, score_source, score_semantics = _regular_time_score(match)
    if home_score is None or away_score is None:
        return None
    competition = match.get("competition") or {}
    competition_name = competition.get("name") if isinstance(competition, dict) else str(competition or "")
    return {
        "provider": provider,
        "provider_match_id": str(match.get("id") or match.get("match_id") or ""),
        "provider_kickoff_utc": kickoff.replace(microsecond=0).isoformat(),
        "date_utc": kickoff.date().isoformat(),
        "competition": competition_name or str(match.get("competition_name") or ""),
        "group": str(match.get("group") or match.get("stage") or ""),
        "home": home,
        "away": away,
        "home_score_90": int(coerce_float(home_score)),
        "away_score_90": int(coerce_float(away_score)),
        "score_source": score_source,
        "score_semantics": score_semantics,
        "is_completed": 1,
        "source": provider,
        "last_updated": last_updated,
    }


def _regular_time_score(match: dict[str, Any]) -> tuple[Any, Any, str, str]:
    score = match.get("score") if isinstance(match.get("score"), dict) else {}
    candidates = [
        ("score.regularTime", score.get("regularTime") if isinstance(score, dict) else None),
        ("score.regular", score.get("regular") if isinstance(score, dict) else None),
        ("score.fullTime", score.get("fullTime") if isinstance(score, dict) else None),
        ("top_level_90", {"home": match.get("home_score_90"), "away": match.get("away_score_90")}),
        ("top_level_home_goals", {"home": match.get("home_goals"), "away": match.get("away_goals")}),
    ]
    for source, payload in candidates:
        if not isinstance(payload, dict):
            continue
        home = payload.get("home")
        away = payload.get("away")
        if home is not None and away is not None:
            return home, away, source, "90-minute regular time"
    return None, None, "", "90-minute regular time"


def _normalise_completed_results(df: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame(columns=COMPLETED_RESULT_COLUMNS)
    for col in COMPLETED_RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[COMPLETED_RESULT_COLUMNS].copy()
    if out.empty:
        return out
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce").dt.date.astype(str)
    out = out.loc[out["date_utc"].astype(str) != "NaT"].copy()
    if out.empty:
        return pd.DataFrame(columns=COMPLETED_RESULT_COLUMNS)
    out["provider_kickoff_utc"] = pd.to_datetime(out["provider_kickoff_utc"], errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["provider_kickoff_utc"] = out["provider_kickoff_utc"].str.replace(r"(\+0000)$", "+00:00", regex=True)
    for col in ["provider", "provider_match_id", "competition", "group", "home", "away", "score_source", "score_semantics", "source", "last_updated"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["home"] = out["home"].map(normalize_team_name)
    out["away"] = out["away"].map(normalize_team_name)
    out["home_score_90"] = pd.to_numeric(out["home_score_90"], errors="coerce")
    out["away_score_90"] = pd.to_numeric(out["away_score_90"], errors="coerce")
    out["is_completed"] = pd.to_numeric(out["is_completed"], errors="coerce").fillna(0).astype(int)
    out = out.dropna(subset=["date_utc", "home", "away", "home_score_90", "away_score_90"]).copy()
    out = out.loc[(out["home"] != "") & (out["away"] != "") & (out["is_completed"] == 1)]
    return out[COMPLETED_RESULT_COLUMNS].reset_index(drop=True)


def _load_local_completed_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    local = read_csv_with_columns(path, COMPLETED_RESULT_COLUMNS)
    out = _normalise_completed_results(local)
    return set_source_attrs(out, SOURCE_LOCAL, _display_path(path), _file_modified(path))


def _filter_completed_results(
    df: pd.DataFrame,
    start: date,
    end: date,
    teams: list[str] | tuple[str, ...] | None,
) -> pd.DataFrame:
    out = _normalise_completed_results(df)
    if out.empty:
        return out
    dates = pd.to_datetime(out["date_utc"], errors="coerce").dt.date
    out = out.loc[(dates >= start) & (dates <= end)].copy()
    if teams:
        wanted = {normalize_team_name(team).lower() for team in teams if str(team).strip()}
        out = out.loc[out["home"].str.lower().isin(wanted) | out["away"].str.lower().isin(wanted)].copy()
    out.attrs = df.attrs.copy()
    return out.reset_index(drop=True)


def _date_window(start_date: str | date | None, end_date: str | date | None) -> tuple[date, date]:
    today = pd.Timestamp.now(tz="UTC").date()
    start = parse_date(start_date) if start_date is not None else today - timedelta(days=2)
    end = parse_date(end_date) if end_date is not None else today
    if end < start:
        start, end = end, start
    return start, end


def _base_diagnostics(start: date, end: date, path: str | Path) -> dict[str, Any]:
    return {
        "status": "",
        "provider_checked": "",
        "date_window_checked": f"{start.isoformat()} to {end.isoformat()}",
        "cache_path_checked": f"data/cache/{COMPLETED_RESULTS_CACHE}",
        "local_path_checked": _display_path(Path(path)),
        "completed_rows_loaded": 0,
        "completed_rows_available": 0,
        "normalized_rows_fetched": 0,
        "raw_rows_loaded": 0,
        "raw_source": "",
    }


def _with_attrs(df: pd.DataFrame, diagnostics: dict[str, Any], warnings: list[str]) -> pd.DataFrame:
    warning = "; ".join(dict.fromkeys(str(item).strip() for item in warnings if str(item).strip()))
    return set_source_attrs(
        df,
        str(diagnostics.get("source_label", "")),
        str(diagnostics.get("source_detail", "")),
        str(diagnostics.get("last_updated", "")) or None,
        warning=warning or None,
    )


def _dedupe_key(row: pd.Series) -> str:
    provider_id = str(row.get("provider_match_id", "") or "").strip()
    if provider_id:
        return f"{row.get('provider')}|{provider_id}"
    return "|".join(
        [
            str(row.get("date_utc", "")),
            str(row.get("home", "")).lower(),
            str(row.get("away", "")).lower(),
            str(int(coerce_float(row.get("home_score_90"), 0))),
            str(int(coerce_float(row.get("away_score_90"), 0))),
        ]
    )


def _team_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("shortName") or payload.get("name") or payload.get("tla") or "").strip()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR.parent))
    except ValueError:
        return str(path)


def _file_modified(path: Path) -> str:
    if not path.exists():
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").replace(microsecond=0).isoformat()
