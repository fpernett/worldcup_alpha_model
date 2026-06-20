from __future__ import annotations

from datetime import date, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from src.cache import (
    cache_last_updated,
    read_dataframe_cache,
    set_source_attrs,
    utc_now_iso,
    write_dataframe_cache,
    write_json_cache,
)
from src.config import DATA_DIR, SOURCE_API, SOURCE_CACHE, SOURCE_LOCAL, get_config
from src.historical_data import build_historical_matches_from_results, load_historical_matches, merge_and_save_historical_matches
from src.odds import ODDS_COLUMNS, load_market_odds, update_market_odds_for_fixtures
from src.ratings import TEAM_RATING_COLUMNS, get_team_ratings
from src.team_behavior import rebuild_team_behavior_csv
from src.utils import csv_status, parse_date, read_csv_with_columns
from src.weather import VENUE_COLUMNS, fetch_weather_for_fixtures


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

RESULT_COLUMNS = [
    "date_utc",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "home_elo",
    "away_elo",
]


def get_upcoming_fixtures(start_date: date, end_date: date, force_refresh: bool = False) -> pd.DataFrame:
    start = parse_date(start_date)
    end = parse_date(end_date)
    cfg = get_config()
    warning = ""

    if cfg.football_configured and not force_refresh:
        cached = read_dataframe_cache("fixtures_latest.csv", max_age_hours=6)
        if cached is not None and not cached.empty:
            return _with_fixture_source(
                _filter_fixture_window(_normalise_fixtures(cached), start, end),
                SOURCE_CACHE,
                "data/cache/fixtures_latest.csv",
                cache_last_updated("fixtures_latest.csv"),
            )

    if cfg.football_configured:
        api_fixtures, error = _fetch_football_data_fixtures(start, end)
        if api_fixtures is not None and not api_fixtures.empty:
            write_dataframe_cache(api_fixtures, "fixtures_latest.csv", "football-data.org API")
            return _with_fixture_source(
                _filter_fixture_window(api_fixtures, start, end),
                SOURCE_API,
                "football-data.org /matches",
                utc_now_iso(),
            )
        warning = error or "Football API returned no usable fixtures."

        cached = read_dataframe_cache("fixtures_latest.csv", max_age_hours=None)
        if cached is not None and not cached.empty:
            return _with_fixture_source(
                _filter_fixture_window(_normalise_fixtures(cached), start, end),
                SOURCE_CACHE,
                "data/cache/fixtures_latest.csv",
                cache_last_updated("fixtures_latest.csv"),
                warning=f"API failed; using cached fixtures. {warning}",
            )

    local = _normalise_fixtures(read_csv_with_columns(DATA_DIR / "fixtures.csv", FIXTURE_COLUMNS))
    return _with_fixture_source(
        _filter_fixture_window(local, start, end),
        SOURCE_LOCAL,
        "data/fixtures.csv",
        _csv_last_modified("fixtures.csv"),
        warning=f"API unavailable; using local CSV. {warning}" if warning else None,
    )


def filter_future_fixtures(
    fixtures: pd.DataFrame,
    now_utc: str | pd.Timestamp | None = None,
    horizon_hours: float | None = None,
    include_past: bool = False,
) -> pd.DataFrame:
    """Add UTC kickoff timestamps and optionally keep only future fixtures.

    `date_utc` and `time_utc` are treated as UTC. Rows with missing or
    unparseable kickoff timestamps are hidden because match modelling needs a
    concrete kickoff time.
    """
    if fixtures is None:
        return pd.DataFrame(columns=FIXTURE_COLUMNS + ["kickoff_utc"])

    if fixtures.empty:
        empty = fixtures.copy()
        if "kickoff_utc" not in empty.columns:
            empty["kickoff_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
        empty.attrs = fixtures.attrs.copy()
        return empty

    out = fixtures.copy()
    attrs = fixtures.attrs.copy()
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    date_part = out.get("date_utc", pd.Series(index=out.index, dtype="object")).astype(str)
    time_part = out.get("time_utc", pd.Series("00:00", index=out.index)).fillna("00:00").astype(str).str.slice(0, 5)
    out["kickoff_utc"] = pd.to_datetime(date_part + " " + time_part, utc=True, errors="coerce")

    mask = out["kickoff_utc"].notna()
    if not include_past:
        mask &= out["kickoff_utc"] >= now
    if horizon_hours is not None:
        mask &= out["kickoff_utc"] <= now + pd.Timedelta(hours=float(horizon_hours))

    filtered = out.loc[mask].sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)
    filtered.attrs = attrs
    return filtered


def fetch_football_results(
    start_date: date | None = None,
    end_date: date | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cfg = get_config()
    start = start_date or (date.today() - timedelta(days=365))
    end = end_date or date.today()
    warning = ""

    if cfg.football_configured and not force_refresh:
        cached = read_dataframe_cache("recent_matches_latest.csv", max_age_hours=24)
        if cached is not None and not cached.empty:
            return set_source_attrs(
                _normalise_results(cached),
                SOURCE_CACHE,
                "data/cache/recent_matches_latest.csv",
                cache_last_updated("recent_matches_latest.csv"),
            )

    if cfg.football_configured:
        results, error = _fetch_football_data_results(start, end)
        if results is not None and not results.empty:
            write_dataframe_cache(results, "recent_matches_latest.csv", "football-data.org API")
            return set_source_attrs(results, SOURCE_API, "football-data.org /matches?status=FINISHED", utc_now_iso())
        warning = error or "Football API returned no usable results."

        cached = read_dataframe_cache("recent_matches_latest.csv", max_age_hours=None)
        if cached is not None and not cached.empty:
            return set_source_attrs(
                _normalise_results(cached),
                SOURCE_CACHE,
                "data/cache/recent_matches_latest.csv",
                cache_last_updated("recent_matches_latest.csv"),
                warning=f"API failed; using cached results. {warning}",
            )

    local = read_csv_with_columns(DATA_DIR / "recent_matches.csv", RESULT_COLUMNS)
    return set_source_attrs(
        _normalise_results(local),
        SOURCE_LOCAL,
        "data/recent_matches.csv",
        _csv_last_modified("recent_matches.csv"),
        warning=f"API unavailable; using local match results if present. {warning}" if warning else None,
    )


def get_source_status(fixtures=None, teams=None, venues=None, odds=None, historical_matches=None, team_behavior=None):
    """
    Return source diagnostics for the dashboard.

    This function accepts the loaded dataframes so app.py can report
    row counts and missing columns. It is deliberately defensive:
    missing inputs should not crash the Streamlit app.
    """
    import pandas as pd
    from pathlib import Path
    from datetime import datetime, timezone

    def safe_len(df):
        try:
            return len(df) if df is not None else 0
        except Exception:
            return 0

    def missing_columns(df, required_cols):
        if df is None:
            return ", ".join(required_cols)
        try:
            return ", ".join([c for c in required_cols if c not in df.columns])
        except Exception:
            return ", ".join(required_cols)

    def file_mtime(path):
        try:
            p = Path(path)
            if not p.exists():
                return ""
            ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            return ts.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return ""

    diagnostics = [
        {
            "input": "fixtures",
            "source": "API/cache/local CSV",
            "rows": safe_len(fixtures),
            "api_configured": bool(getattr(__import__("os"), "environ").get("FOOTBALL_API_KEY")),
            "last_updated": file_mtime("data/fixtures.csv"),
            "warning": "",
            "missing_columns": missing_columns(
                fixtures,
                [
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
                ],
            ),
        },
        {
            "input": "team_ratings",
            "source": "computed/cache/local CSV",
            "rows": safe_len(teams),
            "api_configured": bool(getattr(__import__("os"), "environ").get("RATINGS_API_URL")),
            "last_updated": file_mtime("data/team_ratings.csv"),
            "warning": "",
            "missing_columns": missing_columns(
                teams,
                [
                    "team",
                    "elo",
                    "attack",
                    "defense",
                    "recent_form",
                    "fifa_rank_proxy",
                    "training_temp_c",
                    "training_humidity_pct",
                    "data_quality",
                    "last_updated",
                    "notes",
                ],
            ),
        },
        {
            "input": "venues",
            "source": "weather API/cache/local CSV",
            "rows": safe_len(venues),
            "api_configured": bool(getattr(__import__("os"), "environ").get("WEATHER_API_URL")),
            "last_updated": file_mtime("data/venues.csv"),
            "warning": "",
            "missing_columns": missing_columns(
                venues,
                [
                    "venue",
                    "city",
                    "country",
                    "altitude_m",
                    "temp_c",
                    "humidity_pct",
                    "wind_kmh",
                    "precipitation_mm",
                    "roof_expected_closed",
                ],
            ),
        },
        {
            "input": "market_odds",
            "source": "odds API/cache/local CSV",
            "rows": safe_len(odds),
            "api_configured": bool(getattr(__import__("os"), "environ").get("ODDS_API_URL")),
            "last_updated": file_mtime("data/market_odds.csv"),
            "warning": "",
            "missing_columns": missing_columns(
                odds,
                [
                    "match_id",
                    "market",
                    "selection",
                    "odds",
                ],
            ),
        },
        {
            "input": "historical_matches",
            "source": "cache/local CSV",
            "rows": safe_len(historical_matches),
            "api_configured": bool(getattr(__import__("os"), "environ").get("FOOTBALL_API_KEY")),
            "last_updated": file_mtime("data/historical_matches.csv"),
            "warning": "",
            "missing_columns": missing_columns(
                historical_matches,
                [
                    "match_id",
                    "date_utc",
                    "competition",
                    "competition_type",
                    "team",
                    "opponent",
                    "team_goals",
                    "opponent_goals",
                    "source",
                    "last_updated",
                ],
            ),
        },
        {
            "input": "team_behavior",
            "source": "computed local CSV",
            "rows": safe_len(team_behavior),
            "api_configured": False,
            "last_updated": file_mtime("data/team_behavior.csv"),
            "warning": "",
            "missing_columns": missing_columns(
                team_behavior,
                [
                    "team",
                    "reference_date",
                    "n_matches",
                    "attack_index",
                    "defense_index",
                    "recent_form_index",
                    "overall_data_quality",
                    "last_updated",
                ],
            ),
        },
    ]

    for row in diagnostics:
        if row["missing_columns"]:
            row["warning"] = "Missing required columns"

    return pd.DataFrame(diagnostics)


def update_all_sources(
    start_date: date | None = None,
    end_date: date | None = None,
    update_historical_behavior: bool = True,
) -> dict[str, Any]:
    start = start_date or date.today()
    end = end_date or start
    summary: dict[str, Any] = {"updated_at": utc_now_iso(), "steps": []}

    fixtures = _safe_step(summary, "fixtures", lambda: get_upcoming_fixtures(start, end, force_refresh=True))
    results = _safe_step(
        summary,
        "results",
        lambda: fetch_football_results(start - timedelta(days=365), end, force_refresh=True),
    )
    ratings = _safe_step(summary, "ratings", lambda: get_team_ratings(force_refresh=True))

    historical = None
    behavior = None
    if update_historical_behavior:
        historical = _safe_step(summary, "historical_matches", lambda: _update_historical_matches_from_results(results))
        teams_for_behavior = ratings if isinstance(ratings, pd.DataFrame) else None
        historical_for_behavior = historical if isinstance(historical, pd.DataFrame) else load_historical_matches(use_cache=False)
        behavior = _safe_step(
            summary,
            "team_behavior",
            lambda: rebuild_team_behavior_csv(historical_for_behavior, teams_for_behavior, reference_date=end),
        )

    fixture_frame = fixtures if isinstance(fixtures, pd.DataFrame) else get_upcoming_fixtures(start, end)
    _safe_step(summary, "weather", lambda: fetch_weather_for_fixtures(fixture_frame))
    _safe_step(summary, "odds", lambda: update_market_odds_for_fixtures(fixture_frame))

    summary["fixtures_rows"] = len(fixtures) if isinstance(fixtures, pd.DataFrame) else 0
    summary["team_ratings_rows"] = len(ratings) if isinstance(ratings, pd.DataFrame) else 0
    summary["historical_matches_rows"] = len(historical) if isinstance(historical, pd.DataFrame) else 0
    summary["team_behavior_rows"] = len(behavior) if isinstance(behavior, pd.DataFrame) else 0
    return summary


def _update_historical_matches_from_results(results: Any) -> pd.DataFrame:
    existing = load_historical_matches(use_cache=False)
    if not isinstance(results, pd.DataFrame) or results.empty:
        return existing
    source = results.attrs.get("source_detail") or results.attrs.get("source_label") or "football_results"
    historical_rows = build_historical_matches_from_results(results, source=str(source))
    if historical_rows.empty:
        return existing
    return merge_and_save_historical_matches(historical_rows)


def _fetch_football_data_fixtures(start_date: date, end_date: date) -> tuple[pd.DataFrame | None, str | None]:
    payload, error = _football_data_matches_request(start_date, end_date)
    if payload is None:
        return None, error
    write_json_cache(
        payload,
        f"football_data_fixtures_raw_{start_date.isoformat()}_{end_date.isoformat()}.json",
        "football-data.org API",
    )
    local_context = _normalise_fixtures(read_csv_with_columns(DATA_DIR / "fixtures.csv", FIXTURE_COLUMNS))
    fixtures = _normalise_football_data_fixtures(payload, local_context)
    return fixtures, None


def _fetch_football_data_results(start_date: date, end_date: date) -> tuple[pd.DataFrame | None, str | None]:
    payload, error = _football_data_matches_request(start_date, end_date, status="FINISHED")
    if payload is None:
        return None, error
    write_json_cache(
        payload,
        f"football_data_results_raw_{start_date.isoformat()}_{end_date.isoformat()}.json",
        "football-data.org API",
    )
    results = _normalise_football_data_results(payload)
    return results, None


def _football_data_matches_request(
    start_date: date,
    end_date: date,
    status: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    cfg = get_config()
    if not cfg.football_configured:
        return None, "FOOTBALL_API_URL or FOOTBALL_API_KEY is not configured."

    endpoint = cfg.football_api_url.rstrip("/")
    if not endpoint.endswith("/matches"):
        endpoint = f"{endpoint}/matches"

    params = {"dateFrom": start_date.isoformat(), "dateTo": end_date.isoformat()}
    if status:
        params["status"] = status

    try:
        response = requests.get(
            endpoint,
            headers={"X-Auth-Token": cfg.football_api_key or ""},
            params=params,
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return None, str(exc)

    if not isinstance(payload, dict):
        return None, "Football API response was not a JSON object."
    return payload, None


def _normalise_football_data_fixtures(payload: dict[str, Any], local_context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match in payload.get("matches", []):
        kickoff = pd.to_datetime(match.get("utcDate"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        home = _team_name(match.get("homeTeam", {}))
        away = _team_name(match.get("awayTeam", {}))
        if not home or not away:
            continue

        row = {
            "match_id": str(match.get("id", "")),
            "date_utc": kickoff.date(),
            "time_utc": kickoff.strftime("%H:%M"),
            "competition": (match.get("competition") or {}).get("name", "FIFA World Cup"),
            "group": match.get("group") or match.get("stage") or "",
            "home": home,
            "away": away,
            "venue": "",
            "city": "",
            "country": "",
        }
        row.update(_local_fixture_context(row, local_context))
        rows.append(row)
    return _normalise_fixtures(pd.DataFrame(rows))


def _normalise_football_data_results(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for match in payload.get("matches", []):
        kickoff = pd.to_datetime(match.get("utcDate"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        score = (match.get("score") or {}).get("fullTime") or {}
        home_goals = score.get("home")
        away_goals = score.get("away")
        if home_goals is None or away_goals is None:
            continue
        home = _team_name(match.get("homeTeam", {}))
        away = _team_name(match.get("awayTeam", {}))
        if not home or not away:
            continue
        rows.append(
            {
                "date_utc": kickoff.date(),
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_elo": pd.NA,
                "away_elo": pd.NA,
            }
        )
    return _normalise_results(pd.DataFrame(rows))


def _normalise_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "id": "match_id",
        "fixture_id": "match_id",
        "kickoff_date": "date_utc",
        "kickoff_time": "time_utc",
        "home_team": "home",
        "away_team": "away",
        "stadium": "venue",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    for col in FIXTURE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[FIXTURE_COLUMNS].dropna(subset=["match_id", "date_utc", "home", "away"]).copy()
    out["match_id"] = out["match_id"].astype(str)
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce").dt.date
    out["time_utc"] = out["time_utc"].fillna("00:00").astype(str).str.slice(0, 5)
    out["competition"] = out["competition"].fillna("FIFA World Cup")
    out["group"] = out["group"].fillna("")
    out["venue"] = out["venue"].fillna("")
    out["city"] = out["city"].fillna("")
    out["country"] = out["country"].fillna("")
    out = out.dropna(subset=["date_utc"])
    return out.sort_values(["date_utc", "time_utc", "match_id"]).reset_index(drop=True)


def _normalise_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.dropna(subset=["date_utc", "home", "away", "home_goals", "away_goals"]).copy()
    if out.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce").dt.date
    for col in ["home_goals", "away_goals", "home_elo", "away_elo"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[RESULT_COLUMNS].dropna(subset=["date_utc"]).sort_values("date_utc").reset_index(drop=True)


def _filter_fixture_window(fixtures: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    out = fixtures.copy()
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce").dt.date
    mask = (out["date_utc"] >= start_date) & (out["date_utc"] <= end_date)
    return out.loc[mask].sort_values(["date_utc", "time_utc", "match_id"]).reset_index(drop=True)


def _with_fixture_source(
    df: pd.DataFrame,
    source_label: str,
    source_detail: str,
    last_updated: str,
    warning: str | None = None,
) -> pd.DataFrame:
    return set_source_attrs(df, source_label, source_detail, last_updated, warning)


def _team_name(team_payload: dict[str, Any]) -> str:
    if not isinstance(team_payload, dict):
        return ""
    return str(team_payload.get("shortName") or team_payload.get("name") or team_payload.get("tla") or "").strip()


def _local_fixture_context(row: dict[str, Any], local: pd.DataFrame) -> dict[str, str]:
    if local.empty:
        return {}

    by_id = local.loc[local["match_id"].astype(str) == str(row["match_id"])]
    if by_id.empty:
        by_id = local.loc[
            (local["date_utc"].astype(str) == str(row["date_utc"]))
            & (local["home"].astype(str).str.lower() == str(row["home"]).lower())
            & (local["away"].astype(str).str.lower() == str(row["away"]).lower())
        ]
    if by_id.empty:
        return {}
    context = by_id.iloc[0]
    return {
        "venue": str(context.get("venue", "") or ""),
        "city": str(context.get("city", "") or ""),
        "country": str(context.get("country", "") or ""),
    }


def _safe_step(summary: dict[str, Any], name: str, fn) -> Any:
    try:
        value = fn()
        if isinstance(value, pd.DataFrame):
            step = {
                "name": name,
                "status": "success",
                "rows": len(value),
                "source": value.attrs.get("source_label", ""),
                "detail": value.attrs.get("source_detail", ""),
                "warning": value.attrs.get("warning", ""),
            }
        elif isinstance(value, dict):
            step = {"name": name, **value}
        else:
            step = {"name": name, "status": "success"}
        summary["steps"].append(step)
        return value
    except Exception as exc:
        summary["steps"].append({"name": name, "status": "failed", "message": str(exc)})
        return None


def _csv_last_modified(filename: str) -> str:
    path = DATA_DIR / filename
    if not path.exists():
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz=timezone.utc).isoformat()
