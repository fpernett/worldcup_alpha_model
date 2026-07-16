from __future__ import annotations

import re
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
from src.completed_results import extract_regular_time_score
from src.config import DATA_DIR, SOURCE_API, SOURCE_CACHE, SOURCE_LOCAL, get_config
from src.historical_data import build_historical_matches_from_results, load_historical_matches, merge_and_save_historical_matches
from src.odds import ODDS_COLUMNS, load_market_odds, update_market_odds_for_fixtures
from src.ratings import TEAM_RATING_COLUMNS, get_team_ratings
from src.team_behavior import rebuild_team_behavior_csv
from src.team_names import is_unresolved_team_slot, normalize_team_name, team_name_key
from src.utils import coerce_bool, csv_status, parse_date, read_csv_with_columns
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

INTERNATIONAL_RESULTS_FIXTURE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

POLYMARKET_EVENT_FIXTURE_COLUMNS = [
    "event_id",
    "event_slug",
    "event_title",
    "event_category",
    "event_start_date",
    "event_end_date",
    "event_active",
    "event_closed",
    "start_date",
    "end_date",
    "raw_event_json",
    "source",
    "last_updated",
]

_POLYMARKET_MATCH_TITLE_PATTERN = re.compile(r"^\s*(?P<home>.+?)\s+v(?:s)?\.?\s+(?P<away>.+?)\s*$", re.IGNORECASE)
_POLYMARKET_BASE_MATCH_SLUG_PATTERN = re.compile(r"^fifwc-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}$", re.IGNORECASE)


def get_upcoming_fixtures(start_date: date, end_date: date, force_refresh: bool = False) -> pd.DataFrame:
    start = parse_date(start_date)
    end = parse_date(end_date)
    cfg = get_config()
    warning = ""

    if cfg.football_configured and not force_refresh:
        cached = read_dataframe_cache("fixtures_latest.csv", max_age_hours=6)
        if cached is not None and not cached.empty:
            cached_window = _filter_fixture_window(_normalise_fixtures(cached), start, end)
            if not cached_window.empty:
                return _with_fixture_fallback(
                    cached_window,
                    start,
                    end,
                    SOURCE_CACHE,
                    "data/cache/fixtures_latest.csv",
                    cache_last_updated("fixtures_latest.csv"),
                )
            warning = "Fresh fixtures cache did not contain the requested date window."

    if cfg.football_configured:
        api_fixtures, error = _fetch_football_data_fixtures(start, end)
        if api_fixtures is not None and not api_fixtures.empty:
            api_window = _filter_fixture_window(api_fixtures, start, end)
            if not api_window.empty:
                write_dataframe_cache(api_fixtures, "fixtures_latest.csv", "football-data.org API")
                return _with_fixture_fallback(
                    api_window,
                    start,
                    end,
                    SOURCE_API,
                    "football-data.org /matches",
                    utc_now_iso(),
                )
            error = "Football API returned no fixtures in the requested date window."
        warning = error or "Football API returned no usable fixtures."

        cached = read_dataframe_cache("fixtures_latest.csv", max_age_hours=None)
        if cached is not None and not cached.empty:
            cached_window = _filter_fixture_window(_normalise_fixtures(cached), start, end)
            if not cached_window.empty:
                return _with_fixture_fallback(
                    cached_window,
                    start,
                    end,
                    SOURCE_CACHE,
                    "data/cache/fixtures_latest.csv",
                    cache_last_updated("fixtures_latest.csv"),
                    warning=f"API failed; using cached fixtures. {warning}",
                )
            warning = _combine_warnings(warning, "Cached fixtures did not contain the requested date window.") or ""

    local = _load_local_fixture_pool(start, end)
    return _with_fixture_source(
        _filter_fixture_window(local, start, end),
        SOURCE_LOCAL,
        local.attrs.get("source_detail", "data/fixtures.csv"),
        _csv_last_modified("fixtures.csv"),
        warning=_combine_warnings(
            f"API unavailable; using local CSV. {warning}" if warning else "",
            local.attrs.get("warning", ""),
        ),
    )


def filter_future_fixtures(
    fixtures: pd.DataFrame,
    now_utc: str | pd.Timestamp | None = None,
    horizon_hours: float | None = None,
    include_past: bool = False,
    include_unresolved: bool = False,
) -> pd.DataFrame:
    """Add UTC kickoff timestamps and optionally keep only future fixtures.

    `date_utc` and `time_utc` are treated as UTC. Rows with missing or
    unparseable kickoff timestamps are excluded because match modelling needs a
    concrete kickoff time. Pending bracket slots are excluded by default
    because the model requires actual team rows.
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
    out["has_unresolved_team_slot"] = out.apply(fixture_has_unresolved_team_slot, axis=1)

    mask = out["kickoff_utc"].notna()
    if not include_past:
        mask &= out["kickoff_utc"] >= now
    if horizon_hours is not None:
        mask &= out["kickoff_utc"] <= now + pd.Timedelta(hours=float(horizon_hours))
    unresolved_hidden_count = int((mask & out["has_unresolved_team_slot"].astype(bool)).sum())
    if not include_unresolved:
        mask &= ~out["has_unresolved_team_slot"].astype(bool)

    filtered = out.loc[mask].sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)
    if unresolved_hidden_count and not include_unresolved:
        attrs["unresolved_fixture_count"] = unresolved_hidden_count
        pending_count = int(attrs.get("fixtures_pending_prior_result", 0) or 0)
        manual_count = int(attrs.get("fixtures_requiring_manual_mapping", 0) or 0)
        if manual_count:
            warning = (
                f"{pending_count + manual_count} bracket fixture(s) are excluded from the model selector; "
                f"{pending_count} pending prior match result(s), {manual_count} require manual mapping."
            )
        else:
            warning = f"{pending_count or unresolved_hidden_count} future fixtures pending prior match results."
        attrs["warning"] = _combine_warnings(
            attrs.get("warning", ""),
            warning,
        )
    filtered.attrs = attrs
    return filtered


def _with_fixture_fallback(
    primary: pd.DataFrame,
    start_date: date,
    end_date: date,
    source_label: str,
    source_detail: str,
    last_updated: str,
    warning: str | None = None,
) -> pd.DataFrame:
    """Supplement a partial API/cache schedule with missing local fixtures.

    A fixture provider can return a valid, non-empty response that still omits
    late knockout matches. Treating that response as complete hides curated
    local rows, including bracket slots already resolved from cached sports
    events. Primary rows keep precedence; local rows are added only when their
    date, kickoff time, and team pair are not already present.
    """
    primary_window = _filter_fixture_window(_normalise_fixtures(primary), start_date, end_date)
    local = _load_local_fixture_pool(start_date, end_date)
    local_window = _filter_fixture_window(_normalise_fixtures(local), start_date, end_date)
    if local_window.empty:
        return _with_fixture_source(primary_window, source_label, source_detail, last_updated, warning=warning)

    primary_keys = {_fixture_identity_key(row) for _, row in primary_window.iterrows()}
    missing_local = local_window.loc[
        local_window.apply(
            lambda row: _local_fixture_should_supplement(row, primary_window, primary_keys),
            axis=1,
        )
    ].copy()
    if missing_local.empty:
        return _with_fixture_source(primary_window, source_label, source_detail, last_updated, warning=warning)

    replacement_ids = set(
        missing_local.loc[~missing_local.apply(fixture_has_unresolved_team_slot, axis=1), "match_id"].astype(str)
    )
    primary_for_merge = primary_window.loc[
        ~(
            primary_window["match_id"].astype(str).isin(replacement_ids)
            & primary_window.apply(fixture_has_unresolved_team_slot, axis=1)
        )
    ].copy()
    combined = pd.concat([primary_for_merge, missing_local], ignore_index=True, sort=False)
    combined = _normalise_fixtures(combined).drop_duplicates(subset=["match_id"], keep="first")
    combined = combined.sort_values(["date_utc", "time_utc", "match_id"]).reset_index(drop=True)
    local_detail = str(local.attrs.get("source_detail", "data/fixtures.csv") or "data/fixtures.csv")
    warning = _combine_warnings(
        warning or "",
        "Primary fixture source was incomplete; local schedule fallback supplied "
        f"{len(missing_local)} missing fixture(s).",
        local.attrs.get("warning", ""),
    )
    return _with_fixture_source(
        combined,
        f"{source_label} + {SOURCE_LOCAL}",
        f"{source_detail} + {local_detail}",
        last_updated,
        warning=warning,
    )


def _fixture_identity_key(row: pd.Series) -> tuple[str, str, str, str]:
    kickoff_date = pd.to_datetime(row.get("date_utc"), errors="coerce")
    date_key = "" if pd.isna(kickoff_date) else kickoff_date.date().isoformat()
    time_key = str(row.get("time_utc", "") or "")[:5]
    teams = sorted([team_name_key(row.get("home", "")), team_name_key(row.get("away", ""))])
    return date_key, time_key, teams[0], teams[1]


def _local_fixture_should_supplement(
    local_row: pd.Series,
    primary: pd.DataFrame,
    primary_keys: set[tuple[str, str, str, str]],
) -> bool:
    if _fixture_identity_key(local_row) in primary_keys:
        return False
    match_id = str(local_row.get("match_id", "") or "")
    same_id = primary.loc[primary["match_id"].astype(str) == match_id]
    if same_id.empty:
        return True
    primary_is_unresolved = bool(same_id.apply(fixture_has_unresolved_team_slot, axis=1).all())
    local_is_resolved = not fixture_has_unresolved_team_slot(local_row)
    return primary_is_unresolved and local_is_resolved


def fixture_has_unresolved_team_slot(row: pd.Series | dict[str, Any]) -> bool:
    return is_unresolved_team_slot(row.get("home", "")) or is_unresolved_team_slot(row.get("away", ""))


def _load_local_fixture_pool(start_date: date | None = None, end_date: date | None = None) -> pd.DataFrame:
    """Load curated fixtures plus conservative local fallback fixture sources.

    `data/fixtures.csv` remains the authoritative local schedule. The imported
    international results snapshot can contain future tournament rows with blank
    scores; those rows are useful as a fallback when the curated fixture file is
    incomplete. Because that source has no kickoff time or stadium, fallback rows
    use `23:59` UTC and city-as-venue so they remain selectable but visibly carry
    a source warning.

    Cached Polymarket FIFA event metadata is also used as a read-only discovery
    fallback when a curated knockout slot is still a placeholder but the event
    title already names the teams. Exact local kickoff-slot matches inherit the
    curated match id and venue context, so downstream market mapping continues to
    use the local World Cup schedule identity.
    """
    curated = _normalise_fixtures(read_csv_with_columns(DATA_DIR / "fixtures.csv", FIXTURE_COLUMNS))
    curated["_fixture_source"] = "data/fixtures.csv"
    supplemental = []

    international = _future_fixtures_from_international_results()
    if not international.empty:
        international["_fixture_source"] = "data/international_results.csv"
        supplemental.append(international)

    polymarket_events = _future_fixtures_from_polymarket_events(curated, _polymarket_event_cache_rows())
    if not polymarket_events.empty:
        polymarket_events["_fixture_source"] = "data/polymarket_events_cache.csv"
        supplemental.append(polymarket_events)

    candidate_frames = [curated, *supplemental]
    if _needs_live_polymarket_fixture_refresh(candidate_frames, start_date, end_date):
        live_polymarket_events = _future_fixtures_from_polymarket_events(curated, _live_polymarket_event_rows())
        if not live_polymarket_events.empty:
            live_polymarket_events["_fixture_source"] = "Gamma sports events"
            supplemental.append(live_polymarket_events)

    if not supplemental:
        curated = _normalise_fixtures(curated)
        curated.attrs["source_detail"] = "data/fixtures.csv"
        return curated

    combined = _combine_fixture_source_frames([curated, *supplemental])

    kept_sources = set(combined.get("_fixture_source", pd.Series(dtype=str)).astype(str))
    combined = combined.drop(columns=["_fixture_source", "_order", "_priority", "_dedupe_key"], errors="ignore")
    combined = _normalise_fixtures(combined)

    source_detail = ["data/fixtures.csv"]
    warnings = []
    if "data/international_results.csv" in kept_sources:
        source_detail.append("data/international_results.csv")
        warnings.append(
            "Some fixtures came from data/international_results.csv because data/fixtures.csv is incomplete. "
            "Those fallback rows have date-only kickoff placeholders at 23:59 UTC and city-as-venue."
        )
    if "data/polymarket_events_cache.csv" in kept_sources:
        source_detail.extend(["data/polymarket_events_cache.csv", "data/polymarket_markets_cache.csv"])
        warnings.append(
            "Some fixtures came from cached Polymarket FIFA World Cup event metadata because local bracket rows "
            "were still placeholders. These rows are read-only fixture discovery inputs; verify against the "
            "official schedule when available."
        )
    if "Gamma sports events" in kept_sources:
        source_detail.append("Polymarket Gamma sports events")
        warnings.append(
            "Some fixtures came from live Polymarket FIFA World Cup event metadata because local bracket rows "
            "were still placeholders. These rows are read-only fixture discovery inputs; verify against the "
            "official schedule when available."
        )
    combined.attrs["source_detail"] = " + ".join(dict.fromkeys(source_detail))
    combined.attrs["warning"] = _combine_warnings(*warnings)
    return combined


def _future_fixtures_from_international_results() -> pd.DataFrame:
    source = read_csv_with_columns(DATA_DIR / "international_results.csv", INTERNATIONAL_RESULTS_FIXTURE_COLUMNS)
    if source.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    out = source.copy()
    out["date_utc"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date_utc", "home_team", "away_team"]).copy()
    out = out.loc[
        out["home_score"].isna()
        & out["away_score"].isna()
        & out["tournament"].astype(str).str.contains("World Cup", case=False, na=False)
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)

    out["match_id"] = out.apply(_supplemental_fixture_id, axis=1)
    out["time_utc"] = "23:59"
    out["competition"] = out["tournament"].fillna("FIFA World Cup")
    out["group"] = ""
    out["home"] = out["home_team"]
    out["away"] = out["away_team"]
    out["venue"] = out["city"].fillna("")
    out["city"] = out["city"].fillna("")
    out["country"] = out["country"].fillna("")
    return _normalise_fixtures(out[FIXTURE_COLUMNS])


def _future_fixtures_from_polymarket_events(curated: pd.DataFrame, event_rows: pd.DataFrame | None) -> pd.DataFrame:
    events = event_rows.copy() if event_rows is not None else pd.DataFrame(columns=POLYMARKET_EVENT_FIXTURE_COLUMNS)
    if events.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    for col in POLYMARKET_EVENT_FIXTURE_COLUMNS:
        if col not in events.columns:
            events[col] = pd.NA
    events["_slug"] = events["event_slug"].fillna("").astype(str).str.strip()
    events = events.loc[events["_slug"] != ""].drop_duplicates(subset=["_slug"], keep="last").drop(columns=["_slug"])

    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        if not _is_polymarket_world_cup_event(event):
            continue
        if coerce_bool(event.get("event_closed")):
            continue

        kickoff = _polymarket_event_kickoff(event)
        if pd.isna(kickoff):
            continue
        teams = _polymarket_event_teams(event.get("event_title", ""))
        if teams is None:
            continue
        home, away = teams

        context = _polymarket_event_local_context(kickoff, home, away, curated)
        slug = str(event.get("event_slug", "") or "").strip()
        fallback_id = f"poly_{_slug_part(slug or event.get('event_title', ''))}"
        rows.append(
            {
                "match_id": context.get("match_id") or fallback_id,
                "date_utc": kickoff.date(),
                "time_utc": kickoff.strftime("%H:%M"),
                "competition": context.get("competition") or "FIFA World Cup",
                "group": context.get("group") or "",
                "home": home,
                "away": away,
                "venue": context.get("venue") or "",
                "city": context.get("city") or "",
                "country": context.get("country") or "",
            }
        )

    return _normalise_fixtures(pd.DataFrame(rows, columns=FIXTURE_COLUMNS))


def _needs_live_polymarket_fixture_refresh(
    frames: list[pd.DataFrame],
    start_date: date | None,
    end_date: date | None,
) -> bool:
    if start_date is None or end_date is None:
        return False
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return False
    combined = _normalise_fixtures(_combine_fixture_source_frames(valid_frames))
    window = _filter_fixture_window(combined, start_date, end_date)
    if window.empty:
        return False
    return bool(window.apply(fixture_has_unresolved_team_slot, axis=1).any())


def _combine_fixture_source_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    combined = pd.concat(valid_frames, ignore_index=True, sort=False)
    combined["_order"] = range(len(combined))
    combined["_priority"] = combined.apply(_fixture_source_priority, axis=1)
    combined = (
        combined.sort_values(["_priority", "_order"])
        .drop_duplicates(subset=["match_id"], keep="first")
        .reset_index(drop=True)
    )
    combined["_dedupe_key"] = combined.apply(_fixture_dedupe_key, axis=1)
    return combined.drop_duplicates(subset=["_dedupe_key"], keep="first").reset_index(drop=True)


def _polymarket_event_cache_rows() -> pd.DataFrame:
    frames = [
        read_csv_with_columns(DATA_DIR / "polymarket_events_cache.csv", POLYMARKET_EVENT_FIXTURE_COLUMNS),
        read_csv_with_columns(DATA_DIR / "polymarket_markets_cache.csv", POLYMARKET_EVENT_FIXTURE_COLUMNS),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=POLYMARKET_EVENT_FIXTURE_COLUMNS)

    events = pd.concat(frames, ignore_index=True, sort=False)
    for col in POLYMARKET_EVENT_FIXTURE_COLUMNS:
        if col not in events.columns:
            events[col] = pd.NA
    events["_slug"] = events["event_slug"].fillna("").astype(str).str.strip()
    events = events.loc[events["_slug"] != ""].copy()
    if events.empty:
        return pd.DataFrame(columns=POLYMARKET_EVENT_FIXTURE_COLUMNS)
    events["_updated"] = pd.to_datetime(events["last_updated"], errors="coerce", utc=True)
    events = (
        events.sort_values(["_slug", "_updated"], na_position="first")
        .drop_duplicates(subset=["_slug"], keep="last")
        .drop(columns=["_slug", "_updated"])
        .reset_index(drop=True)
    )
    return events[POLYMARKET_EVENT_FIXTURE_COLUMNS].copy()


def _live_polymarket_event_rows() -> pd.DataFrame:
    try:
        from src.polymarket_sports_discovery import (  # noqa: WPS433
            fetch_all_gamma_events,
            fetch_gamma_sports,
            flatten_gamma_events_to_markets,
        )

        series_ids = [
            str(sport.get("series", "") or "").strip()
            for sport in fetch_gamma_sports()
            if str(sport.get("sport", "") or "").strip().lower() == "fifwc"
            and str(sport.get("series", "") or "").strip()
        ]
        events_payload: list[dict[str, Any]] = []
        for series_id in dict.fromkeys(series_ids):
            events_payload.extend(
                fetch_all_gamma_events(
                    active=True,
                    closed=False,
                    max_pages=2,
                    limit=500,
                    series_id=series_id,
                )
            )
        events = flatten_gamma_events_to_markets(events_payload)
    except Exception:
        return pd.DataFrame(columns=POLYMARKET_EVENT_FIXTURE_COLUMNS)
    if events is None or events.empty:
        return pd.DataFrame(columns=POLYMARKET_EVENT_FIXTURE_COLUMNS)
    for col in POLYMARKET_EVENT_FIXTURE_COLUMNS:
        if col not in events.columns:
            events[col] = pd.NA
    return events[POLYMARKET_EVENT_FIXTURE_COLUMNS].copy()


def _is_polymarket_world_cup_event(row: pd.Series) -> bool:
    slug = str(row.get("event_slug", "") or "").strip().lower()
    title = str(row.get("event_title", "") or "").strip().lower()
    if slug.startswith("fifwc-"):
        return bool(_POLYMARKET_BASE_MATCH_SLUG_PATTERN.match(slug))
    text = " ".join(
        [
            slug,
            str(row.get("event_category", "") or "").strip().lower(),
            title,
        ]
    )
    return slug.startswith("fifwc-") or "soccer-fifwc" in text or "fifa world cup" in text


def _polymarket_event_kickoff(row: pd.Series) -> pd.Timestamp:
    for col in ["event_end_date", "end_date"]:
        kickoff = pd.to_datetime(row.get(col), errors="coerce", utc=True)
        if not pd.isna(kickoff):
            return kickoff
    return pd.NaT


def _polymarket_event_teams(title: Any) -> tuple[str, str] | None:
    match = _POLYMARKET_MATCH_TITLE_PATTERN.match(str(title or "").strip())
    if not match:
        return None
    home = normalize_team_name(match.group("home").strip())
    away = normalize_team_name(match.group("away").strip())
    if not home or not away:
        return None
    if is_unresolved_team_slot(home) or is_unresolved_team_slot(away):
        return None
    return home, away


def _polymarket_event_local_context(
    kickoff: pd.Timestamp,
    home: str,
    away: str,
    curated: pd.DataFrame,
) -> dict[str, str]:
    if curated.empty:
        return {}

    local = curated.copy()
    date_key = kickoff.date().isoformat()
    time_key = kickoff.strftime("%H:%M")
    local_date = pd.to_datetime(local["date_utc"], errors="coerce").dt.date.astype(str)
    same_time = local.loc[(local_date == date_key) & (local["time_utc"].astype(str).str.slice(0, 5) == time_key)].copy()

    unresolved = same_time.loc[same_time.apply(fixture_has_unresolved_team_slot, axis=1)]
    if len(unresolved) == 1:
        return _fixture_context(unresolved.iloc[0])

    home_key = team_name_key(home)
    away_key = team_name_key(away)
    local_home = local["home"].map(team_name_key)
    local_away = local["away"].map(team_name_key)
    same_match = local.loc[(local_date == date_key) & (local_home == home_key) & (local_away == away_key)]
    if not same_match.empty:
        return _fixture_context(same_match.iloc[0])
    return {}


def _fixture_context(row: pd.Series) -> dict[str, str]:
    return {
        "match_id": str(row.get("match_id", "") or ""),
        "competition": str(row.get("competition", "") or ""),
        "group": str(row.get("group", "") or ""),
        "venue": str(row.get("venue", "") or ""),
        "city": str(row.get("city", "") or ""),
        "country": str(row.get("country", "") or ""),
    }


def _fixture_source_priority(row: pd.Series) -> int:
    source = str(row.get("_fixture_source", "") or "")
    if source == "data/fixtures.csv":
        return 20 if fixture_has_unresolved_team_slot(row) else 0
    if source == "data/polymarket_events_cache.csv":
        return 10
    if source == "Gamma sports events":
        return 10
    if source == "data/international_results.csv":
        return 30
    return 40


def _supplemental_fixture_id(row: pd.Series) -> str:
    parts = [
        "intl",
        str(row.get("date_utc", "")),
        _slug_part(row.get("home_team", "")),
        _slug_part(row.get("away_team", "")),
    ]
    return "_".join(part for part in parts if part)


def _fixture_dedupe_key(row: pd.Series) -> str:
    date_key = str(pd.to_datetime(row.get("date_utc"), errors="coerce").date())
    return "|".join([date_key, team_name_key(row.get("home", "")), team_name_key(row.get("away", ""))])


def _slug_part(value: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip()).strip("_")


def _combine_warnings(*warnings: str) -> str | None:
    parts = [str(warning).strip() for warning in warnings if str(warning or "").strip()]
    return " ".join(parts) if parts else None


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
        home_goals, away_goals, _score_source, _score_semantics = extract_regular_time_score(match)
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
