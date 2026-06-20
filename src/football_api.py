from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.cache import read_json_cache, safe_cache_name, write_json_cache
from src.config import CACHE_DIR, SOURCE_API, SOURCE_CACHE, SOURCE_FALLBACK, get_config
from src.utils import parse_date


DEFAULT_FOOTBALL_API_URL = "https://api.football-data.org/v4"
MAX_FOOTBALL_DATA_RANGE_DAYS = 30


def fetch_finished_matches(
    start_date: str | date,
    end_date: str | date,
    force_refresh: bool = False,
) -> tuple[list[dict[str, Any]], str, str]:
    """Fetch finished football-data.org matches, falling back to raw cache.

    Returns `(matches, source_label, warning)`. Missing API configuration,
    failed requests, empty payloads, and malformed cache files are reported as
    warnings instead of raising.
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    if end < start:
        start, end = end, start

    cfg = get_config()
    api_url = cfg.football_api_url or DEFAULT_FOOTBALL_API_URL
    warnings: list[str] = []

    if cfg.football_api_key:
        matches: list[dict[str, Any]] = []
        for chunk_start, chunk_end in _date_chunks(start, end):
            cache_name = _cache_filename(chunk_start, chunk_end)
            if not force_refresh:
                cached_payload = read_json_cache(cache_name, max_age_hours=24)
                cached_matches = _matches_from_payload(cached_payload)
                if cached_matches:
                    matches.extend(cached_matches)
                    continue

            payload, error = _request_finished_matches(api_url, cfg.football_api_key, chunk_start, chunk_end)
            if payload is None:
                warnings.append(f"{chunk_start.isoformat()} to {chunk_end.isoformat()}: {error}")
                cached_payload = read_json_cache(cache_name, max_age_hours=None)
                matches.extend(_matches_from_payload(cached_payload))
                continue
            write_json_cache(payload, cache_name, "football-data.org /matches?status=FINISHED")
            matches.extend(_matches_from_payload(payload))

        deduped = _dedupe_raw_matches(matches)
        if deduped:
            return deduped, SOURCE_API, "; ".join(warnings)
        warnings.append("football-data.org returned no usable finished matches.")
    else:
        warnings.append("FOOTBALL_API_KEY is not configured; using cached/local fallback.")

    cached = load_cached_finished_matches(start, end)
    if cached:
        return cached, SOURCE_CACHE, "; ".join(warnings)
    return [], SOURCE_FALLBACK, "; ".join(warnings)


def load_cached_finished_matches(start_date: str | date, end_date: str | date) -> list[dict[str, Any]]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    if end < start:
        start, end = end, start

    matches: list[dict[str, Any]] = []
    for path in _candidate_cache_files():
        try:
            payload = read_json_cache(path.name, max_age_hours=None)
        except Exception:
            continue
        for match in _matches_from_payload(payload):
            kickoff = pd.to_datetime(match.get("utcDate"), utc=True, errors="coerce")
            if pd.isna(kickoff):
                continue
            match_date = kickoff.date()
            if start <= match_date <= end:
                matches.append(match)
    return _dedupe_raw_matches(matches)


def _request_finished_matches(
    api_url: str,
    api_key: str,
    start: date,
    end: date,
) -> tuple[dict[str, Any] | None, str | None]:
    endpoint = api_url.rstrip("/")
    if not endpoint.endswith("/matches"):
        endpoint = f"{endpoint}/matches"

    try:
        response = requests.get(
            endpoint,
            headers={"X-Auth-Token": api_key},
            params={"dateFrom": start.isoformat(), "dateTo": end.isoformat(), "status": "FINISHED"},
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "Football API response was not a JSON object."
    return payload, None


def _date_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=MAX_FOOTBALL_DATA_RANGE_DAYS - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _cache_filename(start: date, end: date) -> str:
    return safe_cache_name(f"football_data_finished_{start.isoformat()}_{end.isoformat()}.json")


def _candidate_cache_files() -> list[Path]:
    if not CACHE_DIR.exists():
        return []
    patterns = [
        "football_data_finished_*.json",
        "football_data_results_raw_*.json",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(CACHE_DIR.glob(pattern))
    return sorted(set(files))


def _matches_from_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    matches = payload.get("matches") or payload.get("data") or []
    if not isinstance(matches, list):
        return []
    return [match for match in matches if isinstance(match, dict)]


def _dedupe_raw_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for match in matches:
        match_id = str(match.get("id") or "").strip()
        if not match_id:
            match_id = "|".join(
                [
                    str(match.get("utcDate") or ""),
                    str((match.get("homeTeam") or {}).get("name") or ""),
                    str((match.get("awayTeam") or {}).get("name") or ""),
                ]
            )
        deduped[match_id] = match
    return list(deduped.values())
