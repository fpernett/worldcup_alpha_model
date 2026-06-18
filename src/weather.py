from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from src.cache import (
    cache_last_updated,
    read_dataframe_cache,
    safe_cache_name,
    set_source_attrs,
    utc_now_iso,
    write_dataframe_cache,
    write_json_cache,
)
from src.config import DATA_DIR, SOURCE_API, SOURCE_CACHE, SOURCE_FALLBACK, SOURCE_LOCAL, get_config
from src.utils import coerce_bool, coerce_float, read_csv_with_columns


VENUE_COLUMNS = [
    "venue",
    "city",
    "country",
    "latitude",
    "longitude",
    "altitude_m",
    "temp_c",
    "humidity_pct",
    "wind_kmh",
    "precipitation_mm",
    "roof_expected_closed",
    "neutral_site",
]


ENVIRONMENT_COLUMNS = [
    "venue",
    "city",
    "country",
    "latitude",
    "longitude",
    "altitude_m",
    "temp_c",
    "humidity_pct",
    "wind_kmh",
    "precipitation_mm",
    "roof_expected_closed",
    "indoor_adjusted_temp_c",
    "effective_temp_c",
    "effective_humidity_pct",
    "effective_wind_kmh",
    "effective_precipitation_mm",
    "weather_impact_multiplier",
    "environment_source",
    "source_label",
    "last_updated",
    "neutral_site",
]


def load_venues() -> pd.DataFrame:
    venues = _normalise_venues(read_csv_with_columns(DATA_DIR / "venues.csv", VENUE_COLUMNS))
    return set_source_attrs(venues, SOURCE_LOCAL, "data/venues.csv", _csv_last_modified("venues.csv"))


def _normalise_venues(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in VENUE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out.dropna(subset=["venue"]).copy()
    for idx, row in out.iterrows():
        out.at[idx, "latitude"] = coerce_float(row.get("latitude"), 0.0)
        out.at[idx, "longitude"] = coerce_float(row.get("longitude"), 0.0)
        out.at[idx, "altitude_m"] = coerce_float(row.get("altitude_m"), 0.0)
        out.at[idx, "temp_c"] = coerce_float(row.get("temp_c"), 22.0)
        out.at[idx, "humidity_pct"] = coerce_float(row.get("humidity_pct"), 55.0)
        out.at[idx, "wind_kmh"] = coerce_float(row.get("wind_kmh"), 0.0)
        out.at[idx, "precipitation_mm"] = coerce_float(row.get("precipitation_mm"), 0.0)
        out.at[idx, "roof_expected_closed"] = int(coerce_bool(row.get("roof_expected_closed", False)))
        out.at[idx, "neutral_site"] = int(coerce_bool(row.get("neutral_site", True)))

    return out[VENUE_COLUMNS].reset_index(drop=True)


def get_venue_environment(
    venue: str,
    date_utc: Any = None,
    time_utc: Any = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    venues = load_venues()
    match = venues.loc[venues["venue"].astype(str).str.lower() == str(venue).lower()]
    if match.empty:
        base = _neutral_venue(venue)
        return environment_from_venue_row(base, date_utc, time_utc, SOURCE_FALLBACK, "neutral_venue_fallback")

    base = match.iloc[0].copy()
    cfg = get_config()
    cache_name = _weather_cache_filename(str(base["venue"]), date_utc, time_utc)

    if cfg.weather_configured and not force_refresh:
        cached = read_dataframe_cache(cache_name, max_age_hours=6)
        if cached is not None and not cached.empty:
            row = cached.iloc[0].to_dict()
            row["source_label"] = SOURCE_CACHE
            row["environment_source"] = "Open-Meteo cache"
            row["last_updated"] = cache_last_updated(cache_name)
            return row

    if cfg.weather_configured:
        api_weather = _fetch_open_meteo_weather(base, date_utc, time_utc)
        if api_weather is not None:
            env = environment_from_venue_row(
                base,
                date_utc,
                time_utc,
                SOURCE_API,
                "Open-Meteo API",
                weather_values=api_weather,
            )
            write_dataframe_cache(pd.DataFrame([env]), cache_name, "Open-Meteo API")
            return env

    return environment_from_venue_row(base, date_utc, time_utc, SOURCE_LOCAL, "data/venues.csv")


def environment_from_venue_row(
    venue_row: pd.Series | dict[str, Any],
    date_utc: Any = None,
    time_utc: Any = None,
    source_label: str | None = None,
    environment_source: str | None = None,
    weather_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(venue_row)
    weather_values = weather_values or {}

    roof_closed = coerce_bool(row.get("roof_expected_closed", False))
    outdoor_temp = coerce_float(weather_values.get("temp_c", row.get("temp_c")), 22.0)
    outdoor_humidity = coerce_float(weather_values.get("humidity_pct", row.get("humidity_pct")), 55.0)
    outdoor_wind = coerce_float(weather_values.get("wind_kmh", row.get("wind_kmh")), 0.0)
    outdoor_precip = coerce_float(weather_values.get("precipitation_mm", row.get("precipitation_mm")), 0.0)

    weather_multiplier = 0.15 if roof_closed else 1.0
    effective_temp = _effective_indoor_temp(outdoor_temp, roof_closed)
    effective_humidity = 55.0 + (outdoor_humidity - 55.0) * weather_multiplier
    effective_wind = outdoor_wind * weather_multiplier
    effective_precip = outdoor_precip * weather_multiplier

    return {
        "venue": row.get("venue", ""),
        "city": row.get("city", ""),
        "country": row.get("country", ""),
        "latitude": coerce_float(row.get("latitude"), 0.0),
        "longitude": coerce_float(row.get("longitude"), 0.0),
        "date_utc": date_utc,
        "time_utc": time_utc,
        "altitude_m": coerce_float(row.get("altitude_m"), 0.0),
        "temp_c": outdoor_temp,
        "humidity_pct": outdoor_humidity,
        "wind_kmh": outdoor_wind,
        "precipitation_mm": outdoor_precip,
        "roof_expected_closed": int(roof_closed),
        "indoor_adjusted_temp_c": effective_temp,
        "effective_temp_c": round(effective_temp, 1),
        "effective_humidity_pct": round(effective_humidity, 1),
        "effective_wind_kmh": round(effective_wind, 1),
        "effective_precipitation_mm": round(effective_precip, 2),
        "weather_impact_multiplier": weather_multiplier,
        "environment_source": environment_source or "data/venues.csv",
        "source_label": source_label or SOURCE_LOCAL,
        "last_updated": utc_now_iso(),
        "neutral_site": int(coerce_bool(row.get("neutral_site", True))),
    }


def fetch_weather_for_fixtures(fixtures: pd.DataFrame) -> dict[str, Any]:
    if fixtures.empty:
        return {"status": "skipped", "rows": 0, "message": "No fixtures selected for weather refresh."}

    cfg = get_config()
    if not cfg.weather_configured:
        return {
            "status": "skipped",
            "rows": len(fixtures),
            "message": "WEATHER_API_URL is empty; using data/venues.csv.",
            "source": SOURCE_LOCAL,
        }

    rows = []
    errors = []
    for _, fixture in fixtures.iterrows():
        try:
            env = get_venue_environment(
                fixture.get("venue", ""),
                fixture.get("date_utc"),
                fixture.get("time_utc"),
                force_refresh=True,
            )
            rows.append(env)
            if cfg.weather_configured and env.get("source_label") != SOURCE_API:
                errors.append(f"{fixture.get('venue', 'unknown venue')}: weather API unavailable; used {env.get('source_label')}.")
        except Exception as exc:
            errors.append(f"{fixture.get('venue', 'unknown venue')}: {exc}")

    if rows:
        write_dataframe_cache(pd.DataFrame(rows), "venue_environment_latest.csv", "weather refresh")

    status = "success" if rows and not errors else "partial" if rows else "failed"
    return {
        "status": status,
        "rows": len(rows),
        "message": f"Weather refreshed for {len(rows)} fixture venue(s).",
        "errors": errors,
    }


def summarize_environment_adjustment(env: dict[str, Any]) -> str:
    roof = bool(env.get("roof_expected_closed", 0))
    if roof:
        return (
            "Roof/indoor conditions are expected, so outdoor heat, humidity, wind, and precipitation "
            "are strongly dampened before they affect the playing-condition adjustment."
        )
    return (
        "Open-air conditions are used directly. Heat, humidity, wind, precipitation, and altitude "
        "apply conservatively to team climate mismatch and total-goals quality."
    )


def _fetch_open_meteo_weather(venue_row: pd.Series, date_utc: Any, time_utc: Any) -> dict[str, Any] | None:
    cfg = get_config()
    if not cfg.weather_api_url:
        return None

    latitude = coerce_float(venue_row.get("latitude"), 0.0)
    longitude = coerce_float(venue_row.get("longitude"), 0.0)
    if latitude == 0.0 and longitude == 0.0:
        return None

    kickoff = _kickoff_datetime(date_utc, time_utc)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "UTC",
        "start_date": kickoff.date().isoformat(),
        "end_date": kickoff.date().isoformat(),
    }

    try:
        response = requests.get(cfg.weather_api_url.rstrip("/"), params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    write_json_cache(
        payload,
        f"open_meteo_raw_{safe_cache_name(str(venue_row.get('venue', 'venue')))}_{kickoff.date().isoformat()}.json",
        "Open-Meteo API",
    )
    return _closest_hourly_weather(payload, kickoff)


def _closest_hourly_weather(payload: dict[str, Any], kickoff: datetime) -> dict[str, Any] | None:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        return None

    times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
    if len(times) == 0:
        return None

    kickoff_ts = pd.Timestamp(kickoff)
    deltas = abs(times - kickoff_ts)
    if len(deltas) == 0:
        return None
    idx = int(deltas.argmin())

    def hourly_value(name: str, default: float) -> float:
        values = hourly.get(name, [])
        if idx >= len(values):
            return default
        return coerce_float(values[idx], default)

    return {
        "temp_c": hourly_value("temperature_2m", 22.0),
        "humidity_pct": hourly_value("relative_humidity_2m", 55.0),
        "precipitation_mm": hourly_value("precipitation", 0.0),
        "wind_kmh": hourly_value("wind_speed_10m", 0.0),
    }


def _kickoff_datetime(date_utc: Any, time_utc: Any) -> datetime:
    date_part = pd.to_datetime(date_utc, errors="coerce")
    if pd.isna(date_part):
        date_part = pd.Timestamp.now(tz="UTC")
    time_str = str(time_utc or "00:00")[:5]
    combined = pd.to_datetime(f"{date_part.date().isoformat()} {time_str}", utc=True, errors="coerce")
    if pd.isna(combined):
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return combined.to_pydatetime()


def _effective_indoor_temp(outdoor_temp_c: float, roof_closed: bool) -> float:
    if not roof_closed:
        return outdoor_temp_c
    return round(0.85 * 22.0 + 0.15 * outdoor_temp_c, 1)


def _weather_cache_filename(venue: str, date_utc: Any, time_utc: Any) -> str:
    kickoff = _kickoff_datetime(date_utc, time_utc)
    return f"weather_{safe_cache_name(venue)}_{kickoff.strftime('%Y%m%d_%H%M')}.csv"


def _neutral_venue(venue: str) -> pd.Series:
    return pd.Series(
        {
            "venue": venue,
            "city": "",
            "country": "",
            "latitude": 0.0,
            "longitude": 0.0,
            "altitude_m": 0.0,
            "temp_c": 22.0,
            "humidity_pct": 55.0,
            "wind_kmh": 0.0,
            "precipitation_mm": 0.0,
            "roof_expected_closed": 0,
            "neutral_site": 1,
        }
    )


def _csv_last_modified(filename: str) -> str:
    path = DATA_DIR / filename
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
