from __future__ import annotations

from typing import Any

import numpy as np
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
from src.config import DATA_DIR, SOURCE_API, SOURCE_CACHE, SOURCE_LOCAL, get_config
from src.team_names import is_unresolved_team_slot
from src.utils import coerce_float, read_csv_with_columns


ODDS_COLUMNS = ["match_id", "market", "selection", "odds", "source", "last_updated"]
GENERATED_ODDS_SOURCE = "local_model_benchmark"
GENERATED_ODDS_SOURCE_ALIASES = {GENERATED_ODDS_SOURCE, "local_fallback_seed"}
GENERATED_ODDS_WARNING = (
    "Missing fixture odds were filled with generated local model benchmark odds. "
    "These are not live market prices."
)


def load_market_odds(force_cache: bool = False) -> pd.DataFrame:
    cfg = get_config()
    if cfg.odds_configured or force_cache:
        cached = read_dataframe_cache("market_odds_latest.csv", max_age_hours=6)
        if cached is not None:
            return set_source_attrs(
                _normalise_odds(cached),
                SOURCE_CACHE,
                "data/cache/market_odds_latest.csv",
                cache_last_updated("market_odds_latest.csv"),
            )

    local = _normalise_odds(read_csv_with_columns(DATA_DIR / "market_odds.csv", ODDS_COLUMNS))
    return set_source_attrs(local, SOURCE_LOCAL, "data/market_odds.csv", _csv_last_modified("market_odds.csv"))


def get_market_odds(match_id: str | int, force_refresh: bool = False) -> pd.DataFrame:
    match_id = str(match_id)
    cfg = get_config()
    warning = ""

    if cfg.odds_configured and not force_refresh:
        cached = read_dataframe_cache(f"market_odds_{safe_cache_name(match_id)}.csv", max_age_hours=6)
        if cached is not None and not cached.empty:
            return set_source_attrs(
                _normalise_odds(cached),
                SOURCE_CACHE,
                f"data/cache/market_odds_{safe_cache_name(match_id)}.csv",
                cache_last_updated(f"market_odds_{safe_cache_name(match_id)}.csv"),
            )

    if cfg.odds_configured:
        api_odds, error = _fetch_odds_from_api(match_id)
        if api_odds is not None and not api_odds.empty:
            filename = f"market_odds_{safe_cache_name(match_id)}.csv"
            write_dataframe_cache(api_odds, filename, "odds API")
            return set_source_attrs(api_odds, SOURCE_API, "generic odds API", utc_now_iso())
        warning = error or "Odds API returned no usable rows."

    odds = load_market_odds()
    filtered = odds.loc[odds["match_id"].astype(str) == match_id].reset_index(drop=True)
    return set_source_attrs(
        filtered,
        odds.attrs.get("source_label", SOURCE_LOCAL),
        odds.attrs.get("source_detail", "data/market_odds.csv"),
        odds.attrs.get("last_updated", _csv_last_modified("market_odds.csv")),
        warning=f"Odds API unavailable; using fallback. {warning}" if warning else None,
    )


def update_market_odds_for_fixtures(fixtures: pd.DataFrame) -> dict[str, Any]:
    cfg = get_config()
    if fixtures.empty:
        return {"status": "skipped", "rows": 0, "message": "No fixtures selected for odds refresh."}
    if not cfg.odds_configured:
        local = load_market_odds()
        return {
            "status": "skipped",
            "rows": len(local),
            "message": "ODDS_API_URL is empty; using data/market_odds.csv.",
            "source": SOURCE_LOCAL,
        }

    rows = []
    errors = []
    for match_id in fixtures["match_id"].astype(str).unique():
        odds, error = _fetch_odds_from_api(match_id)
        if odds is not None and not odds.empty:
            rows.append(odds)
            write_dataframe_cache(odds, f"market_odds_{safe_cache_name(match_id)}.csv", "odds API")
        elif error:
            errors.append(f"{match_id}: {error}")

    if not rows:
        return {"status": "failed", "rows": 0, "message": "No odds API rows were cached.", "errors": errors}

    all_odds = pd.concat(rows, ignore_index=True)
    write_dataframe_cache(all_odds, "market_odds_latest.csv", "odds API")
    return {
        "status": "success" if not errors else "partial",
        "rows": len(all_odds),
        "message": f"Cached {len(all_odds)} odds row(s).",
        "errors": errors,
        "source": SOURCE_API,
    }


def ensure_market_odds_for_fixtures(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    venues: pd.DataFrame | None = None,
    market_odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return odds with generated benchmark rows for missing fixture markets.

    Real/API/manual odds remain first priority. Generated rows are only a local
    benchmark fallback so the dashboard can keep alpha EV populated when a live
    or manual market source has not been entered yet.
    """
    attrs = getattr(market_odds, "attrs", {}).copy() if market_odds is not None else {}
    odds = _normalise_odds(market_odds.copy() if market_odds is not None else pd.DataFrame(columns=ODDS_COLUMNS))
    generated = build_generated_benchmark_market_odds(fixtures, teams, venues, odds)
    if generated.empty:
        odds.attrs = attrs
        return odds

    valid_mask = _valid_odds_mask(odds)
    combined = pd.concat([odds.loc[valid_mask], generated, odds.loc[~valid_mask]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["match_id", "market", "selection"], keep="first")
    combined = _normalise_odds(combined)
    combined.attrs = attrs
    combined.attrs["source_label"] = _with_generated_source_label(attrs.get("source_label", SOURCE_LOCAL))
    combined.attrs["source_detail"] = _with_generated_source_detail(
        attrs.get("source_detail", "data/market_odds.csv")
    )
    combined.attrs["last_updated"] = utc_now_iso()
    combined.attrs["warning"] = _combine_warnings(attrs.get("warning", ""), GENERATED_ODDS_WARNING)
    combined.attrs["generated_rows"] = int(len(generated))
    return combined


def build_generated_benchmark_market_odds(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    venues: pd.DataFrame | None = None,
    existing_odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    fixtures_df = fixtures.copy() if fixtures is not None else pd.DataFrame()
    if fixtures_df.empty:
        return pd.DataFrame(columns=ODDS_COLUMNS)

    odds = _normalise_odds(existing_odds.copy() if existing_odds is not None else pd.DataFrame(columns=ODDS_COLUMNS))
    existing_keys = _valid_market_keys(odds)
    generated_keys: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    timestamp = utc_now_iso()

    for _, fixture in fixtures_df.iterrows():
        match_id = str(fixture.get("match_id", "") or "").strip()
        home = str(fixture.get("home", "") or "").strip()
        away = str(fixture.get("away", "") or "").strip()
        if not match_id or not home or not away:
            continue
        if is_unresolved_team_slot(home) or is_unresolved_team_slot(away):
            continue

        try:
            probability_map = _benchmark_probability_map(fixture, teams)
        except Exception:
            continue

        for (market, selection), probability in probability_map.items():
            key = (match_id, str(market), str(selection))
            if key in existing_keys or key in generated_keys:
                continue
            odds_value = _fair_decimal_odds(probability)
            if pd.isna(odds_value):
                continue
            generated_keys.add(key)
            rows.append(
                {
                    "match_id": match_id,
                    "market": market,
                    "selection": selection,
                    "odds": odds_value,
                    "source": GENERATED_ODDS_SOURCE,
                    "last_updated": timestamp,
                }
            )

    return _normalise_odds(pd.DataFrame(rows, columns=ODDS_COLUMNS))


def decimal_odds_or_nan(value: Any) -> float:
    odds = coerce_float(value, np.nan)
    if np.isnan(odds) or odds <= 1.0:
        return np.nan
    return odds


def _normalise_odds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ODDS_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[ODDS_COLUMNS].copy()
    if out.empty:
        return out
    out["match_id"] = out["match_id"].astype(str)
    out["market"] = out["market"].astype(str).str.strip()
    out["selection"] = out["selection"].astype(str).str.strip()
    out["odds"] = pd.to_numeric(out["odds"], errors="coerce")
    out["source"] = out["source"].fillna("market_odds_csv")
    out["last_updated"] = out["last_updated"].fillna(utc_now_iso())
    return out.dropna(subset=["match_id", "market", "selection"])


def _fetch_odds_from_api(match_id: str) -> tuple[pd.DataFrame | None, str | None]:
    cfg = get_config()
    if not cfg.odds_api_url:
        return None, "ODDS_API_URL is not configured."

    params = {"match_id": match_id}
    if cfg.odds_api_key:
        params["key"] = cfg.odds_api_key

    try:
        response = requests.get(cfg.odds_api_url, params=params, timeout=20)
        response.raise_for_status()
        payload: Any = response.json()
    except Exception as exc:
        return None, str(exc)

    write_json_cache(payload, f"odds_raw_{safe_cache_name(match_id)}.json", "odds API")

    if isinstance(payload, dict):
        payload = payload.get("odds") or payload.get("markets") or payload.get("data") or []
    if not isinstance(payload, list):
        return None, "Odds API payload did not contain a list."
    df = pd.json_normalize(payload)
    if df.empty:
        return None, "Odds API payload was empty."
    if "decimal_odds" in df.columns and "odds" not in df.columns:
        df["odds"] = df["decimal_odds"]
    if "price" in df.columns and "odds" not in df.columns:
        df["odds"] = df["price"]
    df["match_id"] = str(match_id)
    df["source"] = "odds_api"
    df["last_updated"] = utc_now_iso()
    return _normalise_odds(df), None


def _csv_last_modified(filename: str) -> str:
    path = DATA_DIR / filename
    if not path.exists():
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()


def _benchmark_probability_map(fixture: pd.Series, teams: pd.DataFrame) -> dict[tuple[str, str], float]:
    from src.model import ModelConfig, expected_goals, market_probability_map, outcome_probs, score_matrix
    from src.ratings import neutral_team_rating, rating_row_for_team

    home_name = str(fixture.get("home", "") or "")
    away_name = str(fixture.get("away", "") or "")
    home = rating_row_for_team(teams, home_name)
    away = rating_row_for_team(teams, away_name)
    if home.empty:
        home = neutral_team_rating(home_name)
    if away.empty:
        away = neutral_team_rating(away_name)

    cfg = ModelConfig()
    hxg, axg, _ = expected_goals(home, away, _neutral_benchmark_environment(), cfg)
    probs = outcome_probs(score_matrix(hxg, axg, cfg.max_goals))
    return market_probability_map(home_name, away_name, probs)


def _neutral_benchmark_environment() -> dict[str, Any]:
    return {
        "environment_source": "local_model_benchmark_neutral_environment",
        "source_label": SOURCE_LOCAL,
        "roof_expected_closed": 0,
        "temp_c": 22.0,
        "effective_temp_c": 22.0,
        "humidity_pct": 55.0,
        "effective_humidity_pct": 55.0,
        "wind_kmh": 0.0,
        "effective_wind_kmh": 0.0,
        "precipitation_mm": 0.0,
        "effective_precipitation_mm": 0.0,
        "altitude_m": 0.0,
        "neutral_site": 1,
    }


def _fair_decimal_odds(probability: Any) -> float | pd.NA:
    probability_value = coerce_float(probability, np.nan)
    if np.isnan(probability_value) or probability_value <= 0:
        return pd.NA
    odds = 1.0 / probability_value
    if not np.isfinite(odds) or odds <= 1.0:
        return pd.NA
    return round(float(odds), 4)


def _valid_odds_mask(odds: pd.DataFrame) -> pd.Series:
    if odds is None or odds.empty or "odds" not in odds.columns:
        return pd.Series(dtype=bool)
    values = pd.to_numeric(odds["odds"], errors="coerce")
    return values.gt(1.0) & np.isfinite(values)


def _valid_market_keys(odds: pd.DataFrame) -> set[tuple[str, str, str]]:
    if odds is None or odds.empty:
        return set()
    valid = odds.loc[_valid_odds_mask(odds)].copy()
    if valid.empty:
        return set()
    return {
        (str(row.get("match_id", "")), str(row.get("market", "")), str(row.get("selection", "")))
        for _, row in valid.iterrows()
    }


def _with_generated_source_label(source_label: Any) -> str:
    base = str(source_label or SOURCE_LOCAL).strip() or SOURCE_LOCAL
    if "generated benchmark" in base.lower():
        return base
    return f"{base} + generated benchmark"


def _with_generated_source_detail(source_detail: Any) -> str:
    base = str(source_detail or "data/market_odds.csv").strip() or "data/market_odds.csv"
    generated = "generated local model benchmark odds"
    if generated in base.lower():
        return base
    return f"{base} + {generated}"


def _combine_warnings(*warnings: Any) -> str:
    parts = [str(warning).strip() for warning in warnings if str(warning or "").strip()]
    return "; ".join(dict.fromkeys(parts))
