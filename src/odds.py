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
from src.utils import coerce_float, read_csv_with_columns


ODDS_COLUMNS = ["match_id", "market", "selection", "odds", "source", "last_updated"]


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
