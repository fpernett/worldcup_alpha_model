from __future__ import annotations

import json
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
from src.storage import load_csv
from src.utils import coerce_bool, coerce_float


POLYMARKET_COLUMNS = [
    "market_id",
    "question",
    "slug",
    "event_title",
    "category",
    "start_date",
    "end_date",
    "active",
    "closed",
    "outcomes",
    "yes_price",
    "no_price",
    "liquidity",
    "volume",
    "source",
    "last_updated",
]

DEFAULT_POLYMARKET_GAMMA_API_URL = "https://gamma-api.polymarket.com"
GAMMA_PAGE_LIMIT = 100
GAMMA_MAX_PAGES = 5


def get_polymarket_markets(
    query: str | None = None,
    use_cache: bool = False,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load Polymarket markets from API, cache, or local CSV.

    The function is intentionally read-only. It never places, previews, or signs
    orders. Empty API settings and unexpected response shapes fall back safely.
    """
    warning = ""

    if use_cache and not force_refresh:
        cached = _load_cached_markets(query)
        if cached is not None and not cached.empty:
            return cached
        if cached is not None and cached.empty:
            warning = "Cached Polymarket market file was empty; refreshing Gamma API."

    api_markets, error = _fetch_markets_from_api(query)
    if api_markets is not None and not api_markets.empty:
        write_dataframe_cache(api_markets, "polymarket_markets_normalized.csv", "Polymarket Gamma API")
        return set_source_attrs(api_markets, SOURCE_API, "Polymarket Gamma API", utc_now_iso(), warning=warning or None)
    warning = "; ".join(part for part in [warning, error or "Polymarket Gamma API returned no usable markets."] if part)

    cached = _load_cached_markets(query, warning=f"Polymarket Gamma API failed; using cache. {warning}")
    if cached is not None and not cached.empty:
        return cached

    local = _normalise_markets(load_csv("polymarket_markets.csv", POLYMARKET_COLUMNS), SOURCE_LOCAL)
    local = _filter_query(local, query)
    return set_source_attrs(
        local,
        SOURCE_LOCAL,
        "data/polymarket_markets.csv",
        _csv_last_modified("polymarket_markets.csv"),
        warning=f"Polymarket API unavailable; using local CSV. {warning}" if warning else None,
    )


def update_polymarket_markets(query: str | None = None) -> dict[str, Any]:
    df = get_polymarket_markets(query=query, force_refresh=True)
    return {
        "status": "success" if not df.empty else "skipped",
        "rows": len(df),
        "source": df.attrs.get("source_label", ""),
        "message": f"Loaded {len(df)} Polymarket market row(s) from {df.attrs.get('source_label', 'unknown source')}.",
        "warning": df.attrs.get("warning", ""),
    }


def normalize_price(value: Any) -> float | pd.NA:
    price = coerce_float(value, float("nan"))
    if pd.isna(price):
        return pd.NA
    if price > 1.0:
        price = price / 100.0
    if price < 0.0 or price > 1.0:
        return pd.NA
    return float(price)


def _fetch_markets_from_api(query: str | None) -> tuple[pd.DataFrame | None, str | None]:
    endpoint = _polymarket_endpoint()
    if not endpoint:
        return None, "No Polymarket API URL is configured."

    records: list[dict[str, Any]] = []

    try:
        for page in range(GAMMA_MAX_PAGES):
            params = _gamma_market_params(query=query, offset=page * GAMMA_PAGE_LIMIT)
            response = requests.get(endpoint, params=params, timeout=25)
            response.raise_for_status()
            payload = response.json()
            page_records = _extract_market_records(payload)
            records.extend(page_records)
            if len(page_records) < GAMMA_PAGE_LIMIT:
                break
    except Exception as exc:
        return None, str(exc)

    write_json_cache(records, "polymarket_markets_raw.json", "Polymarket Gamma API")
    if not records:
        return None, "Polymarket API payload did not contain market records."

    df = _normalise_markets(pd.DataFrame(records), SOURCE_API)
    df = _filter_query(df, query)
    return df, None


def _load_cached_markets(query: str | None, warning: str | None = None) -> pd.DataFrame | None:
    cached = read_dataframe_cache("polymarket_markets_normalized.csv", max_age_hours=None)
    if cached is None:
        return None
    df = _normalise_markets(cached, SOURCE_CACHE)
    df = _filter_query(df, query)
    return set_source_attrs(
        df,
        SOURCE_CACHE,
        "data/cache/polymarket_markets_normalized.csv",
        cache_last_updated("polymarket_markets_normalized.csv"),
        warning,
    )


def _normalise_markets(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "id": "market_id",
        "conditionId": "market_id",
        "condition_id": "market_id",
        "title": "question",
        "description": "question",
        "event": "event_title",
        "eventTitle": "event_title",
        "startDate": "start_date",
        "startDateIso": "start_date",
        "endDate": "end_date",
        "endDateIso": "end_date",
        "liquidityNum": "liquidity",
        "liquidityClob": "liquidity",
        "volumeNum": "volume",
        "volumeClob": "volume",
        "updatedAt": "last_updated",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    for col in POLYMARKET_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    rows = []
    for _, row in out.iterrows():
        outcomes = _parse_jsonish(row.get("outcomes"))
        outcome_prices = _parse_jsonish(row.get("outcomePrices", row.get("outcome_prices", pd.NA)))
        yes_price, no_price = _derive_yes_no_prices(row, outcomes, outcome_prices)
        event_title = _event_title(row)
        category = _category(row)

        rows.append(
            {
                "market_id": str(row.get("market_id", "") or "").strip(),
                "question": str(row.get("question", "") or "").strip(),
                "slug": str(row.get("slug", "") or "").strip(),
                "event_title": event_title,
                "category": category,
                "start_date": row.get("start_date", pd.NA),
                "end_date": row.get("end_date", pd.NA),
                "active": int(coerce_bool(row.get("active", True))),
                "closed": int(coerce_bool(row.get("closed", False))),
                "outcomes": _stringify_outcomes(outcomes if outcomes else row.get("outcomes", "")),
                "yes_price": normalize_price(yes_price),
                "no_price": normalize_price(no_price),
                "liquidity": coerce_float(row.get("liquidity"), float("nan")),
                "volume": coerce_float(row.get("volume"), float("nan")),
                "source": row.get("source") if pd.notna(row.get("source")) else source_label,
                "last_updated": row.get("last_updated") if pd.notna(row.get("last_updated")) else utc_now_iso(),
            }
        )

    normalised = pd.DataFrame(rows, columns=POLYMARKET_COLUMNS)
    normalised = normalised[normalised["market_id"].astype(str).str.len() > 0]
    return normalised.reset_index(drop=True)


def _derive_yes_no_prices(row: pd.Series, outcomes: Any, outcome_prices: Any) -> tuple[Any, Any]:
    yes_price = row.get("yes_price", row.get("yesPrice", pd.NA))
    no_price = row.get("no_price", row.get("noPrice", pd.NA))
    if pd.notna(yes_price) or pd.notna(no_price):
        return yes_price, no_price

    if isinstance(outcomes, list) and isinstance(outcome_prices, list):
        price_by_outcome = {
            str(outcome).strip().lower(): outcome_prices[idx]
            for idx, outcome in enumerate(outcomes)
            if idx < len(outcome_prices)
        }
        yes_price = price_by_outcome.get("yes", pd.NA)
        no_price = price_by_outcome.get("no", pd.NA)
        if pd.notna(yes_price) or pd.notna(no_price):
            return yes_price, no_price

    if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
        return outcome_prices[0], outcome_prices[1]
    return pd.NA, pd.NA


def _extract_market_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("markets", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _polymarket_endpoint() -> str | None:
    cfg = get_config()
    base = cfg.polymarket_gamma_api_url or cfg.polymarket_api_url or DEFAULT_POLYMARKET_GAMMA_API_URL
    if not base:
        return None
    endpoint = base.rstrip("/")
    if not endpoint.endswith("/markets"):
        endpoint = f"{endpoint}/markets"
    return endpoint


def _gamma_market_params(query: str | None, offset: int = 0) -> dict[str, Any]:
    params: dict[str, Any] = {
        "limit": GAMMA_PAGE_LIMIT,
        "offset": offset,
        "active": "true",
        "closed": "false",
    }
    if query:
        # Gamma may ignore search params on some deployments; local filtering
        # below remains the source of truth.
        params["search"] = query
    return params


def _filter_query(df: pd.DataFrame, query: str | None) -> pd.DataFrame:
    if df.empty or not query:
        return df
    query_l = query.lower()
    haystack = (
        df["question"].astype(str)
        + " "
        + df["slug"].astype(str)
        + " "
        + df["event_title"].astype(str)
        + " "
        + df["category"].astype(str)
    ).str.lower()
    return df.loc[haystack.str.contains(query_l, regex=False, na=False)].reset_index(drop=True)


def _event_title(row: pd.Series) -> str:
    value = row.get("event_title", "")
    if isinstance(value, dict):
        return str(value.get("title") or value.get("name") or "")
    if isinstance(row.get("events"), list) and row.get("events"):
        event = row.get("events")[0]
        if isinstance(event, dict):
            return str(event.get("title") or event.get("name") or event.get("slug") or "")
    if pd.isna(value):
        return ""
    return str(value)


def _category(row: pd.Series) -> str:
    value = row.get("category", "")
    if isinstance(value, dict):
        return str(value.get("name") or value.get("slug") or "")
    if pd.notna(value) and str(value).strip():
        return str(value).strip()
    if isinstance(row.get("events"), list) and row.get("events"):
        event = row.get("events")[0]
        if isinstance(event, dict):
            event_category = event.get("category")
            if isinstance(event_category, str):
                return event_category.strip()
            if isinstance(event_category, dict):
                return str(event_category.get("name") or event_category.get("slug") or "")
    return ""


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(",") if part.strip()]


def _stringify_outcomes(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def _csv_last_modified(filename: str) -> str:
    path = DATA_DIR / filename
    if not path.exists():
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
