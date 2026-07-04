from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import DATA_DIR, SOURCE_API, SOURCE_CACHE, get_config
from src.utils import coerce_bool, coerce_float, read_csv_with_columns, utc_now_iso


DEFAULT_POLYMARKET_GAMMA_API_URL = "https://gamma-api.polymarket.com"
EVENTS_CACHE_PATH = DATA_DIR / "polymarket_events_cache.csv"
MARKETS_CACHE_PATH = DATA_DIR / "polymarket_markets_cache.csv"

GAMMA_EVENT_MARKET_COLUMNS = [
    "event_id",
    "event_slug",
    "event_title",
    "event_category",
    "event_start_date",
    "event_end_date",
    "event_active",
    "event_closed",
    "market_id",
    "market_slug",
    "question",
    "market_title",
    "market_category",
    "outcomes",
    "yes_price",
    "no_price",
    "liquidity",
    "volume",
    "volume_24hr",
    "start_date",
    "end_date",
    "raw_event_json",
    "raw_market_json",
    "source",
    "last_updated",
]

GAMMA_EVENT_CACHE_COLUMNS = [
    "event_id",
    "event_slug",
    "event_title",
    "event_category",
    "event_start_date",
    "event_end_date",
    "event_active",
    "event_closed",
    "raw_event_json",
    "source",
    "last_updated",
]

DISCOVERY_KEYWORDS = ["world cup", "fifa", "fifwc", "club world cup"]


def fetch_gamma_events(
    active: bool = True,
    closed: bool | None = False,
    limit: int = 500,
    offset: int = 0,
    order: str = "volume_24hr",
    ascending: bool = False,
    tag_id: str | None = None,
    series_id: str | None = None,
    sport_id: str | None = None,
) -> list[dict]:
    records, _has_more = _fetch_gamma_events_page(
        active=active,
        closed=closed,
        limit=limit,
        offset=offset,
        order=order,
        ascending=ascending,
        tag_id=tag_id,
        series_id=series_id,
        sport_id=sport_id,
    )
    return records


def _fetch_gamma_events_page(
    active: bool = True,
    closed: bool | None = False,
    limit: int = 500,
    offset: int = 0,
    order: str = "volume_24hr",
    ascending: bool = False,
    tag_id: str | None = None,
    series_id: str | None = None,
    sport_id: str | None = None,
) -> tuple[list[dict], bool | None]:
    endpoint = _gamma_endpoint("events")
    params: dict[str, Any] = {
        "limit": int(limit),
        "offset": int(offset),
        "active": str(bool(active)).lower(),
        "order": order,
        "ascending": str(bool(ascending)).lower(),
    }
    if closed is not None:
        params["closed"] = str(bool(closed)).lower()
    if tag_id:
        params["tag_id"] = tag_id
    if series_id:
        params["series_id"] = series_id
    if sport_id:
        params["sport_id"] = sport_id
    try:
        response = requests.get(endpoint, params=params, timeout=25)
        response.raise_for_status()
        payload = response.json()
        return _extract_records(payload, ("events", "data", "results")), _has_more(payload)
    except Exception:
        return [], None


def fetch_all_gamma_events(
    active: bool = True,
    closed: bool | None = False,
    max_pages: int = 20,
    limit: int = 500,
    tag_id: str | None = None,
    series_id: str | None = None,
    sport_id: str | None = None,
) -> list[dict]:
    events: list[dict] = []
    for page in range(max_pages):
        offset = page * limit
        page_events, has_more = _fetch_gamma_events_page(
            active=active,
            closed=closed,
            limit=limit,
            offset=offset,
            tag_id=tag_id,
            series_id=series_id,
            sport_id=sport_id,
        )
        events.extend(page_events)
        if has_more is False:
            break
        if has_more is None and len(page_events) < limit:
            break
    return _dedupe_records(events, id_keys=("id", "event_id", "slug"))


def fetch_gamma_sports() -> list[dict]:
    return _fetch_metadata("sports")


def fetch_gamma_tags() -> list[dict]:
    return _fetch_metadata("tags")


def fetch_gamma_series() -> list[dict]:
    return _fetch_metadata("series")


def discover_world_cup_identifiers(tags: list[dict], series: list[dict], sports: list[dict]) -> dict[str, Any]:
    tag_ids, series_ids, sports_ids = [], [], []
    diagnostics = []
    for label, records, bucket in [
        ("tag", tags, tag_ids),
        ("series", series, series_ids),
        ("sport", sports, sports_ids),
    ]:
        for record in records or []:
            text = _record_search_text(record)
            if label == "sport":
                sport_code = _text(record.get("sport", "")).lower()
                if sport_code != "fifwc":
                    continue
            if any(keyword in text for keyword in DISCOVERY_KEYWORDS):
                identifier = _record_id(record)
                if identifier:
                    if label == "sport":
                        series_identifier = _text(record.get("series", ""))
                        if series_identifier:
                            series_ids.append(series_identifier)
                            diagnostics.append(f"{label}:series:{series_identifier}:{_record_name(record)}")
                        else:
                            bucket.append(identifier)
                            diagnostics.append(f"{label}:{identifier}:{_record_name(record)}")
                    else:
                        bucket.append(identifier)
                        diagnostics.append(f"{label}:{identifier}:{_record_name(record)}")
    return {
        "tag_ids": list(dict.fromkeys(tag_ids)),
        "series_ids": list(dict.fromkeys(series_ids)),
        "sports_ids": list(dict.fromkeys(sports_ids)),
        "diagnostics": diagnostics,
    }


def flatten_gamma_events_to_markets(events: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events or []:
        markets = event.get("markets") or []
        if not isinstance(markets, list):
            continue
        event_row = _event_base_row(event)
        for market in markets:
            if not isinstance(market, dict):
                continue
            outcomes = _parse_jsonish(market.get("outcomes"))
            prices = _parse_jsonish(market.get("outcomePrices", market.get("outcome_prices")))
            yes_price, no_price = _derive_yes_no_prices(market, outcomes, prices)
            rows.append(
                {
                    **event_row,
                    "market_id": _text(market.get("id", market.get("market_id", market.get("conditionId", "")))),
                    "market_slug": _text(market.get("slug", "")),
                    "question": _text(market.get("question", market.get("title", ""))),
                    "market_title": _text(market.get("title", "")),
                    "market_category": _market_category(market),
                    "outcomes": _stringify(outcomes if outcomes else market.get("outcomes", "")),
                    "yes_price": _normalise_price(yes_price),
                    "no_price": _normalise_price(no_price),
                    "liquidity": coerce_float(market.get("liquidityNum", market.get("liquidity", market.get("liquidityClob"))), float("nan")),
                    "volume": coerce_float(market.get("volumeNum", market.get("volume", market.get("volumeClob"))), float("nan")),
                    "volume_24hr": coerce_float(market.get("volume24hr", market.get("volume_24hr")), float("nan")),
                    "start_date": market.get("startDate", market.get("start_date", pd.NA)),
                    "end_date": market.get("endDate", market.get("end_date", pd.NA)),
                    "raw_market_json": _stringify(market),
                    "source": SOURCE_API,
                    "last_updated": utc_now_iso(),
                }
            )
    out = pd.DataFrame(rows, columns=GAMMA_EVENT_MARKET_COLUMNS)
    if out.empty:
        return out
    out = out.loc[out["market_id"].astype(str).str.strip() != ""].copy()
    return out.drop_duplicates(subset=["market_id"]).reset_index(drop=True)


def flatten_gamma_events_to_event_cache(events: list[dict]) -> pd.DataFrame:
    rows = []
    for event in events or []:
        row = _event_base_row(event)
        row["source"] = SOURCE_API
        row["last_updated"] = utc_now_iso()
        rows.append({col: row.get(col, pd.NA) for col in GAMMA_EVENT_CACHE_COLUMNS})
    out = pd.DataFrame(rows, columns=GAMMA_EVENT_CACHE_COLUMNS)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["event_id", "event_slug"]).reset_index(drop=True)


def write_polymarket_discovery_cache(events: list[dict], markets_df: pd.DataFrame) -> dict[str, str]:
    event_cache = flatten_gamma_events_to_event_cache(events)
    EVENTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if EVENTS_CACHE_PATH.exists():
        existing_events = read_csv_with_columns(EVENTS_CACHE_PATH, GAMMA_EVENT_CACHE_COLUMNS)
        event_cache = pd.concat([existing_events, event_cache], ignore_index=True)
        event_cache = _dedupe_cache_frame(event_cache, ["event_id", "event_slug"])
    event_cache.to_csv(EVENTS_CACHE_PATH, index=False)
    markets = markets_df.copy() if markets_df is not None else pd.DataFrame(columns=GAMMA_EVENT_MARKET_COLUMNS)
    for col in GAMMA_EVENT_MARKET_COLUMNS:
        if col not in markets.columns:
            markets[col] = pd.NA
    # Full event JSON is stored once in the event cache. Repeating it for every
    # nested market makes the market cache needlessly large.
    markets["raw_event_json"] = ""
    if MARKETS_CACHE_PATH.exists():
        existing_markets = read_csv_with_columns(MARKETS_CACHE_PATH, GAMMA_EVENT_MARKET_COLUMNS)
        markets = pd.concat([existing_markets, markets], ignore_index=True)
        markets = _dedupe_cache_frame(markets, ["market_id"])
    markets[GAMMA_EVENT_MARKET_COLUMNS].to_csv(MARKETS_CACHE_PATH, index=False)
    return {
        "events_cache": str(EVENTS_CACHE_PATH),
        "markets_cache": str(MARKETS_CACHE_PATH),
    }


def _dedupe_cache_frame(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    out = df.copy()
    for col in key_columns:
        if col not in out.columns:
            out[col] = ""
    key = out[key_columns].fillna("").astype(str).agg("|".join, axis=1)
    out = out.assign(_cache_key=key).drop_duplicates(subset=["_cache_key"], keep="last")
    return out.drop(columns=["_cache_key"]).reset_index(drop=True)


def load_polymarket_markets_cache() -> pd.DataFrame:
    df = read_csv_with_columns(MARKETS_CACHE_PATH, GAMMA_EVENT_MARKET_COLUMNS)
    if df.empty:
        return df
    df.attrs["source_label"] = SOURCE_CACHE
    df.attrs["last_updated"] = _cache_last_modified(MARKETS_CACHE_PATH)
    return df


def polymarket_discovery_cache_status() -> dict[str, Any]:
    return {
        "events_cache_path": str(EVENTS_CACHE_PATH),
        "events_cache_exists": EVENTS_CACHE_PATH.exists(),
        "events_cache_rows": _csv_row_count(EVENTS_CACHE_PATH),
        "events_cache_last_updated": _cache_last_modified(EVENTS_CACHE_PATH),
        "markets_cache_path": str(MARKETS_CACHE_PATH),
        "markets_cache_exists": MARKETS_CACHE_PATH.exists(),
        "markets_cache_rows": _csv_row_count(MARKETS_CACHE_PATH),
        "markets_cache_last_updated": _cache_last_modified(MARKETS_CACHE_PATH),
    }


def flattened_markets_to_polymarket_rows(markets_df: pd.DataFrame) -> pd.DataFrame:
    from src.polymarket import POLYMARKET_COLUMNS

    markets = markets_df.copy() if markets_df is not None else pd.DataFrame(columns=GAMMA_EVENT_MARKET_COLUMNS)
    for col in GAMMA_EVENT_MARKET_COLUMNS:
        if col not in markets.columns:
            markets[col] = pd.NA
    rows = []
    for _, row in markets.iterrows():
        rows.append(
            {
                "market_id": row.get("market_id", ""),
                "question": row.get("question", row.get("market_title", "")),
                "slug": row.get("market_slug", ""),
                "event_title": row.get("event_title", ""),
                "category": row.get("event_category", row.get("market_category", "")),
                "start_date": row.get("start_date", row.get("event_start_date", pd.NA)),
                "end_date": row.get("end_date", row.get("event_end_date", pd.NA)),
                "active": int(coerce_bool(row.get("event_active", True))),
                "closed": int(coerce_bool(row.get("event_closed", False))),
                "outcomes": row.get("outcomes", ""),
                "yes_price": row.get("yes_price", pd.NA),
                "no_price": row.get("no_price", pd.NA),
                "liquidity": row.get("liquidity", pd.NA),
                "volume": row.get("volume", pd.NA),
                "source": row.get("source", SOURCE_CACHE),
                "last_updated": row.get("last_updated", ""),
            }
        )
    return pd.DataFrame(rows, columns=POLYMARKET_COLUMNS)


def discover_polymarket_sports_event_markets(
    active: bool = True,
    closed: bool | None = False,
    retrieval_mode: str = "sports_events",
    max_pages: int = 20,
    limit: int = 500,
    write_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    events: list[dict] = []
    layers = []
    identifiers = {"tag_ids": [], "series_ids": [], "sports_ids": [], "diagnostics": []}

    if retrieval_mode in {"sports_events", "auto"}:
        tags = fetch_gamma_tags()
        series = fetch_gamma_series()
        sports = fetch_gamma_sports()
        identifiers = discover_world_cup_identifiers(tags, series, sports)
        for tag_id in identifiers["tag_ids"][:5]:
            layers.append(f"events:tag:{tag_id}")
            events.extend(fetch_all_gamma_events(active=active, closed=closed, max_pages=max_pages, limit=limit, tag_id=tag_id))
        for series_id in identifiers["series_ids"][:5]:
            layers.append(f"events:series:{series_id}")
            events.extend(fetch_all_gamma_events(active=active, closed=closed, max_pages=max_pages, limit=limit, series_id=series_id))
        for sport_id in identifiers["sports_ids"][:5]:
            layers.append(f"events:sport:{sport_id}")
            events.extend(fetch_all_gamma_events(active=active, closed=closed, max_pages=max_pages, limit=limit, sport_id=sport_id))

    if retrieval_mode in {"all_events", "auto"} and not events:
        layers.append("events:all")
        events.extend(fetch_all_gamma_events(active=active, closed=closed, max_pages=max_pages, limit=limit))

    events = _dedupe_records(events, id_keys=("id", "event_id", "slug"))
    markets = flatten_gamma_events_to_markets(events)
    cache_paths: dict[str, str] = {}
    if write_cache and (events or not markets.empty):
        cache_paths = write_polymarket_discovery_cache(events, markets)
    diagnostics = {
        "retrieval_layers_tried": layers,
        "events_fetched": len(events),
        "markets_flattened": len(markets),
        "identifiers": identifiers,
        "cache_paths": cache_paths,
    }
    markets.attrs.update(diagnostics)
    return markets, diagnostics


def _fetch_metadata(kind: str) -> list[dict]:
    try:
        response = requests.get(_gamma_endpoint(kind), params={"limit": 500}, timeout=25)
        response.raise_for_status()
        return _extract_records(response.json(), (kind, "data", "results"))
    except Exception:
        return []


def _gamma_endpoint(kind: str) -> str:
    base = _gamma_base_url()
    return f"{base}/{kind}"


def _gamma_base_url() -> str:
    cfg = get_config()
    base = cfg.polymarket_gamma_api_url or cfg.polymarket_api_url or DEFAULT_POLYMARKET_GAMMA_API_URL
    endpoint = str(base).rstrip("/")
    for suffix in ("/markets", "/events", "/tags", "/series", "/sports"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    return endpoint


def _extract_records(payload: Any, keys: tuple[str, ...]) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for key in ("events", "tags", "series", "sports", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _has_more(payload: Any) -> bool | None:
    if not isinstance(payload, dict):
        return None
    for key in ("has_more", "hasMore", "has_more_events"):
        if key in payload:
            return bool(payload.get(key))
    return None


def _dedupe_records(records: list[dict], id_keys: tuple[str, ...]) -> list[dict]:
    out, seen = [], set()
    for record in records:
        key = ""
        for id_key in id_keys:
            value = _text(record.get(id_key))
            if value:
                key = f"{id_key}:{value}"
                break
        if not key:
            key = json.dumps(record, sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _event_base_row(event: dict) -> dict[str, Any]:
    return {
        "event_id": _text(event.get("id", event.get("event_id", ""))),
        "event_slug": _text(event.get("slug", "")),
        "event_title": _text(event.get("title", event.get("name", event.get("event_title", "")))),
        "event_category": _event_category(event),
        "event_start_date": event.get("startDate", event.get("start_date", pd.NA)),
        "event_end_date": event.get("endDate", event.get("end_date", pd.NA)),
        "event_active": int(coerce_bool(event.get("active", True))),
        "event_closed": int(coerce_bool(event.get("closed", False))),
        "raw_event_json": _stringify(event),
    }


def _event_category(event: dict) -> str:
    for key in ("category", "sport"):
        value = event.get(key)
        if isinstance(value, dict):
            text = _text(value.get("name", value.get("label", value.get("slug", ""))))
        else:
            text = _text(value)
        if text:
            return text
    tags = event.get("tags")
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, dict):
            return _record_name(first)
        return _text(first)
    return ""


def _market_category(market: dict) -> str:
    value = market.get("category", "")
    if isinstance(value, dict):
        return _text(value.get("name", value.get("slug", "")))
    return _text(value)


def _derive_yes_no_prices(market: dict, outcomes: Any, prices: Any) -> tuple[Any, Any]:
    yes_price = market.get("yes_price", market.get("yesPrice", pd.NA))
    no_price = market.get("no_price", market.get("noPrice", pd.NA))
    if pd.notna(yes_price) or pd.notna(no_price):
        return yes_price, no_price
    if isinstance(outcomes, list) and isinstance(prices, list):
        by_outcome = {str(outcome).strip().lower(): prices[idx] for idx, outcome in enumerate(outcomes) if idx < len(prices)}
        if "yes" in by_outcome or "no" in by_outcome:
            return by_outcome.get("yes", pd.NA), by_outcome.get("no", pd.NA)
    if isinstance(prices, list) and len(prices) >= 2:
        return prices[0], prices[1]
    return pd.NA, pd.NA


def _normalise_price(value: Any) -> Any:
    price = coerce_float(value, float("nan"))
    if pd.isna(price):
        return pd.NA
    if price > 1.0:
        price = price / 100.0
    if price < 0.0 or price > 1.0:
        return pd.NA
    return float(price)


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value or "").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(",") if part.strip()]


def _record_search_text(record: dict) -> str:
    parts = [
        _record_name(record),
        _text(record.get("sport", "")),
        _text(record.get("resolution", "")),
        _text(record.get("series", "")),
    ]
    return " ".join(" ".join(parts).lower().split())


def _record_name(record: dict) -> str:
    return _text(record.get("name", record.get("label", record.get("title", record.get("slug", "")))))


def _record_id(record: dict) -> str:
    return _text(record.get("id", record.get("tag_id", record.get("series_id", record.get("slug", "")))))


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return str(value)


def _text(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value or "").strip()


def _cache_last_modified(path: Path) -> str:
    if not path.exists():
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0)
    except OSError:
        return 0
