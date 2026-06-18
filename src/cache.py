from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CACHE_DIR, SOURCE_CACHE


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def safe_cache_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "cache"


def cache_path(filename: str) -> Path:
    return ensure_cache_dir() / filename


def write_json_cache(payload: Any, filename: str, source: str) -> Path:
    path = cache_path(filename)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_cache_metadata(path, source=source, rows=_count_payload_rows(payload))
    return path


def read_json_cache(filename: str, max_age_hours: float | None = None) -> Any | None:
    path = cache_path(filename)
    if not path.exists() or _is_too_old(path, max_age_hours):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_dataframe_cache(df: pd.DataFrame, filename: str, source: str) -> Path:
    path = cache_path(filename)
    df.to_csv(path, index=False)
    write_cache_metadata(path, source=source, rows=len(df))
    return path


def read_dataframe_cache(filename: str, max_age_hours: float | None = None) -> pd.DataFrame | None:
    path = cache_path(filename)
    if not path.exists() or _is_too_old(path, max_age_hours):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    df.attrs["source_label"] = SOURCE_CACHE
    df.attrs["source_detail"] = f"data/cache/{filename}"
    df.attrs["last_updated"] = cache_last_updated(filename)
    return df


def write_cache_metadata(path: Path, source: str, rows: int) -> None:
    meta = {
        "source": source,
        "rows": int(rows),
        "last_updated": utc_now_iso(),
        "file": path.name,
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_cache_metadata(filename: str) -> dict[str, Any]:
    path = cache_path(filename)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def cache_last_updated(filename: str) -> str:
    meta = read_cache_metadata(filename)
    return str(meta.get("last_updated", ""))


def set_source_attrs(
    df: pd.DataFrame,
    source_label: str,
    source_detail: str,
    last_updated: str | None = None,
    warning: str | None = None,
) -> pd.DataFrame:
    df.attrs["source_label"] = source_label
    df.attrs["source_detail"] = source_detail
    df.attrs["last_updated"] = last_updated or utc_now_iso()
    if warning:
        df.attrs["warning"] = warning
    return df


def _is_too_old(path: Path, max_age_hours: float | None) -> bool:
    if max_age_hours is None:
        return False
    age_hours = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0
    return age_hours > max_age_hours


def _count_payload_rows(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("matches", "fixtures", "data", "ratings", "odds", "markets", "results"):
            if isinstance(payload.get(key), list):
                return len(payload[key])
    return 1
