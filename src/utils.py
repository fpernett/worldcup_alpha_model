from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env() -> None:
    """Load optional local API keys without requiring a .env file."""
    load_dotenv(ENV_PATH, override=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def today_iso() -> str:
    return date.today().isoformat()


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def env_value(name: str, default: str | None = None) -> str | None:
    load_project_env()
    value = os.getenv(name, default)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def api_configured(*names: str) -> bool:
    return all(env_value(name) for name in names)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "closed"}


def parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.to_datetime(value).date()


def read_csv_with_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    """Read a CSV and add any missing required columns as NA.

    Missing files return an empty frame with the requested columns. This keeps
    Streamlit and update scripts running even when optional CSVs do not exist.
    """
    required = list(columns)
    if not path.exists():
        return pd.DataFrame(columns=required)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=required)
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def write_cached_dataframe(df: pd.DataFrame, filename: str, source: str) -> Path:
    cache_dir = ensure_cache_dir()
    path = cache_dir / filename
    df.to_csv(path, index=False)
    meta = {
        "source": source,
        "rows": int(len(df)),
        "last_updated": utc_now_iso(),
        "file": filename,
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def read_cached_dataframe(filename: str, max_age_hours: float | None = None) -> pd.DataFrame | None:
    path = CACHE_DIR / filename
    if not path.exists():
        return None
    if max_age_hours is not None:
        age_hours = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0
        if age_hours > max_age_hours:
            return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def csv_status(path: Path, required_columns: Iterable[str] | None = None) -> dict[str, Any]:
    exists = path.exists()
    status: dict[str, Any] = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "exists": exists,
        "rows": 0,
        "last_modified": "",
        "missing_columns": "",
    }
    if not exists:
        return status

    modified = datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0).isoformat()
    status["last_modified"] = modified
    try:
        df = pd.read_csv(path, nrows=5)
        full_rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        status["rows"] = max(full_rows, 0)
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            status["missing_columns"] = ", ".join(missing)
    except Exception as exc:  # defensive status panel only
        status["missing_columns"] = f"read_error: {exc}"
    return status
