from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import DATA_DIR


def data_path(filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / filename


def ensure_csv(filename: str, columns: Iterable[str]) -> Path:
    path = data_path(filename)
    if not path.exists():
        pd.DataFrame(columns=list(columns)).to_csv(path, index=False)
    return path


def load_csv(filename: str, columns: Iterable[str]) -> pd.DataFrame:
    path = ensure_csv(filename, columns)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=list(columns))
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[list(columns)]


def append_csv(filename: str, rows: pd.DataFrame, columns: Iterable[str]) -> None:
    if rows is None or rows.empty:
        return
    path = ensure_csv(filename, columns)
    out = rows.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[list(columns)]
    header = not path.exists() or path.stat().st_size == 0
    out.to_csv(path, mode="a", header=header, index=False)
