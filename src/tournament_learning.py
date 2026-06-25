from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.prediction_ledger import RESULTS_LEDGER_COLUMNS
from src.utils import coerce_float, read_csv_with_columns, utc_now_iso


TOURNAMENT_LEARNING_LEDGER_PATH = DATA_DIR / "tournament_learning_ledger.csv"

TOURNAMENT_LEARNING_LEDGER_COLUMNS = [
    "learning_id",
    "created_utc",
    "match_id",
    "date_utc",
    "competition",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "actual_result",
    "total_goals",
    "btts_actual",
    "used_for_future_calibration",
    "eligible_after_utc",
    "source",
    "notes",
]


def load_tournament_learning_ledger(path: str | Path = TOURNAMENT_LEARNING_LEDGER_PATH) -> pd.DataFrame:
    return read_csv_with_columns(Path(path), TOURNAMENT_LEARNING_LEDGER_COLUMNS)[TOURNAMENT_LEARNING_LEDGER_COLUMNS].copy()


def save_tournament_learning_ledger(
    learning_df: pd.DataFrame | None,
    path: str | Path = TOURNAMENT_LEARNING_LEDGER_PATH,
) -> pd.DataFrame:
    out = _ensure_columns(learning_df, TOURNAMENT_LEARNING_LEDGER_COLUMNS)
    out = _dedupe_learning(out)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def update_tournament_learning_ledger(
    results_ledger_df: pd.DataFrame | None,
    existing_learning_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create one auditable tournament-learning row per completed result."""
    results = _ensure_columns(results_ledger_df, RESULTS_LEDGER_COLUMNS)
    existing = _ensure_columns(existing_learning_df, TOURNAMENT_LEARNING_LEDGER_COLUMNS)
    if results.empty:
        return _dedupe_learning(existing)

    now = utc_now_iso()
    rows: list[dict[str, Any]] = []
    for _, result in results.iterrows():
        match_id = str(result.get("match_id", "") or "").strip()
        if not match_id:
            continue
        home_goals = int(coerce_float(result.get("home_goals"), 0.0))
        away_goals = int(coerce_float(result.get("away_goals"), 0.0))
        total_goals = int(coerce_float(result.get("total_goals"), home_goals + away_goals))
        eligible_after = _eligible_after_utc(result)
        rows.append(
            {
                "learning_id": _learning_id(match_id),
                "created_utc": now,
                "match_id": match_id,
                "date_utc": result.get("date_utc", ""),
                "competition": result.get("competition", ""),
                "home": result.get("home", ""),
                "away": result.get("away", ""),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "actual_result": result.get("actual_result", ""),
                "total_goals": total_goals,
                "btts_actual": int(coerce_float(result.get("btts_actual"), float(home_goals > 0 and away_goals > 0))),
                "used_for_future_calibration": True,
                "eligible_after_utc": eligible_after,
                "source": result.get("result_source", "results_ledger"),
                "notes": "Eligible for future match-specific calibration after eligible_after_utc.",
            }
        )

    new_learning = pd.DataFrame(rows, columns=TOURNAMENT_LEARNING_LEDGER_COLUMNS)
    combined = pd.concat([existing, new_learning], ignore_index=True)
    return _dedupe_learning(combined)


def filter_tournament_learning_asof(
    learning_df: pd.DataFrame | None,
    target_kickoff_utc: Any,
    target_match_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return completed tournament matches eligible before a target kickoff."""
    learning = _ensure_columns(learning_df, TOURNAMENT_LEARNING_LEDGER_COLUMNS)
    if learning.empty:
        diagnostics = _learning_diagnostics(learning, learning, target_kickoff_utc)
        return learning, diagnostics

    target_ts = pd.to_datetime(target_kickoff_utc, errors="coerce", utc=True)
    eligible_ts = pd.to_datetime(learning["eligible_after_utc"], errors="coerce", utc=True)
    eligible = learning.loc[eligible_ts.notna()].copy()
    if not pd.isna(target_ts):
        eligible = eligible.loc[eligible_ts.loc[eligible.index] < target_ts].copy()
    if target_match_id:
        eligible = eligible.loc[eligible["match_id"].astype(str) != str(target_match_id)].copy()
    diagnostics = _learning_diagnostics(learning, eligible, target_kickoff_utc)
    return eligible.reset_index(drop=True), diagnostics


def tournament_learning_summary(path: str | Path = TOURNAMENT_LEARNING_LEDGER_PATH) -> dict[str, Any]:
    learning = load_tournament_learning_ledger(path)
    eligible_ts = pd.to_datetime(learning.get("eligible_after_utc", pd.Series(dtype=str)), errors="coerce", utc=True)
    return {
        "completed_world_cup_matches_available": int(_world_cup_rows(learning).shape[0]),
        "learning_rows": int(len(learning)),
        "latest_completed_match_used": "" if eligible_ts.dropna().empty else eligible_ts.max().date().isoformat(),
        "path": str(Path(path).relative_to(DATA_DIR.parent)) if Path(path).is_absolute() else str(path),
    }


def _ensure_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame(columns=columns)
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].copy()


def _dedupe_learning(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df[TOURNAMENT_LEARNING_LEDGER_COLUMNS].copy()
    out = df.copy()
    out["_created_sort"] = pd.to_datetime(out["created_utc"], errors="coerce", utc=True)
    out = out.sort_values(["match_id", "_created_sort"], na_position="first").drop_duplicates("match_id", keep="last")
    out = out.drop(columns=["_created_sort"]).sort_values(["date_utc", "match_id"]).reset_index(drop=True)
    return out[TOURNAMENT_LEARNING_LEDGER_COLUMNS].copy()


def _eligible_after_utc(result: pd.Series) -> str:
    for col in ["final_whistle_utc", "completed_utc"]:
        if col in result.index:
            ts = pd.to_datetime(result.get(col), errors="coerce", utc=True)
            if not pd.isna(ts):
                return ts.replace(microsecond=0).isoformat()
    ts = pd.to_datetime(result.get("date_utc"), errors="coerce", utc=True)
    if pd.isna(ts):
        return ""
    return (ts + pd.Timedelta(hours=2)).replace(microsecond=0).isoformat()


def _learning_id(match_id: str) -> str:
    return hashlib.sha1(str(match_id).encode("utf-8")).hexdigest()[:16]


def _world_cup_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "competition" not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[df["competition"].astype(str).str.lower().str.contains("world cup", na=False)].copy()


def _learning_diagnostics(all_learning: pd.DataFrame, used_learning: pd.DataFrame, target_kickoff_utc: Any) -> dict[str, Any]:
    world_cup_all = _world_cup_rows(all_learning)
    world_cup_used = _world_cup_rows(used_learning)
    used_eligible = pd.to_datetime(world_cup_used.get("eligible_after_utc", pd.Series(dtype=str)), errors="coerce", utc=True)
    target_ts = pd.to_datetime(target_kickoff_utc, errors="coerce", utc=True)
    lookahead_safe = True
    if not pd.isna(target_ts) and not used_eligible.dropna().empty:
        lookahead_safe = bool((used_eligible.dropna() < target_ts).all())
    return {
        "completed_world_cup_matches_available": int(len(world_cup_all)),
        "completed_world_cup_matches_used": int(len(world_cup_used)),
        "latest_completed_match_used": "" if used_eligible.dropna().empty else used_eligible.max().date().isoformat(),
        "lookahead_safe": lookahead_safe,
    }
