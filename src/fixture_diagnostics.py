from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.data_sources import FIXTURE_COLUMNS
from src.team_names import is_unresolved_team_slot


FIXTURE_AUDIT_COLUMNS = [
    "match_id",
    "date_utc",
    "time_utc",
    "kickoff_utc",
    "competition",
    "group",
    "home",
    "away",
    "venue",
    "included_in_raw_file",
    "inside_selected_window",
    "is_past",
    "hidden_by_past_filter",
    "has_unresolved_team_slot",
    "hidden_by_unresolved_slot",
    "excluded_reason",
    "visible_in_app",
]


def audit_fixture_availability(
    fixtures_df: pd.DataFrame,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    hide_past: bool = True,
    now_utc: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Explain why each fixture is or is not visible in the selected UTC window."""
    fixtures = _normalise_fixture_input(fixtures_df)
    now = _coerce_utc_timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    selected_start = _window_start(start_date)
    selected_end = _window_end(end_date)

    if fixtures.empty:
        audit = pd.DataFrame(columns=FIXTURE_AUDIT_COLUMNS)
        return audit, _summary(audit, selected_start, selected_end, now, "No fixtures loaded from the selected source.")

    audit = fixtures.copy()
    audit["kickoff_utc"] = _kickoff_series(audit)
    audit["included_in_raw_file"] = True
    audit["inside_selected_window"] = audit["kickoff_utc"].notna()
    if selected_start is not None:
        audit["inside_selected_window"] &= audit["kickoff_utc"] >= selected_start
    if selected_end is not None:
        audit["inside_selected_window"] &= audit["kickoff_utc"] <= selected_end
    audit["is_past"] = audit["kickoff_utc"].notna() & (audit["kickoff_utc"] < now)
    audit["hidden_by_past_filter"] = bool(hide_past) & audit["inside_selected_window"] & audit["is_past"]
    audit["has_unresolved_team_slot"] = audit.apply(_has_unresolved_team_slot, axis=1)
    audit["hidden_by_unresolved_slot"] = audit["inside_selected_window"] & audit["has_unresolved_team_slot"]
    audit["visible_in_app"] = (
        audit["inside_selected_window"]
        & ~audit["hidden_by_past_filter"]
        & ~audit["hidden_by_unresolved_slot"]
    )
    audit["excluded_reason"] = audit.apply(_excluded_reason, axis=1)

    audit["kickoff_utc"] = audit["kickoff_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    audit = audit[FIXTURE_AUDIT_COLUMNS].reset_index(drop=True)
    return audit, _summary(audit, selected_start, selected_end, now, _fixture_warning(audit, selected_start, selected_end, now))


def _normalise_fixture_input(fixtures_df: pd.DataFrame | None) -> pd.DataFrame:
    if fixtures_df is None or fixtures_df.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    out = fixtures_df.copy()
    for col in FIXTURE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[FIXTURE_COLUMNS].copy()


def _kickoff_series(df: pd.DataFrame) -> pd.Series:
    date_part = df.get("date_utc", pd.Series(index=df.index, dtype="object")).astype(str)
    time_part = df.get("time_utc", pd.Series("00:00", index=df.index)).fillna("00:00").astype(str).str.slice(0, 5)
    return pd.to_datetime(date_part + " " + time_part, utc=True, errors="coerce")


def _coerce_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _window_start(value: str | date | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    ts = _coerce_utc_timestamp(str(value))
    if "T" not in str(value) and " " not in str(value):
        return ts.normalize()
    return ts


def _window_end(value: str | date | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    ts = _coerce_utc_timestamp(str(value))
    if "T" not in str(value) and " " not in str(value):
        return ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _excluded_reason(row: pd.Series) -> str:
    if pd.isna(row.get("kickoff_utc")):
        return "invalid_kickoff"
    if not bool(row.get("inside_selected_window")):
        return "outside_selected_window"
    if bool(row.get("hidden_by_past_filter")):
        return "hidden_by_past_filter"
    if bool(row.get("hidden_by_unresolved_slot")):
        return "unresolved_team_slot"
    return ""


def _has_unresolved_team_slot(row: pd.Series) -> bool:
    return is_unresolved_team_slot(row.get("home", "")) or is_unresolved_team_slot(row.get("away", ""))


def _fixture_warning(
    audit: pd.DataFrame,
    selected_start: pd.Timestamp | None,
    selected_end: pd.Timestamp | None,
    now: pd.Timestamp,
) -> str:
    if audit.empty:
        return "No fixtures loaded."
    kickoff = pd.to_datetime(audit.get("kickoff_utc", pd.Series(dtype="object")), utc=True, errors="coerce")
    latest = kickoff.max()
    if pd.notna(latest):
        if selected_start is not None and latest < selected_start:
            return (
                f"No fixtures after {latest.strftime('%Y-%m-%dT%H:%M:%SZ')} are available in the loaded fixture source. "
                "Refresh API/cache or update data/fixtures.csv."
            )
    visible = int(audit["visible_in_app"].sum())
    inside = int(audit["inside_selected_window"].sum())
    hidden = int(audit["hidden_by_past_filter"].sum())
    unresolved = int(audit.get("hidden_by_unresolved_slot", pd.Series(dtype=bool)).sum())
    if visible == 0 and inside > 0 and hidden == inside:
        return "Fixtures exist in the selected window, but all are hidden by the past-kickoff filter."
    if visible == 0 and inside > 0 and unresolved > 0:
        return (
            "Fixtures exist in the selected window, but all visible candidates are unresolved bracket slots. "
            "Update data/fixtures.csv with actual teams before modelling them."
        )
    if unresolved > 0:
        return (
            f"{unresolved} fixture(s) in the selected window are hidden because the home or away side "
            "is still an unresolved bracket slot."
        )
    if pd.notna(latest) and selected_end is not None and latest < selected_end:
        return (
            f"The loaded fixture source ends at {latest.strftime('%Y-%m-%dT%H:%M:%SZ')}; "
            "refresh API/cache or update data/fixtures.csv to show later games."
        )
    if visible == 0:
        return "No fixtures are visible for the selected UTC window."
    return ""


def _summary(
    audit: pd.DataFrame,
    selected_start: pd.Timestamp | None,
    selected_end: pd.Timestamp | None,
    now: pd.Timestamp,
    warning: str,
) -> dict[str, Any]:
    kickoff = pd.to_datetime(audit.get("kickoff_utc", pd.Series(dtype="object")), utc=True, errors="coerce")
    return {
        "total_fixtures_loaded": int(len(audit)),
        "earliest_kickoff_utc": _ts_label(kickoff.min()),
        "latest_kickoff_utc": _ts_label(kickoff.max()),
        "selected_start_utc": _ts_label(selected_start),
        "selected_end_utc": _ts_label(selected_end),
        "current_utc": _ts_label(now),
        "fixtures_inside_window": int(audit.get("inside_selected_window", pd.Series(dtype=bool)).sum()),
        "fixtures_hidden_as_past": int(audit.get("hidden_by_past_filter", pd.Series(dtype=bool)).sum()),
        "fixtures_hidden_as_unresolved": int(
            audit.get("hidden_by_unresolved_slot", pd.Series(dtype=bool)).sum()
        ),
        "fixtures_visible": int(audit.get("visible_in_app", pd.Series(dtype=bool)).sum()),
        "fixtures_excluded": int((~audit.get("visible_in_app", pd.Series(dtype=bool))).sum()) if not audit.empty else 0,
        "warning": warning,
    }


def _ts_label(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    ts = _coerce_utc_timestamp(value)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
