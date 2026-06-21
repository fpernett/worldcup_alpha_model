from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.fifa_ranking_import import (
    DEFAULT_FIFA_SOURCE,
    POINTS_COLUMN_CANDIDATES,
    RANK_COLUMN_CANDIDATES,
    TEAM_COLUMN_CANDIDATES,
)
from src.team_names import DEFAULT_TEAM_ALIASES, load_team_name_aliases_df, normalize_team_name, team_name_key
from src.utils import coerce_float, read_csv_with_columns


FIFA_SNAPSHOT_VALIDATION_COLUMNS = [
    "team",
    "canonical_team",
    "snapshot_team",
    "match_status",
    "fifa_rank",
    "fifa_points",
    "source",
    "last_updated",
    "issue",
    "recommended_action",
]

FIFA_SNAPSHOT_MISSING_TEMPLATE_COLUMNS = ["team", "fifa_rank", "fifa_points", "source", "last_updated", "notes"]

READY_MATCH_STATUSES = {
    "matched",
    "matched_missing_rank",
    "matched_missing_points",
    "matched_missing_rank_and_points",
}


def validate_fifa_snapshot(
    fifa_snapshot_df: pd.DataFrame,
    required_teams_df: pd.DataFrame,
    aliases_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate whether a FIFA ranking snapshot is ready for external-prior import."""
    snapshot = _normalise_snapshot_frame(fifa_snapshot_df)
    required = _required_team_frame(required_teams_df)
    alias_map = _alias_map(aliases_df)
    required_teams = _required_canonical_teams(required, alias_map)

    rows: list[dict[str, Any]] = []
    if not snapshot.empty:
        snapshot["_canonical_team"] = snapshot["team"].map(lambda value: _canonical_with_aliases(value, alias_map))
    else:
        snapshot["_canonical_team"] = pd.Series(dtype="object")
    used_snapshot_indexes: set[int] = set()

    for _, required_row in required_teams.iterrows():
        canonical_team = _text_value(required_row.get("canonical_team", ""))
        matches = snapshot.loc[snapshot["_canonical_team"].astype(str).str.lower() == canonical_team.lower()].copy()
        if matches.empty:
            rows.append(
                _validation_row(
                    team=canonical_team,
                    canonical_team=canonical_team,
                    snapshot_team="",
                    match_status="missing_from_snapshot",
                    fifa_rank=pd.NA,
                    fifa_points=pd.NA,
                    source="",
                    last_updated="",
                    issue="Required team is missing from the FIFA snapshot.",
                    recommended_action="add team with FIFA rank and FIFA points",
                )
            )
            continue

        if _ambiguous_matches(matches):
            used_snapshot_indexes.update(int(index) for index in matches.index)
            rows.append(
                _validation_row(
                    team=canonical_team,
                    canonical_team=canonical_team,
                    snapshot_team=_joined_unique(matches, "team"),
                    match_status="ambiguous_match",
                    fifa_rank=pd.NA,
                    fifa_points=pd.NA,
                    source=_joined_unique(matches, "source"),
                    last_updated=_joined_unique(matches, "last_updated"),
                    issue="Multiple snapshot rows matched this required team with different rank or points values.",
                    recommended_action="deduplicate or fix aliases before import",
                )
            )
            continue

        match = matches.iloc[0]
        used_snapshot_indexes.add(int(matches.index[0]))
        rank = _numeric_or_na(match.get("fifa_rank"))
        points = _numeric_or_na(match.get("fifa_points"))
        status, issue, action = _rank_points_status(rank, points)
        rows.append(
            _validation_row(
                team=canonical_team,
                canonical_team=canonical_team,
                snapshot_team=match.get("team", ""),
                match_status=status,
                fifa_rank=rank,
                fifa_points=points,
                source=match.get("source", ""),
                last_updated=match.get("last_updated", ""),
                issue=issue,
                recommended_action=action,
            )
        )

    for index, row in snapshot.iterrows():
        if int(index) in used_snapshot_indexes:
            continue
        rows.append(
            _validation_row(
                team="",
                canonical_team=_text_value(row.get("_canonical_team", "")),
                snapshot_team=row.get("team", ""),
                match_status="unused_fifa_row",
                fifa_rank=row.get("fifa_rank", pd.NA),
                fifa_points=row.get("fifa_points", pd.NA),
                source=row.get("source", ""),
                last_updated=row.get("last_updated", ""),
                issue="Snapshot row does not match a required team.",
                recommended_action="ignore or add alias if this should match a required team",
            )
        )

    validation = pd.DataFrame(rows, columns=FIFA_SNAPSHOT_VALIDATION_COLUMNS)
    summary = summarize_fifa_snapshot_validation(validation, required_count=len(required_teams), snapshot_rows=len(snapshot))
    return validation, summary


def summarize_fifa_snapshot_validation(
    validation_df: pd.DataFrame | None,
    required_count: int | None = None,
    snapshot_rows: int | None = None,
) -> dict[str, Any]:
    validation = validation_df.copy() if validation_df is not None else pd.DataFrame(columns=FIFA_SNAPSHOT_VALIDATION_COLUMNS)
    for col in FIFA_SNAPSHOT_VALIDATION_COLUMNS:
        if col not in validation.columns:
            validation[col] = pd.NA
    statuses = validation["match_status"].astype(str) if not validation.empty else pd.Series(dtype="object")
    required_rows = validation.loc[statuses != "unused_fifa_row"].copy()
    missing_required = int((required_rows["match_status"].astype(str) == "missing_from_snapshot").sum())
    missing_rank = int(required_rows["match_status"].astype(str).isin(["matched_missing_rank", "matched_missing_rank_and_points"]).sum())
    missing_points = int(required_rows["match_status"].astype(str).isin(["matched_missing_points", "matched_missing_rank_and_points"]).sum())
    ambiguous = int((required_rows["match_status"].astype(str) == "ambiguous_match").sum())
    summary = {
        "required_teams": int(required_count if required_count is not None else len(required_rows)),
        "snapshot_rows": int(snapshot_rows if snapshot_rows is not None else _infer_snapshot_rows(validation)),
        "matched_required_teams": int(required_rows["match_status"].astype(str).isin(READY_MATCH_STATUSES).sum()),
        "missing_required_teams": missing_required,
        "matched_missing_rank": missing_rank,
        "matched_missing_points": missing_points,
        "ambiguous_matches": ambiguous,
        "unused_fifa_rows": int((statuses == "unused_fifa_row").sum()) if not validation.empty else 0,
    }
    summary["ready_for_import"] = bool(
        summary["missing_required_teams"] == 0
        and summary["matched_missing_rank"] == 0
        and summary["matched_missing_points"] == 0
        and summary["ambiguous_matches"] == 0
    )
    return summary


def build_missing_fifa_snapshot_template(validation_df: pd.DataFrame | None) -> pd.DataFrame:
    """Build a simple CSV template for missing or incomplete required teams."""
    validation = validation_df.copy() if validation_df is not None else pd.DataFrame(columns=FIFA_SNAPSHOT_VALIDATION_COLUMNS)
    for col in FIFA_SNAPSHOT_VALIDATION_COLUMNS:
        if col not in validation.columns:
            validation[col] = pd.NA
    incomplete_statuses = {
        "missing_from_snapshot",
        "matched_missing_rank",
        "matched_missing_points",
        "matched_missing_rank_and_points",
        "ambiguous_match",
    }
    rows = []
    for _, row in validation.loc[validation["match_status"].isin(incomplete_statuses)].iterrows():
        status = _text_value(row.get("match_status", ""))
        rows.append(
            {
                "team": _text_value(row.get("canonical_team", "")) or _text_value(row.get("team", "")),
                "fifa_rank": row.get("fifa_rank", pd.NA) if status in {"matched_missing_points"} else pd.NA,
                "fifa_points": row.get("fifa_points", pd.NA) if status in {"matched_missing_rank"} else pd.NA,
                "source": _text_value(row.get("source", "")),
                "last_updated": _text_value(row.get("last_updated", "")),
                "notes": _template_note(status),
            }
        )
    return pd.DataFrame(rows, columns=FIFA_SNAPSHOT_MISSING_TEMPLATE_COLUMNS)


def load_latest_fifa_snapshot_validation_report(path: str | Path | None = None) -> tuple[pd.DataFrame, Path | None]:
    if path is not None:
        resolved = _resolve_path(path)
        return read_csv_with_columns(resolved, FIFA_SNAPSHOT_VALIDATION_COLUMNS), resolved if resolved.exists() else None
    reports_dir = DATA_DIR.parent / "reports"
    candidates = sorted(reports_dir.glob("fifa_snapshot_validation_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        return pd.DataFrame(columns=FIFA_SNAPSHOT_VALIDATION_COLUMNS), None
    return read_csv_with_columns(candidates[0], FIFA_SNAPSHOT_VALIDATION_COLUMNS), candidates[0]


def fifa_snapshot_validation_dashboard_status(report_path: str | Path | None = None) -> dict[str, Any]:
    report, resolved = load_latest_fifa_snapshot_validation_report(report_path)
    summary = summarize_fifa_snapshot_validation(report)
    missing_rank_or_points = 0
    if not report.empty and "match_status" in report.columns:
        missing_rank_or_points = int(
            report["match_status"]
            .astype(str)
            .isin(["matched_missing_rank", "matched_missing_points", "matched_missing_rank_and_points"])
            .sum()
        )
    return {
        "required_teams": summary["required_teams"],
        "matched_teams": summary["matched_required_teams"],
        "missing_teams": summary["missing_required_teams"],
        "missing_rank_or_points": missing_rank_or_points,
        "ready_for_import": "yes" if summary["ready_for_import"] else "no",
        "latest_validation_report": _display_path(resolved) if resolved else "none",
    }


def _normalise_snapshot_frame(fifa_snapshot_df: pd.DataFrame | None) -> pd.DataFrame:
    raw = fifa_snapshot_df.copy() if fifa_snapshot_df is not None else pd.DataFrame()
    if raw.empty:
        return pd.DataFrame(columns=["team", "fifa_rank", "fifa_points", "source", "last_updated"])
    team_col = _find_column(raw, TEAM_COLUMN_CANDIDATES)
    if team_col is None:
        return pd.DataFrame(columns=["team", "fifa_rank", "fifa_points", "source", "last_updated"])
    rank_col = _find_column(raw, RANK_COLUMN_CANDIDATES)
    points_col = _find_column(raw, POINTS_COLUMN_CANDIDATES)
    source_col = _find_column(raw, ["source"])
    last_updated_col = _find_column(raw, ["last_updated", "date"])
    out = pd.DataFrame(
        {
            "team": raw[team_col].map(_text_value),
            "fifa_rank": raw[rank_col].map(_numeric_or_na) if rank_col else pd.NA,
            "fifa_points": raw[points_col].map(_numeric_or_na) if points_col else pd.NA,
            "source": raw[source_col].map(_text_value) if source_col else DEFAULT_FIFA_SOURCE,
            "last_updated": raw[last_updated_col].map(_text_value) if last_updated_col else "",
        }
    )
    out = out.loc[out["team"] != ""].copy()
    out["source"] = out["source"].replace("", pd.NA).fillna(DEFAULT_FIFA_SOURCE)
    return out.reset_index(drop=True)


def _required_team_frame(required_teams_df: pd.DataFrame | None) -> pd.DataFrame:
    required = required_teams_df.copy() if required_teams_df is not None else pd.DataFrame()
    if required.empty:
        return pd.DataFrame(columns=["team", "canonical_team", "required_for_model"])
    if "team" not in required.columns:
        required["team"] = pd.NA
    if "canonical_team" not in required.columns:
        required["canonical_team"] = required["team"].map(normalize_team_name)
    if "required_for_model" in required.columns:
        required = required.loc[required["required_for_model"].astype(bool)].copy()
    return required.reset_index(drop=True)


def _required_canonical_teams(required: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in required.iterrows():
        raw = _text_value(row.get("canonical_team", "")) or _text_value(row.get("team", ""))
        canonical = _canonical_with_aliases(raw, alias_map)
        if not canonical:
            continue
        key = canonical.lower()
        if key in seen:
            continue
        rows.append({"team": canonical, "canonical_team": canonical})
        seen.add(key)
    return pd.DataFrame(rows, columns=["team", "canonical_team"])


def _alias_map(aliases_df: pd.DataFrame | None) -> dict[str, str]:
    aliases = {team_name_key(alias): canonical for alias, canonical in DEFAULT_TEAM_ALIASES.items()}
    source = aliases_df.copy() if aliases_df is not None else load_team_name_aliases_df(include_defaults=True)
    for _, row in source.iterrows():
        alias = _text_value(row.get("alias", ""))
        canonical = _text_value(row.get("canonical", ""))
        if alias and canonical:
            aliases[team_name_key(alias)] = canonical
    return aliases


def _canonical_with_aliases(value: Any, aliases: dict[str, str]) -> str:
    raw = _text_value(value)
    if not raw:
        return ""
    return aliases.get(team_name_key(raw), normalize_team_name(raw))


def _rank_points_status(rank: Any, points: Any) -> tuple[str, str, str]:
    rank_missing = pd.isna(rank)
    points_missing = pd.isna(points)
    if rank_missing and points_missing:
        return (
            "matched_missing_rank_and_points",
            "Snapshot row matched, but FIFA rank and FIFA points are missing.",
            "fill FIFA rank and FIFA points before import",
        )
    if rank_missing:
        return ("matched_missing_rank", "Snapshot row matched, but FIFA rank is missing.", "fill FIFA rank before import")
    if points_missing:
        return ("matched_missing_points", "Snapshot row matched, but FIFA points are missing.", "fill FIFA points before import")
    return ("matched", "", "none")


def _ambiguous_matches(matches: pd.DataFrame) -> bool:
    if len(matches) <= 1:
        return False
    comparable = matches[["fifa_rank", "fifa_points"]].copy()
    comparable["fifa_rank"] = comparable["fifa_rank"].map(_numeric_or_na)
    comparable["fifa_points"] = comparable["fifa_points"].map(_numeric_or_na)
    return len(comparable.drop_duplicates()) > 1


def _validation_row(
    team: Any,
    canonical_team: Any,
    snapshot_team: Any,
    match_status: str,
    fifa_rank: Any,
    fifa_points: Any,
    source: Any,
    last_updated: Any,
    issue: str,
    recommended_action: str,
) -> dict[str, Any]:
    return {
        "team": _text_value(team),
        "canonical_team": _text_value(canonical_team),
        "snapshot_team": _text_value(snapshot_team),
        "match_status": match_status,
        "fifa_rank": _numeric_or_na(fifa_rank),
        "fifa_points": _numeric_or_na(fifa_points),
        "source": _text_value(source),
        "last_updated": _text_value(last_updated),
        "issue": issue,
        "recommended_action": recommended_action,
    }


def _template_note(status: str) -> str:
    if status == "missing_from_snapshot":
        return "Missing from FIFA snapshot; fill rank and points from source."
    if status == "ambiguous_match":
        return "Ambiguous FIFA snapshot match; fix duplicate rows or alias mapping."
    return "Incomplete FIFA snapshot row; fill missing values from source."


def _infer_snapshot_rows(validation: pd.DataFrame) -> int:
    if validation.empty:
        return 0
    required_with_snapshot = validation["match_status"].astype(str).isin(READY_MATCH_STATUSES | {"ambiguous_match"}).sum()
    unused = (validation["match_status"].astype(str) == "unused_fifa_row").sum()
    return int(required_with_snapshot + unused)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    columns_by_key = {_column_key(col): col for col in df.columns}
    for candidate in candidates:
        key = _column_key(candidate)
        if key in columns_by_key:
            return columns_by_key[key]
    return None


def _column_key(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _joined_unique(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return ""
    values = [_text_value(value) for value in df[col].dropna()]
    return "; ".join(dict.fromkeys([value for value in values if value]))


def _numeric_or_na(value: Any) -> float | pd._libs.missing.NAType:
    numeric = coerce_float(value, float("nan"))
    if pd.isna(numeric):
        return pd.NA
    return float(numeric)


def _text_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return DATA_DIR.parent / resolved


def _display_path(path: Path | None) -> str:
    if path is None:
        return "none"
    try:
        return str(path.relative_to(DATA_DIR.parent))
    except ValueError:
        return str(path)
