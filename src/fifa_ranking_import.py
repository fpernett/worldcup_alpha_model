from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.external_prior_import import EXTERNAL_STRENGTH_INPUT_COLUMNS
from src.team_names import DEFAULT_TEAM_ALIASES, load_team_name_aliases_df, normalize_team_name, team_name_key
from src.utils import coerce_float, read_csv_with_columns


FIFA_RANKING_SNAPSHOT_COLUMNS = ["team", "fifa_rank", "fifa_points", "source", "last_updated"]

FIFA_RANKING_MATCH_COLUMNS = [
    "team",
    "canonical_team",
    "fifa_rank",
    "fifa_points",
    "source",
    "last_updated",
    "match_status",
    "match_note",
]

TEAM_COLUMN_CANDIDATES = ["team", "country", "name", "ranked_team"]
RANK_COLUMN_CANDIDATES = ["rank", "fifa_rank", "position"]
POINTS_COLUMN_CANDIDATES = ["points", "fifa_points", "total_points", "pts"]

DEFAULT_FIFA_SOURCE = "FIFA ranking snapshot"
DEFAULT_FIFA_NOTES = "Imported from FIFA ranking snapshot; external overall-strength prior only."


def load_fifa_ranking_snapshot(path: str | Path) -> pd.DataFrame:
    """Load a local FIFA ranking snapshot with flexible source column names."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        out = pd.DataFrame(columns=FIFA_RANKING_SNAPSHOT_COLUMNS)
        out.attrs["warning"] = f"missing FIFA ranking snapshot: {resolved}"
        return out
    try:
        raw = pd.read_csv(resolved)
    except pd.errors.EmptyDataError:
        out = pd.DataFrame(columns=FIFA_RANKING_SNAPSHOT_COLUMNS)
        out.attrs["warning"] = "empty FIFA ranking snapshot"
        return out

    team_col = _find_column(raw, TEAM_COLUMN_CANDIDATES)
    if team_col is None:
        out = pd.DataFrame(columns=FIFA_RANKING_SNAPSHOT_COLUMNS)
        out.attrs["warning"] = "missing team column; accepted names: team, country, name, ranked_team"
        return out

    rank_col = _find_column(raw, RANK_COLUMN_CANDIDATES)
    points_col = _find_column(raw, POINTS_COLUMN_CANDIDATES)
    source_col = _find_column(raw, ["source"])
    last_updated_col = _find_column(raw, ["last_updated", "date"])

    out = pd.DataFrame(
        {
            "team": raw[team_col].map(_text_value),
            "fifa_rank": raw[rank_col].map(_numeric_or_na) if rank_col else pd.NA,
            "fifa_points": raw[points_col].map(_numeric_or_na) if points_col else pd.NA,
            "source": raw[source_col].map(_text_value) if source_col else "",
            "last_updated": raw[last_updated_col].map(_text_value) if last_updated_col else "",
        }
    )
    out = out.loc[out["team"].astype(str).str.strip() != ""].copy()
    out["source"] = out["source"].replace("", pd.NA).fillna(DEFAULT_FIFA_SOURCE)
    return out[FIFA_RANKING_SNAPSHOT_COLUMNS].reset_index(drop=True)


def match_fifa_rankings_to_required_teams(
    fifa_df: pd.DataFrame,
    required_teams_df: pd.DataFrame,
    aliases_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Match a normalized FIFA ranking snapshot to required project teams."""
    fifa = _normalise_fifa_df(fifa_df)
    required = _required_team_frame(required_teams_df)
    alias_map = _alias_map(aliases_df)
    required_teams = _required_canonical_teams(required, alias_map)

    if required_teams.empty:
        return pd.DataFrame(columns=FIFA_RANKING_MATCH_COLUMNS)

    fifa["_canonical_team"] = fifa["team"].map(lambda value: _canonical_with_aliases(value, alias_map))
    used_fifa_indexes: set[int] = set()
    rows: list[dict[str, Any]] = []

    for _, req in required_teams.iterrows():
        required_team = _text_value(req.get("team", ""))
        canonical_team = _text_value(req.get("canonical_team", "")) or _canonical_with_aliases(required_team, alias_map)
        matches = fifa.loc[fifa["_canonical_team"].astype(str).str.lower() == canonical_team.lower()].copy()

        if matches.empty:
            rows.append(
                _match_row(
                    required_team,
                    canonical_team,
                    pd.NA,
                    pd.NA,
                    "",
                    "",
                    "unmatched_required_team",
                    "Required team was not found in the FIFA ranking snapshot.",
                )
            )
            continue

        if _is_ambiguous(matches):
            used_fifa_indexes.update(int(index) for index in matches.index)
            rows.append(
                _match_row(
                    required_team,
                    canonical_team,
                    pd.NA,
                    pd.NA,
                    _joined_unique(matches, "source"),
                    _joined_unique(matches, "last_updated"),
                    "ambiguous",
                    "Multiple FIFA rows matched this required team with different rank/points values.",
                )
            )
            continue

        match = matches.iloc[0]
        used_fifa_indexes.add(int(matches.index[0]))
        rank = _numeric_or_na(match.get("fifa_rank"))
        points = _numeric_or_na(match.get("fifa_points"))
        status = "matched"
        note = f"Matched FIFA row: {_text_value(match.get('team', ''))}."
        if pd.isna(rank) or pd.isna(points):
            status = "missing_rank_or_points"
            note = "Matched FIFA row, but rank or points is missing."
        rows.append(
            _match_row(
                required_team,
                canonical_team,
                rank,
                points,
                match.get("source", ""),
                match.get("last_updated", ""),
                status,
                note,
            )
        )

    for index, row in fifa.iterrows():
        if int(index) in used_fifa_indexes:
            continue
        rows.append(
            _match_row(
                row.get("team", ""),
                row.get("_canonical_team", ""),
                row.get("fifa_rank", pd.NA),
                row.get("fifa_points", pd.NA),
                row.get("source", ""),
                row.get("last_updated", ""),
                "unused_fifa_row",
                "FIFA row did not match a required team.",
            )
        )

    return pd.DataFrame(rows, columns=FIFA_RANKING_MATCH_COLUMNS).reset_index(drop=True)


def build_external_strength_from_fifa_matches(match_df: pd.DataFrame | None) -> pd.DataFrame:
    """Convert matched FIFA rows into the external-strength importer input schema."""
    matches = match_df.copy() if match_df is not None else pd.DataFrame(columns=FIFA_RANKING_MATCH_COLUMNS)
    for col in FIFA_RANKING_MATCH_COLUMNS:
        if col not in matches.columns:
            matches[col] = pd.NA
    usable = matches.loc[matches["match_status"].isin(["matched", "missing_rank_or_points"])].copy()
    rows = []
    for _, row in usable.iterrows():
        rows.append(
            {
                "team": _text_value(row.get("canonical_team", "")) or _text_value(row.get("team", "")),
                "fifa_rank": _numeric_or_na(row.get("fifa_rank")),
                "fifa_points": _numeric_or_na(row.get("fifa_points")),
                "external_elo": pd.NA,
                "source": _text_value(row.get("source", "")) or DEFAULT_FIFA_SOURCE,
                "last_updated": _text_value(row.get("last_updated", "")),
                "notes": DEFAULT_FIFA_NOTES,
            }
        )
    return pd.DataFrame(rows, columns=EXTERNAL_STRENGTH_INPUT_COLUMNS)


def update_external_strength_template_from_fifa(
    template_df: pd.DataFrame | None,
    fifa_strength_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Fill blank template cells from matched FIFA rows while preserving manual entries."""
    template = template_df.copy() if template_df is not None else pd.DataFrame(columns=EXTERNAL_STRENGTH_INPUT_COLUMNS)
    fifa = fifa_strength_df.copy() if fifa_strength_df is not None else pd.DataFrame(columns=EXTERNAL_STRENGTH_INPUT_COLUMNS)
    for col in EXTERNAL_STRENGTH_INPUT_COLUMNS:
        if col not in template.columns:
            template[col] = pd.NA
        if col not in fifa.columns:
            fifa[col] = pd.NA
    if template.empty:
        return fifa[EXTERNAL_STRENGTH_INPUT_COLUMNS].reset_index(drop=True)

    out = template[EXTERNAL_STRENGTH_INPUT_COLUMNS].copy()
    fifa_by_team = {
        normalize_team_name(_text_value(row.get("team", ""))).lower(): row
        for _, row in fifa.iterrows()
        if _text_value(row.get("team", ""))
    }
    for idx, row in out.iterrows():
        team_key = normalize_team_name(_text_value(row.get("team", ""))).lower()
        if not team_key or team_key not in fifa_by_team:
            continue
        fifa_row = fifa_by_team[team_key]
        for col in ["fifa_rank", "fifa_points", "source", "last_updated", "notes"]:
            current = row.get(col)
            incoming = fifa_row.get(col)
            if _is_blank(current) and not _is_blank(incoming):
                out.at[idx, col] = incoming
    return out[EXTERNAL_STRENGTH_INPUT_COLUMNS].reset_index(drop=True)


def fifa_ranking_import_summary(match_df: pd.DataFrame | None, required_teams_count: int, fifa_rows_loaded: int) -> dict[str, Any]:
    matches = match_df.copy() if match_df is not None else pd.DataFrame(columns=FIFA_RANKING_MATCH_COLUMNS)
    if matches.empty:
        return {
            "required_teams": int(required_teams_count),
            "fifa_rows_loaded": int(fifa_rows_loaded),
            "matched_required_teams": 0,
            "missing_required_teams": int(required_teams_count),
            "unmatched_fifa_rows": 0,
            "ambiguous_matches": 0,
            "teams_missing_rank_or_points": 0,
        }
    statuses = matches["match_status"].astype(str)
    matched_required = statuses.isin(["matched", "missing_rank_or_points"]).sum()
    return {
        "required_teams": int(required_teams_count),
        "fifa_rows_loaded": int(fifa_rows_loaded),
        "matched_required_teams": int(matched_required),
        "missing_required_teams": int((statuses == "unmatched_required_team").sum() + (statuses == "ambiguous").sum()),
        "unmatched_fifa_rows": int((statuses == "unused_fifa_row").sum()),
        "ambiguous_matches": int((statuses == "ambiguous").sum()),
        "teams_missing_rank_or_points": int((statuses == "missing_rank_or_points").sum()),
    }


def load_fifa_import_report(path: str | Path | None = None) -> pd.DataFrame:
    """Load a saved FIFA import report, or the latest dated report if no path is supplied."""
    if path is not None:
        return read_csv_with_columns(_resolve_path(path), FIFA_RANKING_MATCH_COLUMNS)
    reports_dir = DATA_DIR.parent / "reports"
    candidates = sorted(reports_dir.glob("fifa_ranking_import_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        return pd.DataFrame(columns=FIFA_RANKING_MATCH_COLUMNS)
    return read_csv_with_columns(candidates[0], FIFA_RANKING_MATCH_COLUMNS)


def fifa_ranking_import_dashboard_status(
    snapshot_path: str | Path = DATA_DIR / "raw" / "fifa_rankings_snapshot.csv",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    snapshot = load_fifa_ranking_snapshot(snapshot_path)
    report = load_fifa_import_report(report_path)
    summary = fifa_ranking_import_summary(report, 0, len(snapshot))
    last_updated = ""
    if not report.empty and "last_updated" in report.columns:
        dates = pd.to_datetime(report["last_updated"], errors="coerce")
        if dates.notna().any():
            last_updated = dates.max().date().isoformat()
    if not last_updated and not snapshot.empty:
        dates = pd.to_datetime(snapshot["last_updated"], errors="coerce")
        if dates.notna().any():
            last_updated = dates.max().date().isoformat()
    resolved_snapshot = _resolve_path(snapshot_path)
    return {
        "last_snapshot_file": str(resolved_snapshot.relative_to(DATA_DIR.parent)) if resolved_snapshot.exists() else "missing",
        "fifa_rows_loaded": int(len(snapshot)),
        "matched_teams": int(summary["matched_required_teams"]),
        "missing_teams": int(summary["missing_required_teams"]),
        "unmatched_rows": int(summary["unmatched_fifa_rows"]),
        "last_updated_date": last_updated,
    }


def _normalise_fifa_df(fifa_df: pd.DataFrame | None) -> pd.DataFrame:
    fifa = fifa_df.copy() if fifa_df is not None else pd.DataFrame(columns=FIFA_RANKING_SNAPSHOT_COLUMNS)
    for col in FIFA_RANKING_SNAPSHOT_COLUMNS:
        if col not in fifa.columns:
            fifa[col] = pd.NA
    out = fifa[FIFA_RANKING_SNAPSHOT_COLUMNS].copy()
    out["team"] = out["team"].map(_text_value)
    out = out.loc[out["team"] != ""].copy()
    out["fifa_rank"] = out["fifa_rank"].map(_numeric_or_na)
    out["fifa_points"] = out["fifa_points"].map(_numeric_or_na)
    out["source"] = out["source"].map(_text_value).replace("", DEFAULT_FIFA_SOURCE)
    out["last_updated"] = out["last_updated"].map(_text_value)
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
        raw_team = _text_value(row.get("team", ""))
        canonical = _text_value(row.get("canonical_team", "")) or raw_team
        canonical = _canonical_with_aliases(canonical, alias_map)
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


def _is_ambiguous(matches: pd.DataFrame) -> bool:
    if len(matches) <= 1:
        return False
    comparable = matches[["fifa_rank", "fifa_points"]].copy()
    comparable["fifa_rank"] = comparable["fifa_rank"].map(_numeric_or_na)
    comparable["fifa_points"] = comparable["fifa_points"].map(_numeric_or_na)
    return len(comparable.drop_duplicates()) > 1


def _match_row(
    team: Any,
    canonical_team: Any,
    fifa_rank: Any,
    fifa_points: Any,
    source: Any,
    last_updated: Any,
    match_status: str,
    match_note: str,
) -> dict[str, Any]:
    return {
        "team": _text_value(team),
        "canonical_team": _text_value(canonical_team),
        "fifa_rank": _numeric_or_na(fifa_rank),
        "fifa_points": _numeric_or_na(fifa_points),
        "source": _text_value(source),
        "last_updated": _text_value(last_updated),
        "match_status": match_status,
        "match_note": match_note,
    }


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


def _is_blank(value: Any) -> bool:
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


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
