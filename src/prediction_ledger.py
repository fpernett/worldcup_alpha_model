from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import classify_actual_result, load_completed_matches_for_backtest
from src.config import DATA_DIR
from src.model import ModelConfig, run_match_model, score_matrix
from src.model_policy import get_current_model_policy
from src.ratings import TEAM_RATING_COLUMNS, get_team_ratings
from src.utils import coerce_bool, coerce_float, read_csv_with_columns, utc_now_iso
from src.weather import load_venues


PREDICTION_LEDGER_PATH = DATA_DIR / "prediction_ledger.csv"
RESULTS_LEDGER_PATH = DATA_DIR / "results_ledger.csv"

PREDICTION_LEDGER_COLUMNS = [
    "prediction_id",
    "snapshot_utc",
    "match_id",
    "kickoff_utc",
    "competition",
    "group",
    "home",
    "away",
    "venue",
    "primary_model_mode",
    "model_version",
    "parameter_set_id",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "home_xg",
    "away_xg",
    "over_0_5_prob",
    "over_1_5_prob",
    "over_2_5_prob",
    "over_3_5_prob",
    "btts_yes_prob",
    "source_status",
    "market_snapshot_available",
    "prediction_before_kickoff",
    "notes",
]

RESULTS_LEDGER_COLUMNS = [
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
    "over_0_5_actual",
    "over_1_5_actual",
    "over_2_5_actual",
    "over_3_5_actual",
    "result_source",
    "last_updated",
]

DEFAULT_AUTO_SNAPSHOT_MIN_INTERVAL_MINUTES = 60
DEFAULT_RESULT_READY_DELAY_HOURS = 4.0


def load_prediction_ledger(path: str | Path = PREDICTION_LEDGER_PATH) -> pd.DataFrame:
    return read_csv_with_columns(Path(path), PREDICTION_LEDGER_COLUMNS)[PREDICTION_LEDGER_COLUMNS].copy()


def load_results_ledger(path: str | Path = RESULTS_LEDGER_PATH) -> pd.DataFrame:
    return read_csv_with_columns(Path(path), RESULTS_LEDGER_COLUMNS)[RESULTS_LEDGER_COLUMNS].copy()


def append_prediction_snapshots(
    snapshots_df: pd.DataFrame | None,
    path: str | Path = PREDICTION_LEDGER_PATH,
) -> pd.DataFrame:
    """Append prediction snapshots without overwriting prior ledger rows."""
    path = Path(path)
    new_rows = _ensure_columns(snapshots_df, PREDICTION_LEDGER_COLUMNS)
    existing = load_prediction_ledger(path)
    combined = pd.concat([existing, new_rows[PREDICTION_LEDGER_COLUMNS]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined[PREDICTION_LEDGER_COLUMNS].copy()


def snapshot_predictions_for_fixtures(
    fixtures_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None = None,
    venues_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    cfg: ModelConfig | None = None,
    model_version: str = "baseline_external_calibrated_v1",
    parameter_set_id: str = "baseline_current",
    primary_model_mode: str = "baseline_manual",
    notes: str = "",
    path: str | Path = PREDICTION_LEDGER_PATH,
    append: bool = True,
) -> pd.DataFrame:
    """Snapshot current pre-match predictions for fixture rows.

    The function is append-only by default. Re-snapshotting the same match and
    model creates a new row with a new `snapshot_utc`.
    """
    snapshots, _diagnostics = snapshot_predictions_for_fixtures_with_diagnostics(
        fixtures_df,
        team_ratings_df=team_ratings_df,
        venues_df=venues_df,
        market_odds_df=market_odds_df,
        cfg=cfg,
        model_version=model_version,
        parameter_set_id=parameter_set_id,
        primary_model_mode=primary_model_mode,
        notes=notes,
        path=path,
        append=append,
        include_past=True,
    )
    return snapshots


def snapshot_predictions_for_fixtures_with_diagnostics(
    fixtures_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None = None,
    venues_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    cfg: ModelConfig | None = None,
    model_version: str = "baseline_external_calibrated_v1",
    parameter_set_id: str = "baseline_current",
    primary_model_mode: str = "baseline_manual",
    notes: str = "",
    path: str | Path = PREDICTION_LEDGER_PATH,
    append: bool = True,
    include_past: bool = False,
    selected_match_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Snapshot predictions with explicit write/skip diagnostics.

    `include_past=False` skips snapshots at or after kickoff so dashboard saves
    are valid for honest post-mortem by default. Passing `include_past=True`
    still writes the row but marks `prediction_before_kickoff=False`.
    """
    fixtures = _normalise_fixtures(fixtures_df)
    fixtures_available = int(len(fixtures))
    if selected_match_ids is not None:
        wanted = {str(match_id) for match_id in selected_match_ids if str(match_id).strip()}
        fixtures = fixtures.loc[fixtures["match_id"].astype(str).isin(wanted)].copy() if wanted else fixtures.iloc[0:0].copy()
    fixtures_selected = int(len(fixtures))
    path = Path(path)
    output_path = _display_path(path)
    skip_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    def diagnostics(rows_written: int, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        rows = rows or skip_rows
        skipped = int(len(rows))
        return {
            "fixtures_available": fixtures_available,
            "fixtures_selected": fixtures_selected,
            "fixtures_attempted": int(fixtures_selected - sum(1 for row in rows if row.get("reason_code") in {"past_kickoff", "missing_match_id"})),
            "predictions_written": int(rows_written),
            "predictions_skipped": skipped,
            "skipped_past_kickoff": sum(1 for row in rows if row.get("reason_code") == "past_kickoff"),
            "skipped_missing_model_result": sum(1 for row in rows if row.get("reason_code") == "missing_model_result"),
            "skipped_missing_match_id": sum(1 for row in rows if row.get("reason_code") == "missing_match_id"),
            "skipped_duplicate_policy": 0,
            "output_path": output_path,
            "warnings": warnings.copy(),
            "skip_reasons": rows.copy(),
        }

    if fixtures.empty:
        warning = "No fixtures were selected." if fixtures_available else "No fixtures are available for the requested snapshot."
        warnings.append(warning)
        return pd.DataFrame(columns=PREDICTION_LEDGER_COLUMNS), diagnostics(0)

    snapshot_utc = utc_now_iso()
    teams = team_ratings_df.copy() if team_ratings_df is not None else _load_primary_ratings()
    venues = venues_df.copy() if venues_df is not None else load_venues()
    odds = market_odds_df.copy() if market_odds_df is not None else pd.DataFrame()
    model_cfg = cfg or ModelConfig()
    policy = get_current_model_policy()
    rows: list[dict[str, Any]] = []

    for _, fixture in fixtures.iterrows():
        match_id_raw = str(fixture.get("match_id", "") or "").strip()
        if not match_id_raw:
            skip_rows.append(_skip_row(fixture, "missing_match_id", "Fixture has no match_id, so it was not written to the prediction ledger."))
            continue
        kickoff = _kickoff_utc(fixture)
        before_kickoff = _before_kickoff(snapshot_utc, kickoff)
        if not include_past and not before_kickoff:
            skip_rows.append(_skip_row(fixture, "past_kickoff", "Fixture kickoff is not after the snapshot time."))
            continue
        try:
            result = run_match_model(fixture, teams, venues, odds, model_cfg, model_mode=primary_model_mode)
        except TypeError:
            try:
                result = run_match_model(fixture, teams, venues, odds, model_cfg)
            except Exception as exc:
                skip_rows.append(_skip_row(fixture, "missing_model_result", f"Model result could not be generated: {exc}"))
                continue
        except Exception as exc:
            skip_rows.append(_skip_row(fixture, "missing_model_result", f"Model result could not be generated: {exc}"))
            continue
        if not isinstance(result, dict) or not result.get("probs"):
            skip_rows.append(_skip_row(fixture, "missing_model_result", "Model result was missing probabilities."))
            continue
        probs = result.get("probs", {})
        matrix = result.get("score_matrix")
        over_0_5 = _over_probability(matrix, 0)
        over_1_5 = _over_probability(matrix, 1)
        match_id = match_id_raw
        prediction_id = _prediction_id(snapshot_utc, match_id, model_version, parameter_set_id)
        rows.append(
            {
                "prediction_id": prediction_id,
                "snapshot_utc": snapshot_utc,
                "match_id": match_id,
                "kickoff_utc": kickoff,
                "competition": fixture.get("competition", ""),
                "group": fixture.get("group", ""),
                "home": fixture.get("home", ""),
                "away": fixture.get("away", ""),
                "venue": fixture.get("venue", ""),
                "primary_model_mode": primary_model_mode,
                "model_version": model_version,
                "parameter_set_id": parameter_set_id,
                "home_win_prob": float(probs.get("home_win", 0.0)),
                "draw_prob": float(probs.get("draw", 0.0)),
                "away_win_prob": float(probs.get("away_win", 0.0)),
                "home_xg": float(result.get("hxg", 0.0)),
                "away_xg": float(result.get("axg", 0.0)),
                "over_0_5_prob": over_0_5,
                "over_1_5_prob": over_1_5,
                "over_2_5_prob": float(probs.get("over_2_5", 0.0)),
                "over_3_5_prob": float(probs.get("over_3_5", 0.0)),
                "btts_yes_prob": float(probs.get("btts_yes", 0.0)),
                "source_status": f"policy={policy.get('primary_model_mode', primary_model_mode)}; offline_csv_safe",
                "market_snapshot_available": bool(_market_snapshot_available(odds, match_id)),
                "prediction_before_kickoff": bool(before_kickoff),
                "notes": notes,
            }
        )

    snapshots = pd.DataFrame(rows, columns=PREDICTION_LEDGER_COLUMNS)
    if append:
        append_prediction_snapshots(snapshots, path=path)
    return snapshots, diagnostics(len(snapshots) if append else 0)


def dashboard_parameter_set_id(cfg: ModelConfig | None, baseline_id: str = "baseline_current") -> str:
    """Create a stable parameter id for dashboard snapshots."""
    model_cfg = cfg or ModelConfig()
    default_cfg = ModelConfig()
    if _config_payload(model_cfg) == _config_payload(default_cfg):
        return baseline_id
    raw = "|".join(f"{key}={value}" for key, value in sorted(_config_payload(model_cfg).items()))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"dashboard_custom_{digest}"


def snapshot_selected_match_if_needed(
    fixture_row: pd.Series | dict[str, Any] | pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None = None,
    venues_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    cfg: ModelConfig | None = None,
    model_version: str = "baseline_external_calibrated_v1",
    parameter_set_id: str | None = None,
    primary_model_mode: str = "baseline_manual",
    notes: str = "Auto-saved by dashboard selected-match analysis",
    path: str | Path = PREDICTION_LEDGER_PATH,
    min_interval_minutes: int = DEFAULT_AUTO_SNAPSHOT_MIN_INTERVAL_MINUTES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append a pre-kickoff selected-match snapshot unless a fresh one exists."""
    fixture = _single_fixture_frame(fixture_row)
    parameter_id = parameter_set_id or dashboard_parameter_set_id(cfg)
    match_id = _first_match_id(fixture)
    duplicate = _fresh_existing_prediction(
        match_id,
        model_version=model_version,
        parameter_set_id=parameter_id,
        primary_model_mode=primary_model_mode,
        path=path,
        min_interval_minutes=min_interval_minutes,
    )
    if duplicate:
        diagnostics = {
            "fixtures_available": int(len(fixture)),
            "fixtures_selected": int(len(fixture)),
            "fixtures_attempted": 0,
            "predictions_written": 0,
            "predictions_skipped": 1,
            "skipped_past_kickoff": 0,
            "skipped_missing_model_result": 0,
            "skipped_missing_match_id": 0,
            "skipped_duplicate_policy": 1,
            "output_path": _display_path(Path(path)),
            "warnings": [],
            "skip_reasons": [
                {
                    "match_id": match_id,
                    "reason_code": "duplicate_policy",
                    "reason": f"A fresh pre-kickoff snapshot already exists within {int(min_interval_minutes)} minute(s).",
                }
            ],
        }
        return pd.DataFrame(columns=PREDICTION_LEDGER_COLUMNS), diagnostics
    return snapshot_predictions_for_fixtures_with_diagnostics(
        fixture,
        team_ratings_df=team_ratings_df,
        venues_df=venues_df,
        market_odds_df=market_odds_df,
        cfg=cfg,
        model_version=model_version,
        parameter_set_id=parameter_id,
        primary_model_mode=primary_model_mode,
        notes=notes,
        path=path,
        append=True,
        include_past=False,
        selected_match_ids=[match_id] if match_id else None,
    )


def import_completed_result_for_fixture_if_ready(
    fixture_row: pd.Series | dict[str, Any],
    completed_matches_df: pd.DataFrame | None = None,
    path: str | Path = RESULTS_LEDGER_PATH,
    now_utc: str | pd.Timestamp | None = None,
    result_ready_delay_hours: float = DEFAULT_RESULT_READY_DELAY_HOURS,
    result_source: str = "auto_dashboard_completed_matches",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Import one selected fixture result after a conservative post-game delay.

    The app waits four hours after kickoff by default: roughly three hours for
    regulation, extra time, penalties, and stoppage, plus one hour before trying
    to pull completed-result data.
    """
    fixture = pd.Series(fixture_row).copy()
    path = Path(path)
    existing = load_results_ledger(path)
    match_id = str(fixture.get("match_id", "") or "").strip()
    kickoff = pd.to_datetime(_kickoff_utc(fixture), errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if not match_id:
        return existing, _result_import_diagnostics("skipped_missing_match_id", match_id, "", 0, "Fixture has no match_id.")
    if pd.isna(kickoff):
        return existing, _result_import_diagnostics("skipped_invalid_kickoff", match_id, "", 0, "Fixture kickoff could not be parsed.")
    ready_at = kickoff + pd.Timedelta(hours=float(result_ready_delay_hours))
    if now < ready_at:
        return existing, _result_import_diagnostics(
            "not_ready",
            match_id,
            ready_at.isoformat(),
            0,
            "Result import waits until kickoff plus the post-game delay.",
        )
    if not existing.empty and "match_id" in existing.columns and (existing["match_id"].astype(str) == match_id).any():
        return existing, _result_import_diagnostics("already_imported", match_id, ready_at.isoformat(), 0, "Result already exists in the ledger.")

    completed = completed_matches_df.copy() if completed_matches_df is not None else _completed_candidates_for_fixture(fixture)
    candidate = _completed_result_for_fixture(fixture, completed)
    if candidate.empty:
        return existing, _result_import_diagnostics(
            "result_not_found",
            match_id,
            ready_at.isoformat(),
            0,
            "No completed-result row matched this fixture yet.",
        )

    result_row = candidate.iloc[0].copy()
    result_row["match_id"] = match_id
    for col in ["date_utc", "competition", "home", "away"]:
        if col in fixture.index:
            result_row[col] = fixture.get(col, result_row.get(col, ""))
    combined = import_completed_results(pd.DataFrame([result_row]), path=path, result_source=result_source)
    return combined, _result_import_diagnostics("imported", match_id, ready_at.isoformat(), 1, "Completed result imported.")


def import_completed_results(
    completed_matches_df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    competition: str | None = None,
    path: str | Path = RESULTS_LEDGER_PATH,
    result_source: str = "completed_matches",
) -> pd.DataFrame:
    """Create or update result ledger rows from completed match data."""
    matches = completed_matches_df.copy() if completed_matches_df is not None else load_completed_matches_for_backtest(start_date, end_date)
    if matches is None or matches.empty:
        out = load_results_ledger(path)
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
        return out
    matches = matches.copy()
    if competition and "competition" in matches.columns:
        needle = str(competition).strip().lower()
        matches = matches.loc[matches["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    rows = []
    now = utc_now_iso()
    for _, match in matches.iterrows():
        home_goals = coerce_float(match.get("home_goals"), 0.0)
        away_goals = coerce_float(match.get("away_goals"), 0.0)
        total = home_goals + away_goals
        rows.append(
            {
                "match_id": str(match.get("match_id", "") or _fallback_match_id(match)),
                "date_utc": match.get("date_utc", ""),
                "competition": match.get("competition", ""),
                "home": match.get("home", ""),
                "away": match.get("away", ""),
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                "actual_result": classify_actual_result(home_goals, away_goals),
                "total_goals": int(total),
                "btts_actual": int(home_goals > 0 and away_goals > 0),
                "over_0_5_actual": int(total > 0.5),
                "over_1_5_actual": int(total > 1.5),
                "over_2_5_actual": int(total > 2.5),
                "over_3_5_actual": int(total > 3.5),
                "result_source": result_source,
                "last_updated": now,
            }
        )
    new_results = pd.DataFrame(rows, columns=RESULTS_LEDGER_COLUMNS)
    existing = load_results_ledger(path)
    combined = pd.concat([existing, new_results], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["match_id"], keep="last")
        combined = combined.sort_values(["date_utc", "match_id"]).reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined[RESULTS_LEDGER_COLUMNS].copy()


def _load_primary_ratings() -> pd.DataFrame:
    try:
        return get_team_ratings(model_mode="baseline_manual")
    except TypeError:
        return read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)


def _config_payload(cfg: ModelConfig) -> dict[str, Any]:
    if is_dataclass(cfg):
        raw = asdict(cfg)
    else:
        raw = dict(getattr(cfg, "__dict__", {}))
    payload: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, float):
            payload[key] = round(value, 8)
        else:
            payload[key] = value
    return payload


def _single_fixture_frame(fixture_row: pd.Series | dict[str, Any] | pd.DataFrame | None) -> pd.DataFrame:
    if fixture_row is None:
        return pd.DataFrame()
    if isinstance(fixture_row, pd.DataFrame):
        return fixture_row.head(1).copy()
    return pd.DataFrame([pd.Series(fixture_row)])


def _first_match_id(fixtures: pd.DataFrame) -> str:
    if fixtures is None or fixtures.empty or "match_id" not in fixtures.columns:
        return ""
    return str(fixtures["match_id"].iloc[0] or "").strip()


def _fresh_existing_prediction(
    match_id: str,
    model_version: str,
    parameter_set_id: str,
    primary_model_mode: str,
    path: str | Path,
    min_interval_minutes: int,
) -> bool:
    if not match_id:
        return False
    existing = load_prediction_ledger(path)
    if existing.empty:
        return False
    frame = existing.copy()
    mask = (
        frame["match_id"].astype(str).eq(str(match_id))
        & frame["model_version"].astype(str).eq(str(model_version))
        & frame["parameter_set_id"].astype(str).eq(str(parameter_set_id))
        & frame["primary_model_mode"].astype(str).eq(str(primary_model_mode))
        & frame["prediction_before_kickoff"].astype(str).str.lower().isin({"true", "1", "yes"})
    )
    matched = frame.loc[mask].copy()
    if matched.empty:
        return False
    snapshots = pd.to_datetime(matched["snapshot_utc"], errors="coerce", utc=True).dropna()
    if snapshots.empty:
        return False
    latest = snapshots.max()
    now = pd.to_datetime(utc_now_iso(), errors="coerce", utc=True)
    if pd.isna(now):
        now = pd.Timestamp.now(tz="UTC")
    return bool(latest >= now - pd.Timedelta(minutes=int(min_interval_minutes)))


def _completed_candidates_for_fixture(fixture: pd.Series) -> pd.DataFrame:
    date_label = _date_label(fixture.get("date_utc"))
    teams = [str(fixture.get("home", "") or ""), str(fixture.get("away", "") or "")]
    teams = [team for team in teams if team.strip()]
    return load_completed_matches_for_backtest(start_date=date_label, end_date=date_label, teams=teams or None)


def _completed_result_for_fixture(fixture: pd.Series, completed_matches_df: pd.DataFrame | None) -> pd.DataFrame:
    completed = completed_matches_df.copy() if completed_matches_df is not None else pd.DataFrame()
    if completed.empty:
        return pd.DataFrame()
    for col in ["match_id", "date_utc", "home", "away", "home_goals", "away_goals"]:
        if col not in completed.columns:
            completed[col] = pd.NA
    fixture_match_id = str(fixture.get("match_id", "") or "").strip()
    if fixture_match_id:
        by_id = completed.loc[completed["match_id"].astype(str) == fixture_match_id].copy()
        if not by_id.empty:
            return by_id.head(1)
    date_label = _date_label(fixture.get("date_utc"))
    home_key = _team_key(fixture.get("home", ""))
    away_key = _team_key(fixture.get("away", ""))
    home = completed["home"].map(_team_key)
    away = completed["away"].map(_team_key)
    date = completed["date_utc"].map(_date_label)
    by_fixture = completed.loc[(date == date_label) & (home == home_key) & (away == away_key)].copy()
    return by_fixture.head(1)


def _result_import_diagnostics(
    status: str,
    match_id: str,
    ready_after_utc: str,
    rows_imported: int,
    message: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "match_id": match_id,
        "ready_after_utc": ready_after_utc,
        "rows_imported": int(rows_imported),
        "message": message,
    }


def _date_label(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else ts.date().isoformat()


def _team_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("-", " ").split())


def _normalise_fixtures(fixtures_df: pd.DataFrame | None) -> pd.DataFrame:
    required = [
        "match_id",
        "date_utc",
        "time_utc",
        "competition",
        "group",
        "home",
        "away",
        "venue",
        "city",
        "country",
        "neutral_site",
    ]
    out = _ensure_columns(fixtures_df, required)
    if out.empty:
        return out
    out = out.dropna(subset=["date_utc", "home", "away"]).copy()
    out["home"] = out["home"].astype(str).str.strip()
    out["away"] = out["away"].astype(str).str.strip()
    out = out.loc[(out["home"] != "") & (out["away"] != "")]
    if "neutral_site" in out.columns:
        out["neutral_site"] = out["neutral_site"].map(coerce_bool).astype(int)
    return out.reset_index(drop=True)


def _ensure_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame(columns=columns)
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].copy()


def _over_probability(matrix: pd.DataFrame | None, threshold_goals: int) -> float:
    if matrix is None or matrix.empty:
        return 0.0
    mat = matrix.copy()
    if not {"home_goals", "away_goals", "prob"}.issubset(mat.columns):
        return 0.0
    total = pd.to_numeric(mat["home_goals"], errors="coerce") + pd.to_numeric(mat["away_goals"], errors="coerce")
    return float(pd.to_numeric(mat["prob"], errors="coerce").fillna(0.0).loc[total > float(threshold_goals)].sum())


def _kickoff_utc(row: pd.Series) -> str:
    date_value = str(row.get("date_utc", "") or "").strip()
    time_value = str(row.get("time_utc", "") or "00:00").strip() or "00:00"
    ts = pd.to_datetime(f"{date_value} {time_value}", errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.to_datetime(date_value, errors="coerce", utc=True)
    return "" if pd.isna(ts) else ts.replace(microsecond=0).isoformat()


def _fallback_match_id(row: Any) -> str:
    series = pd.Series(row)
    return "|".join(
        [
            str(series.get("date_utc", "")),
            str(series.get("home", "")),
            str(series.get("away", "")),
        ]
    ).strip("|")


def _prediction_id(snapshot_utc: str, match_id: str, model_version: str, parameter_set_id: str) -> str:
    raw = "|".join([snapshot_utc, match_id, model_version, parameter_set_id])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _skip_row(row: Any, reason_code: str, reason: str) -> dict[str, Any]:
    series = pd.Series(row)
    return {
        "match_id": str(series.get("match_id", "") or ""),
        "date_utc": str(series.get("date_utc", "") or ""),
        "time_utc": str(series.get("time_utc", "") or ""),
        "home": str(series.get("home", "") or ""),
        "away": str(series.get("away", "") or ""),
        "reason_code": reason_code,
        "reason": reason,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR.parent))
    except ValueError:
        return str(path)


def _market_snapshot_available(market_odds_df: pd.DataFrame, match_id: str) -> bool:
    if market_odds_df is None or market_odds_df.empty or "match_id" not in market_odds_df.columns:
        return False
    return bool((market_odds_df["match_id"].astype(str) == str(match_id)).any())


def _before_kickoff(snapshot_utc: str, kickoff_utc: str) -> bool:
    snapshot = pd.to_datetime(snapshot_utc, errors="coerce", utc=True)
    kickoff = pd.to_datetime(kickoff_utc, errors="coerce", utc=True)
    if pd.isna(snapshot) or pd.isna(kickoff):
        return False
    return bool(snapshot < kickoff)
