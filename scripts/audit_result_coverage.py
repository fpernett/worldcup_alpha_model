from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_sources import FIXTURE_COLUMNS  # noqa: E402
from src.match_identity import build_result_fixture_crosswalk, validate_results_ledger_semantics  # noqa: E402
from src.prediction_ledger import PREDICTION_LEDGER_COLUMNS, RESULTS_LEDGER_COLUMNS  # noqa: E402
from src.storage import data_path  # noqa: E402


REPORT_COLUMNS = [
    "match_id",
    "kickoff_utc",
    "date_utc",
    "competition",
    "home",
    "away",
    "prediction_count",
    "latest_snapshot_utc",
    "has_matched_90_minute_result",
    "missing_local_result",
    "result_match_id",
    "result_join_method",
    "sync_would_attempt",
    "sync_skip_reason",
]


def main() -> None:
    args = _parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    predictions = _read_csv_with_columns(Path(args.predictions), PREDICTION_LEDGER_COLUMNS)
    results = validate_results_ledger_semantics(_read_csv_with_columns(Path(args.results_ledger), RESULTS_LEDGER_COLUMNS))
    fixtures = _read_csv_with_columns(Path(args.fixtures), FIXTURE_COLUMNS)

    audit = build_result_coverage_audit(
        predictions,
        results,
        fixtures,
        now_utc=args.now_utc,
        result_ready_delay_minutes=float(args.result_ready_delay_minutes),
        competition=args.competition,
    )
    csv_path = reports_dir / "result_coverage_audit.csv"
    md_path = reports_dir / "result_coverage_audit.md"
    audit.to_csv(csv_path, index=False)
    md_path.write_text(render_result_coverage_audit(audit), encoding="utf-8")

    print(f"total pre-kickoff predictions: {int(audit['prediction_count'].sum()) if not audit.empty else 0:,}")
    print(f"matches with matched 90-minute result: {int(audit['has_matched_90_minute_result'].sum()) if not audit.empty else 0:,}")
    print(f"matches missing local result: {int(audit['missing_local_result'].sum()) if not audit.empty else 0:,}")
    print(f"result_coverage_audit: {_display_path(csv_path)}")
    print(f"result_coverage_audit_report: {_display_path(md_path)}")


def build_result_coverage_audit(
    predictions_df: pd.DataFrame | None,
    results_ledger_df: pd.DataFrame | None,
    fixtures_df: pd.DataFrame | None,
    now_utc: str | pd.Timestamp | None = None,
    result_ready_delay_minutes: float = 15,
    competition: str = "FIFA World Cup",
) -> pd.DataFrame:
    predictions = _ensure_columns(predictions_df, PREDICTION_LEDGER_COLUMNS)
    results = validate_results_ledger_semantics(_ensure_columns(results_ledger_df, RESULTS_LEDGER_COLUMNS))
    fixtures = _ensure_columns(fixtures_df, FIXTURE_COLUMNS)
    pre = _pre_kickoff_predictions(predictions)
    if competition and not pre.empty and "competition" in pre.columns:
        needle = str(competition).strip().lower()
        pre = pre.loc[pre["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    if pre.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    crosswalk = build_result_fixture_crosswalk(results, fixtures)
    result_lookup = _result_lookup(results, crosswalk)
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    fixture_lookup = {str(row.get("match_id", "") or ""): row for _, row in fixtures.iterrows()}

    rows: list[dict[str, Any]] = []
    grouped = pre.sort_values("snapshot_utc").groupby("match_id", dropna=False)
    for match_id_raw, group in grouped:
        match_id = str(match_id_raw or "").strip()
        latest = group.iloc[-1]
        fixture = fixture_lookup.get(match_id)
        source = fixture if fixture is not None else latest
        kickoff = _kickoff_utc(source)
        result_info = result_lookup.get(match_id, {})
        has_result = bool(result_info)
        would_attempt, skip_reason = _sync_attempt_status(source, now, result_ready_delay_minutes)
        rows.append(
            {
                "match_id": match_id,
                "kickoff_utc": kickoff,
                "date_utc": _date_label(source.get("date_utc", latest.get("kickoff_utc", ""))),
                "competition": source.get("competition", latest.get("competition", "")),
                "home": source.get("home", latest.get("home", "")),
                "away": source.get("away", latest.get("away", "")),
                "prediction_count": int(len(group)),
                "latest_snapshot_utc": latest.get("snapshot_utc", ""),
                "has_matched_90_minute_result": has_result,
                "missing_local_result": not has_result,
                "result_match_id": result_info.get("result_match_id", ""),
                "result_join_method": result_info.get("join_method", ""),
                "sync_would_attempt": bool((not has_result) and would_attempt),
                "sync_skip_reason": "" if has_result else skip_reason,
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS).sort_values(["date_utc", "match_id"]).reset_index(drop=True)


def render_result_coverage_audit(audit_df: pd.DataFrame | None) -> str:
    audit = audit_df.copy() if audit_df is not None else pd.DataFrame(columns=REPORT_COLUMNS)
    total_predictions = int(audit["prediction_count"].sum()) if not audit.empty else 0
    matched = int(audit["has_matched_90_minute_result"].sum()) if not audit.empty else 0
    missing = int(audit["missing_local_result"].sum()) if not audit.empty else 0
    lines = [
        "# Result Coverage Audit",
        "",
        f"- Total pre-kickoff predictions: {total_predictions:,}",
        f"- Predicted matches with matched 90-minute result: {matched:,}",
        f"- Predicted matches missing local result: {missing:,}",
        "",
        "## Missing Result By Date",
        "",
    ]
    missing_df = audit.loc[audit["missing_local_result"].astype(bool)].copy() if not audit.empty else pd.DataFrame(columns=REPORT_COLUMNS)
    by_date = missing_df.groupby("date_utc", dropna=False).size().reset_index(name="missing_matches") if not missing_df.empty else pd.DataFrame()
    lines.extend(_markdown_table(by_date, ["date_utc", "missing_matches"]))
    lines.extend(["", "## Missing Result By Match", ""])
    lines.extend(_markdown_table(missing_df.head(100), ["match_id", "date_utc", "home", "away", "sync_would_attempt", "sync_skip_reason"]))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Results are counted only when the local result ledger validates as 90-minute 1X2 eligible.",
            "- `sync_would_attempt` is a dry-run readiness check; this script does not call providers or write ledgers.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pre-kickoff prediction result coverage without writing ledgers.")
    parser.add_argument("--predictions", default=str(data_path("prediction_ledger.csv")))
    parser.add_argument("--results-ledger", default=str(data_path("results_ledger.csv")))
    parser.add_argument("--fixtures", default=str(data_path("fixtures.csv")))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--competition", default="FIFA World Cup")
    parser.add_argument("--result-ready-delay-minutes", type=float, default=15)
    parser.add_argument("--now-utc", default=None)
    return parser.parse_args()


def _read_csv_with_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    return _ensure_columns(df, columns)


def _ensure_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].copy()


def _pre_kickoff_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    out = predictions.copy()
    before = out["prediction_before_kickoff"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    snapshots = pd.to_datetime(out["snapshot_utc"], errors="coerce", utc=True)
    kickoffs = pd.to_datetime(out["kickoff_utc"], errors="coerce", utc=True)
    inferred_before = snapshots.notna() & kickoffs.notna() & (snapshots < kickoffs)
    return out.loc[before | inferred_before].copy()


def _result_lookup(results: pd.DataFrame, crosswalk: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    if not results.empty:
        eligible = results.loc[results["evaluation_eligible_1x2"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
        for _, row in eligible.iterrows():
            match_id = str(row.get("match_id", "") or "").strip()
            if match_id:
                lookup[match_id] = {"result_match_id": match_id, "join_method": "exact_match_id"}
    if crosswalk is not None and not crosswalk.empty:
        mapped = crosswalk.loc[crosswalk["fixture_match_id"].fillna("").astype(str).str.strip() != ""].copy()
        mapped = mapped.loc[mapped["evaluation_eligible_1x2"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
        for _, row in mapped.iterrows():
            fixture_id = str(row.get("fixture_match_id", "") or "").strip()
            if fixture_id and fixture_id not in lookup:
                lookup[fixture_id] = {
                    "result_match_id": str(row.get("result_match_id", "") or ""),
                    "join_method": str(row.get("join_method", "") or "result_fixture_crosswalk"),
                }
    return lookup


def _sync_attempt_status(row: pd.Series, now: pd.Timestamp, result_ready_delay_minutes: float) -> tuple[bool, str]:
    kickoff = pd.to_datetime(_kickoff_utc(row), errors="coerce", utc=True)
    if pd.isna(kickoff):
        return False, "invalid_kickoff"
    home = str(row.get("home", "") or "").strip().lower()
    away = str(row.get("away", "") or "").strip().lower()
    unresolved_terms = ("winner ", "loser ", "winner match", "loser match", "tbd", "to be determined")
    if any(term in home for term in unresolved_terms) or any(term in away for term in unresolved_terms):
        return False, "unresolved_team"
    ready_at = kickoff + pd.Timedelta(minutes=float(result_ready_delay_minutes))
    if ready_at > now:
        return False, "not_ready"
    return True, "would_attempt"


def _kickoff_utc(row: pd.Series | dict[str, Any]) -> str:
    value = row.get("kickoff_utc", "") if hasattr(row, "get") else ""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.notna(parsed):
        return parsed.isoformat()
    date_value = str(row.get("date_utc", "") or "").strip() if hasattr(row, "get") else ""
    time_value = str(row.get("time_utc", "") or "").strip() if hasattr(row, "get") else ""
    parsed = pd.to_datetime(f"{date_value} {time_value}", errors="coerce", utc=True)
    return "" if pd.isna(parsed) else parsed.isoformat()


def _date_label(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df is None or df.empty:
        return ["No rows."]
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    table = out[columns].fillna("").astype(str)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_escape_markdown_cell(row[col]) for col in columns) + " |")
    return lines


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
