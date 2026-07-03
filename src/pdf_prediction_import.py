from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from dateutil import parser as date_parser

from src.match_identity import normalize_match_date, normalize_team_name
from src.prediction_ledger import PREDICTION_LEDGER_COLUMNS
from src.team_names import team_name_key
from src.utils import coerce_float


LOCAL_REPORT_TZ = ZoneInfo("Europe/Stockholm")
UTC = ZoneInfo("UTC")

PDF_AUDIT_COLUMNS = [
    "source_file",
    "file_path",
    "file_modified_time_local",
    "text_extractor",
    "parsed_home_team",
    "parsed_away_team",
    "parsed_kickoff_utc",
    "parsed_report_timestamp_text",
    "parsed_report_timestamp_utc",
    "timestamp_source",
    "timestamp_confidence",
    "parsed_home_win_prob",
    "parsed_draw_prob",
    "parsed_away_win_prob",
    "parsed_home_xg",
    "parsed_away_xg",
    "match_id",
    "competition",
    "round",
    "venue",
    "primary_model",
    "behavior_layer_status",
    "model_confidence",
    "best_local_ev_text",
    "best_polymarket_gap_text",
    "market_join_status_text",
    "home_data_support_matches",
    "away_data_support_matches",
    "h2h_matches",
    "training_matches",
    "valid_pre_kickoff_snapshot",
    "import_candidate_status",
    "import_status",
    "import_warning",
]

PDF_LEDGER_EXTRA_COLUMNS = [
    "source_type",
    "source_file",
    "generated_at_source",
    "generated_at_confidence",
    "home_win_prob_raw",
    "draw_prob_raw",
    "away_win_prob_raw",
    "primary_model",
    "behavior_layer_status",
    "best_local_ev_text",
    "best_polymarket_gap_text",
    "market_join_status_text",
    "home_data_support_matches",
    "away_data_support_matches",
    "h2h_matches",
    "training_matches",
    "valid_pre_kickoff_snapshot",
    "import_status",
    "import_warning",
]

PDF_LEDGER_COLUMNS = list(dict.fromkeys([*PREDICTION_LEDGER_COLUMNS, *PDF_LEDGER_EXTRA_COLUMNS]))

TIMESTAMP_OVERRIDE_COLUMNS = ["source_file", "generated_at_utc", "reason", "reviewer", "reviewed_at_utc"]

MERGE_AUDIT_COLUMNS = [
    "source_type",
    "source_file",
    "prediction_id",
    "match_id",
    "snapshot_utc",
    "kickoff_utc",
    "merge_status",
    "snapshot_horizon_hours",
    "snapshot_group",
]


@dataclass
class ExtractedPdf:
    text: str
    extractor: str
    metadata: dict[str, Any]
    warning: str = ""


def extract_pdf_text(path: str | Path) -> ExtractedPdf:
    """Extract PDF text using PyMuPDF, pdfplumber, then pypdf.

    OCR is intentionally not attempted. If all three text extractors fail, the
    caller receives an empty text string and can mark the PDF unsupported.
    """
    pdf_path = Path(path)
    warnings: list[str] = []

    text, metadata, warning = _extract_with_pymupdf(pdf_path)
    if text.strip():
        return ExtractedPdf(text=text, extractor="pymupdf", metadata=metadata, warning=warning)
    if warning:
        warnings.append(warning)

    text, metadata_fallback, warning = _extract_with_pdfplumber(pdf_path)
    if text.strip():
        return ExtractedPdf(text=text, extractor="pdfplumber", metadata=metadata or metadata_fallback, warning="; ".join(warnings))
    if warning:
        warnings.append(warning)

    text, metadata_fallback, warning = _extract_with_pypdf(pdf_path)
    if text.strip():
        return ExtractedPdf(text=text, extractor="pypdf", metadata=metadata or metadata_fallback, warning="; ".join(warnings))
    if warning:
        warnings.append(warning)

    return ExtractedPdf(text="", extractor="unsupported", metadata=metadata, warning="; ".join(warnings))


def parse_pdf_prediction_report(
    path: str | Path,
    *,
    fixtures_df: pd.DataFrame | None = None,
    timestamp_overrides: dict[str, dict[str, Any]] | None = None,
    existing_predictions_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    pdf_path = Path(path)
    extracted = extract_pdf_text(pdf_path)
    modified = _file_modified_time(pdf_path)
    override = (timestamp_overrides or {}).get(pdf_path.name)
    parsed = parse_prediction_report_text(
        extracted.text,
        source_file=pdf_path.name,
        file_path=str(pdf_path),
        file_modified_time_local=modified,
        text_extractor=extracted.extractor,
        pdf_metadata=extracted.metadata,
        timestamp_override=override,
        fixtures_df=fixtures_df,
    )
    if extracted.warning and parsed.get("import_warning"):
        parsed["import_warning"] = f"{parsed['import_warning']}; {extracted.warning}"
    elif extracted.warning:
        parsed["import_warning"] = extracted.warning
    if extracted.extractor == "unsupported":
        parsed["import_candidate_status"] = "unsupported_pdf_format"
        parsed["import_status"] = "unsupported_pdf_format"
        parsed["valid_pre_kickoff_snapshot"] = False
        parsed["import_warning"] = "All text extractors failed; OCR was not attempted."
    if _is_duplicate_existing_snapshot(parsed, existing_predictions_df):
        parsed["import_candidate_status"] = "duplicate_existing_snapshot"
        parsed["import_status"] = "duplicate_existing_snapshot"
        parsed["valid_pre_kickoff_snapshot"] = False
        parsed["import_warning"] = "An existing prediction snapshot has the same match_id and snapshot timestamp."
    return _ensure_columns_dict(parsed, PDF_AUDIT_COLUMNS)


def parse_prediction_report_text(
    text: str,
    *,
    source_file: str = "",
    file_path: str = "",
    file_modified_time_local: str = "",
    text_extractor: str = "text_fixture",
    pdf_metadata: dict[str, Any] | None = None,
    timestamp_override: dict[str, Any] | None = None,
    fixtures_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    lines = _clean_lines(text)
    home, away = _parse_fixture_teams(lines)
    competition, round_name, kickoff_utc, venue = _parse_fixture_meta(lines)
    home_xg, away_xg = _parse_expected_goals(lines)
    home_prob, draw_prob, away_prob = _parse_outcome_probabilities(lines)
    generated_at, timestamp_text, timestamp_source, timestamp_confidence = _parse_report_timestamp(
        text,
        pdf_metadata or {},
        timestamp_override,
        file_modified_time_local,
    )
    match_id = _match_fixture_id(home, away, kickoff_utc, fixtures_df)
    model_confidence = _value_after_label(lines, "Model confidence")
    primary_model = _value_after_label(lines, "Primary model")
    behavior_status = _value_after_label(lines, "Behavior layer")
    best_local = _value_after_label(lines, "Best local EV")
    best_poly = _value_after_label(lines, "Best Polymarket gap")
    support = _parse_data_support(lines, home, away)
    market_status = _parse_market_status(text)

    status, warning, valid = _classify_import_candidate(
        home=home,
        away=away,
        kickoff_utc=kickoff_utc,
        generated_at_utc=generated_at,
        timestamp_source=timestamp_source,
        home_prob=home_prob,
        draw_prob=draw_prob,
        away_prob=away_prob,
        match_id=match_id,
    )
    return {
        "source_file": source_file,
        "file_path": file_path,
        "file_modified_time_local": file_modified_time_local,
        "text_extractor": text_extractor,
        "parsed_home_team": home,
        "parsed_away_team": away,
        "parsed_kickoff_utc": kickoff_utc,
        "parsed_report_timestamp_text": timestamp_text,
        "parsed_report_timestamp_utc": generated_at,
        "timestamp_source": timestamp_source,
        "timestamp_confidence": timestamp_confidence,
        "parsed_home_win_prob": home_prob,
        "parsed_draw_prob": draw_prob,
        "parsed_away_win_prob": away_prob,
        "parsed_home_xg": home_xg,
        "parsed_away_xg": away_xg,
        "match_id": match_id,
        "competition": competition,
        "round": round_name,
        "venue": venue,
        "primary_model": primary_model,
        "behavior_layer_status": behavior_status,
        "model_confidence": model_confidence,
        "best_local_ev_text": best_local,
        "best_polymarket_gap_text": best_poly,
        "market_join_status_text": market_status,
        "home_data_support_matches": support["home_data_support_matches"],
        "away_data_support_matches": support["away_data_support_matches"],
        "h2h_matches": support["h2h_matches"],
        "training_matches": support["training_matches"],
        "valid_pre_kickoff_snapshot": valid,
        "import_candidate_status": status,
        "import_status": status,
        "import_warning": warning,
    }


def audit_rows_to_prediction_ledger(audit_df: pd.DataFrame | None) -> pd.DataFrame:
    audit = audit_df.copy() if audit_df is not None else pd.DataFrame(columns=PDF_AUDIT_COLUMNS)
    rows: list[dict[str, Any]] = []
    if audit.empty:
        return pd.DataFrame(columns=PDF_LEDGER_COLUMNS)
    ready = audit.loc[
        audit["valid_pre_kickoff_snapshot"].astype(bool)
        & audit["import_candidate_status"].astype(str).eq("ready_for_import")
    ].copy()
    for _, row in ready.iterrows():
        snapshot = str(row.get("parsed_report_timestamp_utc", "") or "")
        match_id = str(row.get("match_id", "") or "")
        source_file = str(row.get("source_file", "") or "")
        model_version = "pdf_report_backfill_v1"
        parameter_set_id = "pdf_report"
        prediction_id = _stable_id("pdf_report", source_file, match_id, snapshot)
        ledger_row = {col: pd.NA for col in PDF_LEDGER_COLUMNS}
        ledger_row.update(
            {
                "prediction_id": prediction_id,
                "snapshot_utc": snapshot,
                "match_id": match_id,
                "kickoff_utc": row.get("parsed_kickoff_utc", ""),
                "competition": row.get("competition", "FIFA World Cup"),
                "group": row.get("round", ""),
                "home": row.get("parsed_home_team", ""),
                "away": row.get("parsed_away_team", ""),
                "venue": row.get("venue", ""),
                "primary_model_mode": row.get("primary_model", ""),
                "model_version": model_version,
                "parameter_set_id": parameter_set_id,
                "home_win_prob": row.get("parsed_home_win_prob", pd.NA),
                "draw_prob": row.get("parsed_draw_prob", pd.NA),
                "away_win_prob": row.get("parsed_away_win_prob", pd.NA),
                "home_xg": row.get("parsed_home_xg", pd.NA),
                "away_xg": row.get("parsed_away_xg", pd.NA),
                "model_confidence": row.get("model_confidence", ""),
                "source_status": "source=pdf_report; timestamp_validated",
                "market_snapshot_available": _market_snapshot_available(row.get("market_join_status_text", "")),
                "prediction_before_kickoff": True,
                "notes": f"source_file={source_file}; generated_at_source={row.get('timestamp_source', '')}; generated_at_confidence={row.get('timestamp_confidence', '')}",
                "source_type": "pdf_report",
                "source_file": source_file,
                "generated_at_source": row.get("timestamp_source", ""),
                "generated_at_confidence": row.get("timestamp_confidence", ""),
                "home_win_prob_raw": row.get("parsed_home_win_prob", pd.NA),
                "draw_prob_raw": row.get("parsed_draw_prob", pd.NA),
                "away_win_prob_raw": row.get("parsed_away_win_prob", pd.NA),
                "primary_model": row.get("primary_model", ""),
                "behavior_layer_status": row.get("behavior_layer_status", ""),
                "best_local_ev_text": row.get("best_local_ev_text", ""),
                "best_polymarket_gap_text": row.get("best_polymarket_gap_text", ""),
                "market_join_status_text": row.get("market_join_status_text", ""),
                "home_data_support_matches": row.get("home_data_support_matches", pd.NA),
                "away_data_support_matches": row.get("away_data_support_matches", pd.NA),
                "h2h_matches": row.get("h2h_matches", pd.NA),
                "training_matches": row.get("training_matches", pd.NA),
                "valid_pre_kickoff_snapshot": True,
                "import_status": row.get("import_candidate_status", ""),
                "import_warning": row.get("import_warning", ""),
            }
        )
        rows.append(ledger_row)
    out = pd.DataFrame(rows, columns=PDF_LEDGER_COLUMNS)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["source_file", "match_id", "snapshot_utc", "home_win_prob", "draw_prob", "away_win_prob"]).reset_index(drop=True)


def load_timestamp_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    override_path = Path(path)
    if not override_path.exists():
        return {}
    df = pd.read_csv(override_path)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        source_file = str(row.get("source_file", "") or "").strip()
        generated = str(row.get("generated_at_utc", "") or "").strip()
        if source_file and generated:
            out[source_file] = row.to_dict()
    return out


def build_pdf_backfill_inventory(
    pdf_dir: str | Path,
    *,
    fixtures_df: pd.DataFrame | None = None,
    timestamp_overrides: dict[str, dict[str, Any]] | None = None,
    existing_predictions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    directory = Path(pdf_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        parse_pdf_prediction_report(
            path,
            fixtures_df=fixtures_df,
            timestamp_overrides=timestamp_overrides,
            existing_predictions_df=existing_predictions_df,
        )
        for path in sorted(directory.glob("*.pdf"))
    ]
    return pd.DataFrame(rows, columns=PDF_AUDIT_COLUMNS)


def render_pdf_backfill_inventory(inventory_df: pd.DataFrame | None) -> str:
    inventory = inventory_df.copy() if inventory_df is not None else pd.DataFrame(columns=PDF_AUDIT_COLUMNS)
    found = len(inventory)
    imported = int((inventory["import_candidate_status"].astype(str) == "ready_for_import").sum()) if not inventory.empty else 0
    rejected = found - imported
    lines = [
        "# PDF Backfill Inventory",
        "",
        "Historical PDF reports are accepted only when a report timestamp proves the snapshot was generated before kickoff. OCR is not used.",
        "",
        f"- PDFs found: `{found}`",
        f"- Ready for import: `{imported}`",
        f"- Rejected or review-required: `{rejected}`",
        "",
        "## Rejection Reasons",
        "",
        _markdown_table(_status_counts(inventory)),
        "",
        "## Files",
        "",
        _markdown_table(_inventory_display(inventory)),
    ]
    return "\n".join(lines)


def create_timestamp_override_template(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        pd.DataFrame(columns=TIMESTAMP_OVERRIDE_COLUMNS).to_csv(target, index=False)


def build_enriched_prediction_ledger(
    prediction_ledger_df: pd.DataFrame | None,
    backfill_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = prediction_ledger_df.copy() if prediction_ledger_df is not None else pd.DataFrame()
    pdf = backfill_df.copy() if backfill_df is not None else pd.DataFrame()
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    if not existing.empty:
        app_rows = existing.copy()
        app_rows["source_type"] = app_rows.get("source_type", "app_snapshot")
        app_rows["source_type"] = app_rows["source_type"].fillna("app_snapshot").replace("", "app_snapshot")
        app_rows["source_file"] = app_rows.get("source_file", "")
        frames.append(app_rows)

    if not pdf.empty:
        pdf_rows = pdf.copy()
        pdf_rows["source_type"] = "pdf_report"
        before = len(pdf_rows)
        pdf_rows = pdf_rows.drop_duplicates(subset=["source_file", "match_id", "snapshot_utc", "home_win_prob", "draw_prob", "away_win_prob"]).copy()
        for _, row in pdf_rows.iterrows():
            audit_rows.append(_merge_audit_row(row, "merged"))
        duplicate_count = before - len(pdf_rows)
        if duplicate_count:
            audit_rows.append({"source_type": "pdf_report", "source_file": "", "prediction_id": "", "match_id": "", "snapshot_utc": "", "kickoff_utc": "", "merge_status": f"deduplicated_{duplicate_count}_pdf_rows", "snapshot_horizon_hours": pd.NA, "snapshot_group": ""})
        frames.append(pdf_rows)

    if not frames:
        return pd.DataFrame(), pd.DataFrame(columns=MERGE_AUDIT_COLUMNS)

    columns = list(dict.fromkeys([*(col for frame in frames for col in frame.columns), "snapshot_horizon_hours", "snapshot_group"]))
    combined = pd.concat([_ensure_frame_columns(frame, columns) for frame in frames], ignore_index=True)
    combined["snapshot_horizon_hours"] = combined.apply(_snapshot_horizon_hours, axis=1)
    combined["snapshot_group"] = combined["snapshot_horizon_hours"].map(snapshot_group)
    for idx, row in combined.iterrows():
        if not audit_rows or str(row.get("source_type", "")) != "pdf_report":
            audit_rows.append(_merge_audit_row(row, "preserved" if str(row.get("source_type", "")) == "app_snapshot" else "merged"))
    audit = pd.DataFrame(audit_rows, columns=MERGE_AUDIT_COLUMNS)
    return combined, audit


def snapshot_group(hours: Any) -> str:
    value = coerce_float(hours, float("nan"))
    if pd.isna(value):
        return "unknown"
    if value < 0:
        return "invalid_post_kickoff"
    if value <= 6:
        return "late_0_to_6h"
    if value <= 24:
        return "same_day_6_to_24h"
    return "early_gt_24h"


def calibration_status_from_sample_size(evaluated_matches: int, comparison_df: pd.DataFrame | None = None) -> str:
    if evaluated_matches < 30:
        return "sample_too_small"
    if evaluated_matches <= 100:
        return "preliminary"
    comparison = comparison_df.copy() if comparison_df is not None else pd.DataFrame()
    if not comparison.empty and (comparison.get("production_status", pd.Series(dtype=str)).astype(str) == "production_eligible").any():
        return "production_candidate"
    return "preliminary"


def render_prediction_source_coverage(
    *,
    predictions_df: pd.DataFrame,
    pdf_audit_df: pd.DataFrame | None,
    evaluation_df: pd.DataFrame,
    calibration_status: str,
) -> str:
    predictions = predictions_df.copy() if predictions_df is not None else pd.DataFrame()
    pdf_audit = pdf_audit_df.copy() if pdf_audit_df is not None else pd.DataFrame(columns=PDF_AUDIT_COLUMNS)
    evaluation = evaluation_df.copy() if evaluation_df is not None else pd.DataFrame()
    prediction_sources = predictions.get("source_type", pd.Series(dtype=str)).fillna("app_snapshot").replace("", "app_snapshot")
    evaluated_sources = evaluation.get("prediction_source", pd.Series(dtype=str)).fillna("unknown").replace("", "unknown")
    rows = [
        {"metric": "app_snapshots_available", "value": int((prediction_sources == "app_snapshot").sum())},
        {"metric": "pdf_snapshots_available", "value": int((prediction_sources == "pdf_report").sum())},
        {"metric": "pdfs_found", "value": int(len(pdf_audit))},
        {"metric": "pdfs_imported", "value": int((pdf_audit.get("import_candidate_status", pd.Series(dtype=str)).astype(str) == "ready_for_import").sum())},
        {"metric": "pdfs_rejected", "value": int((pdf_audit.get("import_candidate_status", pd.Series(dtype=str)).astype(str) != "ready_for_import").sum()) if not pdf_audit.empty else 0},
        {"metric": "valid_pre_kickoff_pdf_snapshots", "value": int(pdf_audit.get("valid_pre_kickoff_snapshot", pd.Series(dtype=bool)).astype(bool).sum()) if not pdf_audit.empty else 0},
        {"metric": "evaluated_app_snapshots", "value": int((evaluated_sources == "app_snapshot").sum())},
        {"metric": "evaluated_pdf_snapshots", "value": int((evaluated_sources == "pdf_report").sum())},
        {"metric": "total_evaluated_matches", "value": int(len(evaluation))},
        {"metric": "calibration_status", "value": calibration_status},
    ]
    lines = [
        "## Prediction Source Coverage",
        "",
        "Historical PDF reports can be used as archived prediction snapshots only if their report timestamp is before kickoff. Post-kickoff or ambiguous PDFs are excluded from accuracy claims.",
        "",
        _markdown_table(pd.DataFrame(rows)),
    ]
    if not pdf_audit.empty:
        lines.extend(["", "PDF rejection reasons:", "", _markdown_table(_status_counts(pdf_audit))])
    return "\n".join(lines)


def _extract_with_pymupdf(path: Path) -> tuple[str, dict[str, Any], str]:
    try:
        import pymupdf

        doc = pymupdf.open(path)
        text = "\n".join(page.get_text() or "" for page in doc)
        return text, dict(doc.metadata or {}), ""
    except Exception as exc:  # pragma: no cover - depends on optional package/runtime
        return "", {}, f"PyMuPDF extraction failed: {exc}"


def _extract_with_pdfplumber(path: Path) -> tuple[str, dict[str, Any], str]:
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            metadata = dict(pdf.metadata or {})
        return text, metadata, ""
    except Exception as exc:  # pragma: no cover - depends on optional package/runtime
        return "", {}, f"pdfplumber extraction failed: {exc}"


def _extract_with_pypdf(path: Path) -> tuple[str, dict[str, Any], str]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        metadata = {str(key).lstrip("/"): value for key, value in dict(reader.metadata or {}).items()}
        return text, metadata, ""
    except Exception as exc:  # pragma: no cover - depends on optional package/runtime
        return "", {}, f"pypdf extraction failed: {exc}"


def _clean_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines() if re.sub(r"\s+", " ", line).strip()]


def _parse_fixture_teams(lines: list[str]) -> tuple[str, str]:
    for line in lines:
        if " vs " not in line:
            continue
        left, right = [part.strip(" .") for part in line.split(" vs ", 1)]
        if left and right:
            return normalize_team_name(left), normalize_team_name(right)
    return "", ""


def _parse_fixture_meta(lines: list[str]) -> tuple[str, str, str, str]:
    for line in lines:
        if "FIFA World Cup |" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        competition = parts[0].strip()
        round_name = parts[1].strip() if len(parts) > 1 else ""
        kickoff = parts[2].strip() if len(parts) > 2 else ""
        venue = parts[3].strip() if len(parts) > 3 else ""
        kickoff_ts = pd.to_datetime(kickoff, errors="coerce", utc=True)
        kickoff_utc = "" if pd.isna(kickoff_ts) else kickoff_ts.isoformat()
        return competition, round_name, kickoff_utc, venue
    return "FIFA World Cup", "", "", ""


def _parse_expected_goals(lines: list[str]) -> tuple[Any, Any]:
    value = _value_after_label(lines, "Expected goals")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)", value)
    if not match:
        return pd.NA, pd.NA
    return float(match.group(1)), float(match.group(2))


def _parse_outcome_probabilities(lines: list[str]) -> tuple[Any, Any, Any]:
    try:
        start = next(idx for idx, line in enumerate(lines) if line == "Probability %")
        end = next((idx for idx in range(start + 1, len(lines)) if lines[idx] == "Match Outcome"), min(len(lines), start + 20))
    except StopIteration:
        return pd.NA, pd.NA, pd.NA
    segment = lines[start + 1 : end]
    percentages = [_parse_percent(line) for line in segment]
    percentages = [value for value in percentages if pd.notna(value)]
    if len(percentages) < 3:
        return pd.NA, pd.NA, pd.NA
    return tuple(percentages[:3])


def _parse_percent(text: str) -> Any:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", str(text or ""))
    if not match:
        return pd.NA
    return float(match.group(1)) / 100.0


def _parse_report_timestamp(
    text: str,
    metadata: dict[str, Any],
    override: dict[str, Any] | None,
    file_modified_time_local: str,
) -> tuple[str, str, str, str]:
    if override:
        generated = _parse_to_utc_iso(override.get("generated_at_utc"), default_tz=UTC)
        return generated, str(override.get("generated_at_utc", "") or ""), "manual_override", "high"

    timestamp_match = re.findall(r"\b\d{1,2}/\d{1,2}/\d{4},?\s+\d{1,2}:\d{2}\b", str(text or ""))
    if timestamp_match:
        raw = timestamp_match[-1]
        generated = _parse_to_utc_iso(raw, default_tz=LOCAL_REPORT_TZ, dayfirst=True)
        if generated:
            return generated, raw, "parsed_from_pdf", "medium"

    metadata_value = _metadata_timestamp(metadata)
    if metadata_value:
        generated = _parse_pdf_metadata_date(metadata_value)
        if generated:
            return generated, metadata_value, "parsed_from_pdf", "high"

    if file_modified_time_local:
        generated = _parse_to_utc_iso(file_modified_time_local, default_tz=LOCAL_REPORT_TZ)
        return generated, file_modified_time_local, "file_modified_time", "low"
    return "", "", "unavailable", "unavailable"


def _metadata_timestamp(metadata: dict[str, Any]) -> str:
    for key in ["creationDate", "CreationDate", "creation_date", "modDate", "ModDate"]:
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def _parse_pdf_metadata_date(value: str) -> str:
    text = str(value or "").strip()
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(Z|[+-]\d{2}'?\d{2}'?)?", text)
    if not match:
        return _parse_to_utc_iso(text, default_tz=UTC)
    year, month, day, hour, minute, second, offset = match.groups()
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    if offset and offset != "Z":
        sign = 1 if offset.startswith("+") else -1
        digits = re.sub(r"[^0-9]", "", offset)
        offset_minutes = sign * (int(digits[:2]) * 60 + int(digits[2:4]))
        tz = ZoneInfo("UTC")
        dt = dt.replace(tzinfo=tz) - pd.Timedelta(minutes=offset_minutes).to_pytimedelta()
    else:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _parse_to_utc_iso(value: Any, *, default_tz: ZoneInfo, dayfirst: bool = False) -> str:
    if value is None or str(value).strip() == "":
        return ""
    try:
        dt = date_parser.parse(str(value), dayfirst=dayfirst)
    except Exception:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    return dt.astimezone(UTC).isoformat()


def _match_fixture_id(home: str, away: str, kickoff_utc: str, fixtures_df: pd.DataFrame | None) -> str:
    fixtures = fixtures_df.copy() if fixtures_df is not None else pd.DataFrame()
    if fixtures.empty or not home or not away or not kickoff_utc:
        return ""
    for col in ["match_id", "home", "away", "date_utc", "time_utc", "kickoff_utc"]:
        if col not in fixtures.columns:
            fixtures[col] = ""
    missing = fixtures["kickoff_utc"].fillna("").astype(str).str.strip().eq("")
    fixtures.loc[missing, "kickoff_utc"] = fixtures.loc[missing, "date_utc"].astype(str) + "T" + fixtures.loc[missing, "time_utc"].astype(str) + ":00+00:00"
    target_ts = pd.to_datetime(kickoff_utc, errors="coerce", utc=True)
    home_key = team_name_key(normalize_team_name(home))
    away_key = team_name_key(normalize_team_name(away))
    fixtures["_home_key"] = fixtures["home"].map(lambda value: team_name_key(normalize_team_name(value)))
    fixtures["_away_key"] = fixtures["away"].map(lambda value: team_name_key(normalize_team_name(value)))
    fixtures["_kickoff_ts"] = pd.to_datetime(fixtures["kickoff_utc"], errors="coerce", utc=True)
    matched = fixtures.loc[
        fixtures["_home_key"].eq(home_key)
        & fixtures["_away_key"].eq(away_key)
        & fixtures["_kickoff_ts"].eq(target_ts)
    ]
    if len(matched) == 1:
        return str(matched.iloc[0].get("match_id", "") or "")
    date_key = normalize_match_date(kickoff_utc)
    matched = fixtures.loc[
        fixtures["_home_key"].eq(home_key)
        & fixtures["_away_key"].eq(away_key)
        & fixtures["_kickoff_ts"].map(normalize_match_date).eq(date_key)
    ]
    return str(matched.iloc[0].get("match_id", "") or "") if len(matched) == 1 else ""


def _parse_data_support(lines: list[str], home: str, away: str) -> dict[str, Any]:
    return {
        "home_data_support_matches": _number_after_label(lines, f"{home.lower()} matches"),
        "away_data_support_matches": _number_after_label(lines, f"{away.lower()} matches"),
        "h2h_matches": _number_after_label(lines, "H2H matches"),
        "training_matches": _number_after_label(lines, "Training matches"),
    }


def _number_after_label(lines: list[str], label: str) -> Any:
    label_key = str(label or "").lower()
    for idx, line in enumerate(lines):
        if line.lower() != label_key or idx + 1 >= len(lines):
            continue
        value = re.sub(r"[^0-9]", "", lines[idx + 1])
        if value:
            return int(value)
    return pd.NA


def _parse_market_status(text: str) -> str:
    lower = str(text or "").lower()
    if "joined market price" in lower:
        return "Joined market price."
    if "no polymarket event resolved" in lower:
        return "No Polymarket event resolved."
    if "no matching market prices" in lower or "no polymarket price joined" in lower:
        return "Event resolved; relevant market found but no usable price."
    return ""


def _classify_import_candidate(
    *,
    home: str,
    away: str,
    kickoff_utc: str,
    generated_at_utc: str,
    timestamp_source: str,
    home_prob: Any,
    draw_prob: Any,
    away_prob: Any,
    match_id: str,
) -> tuple[str, str, bool]:
    if not home or not away or not kickoff_utc or not match_id:
        return "failed_missing_fixture", "Could not resolve PDF fixture to fixtures.csv.", False
    if timestamp_source in {"unavailable", "file_modified_time"} or not generated_at_utc:
        return "needs_manual_timestamp_review", "No reliable PDF report timestamp was parsed.", False
    probs = [home_prob, draw_prob, away_prob]
    if not all(pd.notna(value) for value in probs):
        return "failed_missing_probabilities", "Could not parse a complete home/draw/away probability triplet.", False
    total = sum(float(value) for value in probs)
    if total < 0.98 or total > 1.02:
        return "failed_probability_sum", f"Parsed probabilities sum to {total:.4f}, outside 0.98-1.02.", False
    generated = pd.to_datetime(generated_at_utc, errors="coerce", utc=True)
    kickoff = pd.to_datetime(kickoff_utc, errors="coerce", utc=True)
    if pd.isna(generated) or pd.isna(kickoff):
        return "needs_manual_timestamp_review", "Generated or kickoff timestamp could not be parsed.", False
    if generated >= kickoff:
        return "generated_after_kickoff", "PDF report timestamp is at or after kickoff.", False
    return "ready_for_import", "", True


def _is_duplicate_existing_snapshot(parsed: dict[str, Any], existing_predictions_df: pd.DataFrame | None) -> bool:
    existing = existing_predictions_df.copy() if existing_predictions_df is not None else pd.DataFrame()
    if existing.empty:
        return False
    match_id = str(parsed.get("match_id", "") or "")
    snapshot = str(parsed.get("parsed_report_timestamp_utc", "") or "")
    if not match_id or not snapshot:
        return False
    if "snapshot_utc" not in existing.columns or "match_id" not in existing.columns:
        return False
    return bool(
        (
            existing["match_id"].fillna("").astype(str).eq(match_id)
            & existing["snapshot_utc"].fillna("").astype(str).eq(snapshot)
        ).any()
    )


def _market_snapshot_available(text: Any) -> bool:
    lower = str(text or "").lower()
    return "event resolved" in lower or "joined market price" in lower


def _value_after_label(lines: list[str], label: str) -> str:
    for idx, line in enumerate(lines):
        if line.lower() == label.lower() and idx + 1 < len(lines):
            return lines[idx + 1]
    return ""


def _file_modified_time(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    except OSError:
        return ""


def _ensure_columns_dict(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {col: row.get(col, pd.NA) for col in columns}


def _ensure_frame_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].copy()


def _snapshot_horizon_hours(row: pd.Series) -> Any:
    snapshot = pd.to_datetime(row.get("snapshot_utc"), errors="coerce", utc=True)
    kickoff = pd.to_datetime(row.get("kickoff_utc"), errors="coerce", utc=True)
    if pd.isna(snapshot) or pd.isna(kickoff):
        return pd.NA
    return (kickoff - snapshot).total_seconds() / 3600.0


def _merge_audit_row(row: pd.Series, status: str) -> dict[str, Any]:
    horizon = _snapshot_horizon_hours(row)
    return {
        "source_type": row.get("source_type", ""),
        "source_file": row.get("source_file", ""),
        "prediction_id": row.get("prediction_id", ""),
        "match_id": row.get("match_id", ""),
        "snapshot_utc": row.get("snapshot_utc", ""),
        "kickoff_utc": row.get("kickoff_utc", ""),
        "merge_status": status,
        "snapshot_horizon_hours": horizon,
        "snapshot_group": snapshot_group(horizon),
    }


def _stable_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _status_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "import_candidate_status" not in df.columns:
        return pd.DataFrame(columns=["import_candidate_status", "count"])
    return (
        df["import_candidate_status"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .rename_axis("import_candidate_status")
        .reset_index(name="count")
    )


def _inventory_display(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_file",
        "parsed_home_team",
        "parsed_away_team",
        "parsed_kickoff_utc",
        "parsed_report_timestamp_utc",
        "timestamp_source",
        "timestamp_confidence",
        "parsed_home_win_prob",
        "parsed_draw_prob",
        "parsed_away_win_prob",
        "import_candidate_status",
        "import_warning",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df[[col for col in columns if col in df.columns]].copy()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row.get(col, "")
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
