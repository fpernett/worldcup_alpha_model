from __future__ import annotations

import pandas as pd

from src.evaluation import build_formal_evaluation_dataset
from src.pdf_prediction_import import (
    audit_rows_to_prediction_ledger,
    build_enriched_prediction_ledger,
    load_timestamp_overrides,
    parse_prediction_report_text,
    snapshot_group,
)


def _report_text(timestamp: str = "30/06/2026, 21:37", home_prob: str = "17.1%") -> str:
    return f"""
FIFA World Cup Match Alpha Model
Mexico vs Ecuador
FIFA World Cup | Round of 32 | 2026-07-01 01:00 UTC | Estadio Azteca
Primary model
Baseline external-calibrated
Behavior layer
Diagnostic only
Alpha Read
Model confidence
Moderate
Expected goals
0.77 - 1.66
Best local EV
n/a
Best Polymarket gap
No joined market price
Data Support
mexico matches
900
ecuador matches
600
H2H matches
12
Training matches
1,500
Goal Distribution And Outcome
Goals
Probability %
Mexico win
{home_prob}
Draw
24.2%
Ecuador win
58.8%
Match Outcome
{timestamp}
World Cup Alpha Model
"""


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "wc2026_79",
                "date_utc": "2026-07-01",
                "time_utc": "01:00",
                "competition": "FIFA World Cup",
                "group": "Round of 32",
                "home": "Mexico",
                "away": "Ecuador",
                "venue": "Estadio Azteca",
            }
        ]
    )


def test_parse_fixture_identity_kickoff_probabilities_and_support() -> None:
    parsed = parse_prediction_report_text(_report_text(), source_file="mex.pdf", fixtures_df=_fixtures())

    assert parsed["parsed_home_team"] == "Mexico"
    assert parsed["parsed_away_team"] == "Ecuador"
    assert parsed["parsed_kickoff_utc"] == "2026-07-01T01:00:00+00:00"
    assert parsed["match_id"] == "wc2026_79"
    assert parsed["parsed_home_win_prob"] == 0.171
    assert parsed["parsed_draw_prob"] == 0.242
    assert parsed["parsed_away_win_prob"] == 0.588
    assert parsed["home_data_support_matches"] == 900
    assert parsed["training_matches"] == 1500
    assert parsed["import_candidate_status"] == "ready_for_import"


def test_report_timestamp_without_timezone_assumes_stockholm_medium_confidence() -> None:
    parsed = parse_prediction_report_text(_report_text(), source_file="mex.pdf", fixtures_df=_fixtures())

    assert parsed["parsed_report_timestamp_utc"] == "2026-06-30T19:37:00+00:00"
    assert parsed["timestamp_source"] == "parsed_from_pdf"
    assert parsed["timestamp_confidence"] == "medium"
    assert parsed["valid_pre_kickoff_snapshot"] is True


def test_pdf_metadata_timestamp_is_used_when_text_timestamp_missing() -> None:
    parsed = parse_prediction_report_text(
        _report_text(timestamp=""),
        source_file="mex.pdf",
        fixtures_df=_fixtures(),
        pdf_metadata={"creationDate": "D:20260630193745+00'00'"},
    )

    assert parsed["parsed_report_timestamp_utc"] == "2026-06-30T19:37:45+00:00"
    assert parsed["timestamp_confidence"] == "high"
    assert parsed["import_candidate_status"] == "ready_for_import"


def test_generated_after_kickoff_rejected() -> None:
    parsed = parse_prediction_report_text(
        _report_text(timestamp="01/07/2026, 03:30"),
        source_file="late.pdf",
        fixtures_df=_fixtures(),
    )

    assert parsed["import_candidate_status"] == "generated_after_kickoff"
    assert parsed["valid_pre_kickoff_snapshot"] is False


def test_missing_timestamp_requires_manual_review() -> None:
    parsed = parse_prediction_report_text(
        _report_text(timestamp=""),
        source_file="missing.pdf",
        fixtures_df=_fixtures(),
    )

    assert parsed["import_candidate_status"] == "needs_manual_timestamp_review"
    assert parsed["valid_pre_kickoff_snapshot"] is False


def test_probability_sum_validation() -> None:
    parsed = parse_prediction_report_text(
        _report_text(home_prob="50.0%"),
        source_file="bad_probs.pdf",
        fixtures_df=_fixtures(),
    )

    assert parsed["import_candidate_status"] == "failed_probability_sum"


def test_manual_timestamp_override(tmp_path) -> None:
    override_path = tmp_path / "overrides.csv"
    pd.DataFrame(
        [
            {
                "source_file": "mex.pdf",
                "generated_at_utc": "2026-06-30T18:00:00+00:00",
                "reason": "reviewed footer",
                "reviewer": "tester",
                "reviewed_at_utc": "2026-07-03T00:00:00+00:00",
            }
        ]
    ).to_csv(override_path, index=False)
    overrides = load_timestamp_overrides(override_path)

    parsed = parse_prediction_report_text(
        _report_text(timestamp=""),
        source_file="mex.pdf",
        fixtures_df=_fixtures(),
        timestamp_override=overrides["mex.pdf"],
    )

    assert parsed["parsed_report_timestamp_utc"] == "2026-06-30T18:00:00+00:00"
    assert parsed["timestamp_source"] == "manual_override"
    assert parsed["timestamp_confidence"] == "high"


def test_duplicate_pdf_exclusion_via_existing_snapshot() -> None:
    parsed = parse_prediction_report_text(_report_text(), source_file="mex.pdf", fixtures_df=_fixtures())
    existing = pd.DataFrame([{"match_id": parsed["match_id"], "snapshot_utc": parsed["parsed_report_timestamp_utc"]}])
    audit = pd.DataFrame([parsed])
    audit.loc[0, "import_candidate_status"] = (
        "duplicate_existing_snapshot"
        if ((existing["match_id"] == parsed["match_id"]) & (existing["snapshot_utc"] == parsed["parsed_report_timestamp_utc"])).any()
        else audit.loc[0, "import_candidate_status"]
    )

    assert audit.loc[0, "import_candidate_status"] == "duplicate_existing_snapshot"


def test_enriched_ledger_merge_and_snapshot_groups() -> None:
    parsed = parse_prediction_report_text(_report_text(), source_file="mex.pdf", fixtures_df=_fixtures())
    backfill = audit_rows_to_prediction_ledger(pd.DataFrame([parsed]))
    existing = pd.DataFrame(
        [
            {
                "prediction_id": "app1",
                "snapshot_utc": "2026-06-30T10:00:00+00:00",
                "match_id": "wc2026_79",
                "kickoff_utc": "2026-07-01T01:00:00+00:00",
                "home": "Mexico",
                "away": "Ecuador",
                "home_win_prob": 0.2,
                "draw_prob": 0.2,
                "away_win_prob": 0.6,
            }
        ]
    )

    duplicated_backfill = pd.concat([backfill, backfill], ignore_index=True)
    enriched, audit = build_enriched_prediction_ledger(existing, duplicated_backfill)

    assert len(enriched) == 2
    assert set(enriched["source_type"]) == {"app_snapshot", "pdf_report"}
    assert snapshot_group(-1) == "invalid_post_kickoff"
    assert "same_day_6_to_24h" in set(enriched["snapshot_group"])
    assert not audit.empty


def test_pdf_backfill_row_evaluates_against_results_ledger_latest_snapshot() -> None:
    parsed = parse_prediction_report_text(_report_text(), source_file="mex.pdf", fixtures_df=_fixtures())
    backfill = audit_rows_to_prediction_ledger(pd.DataFrame([parsed]))
    results = pd.DataFrame(
        [
            {
                "match_id": "wc2026_79",
                "date_utc": "2026-07-01",
                "competition": "FIFA World Cup",
                "home": "Mexico",
                "away": "Ecuador",
                "home_goals": 2,
                "away_goals": 0,
                "actual_result": "home_win",
                "result_semantics": "90-minute regular time",
                "result_source": "test",
            }
        ]
    )

    evaluation = build_formal_evaluation_dataset(backfill, results, fixtures_df=_fixtures())

    assert len(evaluation) == 1
    assert evaluation.iloc[0]["prediction_source"] == "pdf_report"
    assert evaluation.iloc[0]["actual_result_1x2"] == "home_win"


def test_post_kickoff_pdf_not_written_to_clean_backfill() -> None:
    parsed = parse_prediction_report_text(
        _report_text(timestamp="01/07/2026, 03:30"),
        source_file="late.pdf",
        fixtures_df=_fixtures(),
    )

    backfill = audit_rows_to_prediction_ledger(pd.DataFrame([parsed]))

    assert backfill.empty
