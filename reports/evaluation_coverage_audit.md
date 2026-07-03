# Evaluation Coverage Audit - 2026-07-03

Evaluation uses the latest valid pre-kickoff prediction snapshot per resolved completed match and scores only 90-minute 1X2 outcomes.

| metric | value |
| --- | --- |
| usable_evaluated_predictions_before_fix | 1.0 |
| prediction_snapshots | 78.0 |
| unique_prediction_match_ids | 35.0 |
| exact_match_id_joins_possible | 2.0 |
| schedule_bridge_joins_possible | 0.0 |
| fixture_bridge_joins_possible | 0.0 |
| normalized_team_date_joins_possible | 0.0 |
| home_away_order_mismatch_matches | 0.0 |
| usable_evaluated_predictions_after_latest_selection | 1.0 |
| excluded_predictions | 77.0 |
| join_status:no_result_found | 70.0 |
| join_status:prediction_generated_after_kickoff | 3.0 |
| join_status:result_not_completed | 3.0 |
| join_status:duplicate_snapshot_not_selected | 1.0 |
| join_status:exact_match_id_join | 1.0 |

The previous Milestone 2 dataset had `1` usable row(s). The regenerated dataset has `1` usable row(s) after duplicate-snapshot selection.

Top exclusion reasons:

| join_status | count |
| --- | --- |
| no_result_found | 70 |
| prediction_generated_after_kickoff | 3 |
| result_not_completed | 3 |
| duplicate_snapshot_not_selected | 1 |

## Prediction Source Coverage

Historical PDF reports can be used as archived prediction snapshots only if their report timestamp is before kickoff. Post-kickoff or ambiguous PDFs are excluded from accuracy claims.

| metric | value |
| --- | --- |
| app_snapshots_available | 61 |
| pdf_snapshots_available | 17 |
| pdfs_found | 21 |
| pdfs_imported | 17 |
| pdfs_rejected | 4 |
| valid_pre_kickoff_pdf_snapshots | 17 |
| evaluated_app_snapshots | 1 |
| evaluated_pdf_snapshots | 0 |
| total_evaluated_matches | 1 |
| calibration_status | sample_too_small |

PDF rejection reasons:

| import_candidate_status | count |
| --- | --- |
| ready_for_import | 17 |
| failed_missing_fixture | 3 |
| failed_missing_probabilities | 1 |
