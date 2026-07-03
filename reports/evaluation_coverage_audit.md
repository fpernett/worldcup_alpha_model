# Evaluation Coverage Audit - 2026-07-03

Evaluation uses the latest valid pre-kickoff prediction snapshot per resolved completed match and scores only 90-minute 1X2 outcomes.

| metric | value |
| --- | --- |
| usable_evaluated_predictions_before_fix | 2.0 |
| prediction_snapshots | 62.0 |
| unique_prediction_match_ids | 35.0 |
| exact_match_id_joins_possible | 2.0 |
| fixture_bridge_joins_possible | 0.0 |
| normalized_team_date_joins_possible | 0.0 |
| home_away_order_mismatch_matches | 0.0 |
| usable_evaluated_predictions_after_latest_selection | 1.0 |
| excluded_predictions | 61.0 |
| join_status:no_result_found | 55.0 |
| join_status:prediction_generated_after_kickoff | 3.0 |
| join_status:result_not_completed | 2.0 |
| join_status:duplicate_snapshot_not_selected | 1.0 |
| join_status:exact_match_id_join | 1.0 |

The previous Milestone 2 dataset had `2` usable row(s). The regenerated dataset has `1` usable row(s) after duplicate-snapshot selection.

Top exclusion reasons:

| join_status | count |
| --- | --- |
| no_result_found | 55 |
| prediction_generated_after_kickoff | 3 |
| result_not_completed | 2 |
| duplicate_snapshot_not_selected | 1 |