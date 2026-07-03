# Result/Prediction Join Audit - 2026-07-03

This audit reconciles saved pre-match prediction snapshots with the existing completed-results ledger. It does not scrape PDFs or build a separate result-ingestion path.

## Coverage

- Prediction snapshots: `62`
- Unique prediction match_ids: `35`
- Result ledger rows: `47`
- Unique result match_ids: `47`
- Completed-results source rows: `6`
- Exact match_id joins possible: `2`
- Schedule bridge joins possible: `0`
- Fixture-bridge joins possible: `0`
- Normalized team/date joins possible: `0`
- Usable evaluated predictions before this fix: `1`
- Usable evaluated predictions after latest-snapshot selection: `1`

## Exclusions

| join_status | count |
| --- | --- |
| no_result_found | 55 |
| prediction_generated_after_kickoff | 3 |
| result_not_completed | 2 |
| duplicate_snapshot_not_selected | 1 |

## Do we still need PDF backfill?

Structured CSVs currently provide `1` usable evaluated row(s), below the `30`-row high-confidence calibration threshold. PDFs may materially increase the sample only if their generated-before-kickoff timestamps can be verified.

PDF import remains secondary. Structured CSV joins must be correct first, and PDF timestamps need separate provenance review before they can support calibration claims.