# Product Readiness Report

Date: 2026-07-03

## Milestone 1 fixes

- Removed production-facing TODO/placeholder language from expected-goals reporting.
- Added staged xG decomposition: baseline, rating-adjusted, form-adjusted, venue/weather-adjusted, and final adjusted xG.
- Replaced unresolved bracket warnings with count-aware fixture status copy.
- Added fixture Data Health counts for resolved automatically, pending prior result, requires manual mapping, and source fixture unresolved.
- Tightened Polymarket market join states to exact subscriber-facing statuses:
  - No Polymarket event resolved.
  - Event resolved; no relevant market found.
  - Event resolved; relevant market found but no usable price.
  - Joined market price.
  - Model-only fair value.
- Changed headline Polymarket logic so a gap headline is shown only from eligible joined price rows. Otherwise the card shows `Polymarket price status: No joined market price`.
- Changed local odds headline logic so generated/model-only odds are labeled `Model-only edge`; missing local odds show `No local odds`.
- Made calibration presentation distinguish raw baseline probabilities, review-only calibrated candidates, and small-sample conservative shrinkage.

## Streamlit verification

- App start verified with `.venv/bin/python -m streamlit run app.py --server.port 8501 --server.headless true`.
- Initial rendered Streamlit page was inspected through Streamlit's app-testing runtime after the server start. The Data Health copy now says `3 future fixtures pending prior match results.` and no longer shows the old unresolved bracket-slot warning.
- The in-app browser and Chrome-control surfaces were unavailable in this Codex session, so fixture-level subscriber reports were generated through the same model, xG, Polymarket join, market-table, and headline helper functions used by `app.py`.
- Fresh verification artifacts are saved under `reports/ui_verification/`.

## Test verification

- Deterministic smoke tests cover Argentina vs Cape Verde and Australia vs Egypt report output in `tests/test_ui_report_smoke.py`.
- Focused tests cover fixture resolution, fixture diagnostics, xG decomposition, Polymarket join statuses, market table display, and calibration status.
- Final validation passed: `git diff --check`, `.venv/bin/python -m py_compile app.py src/*.py scripts/*.py tests/*.py`, and `.venv/bin/python -m pytest` with `380 passed`.

## Fixture readiness

### Argentina vs Cape Verde

Status: production-clean for the verified report artifact.

- Unresolved bracket warning: gone.
- Base xG TODO: gone.
- Unexplained n/a: gone from generated subscriber report.
- Blank cells: gone from generated subscriber tables.
- Polymarket gap: backed by joined market price rows at generation time.
- Market join status: `Joined market price.`

### Australia vs Egypt

Status: production-clean for the verified report artifact.

- Unresolved bracket warning: gone.
- Base xG TODO: gone.
- Unexplained n/a: gone from generated subscriber report.
- Blank cells: gone from generated subscriber tables.
- Polymarket gap: backed by joined market price rows at generation time.
- Market join status: `Joined market price.`

## Calibration status

- The primary model remains the raw baseline external-calibrated model.
- Walk-forward calibrated variants are review-only unless they beat the raw baseline on both Brier score and log loss.
- When completed pre-kickoff history is too small, the UI shows: `Calibration sample too small; using conservative shrinkage.`
- The UI must not imply the model is fully calibrated for paid users until larger walk-forward history supports that claim.

## Known model limitations

- The model is a transparent statistical screen, not betting, staking, investment, wallet, order-placement, or execution software.
- Generated local benchmark odds are model-only context, not tradable market prices.
- Polymarket price availability depends on event resolution, supported market semantics, mapping confidence, and price freshness.
- Weather, venue, and training-climate effects are approximate and intentionally conservative.
- Backtesting quality depends on append-only pre-kickoff prediction snapshots and validated 90-minute result rows.
- RPS remains unavailable until ranked-probability scoring is implemented and backtested.

## Subscription readiness

Decision: not ready for broad paid-subscriber launch.

The Milestone 1 subscriber-facing issues are fixed for the two verified reports, and the app starts locally. The remaining blockers are:

- Browser-based click-through QA needs to be repeated in a working browser-control session or manually by a human reviewer.
- Calibration history is still too small to market the model as fully calibrated.
- Polymarket market-price freshness and semantic-match checks should be monitored over multiple live fixtures.
- More completed prediction snapshots are needed before performance claims can be made responsibly.

Current recommended status: internal beta / controlled reviewer access only.

## Milestone 2 predictive calibration addendum

Date: 2026-07-03

Milestone 2 adds a formal evaluation pipeline and does not change the Milestone 1 production-clean report status above.

- Generated `reports/evaluation_dataset.csv` from pre-kickoff prediction snapshots joined to completed 90-minute results.
- Generated calibration outputs: `reports/calibration_summary.csv`, `reports/calibration_by_bin.csv`, `reports/calibration_by_confidence.csv`, `reports/calibration_by_match_context.csv`, `reports/worst_model_misses.csv`, and `reports/calibration_report.md`.
- Generated walk-forward candidate outputs: `reports/calibration_model_comparison.csv` and `reports/walk_forward_calibration_results.csv`.
- Generated market benchmark outputs: `reports/model_vs_market_benchmark.csv` and `reports/model_vs_market_report.md`.
- Generated recent diagnostic post-mortem outputs: `reports/recent_match_postmortem.csv` and `reports/recent_match_postmortem.md`.
- Added evidence-based confidence scoring with explicit caps for small calibration sample, missing market joins, behavior disagreement, venue uncertainty, knockout/draw risk, and extreme favorite probabilities.

Current Milestone 2 evidence:

- Usable evaluated prediction snapshots: 2.
- Raw model on usable rows: Brier 0.4498, log loss 0.7929, top-pick accuracy 100.0%.
- Calibrated probabilities are not production-eligible; the sample is too small for walk-forward promotion.
- Market benchmark sample is too small for reliable conclusions.
- Mexico vs Ecuador is documented as a diagnostic miss; it should trigger low-confidence, behavior-disagreement, venue-context, and calibration cautions in similar future cases.

Current subscription readiness after Milestone 2: still not ready for broad paid-subscriber launch.

Exact remaining blockers:

- Need materially more completed pre-kickoff prediction snapshots before accuracy, calibration, or market-edge claims are credible.
- Need semantically matched 90-minute 1X2 market samples before claiming the model beats or complements market-implied probabilities.
- Need browser/manual click-through QA in addition to generated-report and unit-test verification.
- Need result-ledger coverage for the recent diagnostic knockout matches before those cases can enter formal calibration metrics rather than diagnostic post-mortem only.
