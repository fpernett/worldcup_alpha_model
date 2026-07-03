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
