# UI Verification Notes

Method: Streamlit was started locally with `.venv/bin/python -m streamlit run app.py --server.port 8501 --server.headless true`. In-app browser and Chrome-control surfaces were unavailable in this Codex session, so fixture-level artifacts were generated through the same model, xG, Polymarket join, market-table, and headline helpers used by `app.py`.

| Fixture | Report | Unresolved bracket warning | Base xG TODO | Unexplained n/a | Blank cells | Fake Polymarket gap | Vague market join status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Argentina vs Cape Verde | reports/ui_verification/report_argentina_cape_verde_after_fix.html | Gone | Gone | Gone | Gone | Gone; headline gap is backed by joined market prices | Gone; status is `Joined market price.` |
| Australia vs Egypt | reports/ui_verification/report_australia_egypt_after_fix.html | Gone | Gone | Gone | Gone | Gone; headline gap is backed by joined market prices | Gone; status is `Joined market price.` |

Fresh artifact check: no generated HTML report contains `TODO`, `base xG is not separately stored`, `n/a`, `Best Polymarket gap`, or `No price joined`.
