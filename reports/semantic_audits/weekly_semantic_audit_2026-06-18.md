# Weekly Semantic Audit - 2026-06-18

## Audit Metadata

- Audit date: `2026-06-18T14:26:58+00:00`
- Git branch: `polymarket-alpha-layer`
- Latest commit hash: `08c9306`
- Previous audit report: `reports/semantic_audits/weekly_semantic_audit_2026-06-18.md`
- Change detection basis: Compared against previous audit commit 08c9306.

## Recent Commits

```text
08c9306 gitignore update
be45daa Unknown changes
9fa7a0a Intergration of Gamma API from Polymarket
4fd6dfc gitignore update
5fcd7e7 Creation of the app and applying second layer
```

## Git Status

```text
M .gitignore
 M README.md
 M app.py
?? reports/
?? scripts/weekly_semantic_audit.py
?? src/market_tables.py
?? src/report_charts.py
?? src/report_metrics.py
?? src/schema_registry.py
?? src/timeline.py
?? tests/test_match_report.py
?? tests/test_schema_registry.py
```

## Files Changed Since Last Audit

- `.gitignore`
- `README.md`
- `app.py`
- `scripts/weekly_semantic_audit.py`
- `src/market_tables.py`
- `src/report_charts.py`
- `src/report_metrics.py`
- `src/schema_registry.py`
- `src/timeline.py`
- `tests/test_match_report.py`
- `tests/test_schema_registry.py`

## CSV Schema Checks

| File | Status | Missing columns | Extra columns | Order matches |
| --- | --- | --- | --- | --- |
| `data/fixtures.csv` | ok | - | - | yes |
| `data/team_ratings.csv` | ok | - | - | yes |
| `data/venues.csv` | ok | - | - | yes |
| `data/market_odds.csv` | ok | - | - | yes |
| `data/polymarket_markets.csv` | ok | - | - | yes |
| `data/market_mappings.csv` | ok | - | - | yes |
| `data/prediction_log.csv` | ok | - | - | yes |
| `data/results_log.csv` | ok | - | - | yes |

## Missing Required Columns

- None detected.

## Model-Definition Files Changed

- None detected.

## Metric-Definition Files Changed

- `src/market_tables.py`
- `src/report_metrics.py`

## API / Source Files Changed

- None detected.

## Polymarket Mapping Files Changed

- None detected.

## Dashboard Files Changed

- `app.py`
- `src/market_tables.py`
- `src/report_charts.py`
- `src/report_metrics.py`
- `src/timeline.py`

## Documentation Files Changed

- `README.md`

## Semantic Layer Documentation Changed

- None detected.

## Validation Results

### Compile app.py

- Command: `/Users/fraper/Documents/Polymarket/worldcup_alpha_model/.venv/bin/python -m py_compile app.py`

- Status: passed

### Compile src/*.py

- Command: `/Users/fraper/Documents/Polymarket/worldcup_alpha_model/.venv/bin/python -m py_compile src/alpha.py src/backtesting.py src/cache.py src/climate.py src/config.py src/data_sources.py src/feature_engineering.py src/market_mapping.py src/market_tables.py src/model.py src/odds.py src/polymarket.py src/ratings.py src/report_charts.py src/report_metrics.py src/schema_registry.py src/sensitivity.py src/storage.py src/timeline.py src/utils.py src/weather.py`

- Status: passed

### pytest

- Command: `/Users/fraper/Documents/Polymarket/worldcup_alpha_model/.venv/bin/python -m pytest -q`

- Status: passed


```text
.....................                                                    [100%]
21 passed in 1.42s
```

## Recommended Semantic-Layer Updates

- `Review metric-definition changes and update fair odds, alpha, sensitivity, report, or backtesting definitions where needed.`
- `Review dashboard output definitions and README screenshots/prose if tab labels, tables, or report sections changed.`
- `Reconcile README, AGENTS.md, and semantic-layer documentation so local code remains the source of truth.`
