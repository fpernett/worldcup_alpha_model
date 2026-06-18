# FIFA World Cup Match Alpha Model

Local Streamlit dashboard for transparent FIFA World Cup match probability estimates.

The app estimates expected goals, 1X2 probabilities, fair odds, scoreline probabilities, totals, BTTS, handicap approximations, environmental adjustments, model confidence, and alpha EV versus optional market odds.

This project is for statistical alpha estimates only. It does not provide investment advice, staking advice, bet sizing, Kelly sizing, or trading recommendations.

## 1. Installation

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The app is designed for macOS with Python 3.10+.

## 2. Run The App

```bash
source .venv/bin/activate
streamlit run app.py
```

The dashboard opens in your browser. Select a date window, then select one or more matches to model. The app does not model every fixture unless you select those fixtures.

For each selected match, open the **Full report** tab for a report-style view with Alpha Read, data support, goal distribution, outcome donut, league context, score timeline, scoreline matrix, expected goals, team ratings, climate factors, model information, and grouped market value tables.

## 3. Update Data

The app runs without API keys and falls back to CSV files. The sidebar button **Update API data** refreshes only the sources that are configured. Each refresh step reports `success`, `skipped`, `partial`, or `failed`.

Refresh all configured sources and local cache:

```bash
source .venv/bin/activate
python scripts/update_all.py --start-date 2026-06-17 --end-date 2026-06-24
```

Individual update commands:

```bash
python scripts/update_fixtures.py --start-date 2026-06-17 --end-date 2026-06-24
python scripts/update_team_ratings.py
python scripts/update_weather.py --date 2026-06-17 --time 12:00
```

Successful API responses are cached in `data/cache/` as raw JSON plus normalized CSV where useful. Manual CSV files remain valid overrides and fallbacks.

## 4. Optional API Keys

Copy `.env.example` to `.env` and fill only the services you have:

```bash
cp .env.example .env
```

Supported optional variables:

```text
FOOTBALL_API_URL=https://api.football-data.org/v4
FOOTBALL_API_KEY=
RATINGS_API_URL=
WEATHER_API_URL=https://api.open-meteo.com/v1/forecast
WEATHER_API_KEY=
ODDS_API_URL=
ODDS_API_KEY=
POLYMARKET_API_URL=
POLYMARKET_CLOB_API_URL=
POLYMARKET_GAMMA_API_URL=https://gamma-api.polymarket.com
```

If an API key or API URL is missing, or an API call fails, the app continues with cache or local CSV data.

### Football Fixtures And Results

`FOOTBALL_API_URL` supports football-data.org. Use the base URL:

```text
https://api.football-data.org/v4
```

The request uses:

```text
GET /matches?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD
X-Auth-Token: FOOTBALL_API_KEY
```

The response is normalized into the same schema as `data/fixtures.csv`. Match results are cached into `data/cache/recent_matches_latest.csv` when available and can feed local Elo, attack, defense, and recent-form calculations.

### Weather

`WEATHER_API_URL` supports Open-Meteo:

```text
https://api.open-meteo.com/v1/forecast
```

No weather API key is required. The app sends venue latitude/longitude and requests hourly:

- `temperature_2m`
- `relative_humidity_2m`
- `precipitation`
- `wind_speed_10m`

The weather row closest to kickoff UTC is selected. If the roof is expected to be closed, outdoor weather effects are strongly reduced for effective playing conditions.

## 5. Manual Data Files

### Fixtures

Edit `data/fixtures.csv` to add or change games:

```text
match_id,date_utc,time_utc,competition,group,home,away,venue,city,country
```

Use UTC dates and times. The `home`, `away`, and `venue` names should match `data/team_ratings.csv` and `data/venues.csv` where possible.

### Team Ratings

Edit `data/team_ratings.csv`:

```text
team,elo,attack,defense,recent_form,fifa_rank_proxy,training_temp_c,training_humidity_pct,data_quality,last_updated,notes
```

Attack, defense, and recent form are 0-1 values. If recent match history is later added in `data/recent_matches.csv`, the app can recalculate attack, defense, and form with recency weighting.

### Venues

Edit `data/venues.csv`:

```text
venue,city,country,latitude,longitude,altitude_m,temp_c,humidity_pct,wind_kmh,precipitation_mm,roof_expected_closed,neutral_site
```

Use `roof_expected_closed = 1` for indoor or roof-closed matches. Latitude and longitude are required for Open-Meteo weather refreshes. If they are missing or zero, weather falls back to local CSV values.

### Market Odds

Edit `data/market_odds.csv`:

```text
match_id,market,selection,odds,source,last_updated
```

Decimal odds are used. Missing odds do not break the app; alpha EV is left blank.

### Polymarket Markets

The app uses the public Polymarket Gamma API as the default read-only market source:

```text
https://gamma-api.polymarket.com/markets
```

No Polymarket API key is required for this market-data fetch. The sidebar button
**Update Polymarket markets** refreshes Gamma data, writes the cache files, and
then filters the loaded markets by the sidebar search query.

Edit `data/polymarket_markets.csv` to paste Polymarket market data manually:

```text
market_id,question,slug,event_title,category,start_date,end_date,active,closed,outcomes,yes_price,no_price,liquidity,volume,source,last_updated
```

Prices can be entered as probabilities (`0.42`) or cents (`42`). The app normalizes them to probabilities internally.

If `POLYMARKET_GAMMA_API_URL` or `POLYMARKET_API_URL` is configured, that URL
overrides the default Gamma base URL. Successful API responses are cached in:

```text
data/cache/polymarket_markets_raw.json
data/cache/polymarket_markets_normalized.csv
```

If Gamma is unavailable or the API call fails, the app uses cache or
`data/polymarket_markets.csv`.

### Manual Market Mappings

Edit `data/market_mappings.csv` to confirm or override a fuzzy mapping:

```text
match_id,market_id,market_type,model_side,polymarket_side,manual_confirmed,notes
```

Example:

```text
66457042,pm_market_id,match_winner_home,home_win,YES,1,England win market confirmed manually
```

Manual mappings override automatic fuzzy matching.

## 6. How The Model Works

Expected goals are based on:

- Elo difference
- team attack strength
- opponent defense strength
- recent form
- match venue temperature
- humidity
- altitude
- wind
- precipitation
- roof status
- difference between venue conditions and each team's usual training climate

Attack and defense can be calculated from recent match history:

- attack = recency-weighted goals scored per match, lightly adjusted for opponent strength
- defense = inverse of recency-weighted goals conceded per match, lightly adjusted for opponent strength
- recent form = win 1.0, draw 0.5, loss 0.0, with recency weighting

The score matrix uses an independent Poisson model. Probabilities are normalized over the displayed score grid.

Environmental effects are deliberately conservative. Roof-closed venues reduce outdoor weather impact by 85%. Altitude mostly matters above about 1,200 meters. Wind and precipitation mainly reduce total-goals quality rather than heavily favoring one team.

## 7. Full Match Report

The **Full report** tab is built from local model outputs and local/API data already loaded by the app. It does not scrape AlphaMetri or any login-protected page.

Report sections:

- **Alpha Read**: model confidence, expected goals, best local EV, and best Polymarket alpha gap when market data is available.
- **Data support**: historical match counts for each team, H2H count, training data range, and fallback warnings.
- **Goal distribution**: marginal home and away goal probabilities from the score matrix.
- **Match outcome donut**: home win, draw, and away win probabilities.
- **League context**: historical or fallback baseline comparison for goals, result rates, BTTS, and over 2.5.
- **Score timeline**: cumulative goal probabilities from a simple xG hazard curve.
- **Scoreline matrix**: compact 0-4 heatmap plus expandable full matrix.
- **Expected goals chart**: adjusted xG with simple uncertainty bands.
- **Team ratings**: attack and defense inputs with percentile labels.
- **Climate factors**: altitude, temperature, humidity, precipitation, and wind categories with conservative multipliers.
- **Model information**: model version, training data count/range, source status, and backtest placeholder.
- **Market value tables**: grouped decimal-odds alpha and Polymarket alpha screens.

Real metrics in v1:

- model probabilities, expected goals, scoreline probabilities, fair odds, alpha EV, and Polymarket alpha gaps;
- historical support counts when `data/recent_matches.csv` or cached football results exist;
- climate factors derived from the same venue/environment inputs used by the model.

Placeholders or approximations in v1:

- RPS is explicitly shown as `RPS placeholder / not yet backtested`;
- base xG is not separately stored yet, so adjusted xG is used as the base value in the expected-goals report chart;
- league context uses fallback baselines when historical match data is unavailable.

## 8. Alpha EV

Fair odds:

```text
fair_odds = 1 / probability
```

Alpha EV:

```text
alpha_EV = model_probability x market_decimal_odds - 1
```

Example:

```text
model probability = 0.60
market odds = 2.00
alpha_EV = 0.60 x 2.00 - 1 = 0.20
```

That means the offered decimal odds are above the model's fair estimate. It is not a staking recommendation.

### Polymarket Alpha Gap

For Polymarket YES/NO prices:

```text
fair_price_cents = model_probability x 100
alpha_gap_cents = fair_price_cents - polymarket_price_cents
alpha_EV = model_probability / market_probability - 1
```

Example: if England win probability is `0.491`, the YES fair price is `49.1` cents. If the Polymarket YES price is `42.0` cents, the alpha gap is `+7.1` cents.

Signal labels are simple screens:

- Strong: alpha gap at least 8 cents, high/manual mapping confidence, and enough liquidity.
- Moderate: alpha gap at least 5 cents.
- Weak: alpha gap at least 3 cents.
- No signal: alpha gap between -3 and +3 cents.
- Avoid: alpha gap below -3 cents.

These labels are not staking advice.

## 9. Model Confidence

Confidence is shown as High, Moderate, or Low.

It depends on:

- team rating availability
- recent match-history availability
- venue/weather availability
- market odds availability
- how much fallback/manual data was used

Example: Moderate confidence may mean team ratings and venue data are available, but recent match history is partly inferred.

## 10. Source Diagnostics

The sidebar shows:

- fixtures source
- ratings source
- weather source
- odds source
- last updated timestamp
- API configured status
- warnings or missing columns

Source labels are `API`, `cache`, or `local CSV`.

## 11. Prediction Snapshots And Backtesting

Use the **Polymarket alpha** tab for a selected match, then click **Save prediction snapshot**. Rows append to:

```text
data/prediction_log.csv
```

After the match is complete, add results to:

```text
data/results_log.csv
```

Required result columns:

```text
match_id,home,away,home_goals,away_goals,result_home_win,result_draw,result_away_win,over_2_5,under_2_5,btts_yes,btts_no,completed,result_source,last_updated
```

The **Backtesting** tab loads both logs and reports simple Brier score, log loss, mean alpha gap, hit rate by signal strength, and calibration buckets when completed results exist.

## 12. Weekly Semantic Audit

Run a lightweight local audit when you want to check whether the code, CSV schemas, dashboard outputs, or semantic-layer documentation have drifted:

```bash
source .venv/bin/activate
python scripts/weekly_semantic_audit.py
```

The audit is read-only except for writing a Markdown report to:

```text
reports/semantic_audits/weekly_semantic_audit_YYYY-MM-DD.md
```

It checks:

- current Git status and recent commits;
- files changed since the previous semantic audit;
- CSV headers against `src/schema_registry.py`;
- model, metric, API/source, Polymarket mapping, dashboard, README, AGENTS, and semantic-layer documentation changes;
- `python -m py_compile app.py`;
- `python -m py_compile src/*.py`;
- `pytest -q` when tests are present.

The audit does not require internet access, API keys, a cloud scheduler, or Polymarket credentials. It does not modify model code, CSV data, cache files, commits, or notifications.

`data/cache/` and `.env` remain ignored. Small Markdown reports under `reports/semantic_audits/` are allowed through `.gitignore` so they can be tracked if useful.

## 13. Validate The Model

Run:

```bash
source .venv/bin/activate
python -m pytest -q
python scripts/update_all.py --start-date 2026-06-17 --end-date 2026-06-18
```

The tests check probability sums, fair odds, score matrix normalization, missing odds, missing weather, roof-closed weather dampening, Polymarket price normalization, YES/NO mapping, alpha gaps, sensitivity output shape, prediction-log append, missing-API fallback, and Full report helper outputs.

## 14. Known Limitations

- Ratings are transparent priors unless connected to a real ratings API or recent match-history file.
- Injuries, lineups, tactical changes, and motivation are not automatically modeled.
- Weather is approximate unless a weather API is configured.
- Polymarket fuzzy matching can be wrong; use `data/market_mappings.csv` for confirmed mappings.
- Low-liquidity markets and missing prices should be treated as data warnings, not opportunities.
- The Poisson model assumes independent scoring rates.
- Training climate is approximate and should not be overinterpreted.
- Full report v1 uses simple report-layer uncertainty bands and a score-timeline hazard approximation.
- The app does not include trade execution, staking, bet sizing, Kelly criterion, or investment recommendations.
