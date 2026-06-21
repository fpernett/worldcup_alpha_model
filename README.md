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

Open the **Historical behavior** tab to inspect the calibrated recent-window behavior layer, including attacking/defensive behavior, schedule strength, expected-performance residuals, behavior driver matches, opponent/competition breakdowns, goal-market rates, recent-vs-all-time diagnostics, environment response samples, warnings, and the final model input impact.

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

Historical match ingestion is opt-in from the all-source script:

```bash
source .venv/bin/activate
.venv/bin/python scripts/update_all.py --start-date 2022-01-01 --end-date 2026-06-18 --update-historical
```

Or run it directly:

```bash
.venv/bin/python scripts/update_historical_matches.py --start-date 2022-01-01 --end-date 2026-06-18 --rebuild-behavior
```

Import historical match results from any local CSV:

```bash
.venv/bin/python scripts/import_historical_csv.py --input data/raw/results.csv --source public_csv --rebuild-behavior
```

Optional team subset:

```bash
.venv/bin/python scripts/update_historical_matches.py --start-date 2022-01-01 --end-date 2026-06-18 --teams England Croatia Switzerland "Bosnia and Herzegovina"
.venv/bin/python scripts/import_historical_csv.py --input data/raw/results.csv --source public_csv --teams England Croatia Switzerland "Bosnia and Herzegovina" --rebuild-behavior
```

The historical refresh attempts to:

1. fetch recent match results when a football API is configured;
2. merge those results into `data/historical_matches.csv` in team-match long format;
3. rebuild `data/team_behavior.csv`;
4. report the source status for historical matches and team behavior.

If API access is unavailable, the refresh uses local CSV fallback data and leaves manual historical rows intact.

Audit the calibrated behavior blend:

```bash
.venv/bin/python scripts/audit_behavior_calibration.py
```

The audit prints model-team behavior diagnostics and saves `reports/behavior_calibration_audit_YYYY-MM-DD.md`.

Rebuild rolling Elo on historical rows and refresh behavior:

```bash
.venv/bin/python scripts/rebuild_elo.py --rebuild-behavior
```

Audit recent schedule strength:

```bash
.venv/bin/python scripts/audit_schedule_strength.py
```

The schedule audit prints opponent-Elo diagnostics and saves `reports/schedule_strength_audit_YYYY-MM-DD.md`.

Audit Elo calibration and residual-based behavior:

```bash
.venv/bin/python scripts/audit_elo_calibration.py
.venv/bin/python scripts/audit_performance_residuals.py
```

These audits compare rolling Elo with manual ratings, then show whether recent attack and defense were better or worse than expected given opponent Elo. Reports are saved under `reports/`.

Audit behavior drivers and final model inputs:

```bash
.venv/bin/python scripts/audit_behavior_drivers.py --teams Norway England Argentina Brazil Uruguay Croatia Morocco "United States" Mexico
.venv/bin/python scripts/audit_model_inputs.py
```

The driver audit shows top attack/defense driver matches, opponent-tier mix, competition-type mix, and behavior warnings. The model-input audit shows the actual attack, defense, and recent-form values used by the model after manual-rating blends and delta caps.

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

Historical match ingestion uses the same endpoint with `status=FINISHED`, chunks long date ranges, writes raw cache files under `data/cache/`, and then converts match results into two long-format historical rows. For v1, xG, shots, possession, and weather fields are optional and can remain blank.

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

Manual ratings remain the base override values. When `data/team_behavior.csv` has acceptable quality, the app conservatively blends behavior into model inputs and exposes `attack_source`, `defense_source`, `recent_form_source`, and `behavior_blend_used` in the dashboard.

### Historical Matches

Edit `data/historical_matches.csv` to add historical team-match rows:

```text
match_id,date_utc,competition,competition_type,team,opponent,is_home,is_neutral,venue,city,country,team_goals,opponent_goals,team_xg,opponent_xg,shots_for,shots_against,shots_on_target_for,shots_on_target_against,possession_pct,opponent_elo,team_elo_pre,temperature_c,humidity_pct,altitude_m,wind_kmh,precipitation_mm,roof_closed,source,last_updated
```

The table is long format. If Switzerland played Bosnia, add one row for Switzerland vs Bosnia and one row for Bosnia vs Switzerland. Missing xG, shots, possession, or weather values can be left blank.

Required v1 result fields are date, teams, opponents, goals, and competition when available. Optional fields are:

- `team_xg`, `opponent_xg`
- `shots_for`, `shots_against`
- `shots_on_target_for`, `shots_on_target_against`
- `possession_pct`
- `temperature_c`, `humidity_pct`, `wind_kmh`, `precipitation_mm`, `roof_closed`

Manual rows are preserved during ingestion when `source = manual`. New API rows are deduplicated by `match_id + team + opponent`; if `match_id` is missing, the fallback key uses date, team, opponent, and score.

### Generic Historical Results CSV Import

`scripts/import_historical_csv.py` imports match-result CSVs from any local source and converts each source match into two long-format rows.

Minimum accepted input format:

```text
date,home_team,away_team,home_score,away_score
2026-06-01,England,Croatia,2,1
```

Optional columns:

```text
tournament,competition,city,country,neutral,venue
```

The importer accepts flexible column names such as `home`, `away`, `home_goals`, `away_goals`, `match_date`, `stadium`, and `neutral_site`. Team names are normalized through `data/team_name_aliases.csv`, and competition type is classified transparently from the tournament or competition string.

After import, diagnostics report source rows read, rows converted, rows added, rows deduplicated, final historical rows, final behavior rows, teams with zero rows, and unknown team-name warnings.

### Team Name Aliases

Edit `data/team_name_aliases.csv` when an API source uses a different country name:

```text
alias,canonical
```

Examples include `USA -> United States`, `Korea Republic -> South Korea`, and `Czech Republic -> Czechia`.

### Team Behavior

`data/team_behavior.csv` is rebuilt from historical matches:

```text
team,reference_date,behavior_window_start,behavior_window_end,matches_available_all_time,matches_used_recent,oldest_match_used,latest_match_used,behavior_config_name,n_matches,weighted_goals_for,weighted_goals_for_raw,weighted_goals_for_adjusted,weighted_goals_against,weighted_goals_against_raw,weighted_goals_against_adjusted,weighted_goal_difference,all_time_goals_for,all_time_goals_against,attack_index,attack_index_raw,attack_index_adjusted_old,attack_index_adjusted,attack_index_residual,attack_index_residual_raw,attack_index_residual_robust,attack_index_final,defense_index,defense_index_raw,defense_index_adjusted_old,defense_index_adjusted,defense_index_residual,defense_index_residual_raw,defense_index_residual_robust,defense_index_final,weighted_goal_for_residual,weighted_goal_for_residual_raw,weighted_goal_for_residual_robust,weighted_goal_against_residual,weighted_goal_against_residual_raw,weighted_goal_against_residual_robust,weighted_result_residual,residual_coverage_recent,mean_blowout_weight_recent,top_3_attack_residual_share,top_3_defense_residual_share,recent_form_index,weighted_btts_rate,weighted_over_2_5_rate,clean_sheet_rate,failed_to_score_rate,environment_response_index,environment_sample_size,attack_data_quality,defense_data_quality,form_data_quality,environment_data_quality,overall_data_quality,behavior_warning,sample_size_warning,staleness_warning,opponent_quality_warning,mean_opponent_elo_recent,median_opponent_elo_recent,min_opponent_elo_recent,max_opponent_elo_recent,opponent_elo_coverage_recent,strong_opponent_match_count,weak_opponent_match_count,schedule_strength_label,schedule_strength_warning,opponent_adjustment_warning,residual_warning,residual_concentration_warning,last_updated
```

Rebuild it with the sidebar **Update API data** button, the historical API ingestion command, or the generic CSV importer:

```bash
.venv/bin/python scripts/update_historical_matches.py --start-date 2022-01-01 --end-date 2026-06-18 --rebuild-behavior
.venv/bin/python scripts/import_historical_csv.py --input data/raw/results.csv --source public_csv --rebuild-behavior
.venv/bin/python scripts/rebuild_elo.py --rebuild-behavior
```

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

Attack and defense can be calibrated from recent match history:

- primary behavior uses the latest 4 years only, capped at the latest 40 matches per team;
- attack = recency- and competition-weighted scoring behavior, with raw, old opponent-adjusted, residual, and final indices retained for diagnostics;
- defense = recency- and competition-weighted goals conceded behavior, with raw, old opponent-adjusted, residual, and final indices retained for diagnostics;
- recent form = points, win/draw/loss rates, and capped goal difference with recency weighting.

Historical Behavior Calibration v1 uses `data/historical_matches.csv` to calculate:

- attacking behavior: weighted goals, xG, shots, scoring rate, multi-goal rate, and a 0-1 attack index;
- defensive behavior: weighted goals/xG/shots against, clean sheets, multi-conceded rate, and a 0-1 defense index where higher is better;
- recent form: weighted points per match, result rates, goal difference, and a 0-1 recent-form index;
- goal-market behavior: BTTS, over/under 2.5, clean-sheet, and failed-to-score rates;
- environment response: capped descriptive deltas under heat, humidity, altitude, wind, rain, and roof-closed conditions.
- diagnostics: all-time goals for/against, recent-vs-all-time warnings, sample-size warnings, staleness warnings, and opponent-quality warnings.

Historical Data Ingestion v1 populates `data/historical_matches.csv` with match-result history only. It does not require xG or advanced event data.

Rolling Elo v1 uses imported historical match results to compute pre-match team and opponent Elo:

- one real match is deduplicated from the two long-format rows;
- default Elo starts at `1500`;
- neutral-site matches do not receive home advantage;
- non-neutral matches use a small `35` Elo home advantage in the expected-result calculation;
- K factors are transparent: World Cup `45`, qualifiers and continental tournaments `35`, Nations League `25`, friendlies `15`, other `20`;
- margin of victory uses a conservative logarithmic multiplier;
- each match update is capped to avoid extreme rating jumps.

Run it after importing or editing historical results:

```bash
.venv/bin/python scripts/rebuild_elo.py --rebuild-behavior
```

Elo Calibration + Expected Performance Residuals v1 validates that rolling Elo is behaving plausibly and uses it to estimate what each team should have done in each match:

```text
expected_result_score = 1 / (1 + 10 ** ((opponent_elo - adjusted_team_elo) / 400))
expected_goals_for = 1.35 * exp((team_elo_pre - opponent_elo) / 900)
```

Expected goals are capped between `0.25` and `3.50`. Result residual is actual result score minus expected result score. Goals-for residual is actual goals for minus expected goals for. Goals-against residual is actual goals against minus expected goals against.

Simple opponent boosting was useful as a first diagnostic, but it could make almost every strong-schedule team look better on both attack and defense. Residual behavior is more discriminative because it asks whether the team actually beat, matched, or missed the Elo-based expectation.

Residual Robustness v1 keeps raw residuals for diagnostics, then uses capped robust residuals for final behavior:

- `cap_residual` clips goal residuals to `[-1.75, +1.75]`.
- `blowout_weight` leaves normal margins unchanged and down-weights large-margin matches, with a floor of `0.55`.
- `top_3_attack_residual_share` and `top_3_defense_residual_share` flag whether the residual signal is concentrated in only a few matches.
- `residual_concentration_warning` is raised when either top-three share is at least `60%`.

Final behavior indices use robust residual calibration conservatively:

```text
attack_index_final = 0.70 * attack_index_raw + 0.30 * attack_index_residual_robust
defense_index_final = 0.70 * defense_index_raw + 0.30 * defense_index_residual_robust
```

The dashboard and audits still show the old opponent-adjusted indices as diagnostics:

- `attack_index_raw` and `defense_index_raw`: recent weighted behavior before opponent adjustment.
- `attack_index_adjusted_old` and `defense_index_adjusted_old`: old direct opponent-quality adjustment.
- `attack_index_residual_raw` and `defense_index_residual_raw`: uncapped performance versus Elo-based expected goals.
- `attack_index_residual_robust` and `defense_index_residual_robust`: capped, blowout-weighted performance versus Elo-based expected goals.
- `attack_index_final` and `defense_index_final`: primary behavior inputs used for rating blends.

Audit Elo and residual behavior with:

```bash
.venv/bin/python scripts/audit_elo_calibration.py
.venv/bin/python scripts/audit_performance_residuals.py
```

Behavior Driver Report + Final Model Input Audit v1 adds transparency without changing the core formula:

- top attack and defense driver matches show which recent games contribute most to behavior diagnostics;
- opponent-tier breakdown separates elite, strong, average, and weak opponents;
- competition breakdown separates World Cup, qualifiers, continental tournaments, Nations League, friendlies, and other matches;
- final model input audit compares manual ratings, behavior indices, adjusted model inputs, deltas, cap flags, and warnings;
- warnings flag high behavior versus low manual priors, capped behavior deltas, friendly-heavy or weak-opponent-heavy behavior, raw attack that is much higher than robust residual attack, and high top-three residual concentration.

All-time behavior remains diagnostic only. Primary attack, defense, form, and goal-market behavior use the recent calibrated window.

Recency weighting:

```text
weight = exp(-days_since_match / half_life_days)
```

Default `half_life_days` is 365. The default behavior config is:

```text
lookback_years = 4
max_matches = 40
min_matches = 10
half_life_days = 365
friendly_weight = 0.45
nations_league_weight = 0.90
qualifier_weight = 1.15
continental_weight = 1.25
world_cup_weight = 1.50
opponent_adjustment_strength = 0.25
goal_contribution_cap = 4.00
min_opponent_elo_coverage = 0.65
min_residual_coverage = 0.65
residual_cap = 1.75
residual_disagreement_threshold = 0.25
residual_concentration_warning_threshold = 0.60
max_behavior_blend = 0.30
max_form_blend = 0.50
```

Competition weighting:

```text
world_cup = 1.50
world_cup_qualifier = 1.15
continental_tournament = 1.25
nations_league = 0.90
friendly = 0.45
other = 0.80
```

The match weight is recency weight times competition weight. Opponent quality is then applied conservatively to goals-for and goals-against contributions. If `opponent_elo` is missing, the app uses manual team ratings where available, otherwise a transparent recent-results fallback. The opponent modifier is capped between `0.75` and `1.25`.

Opponent-quality and residual behavior diagnostics:

- `attack_index_raw` and `defense_index_raw` show behavior before opponent-Elo adjustment.
- `attack_index_adjusted_old` and `defense_index_adjusted_old` show the old direct opponent-Elo adjustment.
- `attack_index_final` and `defense_index_final` are the primary model behavior inputs.
- scoring against stronger opponents receives slightly more credit;
- scoring against weaker opponents receives slightly less credit;
- conceding against stronger opponents is penalized slightly less;
- conceding against weaker opponents is penalized slightly more.
- residual warnings flag cases where the old opponent adjustment boosted a team but its actual goals or goals allowed were worse than Elo expectation.

Schedule strength:

```text
strong opponent = opponent_elo >= 1700
weak opponent = opponent_elo <= 1350
```

The behavior table records mean, median, min, and max recent opponent Elo, counts of strong and weak opponents, a schedule label, and warnings. Audit it with:

```bash
.venv/bin/python scripts/audit_schedule_strength.py
```

Reliability grades:

```text
high = at least 25 recent matches and latest match within 180 days
moderate = at least 15 recent matches and latest match within 365 days
low = at least 8 recent matches
insufficient = fewer than 8 recent matches
```

Behavior blending into model inputs:

```text
high/moderate: attack and defense blend up to 30%; recent form blends up to 50%
low: attack and defense blend up to 15%; recent form blends up to 25%
insufficient: manual values only
```

Manual ratings remain the stable base. For fixture teams missing from `data/team_ratings.csv`, the app creates a neutral base row and applies the same capped behavior blend if behavior data exist.

If schedule strength is weak and the old adjusted attack index is below the raw attack index, the attack behavior blend is cut in half. If opponent-Elo or residual coverage is poor, the app uses manual inputs rather than forcing an unreliable behavior blend. If residual warnings are present, behavior blend weights are cut in half.

Behavior movement caps:

```text
attack_delta_cap = 0.12
defense_delta_cap = 0.12
recent_form_delta_cap = 0.18
```

These caps prevent historical behavior from making implausibly large changes to the model inputs.

The score matrix uses an independent Poisson model. Probabilities are normalized over the displayed score grid.

Environmental effects are deliberately conservative. Roof-closed venues reduce outdoor weather impact by 85%. Altitude mostly matters above about 1,200 meters. Wind and precipitation mainly reduce total-goals quality rather than heavily favoring one team.

Environment response uses historical thresholds of hot `>= 28 C`, humid `>= 70%`, altitude `>= 1000 m`, windy `>= 20 km/h`, and rain `> 0 mm`. Samples below 5 matches are labeled low confidence. Deltas and response indices are capped so environment history cannot dominate the model.

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
- **Historical behavior**: recency-weighted team behavior summary, schedule strength, expected-performance residuals, behavior driver matches, opponent/competition breakdowns, recent long-format match history, environment response, and manual-vs-behavior model input impact.
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
.venv/bin/python -m py_compile app.py src/*.py scripts/*.py tests/*.py
.venv/bin/python scripts/update_historical_matches.py --start-date 2022-01-01 --end-date 2026-06-18 --rebuild-behavior
.venv/bin/python scripts/rebuild_elo.py --rebuild-behavior
.venv/bin/python scripts/audit_behavior_calibration.py
.venv/bin/python scripts/audit_schedule_strength.py
.venv/bin/python scripts/audit_elo_calibration.py
.venv/bin/python scripts/audit_performance_residuals.py
.venv/bin/python scripts/audit_behavior_drivers.py --teams Norway England Argentina Brazil Uruguay Croatia Morocco "United States" Mexico
.venv/bin/python scripts/audit_model_inputs.py
```

The tests check probability sums, fair odds, score matrix normalization, missing odds, missing weather, roof-closed weather dampening, Polymarket price normalization, YES/NO mapping, alpha gaps, sensitivity output shape, prediction-log append, missing-API fallback, historical CSV import, rolling Elo, Elo calibration, expected-performance residuals, opponent-quality adjustment, behavior driver reports, final model input audits, schedule-strength diagnostics, behavior calibration, rating blend caps, audit-script output, and Full report helper outputs.

## 14. Known Limitations

- Ratings are transparent priors unless connected to a real ratings API or recent match-history file.
- Historical behavior is descriptive, not causal proof.
- Friendlies may not reflect full competitive strength.
- Opponent quality matters, and opponent Elo may be missing or stale.
- Opponent quality fallback is transparent but approximate when no external Elo exists.
- Rolling Elo depends on the completeness and ordering of imported historical results.
- Rolling Elo is not an official FIFA rating and should be treated as an internal schedule-strength estimate.
- Residual performance is an approximation from Elo and goals, not causal proof of team quality.
- Expected goals from Elo are transparent and capped, but they are not a substitute for real xG or lineup-level information.
- Recent behavior can differ sharply from all-time history; the dashboard flags those cases instead of treating them as causal proof.
- Environmental response requires enough previous matches to be meaningful.
- Environmental effects are capped and conservative.
- Manual ratings remain available as overrides and are the base for behavior blending.
- football-data.org historical national-team resources may be subscription-restricted; use the generic CSV importer when API access is unavailable.
- Historical ingestion depends on source coverage and naming; use `data/team_name_aliases.csv` and manual rows to patch gaps.
- Generic CSV import depends on source CSV quality; duplicate same-day matches with identical teams and scores may need manual review.
- Historical ingestion v1 does not enrich weather, xG, shots, or possession automatically.
- Injuries, lineups, tactical changes, and motivation are not automatically modeled.
- Weather is approximate unless a weather API is configured.
- Polymarket fuzzy matching can be wrong; use `data/market_mappings.csv` for confirmed mappings.
- Low-liquidity markets and missing prices should be treated as data warnings, not opportunities.
- The Poisson model assumes independent scoring rates.
- Training climate is approximate and should not be overinterpreted.
- Full report v1 uses simple report-layer uncertainty bands and a score-timeline hazard approximation.
- The app does not include trade execution, staking, bet sizing, Kelly criterion, or investment recommendations.
