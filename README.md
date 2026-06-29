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

Fixture visibility can be audited without changing model inputs:

```bash
.venv/bin/python scripts/audit_fixture_availability.py \
  --start-date 2026-06-22 \
  --end-date 2026-06-29 \
  --hide-past
```

The audit explains which fixtures are loaded, inside the selected UTC window, hidden as past kickoffs, or excluded. The dashboard also shows fixture availability diagnostics so a selected range such as upcoming 48 hours or next 7 days is not confused with a today-only fixture list.

If the loaded fixture source ends before the selected range, the app shows a Data Health warning instead of silently looking like a today-only app. Refresh API/cache or update `data/fixtures.csv`; the app does not fabricate future fixtures while offline.

### Team Ratings

Edit `data/team_ratings.csv`:

```text
team,elo,attack,defense,recent_form,fifa_rank_proxy,training_temp_c,training_humidity_pct,data_quality,last_updated,notes
```

Attack, defense, and recent form are 0-1 values. If recent match history is later added in `data/recent_matches.csv`, the app can recalculate attack, defense, and form with recency weighting.

Manual and external-benchmark-calibrated ratings remain the primary override values. The historical behavior layer is preserved as a diagnostic/explanatory layer; default primary predictions use behavior blend `0.00`.

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

Historical data binding can be audited for a selected match:

```bash
.venv/bin/python scripts/audit_historical_binding.py \
  --home Jordan \
  --away Algeria \
  --as-of-date 2026-06-23
```

The audit canonicalizes team names through `data/team_name_aliases.csv`, counts historical rows for both teams, counts head-to-head rows, checks rating and behavior rows, and reports close-name suggestions when a canonical team is not found. The app uses this binding audit in the report so imported long-format historical rows are not mistaken for missing data.

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
match_id,market_id,market_type,model_side,polymarket_side,manual_confirmed,mapping_confidence,mapping_score,mapping_reason,reject_reason,notes
```

Example:

```text
66457042,pm_market_id,match_winner_home,home_win,YES,1,manual,100,England win market confirmed manually,,England win market confirmed manually
```

Manual mappings override automatic fuzzy matching.

### Polymarket Match Search Precision

Country-only Polymarket searches can return unrelated political or macro markets. For example, searching only `Colombia` can return election, president, inflation, or polling markets instead of a football match market.

The selected-match search now builds matchup-specific queries such as:

```text
Colombia DR Congo
Colombia vs DR Congo
Colombia v DR Congo
COL COD
World Cup Colombia DR Congo
FIFA World Cup Colombia DR Congo
Will Colombia beat DR Congo
Will DR Congo beat Colombia
Colombia draw DR Congo
```

The mapper scores returned markets before they can be used:

- positive signals: both teams or aliases found, matchup words such as `vs` or `beat`, sports terms such as `World Cup`, `FIFA`, `soccer`, `football`, and sports-like category labels;
- rejection signals: political/non-sports terms such as `election`, `president`, `congress`, `poll`, `vote`, `war`, `tariff`, `inflation`, `crypto`, `bitcoin`, `fed`, and `interest rate`;
- country-only or one-team-only markets are diagnostics only and are not auto-mapped for selected-match alpha.

Automatic mapping is conservative:

- only `high` confidence candidates are auto-suggested;
- automatic suggestions always keep `manual_confirmed = False`;
- medium/low candidates remain visible only as diagnostics;
- rejected candidates are not used in alpha tables.

Audit a matchup search:

```bash
.venv/bin/python scripts/audit_polymarket_match_search.py \
  --home Colombia \
  --away "DR Congo" \
  --competition "World Cup"
```

This writes:

```text
reports/polymarket_match_search_YYYY-MM-DD.csv
reports/polymarket_match_search_YYYY-MM-DD.md
```

To manually confirm a mapping, copy the chosen candidate `market_id` into `data/market_mappings.csv`, set the correct `market_type`, `model_side`, `polymarket_side`, and set `manual_confirmed` to `1`. Manual rows remain the highest-priority override.

### Polymarket Sports Event Discovery

Some football markets are nested under Polymarket Gamma events, series, or tags and may not appear in direct text-query results. The app now uses layered read-only discovery:

1. direct matchup query search;
2. active Gamma sports, World Cup, FIFA, soccer, or football events when identifiers are discoverable;
3. broader active Gamma events ordered by volume/date;
4. local discovery caches.

Nested event markets are flattened and then scored with the same matchup-aware filters described above. Political, macro, crypto, and one-team-only markets are rejected or kept as diagnostics; they are not used in alpha tables.

Broad discovery writes local fallback caches:

```text
data/polymarket_events_cache.csv
data/polymarket_markets_cache.csv
```

Audit broad sports-event discovery for a matchup:

```bash
.venv/bin/python scripts/audit_polymarket_sports_discovery.py \
  --home Colombia \
  --away "DR Congo" \
  --competition "World Cup" \
  --fixture-date 2026-06-23
```

If no candidate is found, it may mean no match market exists yet, the market is closed/resolved, the event was not returned by active filters, the cache is stale, or team wording differs from `data/team_name_aliases.csv`. Try:

```bash
.venv/bin/python scripts/audit_polymarket_sports_discovery.py \
  --home Colombia \
  --away "DR Congo" \
  --competition "World Cup" \
  --fixture-date 2026-06-23 \
  --retrieval-mode all_events
```

You can also run with `--include-closed` or inspect Polymarket manually and add a confirmed row to `data/market_mappings.csv`.

### Polymarket URL And Slug Resolver

Some Polymarket sports events use abbreviations that differ from FIFA-style country codes. For example, DR Congo may appear as `CDR` in a Polymarket sports slug even though other systems may use `COD`:

```text
https://polymarket.com/sports/world-cup/fifwc-col-cdr-2026-06-23
```

The resolver must preserve known Polymarket-specific country codes and local-date slugs. For example, Netherlands can appear as `NLD` rather than `NED`, and late UTC kickoffs can use the previous local date in the Polymarket URL:

```text
https://polymarket.com/sports/world-cup/fifwc-nld-mar-2026-06-29
```

That URL maps to Netherlands vs Morocco for the `2026-06-30 01:00 UTC` fixture. Do not remove the `NLD -> Netherlands` alias or the prior-local-date slug candidate logic unless Polymarket changes this convention.

The URL/slug resolver parses exact sports event URLs, validates the slug against the selected fixture, fetches the event from Gamma by exact slug, and extracts nested markets for local scoring. This is different from automatic search: automatic search tries to discover possible markets, while the URL resolver starts from a user-supplied event URL or slug.

Supported parsed fields include:

- event slug, such as `fifwc-col-cdr-2026-06-23`;
- Polymarket team codes, such as `COL` and `CDR`;
- fixture date hint, such as `2026-06-23`;
- competition path, such as `sports/world-cup`.

Audit an exact URL:

```bash
.venv/bin/python scripts/audit_polymarket_url_resolver.py \
  --url https://polymarket.com/sports/world-cup/fifwc-col-cdr-2026-06-23 \
  --home Colombia \
  --away "DR Congo" \
  --fixture-date 2026-06-23 \
  --competition "World Cup"
```

This writes:

```text
reports/polymarket_url_resolver_YYYY-MM-DD.csv
reports/polymarket_url_resolver_YYYY-MM-DD.md
```

In the dashboard, paste the exact URL into **Optional Polymarket event URL or slug**. The extracted markets are added to the candidate mapping pool, but mappings remain `manual_confirmed = False` unless you explicitly add a manual row to `data/market_mappings.csv`.

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

- calibrated behavior diagnostics use the latest 4 years only, capped at the latest 40 matches per team;
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
- `attack_index_final` and `defense_index_final`: calibrated behavior diagnostics used when inspecting behavior-adjusted blends.

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
- `attack_index_final` and `defense_index_final` are the calibrated behavior diagnostics used by behavior-adjusted audit modes.
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

Behavior diagnostic blend rules:

```text
high/moderate: attack and defense blend up to 30%; recent form blends up to 50%
low: attack and defense blend up to 15%; recent form blends up to 25%
insufficient: manual values only
```

Manual/external-calibrated ratings remain the stable primary base. Diagnostic behavior-adjusted runs use these same capped blend rules, but the default dashboard probability uses behavior blend `0.00`.

If schedule strength is weak and the old adjusted attack index is below the raw attack index, the attack behavior blend is cut in half. If opponent-Elo or residual coverage is poor, the app uses manual inputs rather than forcing an unreliable behavior blend. If residual warnings are present, behavior blend weights are cut in half.

Behavior movement caps:

```text
attack_delta_cap = 0.12
defense_delta_cap = 0.12
recent_form_delta_cap = 0.18
```

These caps prevent historical behavior from making implausibly large changes when the diagnostic behavior-adjusted mode is inspected.

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
- **Tournament context**: experimental/off by default; retained for offline audits, not active dashboard alpha.

Real metrics in v1:

- model probabilities, expected goals, scoreline probabilities, fair odds, alpha EV, and Polymarket alpha gaps;
- historical support counts when `data/recent_matches.csv` or cached football results exist;
- climate factors derived from the same venue/environment inputs used by the model.

Placeholders or approximations in v1:

- RPS is explicitly shown as `RPS placeholder / not yet backtested`;
- base xG is not separately stored yet, so adjusted xG is used as the base value in the expected-goals report chart;
- league context uses fallback baselines when historical match data is unavailable.
- tournament context is a diagnostic layer; it is not the primary model unless strict as-of validation later supports that change.

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

### Tournament Context Layer v1

Tournament Context Layer v1 is experimental and off by default in the dashboard. It remains in the codebase for audit and backtesting work, but the active dashboard, Market Value Tables, Top Alpha Signals, and Polymarket Alpha tab use baseline model probabilities only.

The feature was disabled from the active selected-match view after local standings inputs proved incomplete for a last group-stage fixture. Do not use context-adjusted probabilities as active alpha inputs until the standings ledger and strict as-of validation are reliable.

When enabled for offline audit, Tournament Context Layer v1 estimates transparent qualification incentives from completed group results and remaining fixtures. It is designed for cases where one team can probably accept a draw while the other likely needs to win.

The layer:

- builds group standings as of the selected kickoff using only completed matches before kickoff;
- classifies each team as `draw_enough`, `needs_win`, `must_not_lose`, `already_qualified_likely`, `already_eliminated_likely`, or `unknown`;
- classifies match context as `one_needs_win_other_draw_enough`, `both_need_win`, `both_draw_ok`, `one_safe_other_needs_win`, `dead_rubber`, `knockout_must_advance`, or `unknown`;
- applies only small optional diagnostic probability shifts, capped at 5 percentage points on 1X2 outcomes;
- keeps baseline probabilities and baseline alpha gaps visible.

For knockout matches, the layer does not apply group-stage “draw is enough” logic. It flags advancement context separately because 90-minute 1X2 markets and qualification markets are different.

Audit a fixture:

```bash
.venv/bin/python scripts/audit_tournament_context.py \
  --home Uruguay \
  --away Spain \
  --fixture-date 2026-06-26 \
  --competition "World Cup" \
  --group H
```

Reports are saved to:

```text
reports/tournament_context_YYYY-MM-DD.csv
reports/tournament_context_YYYY-MM-DD.md
```

The context layer remains diagnostic until `evaluate_context_layer_on_completed_matches(...)` or a stricter as-of validation flow shows improved Brier score, log loss, draw calibration, and totals calibration.

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

For Polymarket football match markets, result rows should represent regular
time plus stoppage only. If a source exposes both 90-minute and final
extra-time scores, the prediction/result ledger importer uses the 90-minute
columns for 1X2/win, totals, BTTS, and backtesting outcomes.

The **Backtesting** tab can load both logs and report simple Brier score, log loss, mean alpha gap, hit rate by signal strength, and calibration buckets when completed results exist. These diagnostics are opt-in so normal match selection does not parse heavy logs or run expensive backtests automatically.

If `data/prediction_log.csv` becomes unreadable because old and new dashboard schemas were appended into the same file, quarantine and rebuild it with:

```bash
.venv/bin/python scripts/repair_prediction_log.py
```

The script preserves the malformed file as `data/prediction_log_corrupt_backup_YYYY-MM-DD_HHMMSS.csv` and recreates `data/prediction_log.csv` with the current schema.

### Behavior-Adjusted Completed-Match Backtesting

Behavior-Adjusted Backtesting v1 compares two model modes on completed matches loaded from `data/fixtures.csv` and `data/historical_matches.csv`:

- `baseline_manual`: uses only the manual base ratings in `data/team_ratings.csv`.
- `behavior_adjusted`: explicitly requests conservative historical behavior blending and delta caps for evaluation.

Run:

```bash
.venv/bin/python scripts/run_backtest.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --save-report
```

The script writes:

```text
reports/backtest_predictions_YYYY-MM-DD.csv
reports/backtest_report_YYYY-MM-DD.md
```

Key metrics:

- **Brier score 1X2**: squared error across home win, draw, and away win probabilities. Lower is better.
- **Log loss 1X2**: penalty for assigning low probability to the actual result. Lower is better.
- **Most-likely result accuracy**: how often the highest-probability 1X2 outcome occurred. Higher is better.
- **Mean probability assigned to actual result**: average model probability on what actually happened. Higher is better.
- **Over 2.5 and BTTS Brier**: probability error for goal markets. Lower is better.
- **Total goals MAE**: absolute error between predicted total xG and actual total goals. Lower is better.

Inspect the per-match comparison to see where behavior adjustment increased or reduced the probability assigned to the actual result. The dashboard **Backtesting** tab shows the same comparison for a selected date window.

Important v1 limitation: this backtest uses the current `data/team_behavior.csv`. It may have look-ahead bias because behavior metrics are not rebuilt as of each historical match date. Backtesting v2 should rebuild Elo and behavior snapshots as of each match date before scoring that match.

### Strict As-Of-Date Backtesting

Strict As-Of-Date Backtesting v2 scores completed matches using only information available before each match date. For every historical prediction date, it:

- filters `data/historical_matches.csv` to rows where `date_utc < match date`;
- excludes the completed match itself from behavior inputs;
- rebuilds rolling Elo from the pre-match slice;
- rebuilds recent team behavior from the pre-match slice;
- runs `baseline_manual` and `behavior_adjusted_asof` with those as-of inputs.

Run:

```bash
.venv/bin/python scripts/run_asof_backtest.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --save-report
```

The script writes:

```text
reports/asof_backtest_predictions_YYYY-MM-DD.csv
reports/asof_backtest_report_YYYY-MM-DD.md
```

The strict report includes the same Brier score, log loss, accuracy, mean actual-result probability, goal-market Brier, and total-goals MAE metrics as v1. It also adds:

- `lookahead_safe_rate`: share of prediction rows that used no future behavior data;
- `mean_home_behavior_matches_used` and `mean_away_behavior_matches_used`;
- `matches_with_insufficient_asof_behavior`;
- per-match latest behavior match used for home and away teams.

Why this matters: v1 can look better because current behavior data may include matches that happened after the historical prediction date. V2 is more trustworthy because it prevents that look-ahead bias. V2 metrics may be worse than v1 if a team had little pre-match history available, but those lower-confidence rows are explicitly labeled.

The dashboard **Backtesting** tab shows a **Strict As-Of-Date Backtest** section under the existing behavior-adjusted backtest. If any row has `lookahead_safe = False`, inspect the diagnostics before trusting the metrics.

### Behavior Blend Sensitivity

Behavior Blend Sensitivity v1 tests whether the historical behavior layer should influence model inputs at all, and if so, how strongly. It keeps the core match model formula unchanged and reruns strict as-of-date backtests across behavior blend multipliers.

Default multipliers:

```text
0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00
```

Interpretation:

- `0.00`: no behavior adjustment; this should match strict `baseline_manual` or be very close.
- `1.00`: the full historical behavior blend tested as a diagnostic alternative.
- values below `1.00`: weakened behavior influence.

Run:

```bash
.venv/bin/python scripts/run_blend_sensitivity.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --save-report
```

The script writes:

```text
reports/blend_sensitivity_predictions_YYYY-MM-DD.csv
reports/blend_sensitivity_metrics_YYYY-MM-DD.csv
reports/blend_sensitivity_team_YYYY-MM-DD.csv
reports/blend_sensitivity_report_YYYY-MM-DD.md
```

How to read the output:

- `delta_brier_vs_baseline < 0`: that blend improved 1X2 Brier score versus `0.00`.
- `delta_log_loss_vs_baseline < 0`: that blend improved log loss versus `0.00`.
- `delta_actual_prob_vs_baseline > 0`: that blend assigned more probability to the actual result.
- Team-level helped/hurt counts show whether current/default behavior tends to help or hurt specific teams in the sample.

Decision rule:

- If no blend improves both Brier and log loss, keep behavior diagnostic-only.
- If a weak blend from `0.05` to `0.15` improves both, consider a reduced behavior blend.
- If `1.00` improves both, consider promoting the full behavior blend after further QA.
- Otherwise, investigate before changing defaults.

This sensitivity step should be run before changing model defaults. It is evaluation-only and does not provide staking, sizing, trade execution, or betting advice.

### Default Model Policy

Default Model Policy v1 makes the strict-evidence-supported model the primary dashboard model:

- primary model mode: `baseline_manual`;
- primary rating inputs: manual or external-benchmark-calibrated rows from `data/team_ratings.csv`;
- behavior status: `diagnostic_only`;
- behavior default blend: `0.00`;
- last validation report: `reports/blend_sensitivity_report_2026-06-21.md`.

Why baseline is primary: strict as-of-date blend sensitivity found that no tested behavior blend improved both Brier score and log loss versus the no-behavior baseline. The full behavior blend (`1.00`) was worse than baseline in that strict test, so the main probability, fair odds, decimal market alpha, and Polymarket alpha use the primary baseline model by default.

Behavior is still preserved as an explanatory layer. The dashboard shows behavior-adjusted as-of probabilities, behavior driver matches, residuals, schedule strength, and disagreement warnings, but behavior-only differences are not labelled as primary alpha.

Market/export fields now distinguish the two layers:

- `primary_model_probability`: probability used for fair odds and alpha by default;
- `behavior_diagnostic_probability`: behavior-adjusted context probability when available;
- `behavior_probability_delta`: behavior minus primary probability;
- `model_policy`: current policy label;
- `edge_source`: `primary_model` unless the user explicitly toggles diagnostic behavior probability visibility;
- `primary_model_mode`, `behavior_status`, `behavior_blend_used`, and `strict_validation_summary`: report/export policy metadata.

Evidence needed to promote behavior to primary: rerun strict as-of-date backtesting and blend sensitivity on a larger, clean sample. Behavior should improve both Brier score and log loss, remain lookahead-safe, and avoid unstable team-level helped/hurt patterns before the default blend is increased.

Rerun the strict checks:

```bash
.venv/bin/python scripts/run_asof_backtest.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --save-report

.venv/bin/python scripts/run_blend_sensitivity.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --save-report
```

### Backtest QA And Rating Coverage

Before interpreting behavior-adjusted backtest metrics, run the rating coverage audit:

```bash
.venv/bin/python scripts/audit_backtest_quality.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup"
```

The audit writes:

```text
reports/backtest_quality_audit_YYYY-MM-DD.md
reports/backtest_quality_matches_YYYY-MM-DD.csv
```

This QA step checks:

- whether completed-match teams have exact rows in `data/team_ratings.csv`;
- whether behavior-adjusted ratings and `data/team_behavior.csv` rows exist;
- whether aliases such as `Curacao` / `Curaçao`, `USA` / `United States`, `Bosnia-H.` / `Bosnia and Herzegovina`, and `Czech Republic` / `Czechia` may be needed;
- whether the model used neutral fallback inputs;
- whether 1X2 probabilities look suspiciously flat.

Neutral fallback means the model did not find an exact team row and used default values near `attack=0.55`, `defense=0.55`, and `recent_form=0.55`. This can make mismatched teams look nearly equal and produce unrealistic probabilities. For example, if Germany and Curaçao are missing from the exact rating table used by the model, Germany vs Curaçao can look close to a 36% / 27% / 36% match even though the score and football prior suggest a large gap.

Fixes should be manual and transparent:

- add missing teams to `data/team_ratings.csv`;
- add aliases to `data/team_name_aliases.csv`;
- rebuild behavior/Elo after changing historical names if needed;
- rerun the QA audit until neutral fallback and alias warnings are resolved.

Backtest metrics should not be trusted until the QA section is clean. This is a validation layer only; it does not change the model formula or provide staking, sizing, or trading advice.

### Team Rating Coverage Completion

Run a read-only rating coverage audit before interpreting completed-match backtests:

```bash
.venv/bin/python scripts/audit_rating_coverage.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup"
```

The audit saves:

```text
reports/rating_coverage_audit_YYYY-MM-DD.md
reports/rating_coverage_teams_YYYY-MM-DD.csv
```

Generate conservative proposed rows for missing required teams:

```bash
.venv/bin/python scripts/propose_missing_team_ratings.py \
  --include-backtest \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup"
```

This writes proposals to:

```text
data/team_ratings_proposed.csv
data/team_name_aliases_proposed.csv
```

The proposal script does **not** modify `data/team_ratings.csv` unless `--write` is passed. With `--write`, it appends only missing teams and preserves existing manual rows:

```bash
.venv/bin/python scripts/propose_missing_team_ratings.py \
  --include-backtest \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --write
```

Generated rating rows are transparent priors, not silent model truth:

- If behavior data exists, generated values use `70%` neutral base and `30%` behavior index.
- If behavior data is missing, generated values stay at neutral `0.55`.
- Generated rows are clipped to `0.35-0.85`.
- `data_quality` is marked `generated_from_behavior` or `generated_neutral_placeholder_low`.
- Notes explicitly say manual review is recommended.

Alias proposals should be reviewed before adding them to `data/team_name_aliases.csv`. To append proposed aliases explicitly:

```bash
.venv/bin/python scripts/propose_missing_team_ratings.py \
  --include-backtest \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --write-aliases
```

Backtest metrics should not be trusted while required teams are missing manual rows or using neutral fallback ratings.

### Manual Rating Review

Generated ratings are conservative placeholders created to avoid neutral fallback. They are useful for coverage, but important teams should be reviewed before treating them as expert priors.

Run the review audit:

```bash
.venv/bin/python scripts/audit_rating_review.py
```

The audit saves:

```text
reports/rating_review_audit_YYYY-MM-DD.md
reports/rating_review_teams_YYYY-MM-DD.csv
```

Create conservative review proposals for important teams:

```bash
.venv/bin/python scripts/propose_reviewed_ratings.py \
  --teams Argentina Brazil France Germany Spain Netherlands Portugal England Uruguay Morocco "United States" Norway
```

This writes:

```text
data/team_ratings_review_proposals.csv
```

The proposal script does **not** modify `data/team_ratings.csv` unless `--write` is passed. With `--write`, it updates only rows where `data_quality == generated_from_behavior`, sets `data_quality = manual_review_candidate`, and preserves rows marked `manual_csv` or `manual_reviewed`.

Rating statuses:

- `manual_reviewed`: human-reviewed prior, preserved by proposal scripts.
- `manual_existing`: existing manual CSV prior.
- `generated_from_behavior`: conservative placeholder from behavior/Elo, needs review.
- `neutral_placeholder`: neutral generated placeholder, highest review risk.

Review proposal formula:

```text
reviewed_attack = 0.60 * current_attack + 0.25 * behavior_attack_final + 0.15 * elo_scaled_attack
reviewed_defense = 0.60 * current_defense + 0.25 * behavior_defense_final + 0.15 * elo_scaled_defense
reviewed_recent_form = 0.60 * current_recent_form + 0.40 * behavior_recent_form
```

Proposed values are clipped to `0.35-0.90` and remain marked as candidates until a human explicitly promotes them to reviewed priors. The dashboard **Backtesting** tab shows rating review counts next to Rating Coverage so generated rows remain visible during QA.

### External Prior Review

External Prior Review v1 adds one more quality-control layer before generated ratings are accepted as reviewed priors. It compares current/generated ratings and review proposals against independent or manually entered reference priors. This is an audit workflow only; it does not change the model formula and does not silently replace ratings.

Reference priors live in:

```text
data/team_rating_external_priors.csv
```

Required columns:

```text
team,reference_attack,reference_defense,reference_recent_form,reference_overall_strength,source,last_updated,notes
```

Use 0-1 values for the reference fields. Blank reference values are allowed and mean the team has not yet been externally checked. The v1 file is seeded with placeholder rows, not invented ratings. Fill `source`, `last_updated`, and `notes` with enough detail for another reviewer to understand where the prior came from.

Audit external priors:

```bash
.venv/bin/python scripts/audit_external_priors.py
```

The audit saves:

```text
reports/external_prior_audit_YYYY-MM-DD.md
reports/external_prior_comparison_YYYY-MM-DD.csv
```

Build the manual review worksheet:

```bash
.venv/bin/python scripts/build_rating_review_sheet.py \
  --teams Argentina Brazil France Germany Spain Netherlands Portugal England Uruguay Morocco "United States" Norway \
  --output data/manual_rating_review_sheet.csv
```

The worksheet includes current ratings, generated review proposals, external priors, recommended values, and empty human-review fields. If an external prior exists, recommended values blend the proposal with the reference prior:

```text
recommended_attack = 0.50 * proposal_attack + 0.50 * reference_attack
recommended_defense = 0.50 * proposal_defense + 0.50 * reference_defense
recommended_recent_form = 0.60 * proposal_recent_form + 0.40 * reference_recent_form
```

If no external prior exists, the worksheet keeps the proposal but marks the row as needing external-prior review. To approve a row, edit `review_status` to `approved` after human review, then run:

```bash
.venv/bin/python scripts/build_rating_review_sheet.py \
  --output data/manual_rating_review_sheet.csv \
  --write
```

With `--write`, only rows marked `review_status = approved` are applied, and only rows currently marked `generated_from_behavior` or `manual_review_candidate` can be updated. Rows marked `manual_reviewed` are never overwritten by this script. Approved rows become `manual_reviewed` in `data/team_ratings.csv`.

This step should happen before strict backtesting v2, because strict as-of-date evaluation is only useful when the manual priors being tested are credible and clearly sourced.

### External Prior Import Helper

External Prior Import Helper v1 fills `data/team_rating_external_priors.csv` from a manually prepared local CSV. It does not scrape websites, does not require login-protected sources, and does not invent values. The helper converts only explicitly supplied rank, points, or Elo fields into `reference_overall_strength`; attack, defense, and recent form stay blank unless the input file explicitly includes those optional columns.

Prepare:

```text
data/raw/external_team_strength.csv
```

Required input column:

```text
team
```

Recommended input columns:

```text
team,fifa_rank,fifa_points,external_elo,source,last_updated,notes
```

Optional direct prior columns:

```text
reference_attack,reference_defense,reference_recent_form
```

At least one of `fifa_rank`, `fifa_points`, or `external_elo` should be populated for a team to receive an overall-strength prior. If all three are blank, the helper keeps `reference_overall_strength` blank and reports `missing rank/Elo data`.

Conversion formulas:

```text
rank_strength = 0.90 - ((rank - 1) / (max_rank - 1)) * 0.55
points_strength = 0.35 + ((points - min_points) / (max_points - min_points)) * 0.55
elo_strength = 0.35 + ((external_elo - min_elo) / (max_elo - min_elo)) * 0.55
reference_overall_strength = average(available strengths)
```

Defaults:

```text
max_rank = 210
min_points = 900
max_points = 1900
min_elo = 1200
max_elo = 2200
```

All scaled values are clipped to `0.35-0.90`.

Dry run:

```bash
.venv/bin/python scripts/import_external_priors.py \
  --input data/raw/external_team_strength.csv \
  --output data/team_rating_external_priors.csv
```

Default behavior writes proposed updates only:

```text
data/team_rating_external_priors_proposed.csv
```

It does not modify `data/team_rating_external_priors.csv`.

Apply matching, nonblank updates explicitly:

```bash
.venv/bin/python scripts/import_external_priors.py \
  --input data/raw/external_team_strength.csv \
  --output data/team_rating_external_priors.csv \
  --write
```

After importing external priors, rerun:

```bash
.venv/bin/python scripts/audit_external_priors.py
.venv/bin/python scripts/build_rating_review_sheet.py \
  --teams Argentina Brazil France Germany Spain Netherlands Portugal England Uruguay Morocco "United States" Norway \
  --output data/manual_rating_review_sheet.csv
```

### FIFA Ranking Import

FIFA Ranking Import v1 reduces manual entry by matching a local FIFA ranking snapshot to the teams required by fixtures and the completed-match backtest window. It does not scrape websites, does not require a login-protected source, and does not invent ranks or points. If official automated downloads are unavailable or brittle, save a public snapshot locally and record its source/date in the CSV.

Prepare a local snapshot:

```text
data/raw/fifa_rankings_snapshot.csv
```

Preferred normalized format:

```text
team,fifa_rank,fifa_points,source,last_updated
Argentina,1,1877.27,FIFA ranking snapshot,2026-06-21
```

The importer also accepts flexible column names:

```text
team column: team, country, name, ranked_team
rank column: rank, fifa_rank, position
points column: points, fifa_points, total_points, pts
optional date/source columns: date, last_updated, source
```

Team names are matched through `data/team_name_aliases.csv`, including common FIFA variants such as `USA -> United States`, `Korea Republic -> South Korea`, `IR Iran -> Iran`, `Türkiye -> Turkey`, `Congo DR -> DR Congo`, `Czech Republic -> Czechia`, and `Bosnia-Herzegovina -> Bosnia and Herzegovina`.

Validate the snapshot before importing it:

```bash
.venv/bin/python scripts/validate_fifa_snapshot.py \
  --input data/raw/fifa_rankings_snapshot.csv \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup"
```

The validator checks required World Cup teams, aliases, rank values, points values, ambiguous matches, and unused snapshot rows. It saves:

```text
reports/fifa_snapshot_validation_YYYY-MM-DD.csv
reports/fifa_snapshot_validation_YYYY-MM-DD.md
```

If the snapshot is not ready, create a fill-in template for missing or incomplete teams:

```bash
.venv/bin/python scripts/validate_fifa_snapshot.py \
  --input data/raw/fifa_rankings_snapshot.csv \
  --missing-output data/raw/fifa_snapshot_missing_teams.csv
```

Create an external-strength file from the FIFA snapshot without modifying priors:

```bash
.venv/bin/python scripts/import_fifa_rankings.py \
  --input data/raw/fifa_rankings_snapshot.csv \
  --output data/raw/external_team_strength_from_fifa.csv
```

The command prints required teams, FIFA rows loaded, matched teams, missing required teams, unmatched FIFA rows, ambiguous matches, and teams missing rank/points. It also saves:

```text
reports/fifa_ranking_import_YYYY-MM-DD.csv
reports/fifa_ranking_import_YYYY-MM-DD.md
```

To fill blank cells in the missing-team template:

```bash
.venv/bin/python scripts/import_fifa_rankings.py \
  --input data/raw/fifa_rankings_snapshot.csv \
  --output data/raw/external_team_strength_from_fifa.csv \
  --update-template
```

Full external benchmark workflow:

```bash
.venv/bin/python scripts/validate_fifa_snapshot.py \
  --input data/raw/fifa_rankings_snapshot.csv \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup"
.venv/bin/python scripts/import_fifa_rankings.py \
  --input data/raw/fifa_rankings_snapshot.csv \
  --output data/raw/external_team_strength_from_fifa.csv
.venv/bin/python scripts/import_external_priors.py \
  --input data/raw/external_team_strength_from_fifa.csv \
  --output data/team_rating_external_priors.csv \
  --write
.venv/bin/python scripts/calibrate_ratings_from_external.py --write
.venv/bin/python scripts/run_asof_backtest.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --save-report
.venv/bin/python scripts/run_blend_sensitivity.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --save-report
```

The importer can also update `data/team_rating_external_priors.csv` directly with an explicit flag:

```bash
.venv/bin/python scripts/import_fifa_rankings.py \
  --input data/raw/fifa_rankings_snapshot.csv \
  --output data/raw/external_team_strength_from_fifa.csv \
  --write-external-priors
```

Manual CSV overrides remain available. Edit `data/team_name_aliases.csv` for name mismatches and edit `data/team_rating_external_priors.csv` only when you have source-backed benchmark values.

### External Benchmark Rating Calibration

External Benchmark Rating Calibration v1 replaces row-by-row manual approval with a transparent benchmark check. It compares internal ratings against independent FIFA/Elo-style overall-strength priors, creates conservative calibrated proposals, and only writes those proposals to `data/team_ratings.csv` when `--write` is passed.

The benchmark source remains a local CSV snapshot:

```text
data/raw/external_team_strength.csv
```

Required columns:

```text
team,fifa_rank,fifa_points,external_elo,source,last_updated,notes
```

The import script also works when only `fifa_rank` and `fifa_points` are populated. Optional direct columns `reference_attack`, `reference_defense`, and `reference_recent_form` are preserved if supplied, but the app does not infer those from overall strength.

Audit external benchmark coverage for teams required by fixtures and the current completed-match backtest window:

```bash
.venv/bin/python scripts/audit_external_benchmark_coverage.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup"
```

Build a clean manual-entry template for missing benchmark snapshots:

```bash
.venv/bin/python scripts/build_external_team_strength_template.py
```

This writes:

```text
data/raw/external_team_strength_missing_template.csv
```

Template columns:

```text
team,fifa_rank,fifa_points,external_elo,source,last_updated,notes
```

By default, the template includes only required teams missing `reference_overall_strength`. Use `--all-required-teams` to output every required team. Fill FIFA rank and FIFA points manually from an official/public snapshot, and optionally add public Elo. Do not invent values for teams without a source.

Import the filled missing-team template:

```bash
.venv/bin/python scripts/import_external_priors.py \
  --input data/raw/external_team_strength_missing_template.csv \
  --output data/team_rating_external_priors.csv \
  --write
```

Import benchmark inputs into the external prior table:

```bash
.venv/bin/python scripts/import_external_benchmarks.py \
  --input data/raw/external_team_strength.csv \
  --output data/team_rating_external_priors.csv
```

Apply the imported benchmark priors explicitly:

```bash
.venv/bin/python scripts/import_external_benchmarks.py \
  --input data/raw/external_team_strength.csv \
  --output data/team_rating_external_priors.csv \
  --write
```

Scaling formulas:

```text
rank_strength = 0.90 - ((rank - 1) / (210 - 1)) * 0.55
points_strength = 0.35 + ((points - 900) / (1900 - 900)) * 0.55
elo_strength = 0.35 + ((external_elo - 1200) / (2200 - 1200)) * 0.55
external_overall_strength = average(available scaled strengths)
```

All benchmark strengths are clipped to `0.35-0.90`; missing values stay missing.

Calibration formulas for `generated_from_behavior` rows:

```text
calibrated_attack = 0.45 * current_attack + 0.25 * behavior_attack_final + 0.30 * external_overall_strength
calibrated_defense = 0.45 * current_defense + 0.25 * behavior_defense_final + 0.30 * external_overall_strength
calibrated_recent_form = 0.50 * current_recent_form + 0.30 * behavior_recent_form + 0.20 * external_overall_strength
```

Calibration formulas for `manual_existing` rows:

```text
calibrated_attack = 0.70 * current_attack + 0.30 * external_overall_strength
calibrated_defense = 0.70 * current_defense + 0.30 * external_overall_strength
calibrated_recent_form = 0.80 * current_recent_form + 0.20 * external_overall_strength
```

Delta caps keep benchmark calibration conservative:

```text
attack max change = 0.08
defense max change = 0.08
recent_form max change = 0.10
```

Generate calibrated proposals without modifying ratings:

```bash
.venv/bin/python scripts/calibrate_ratings_from_external.py
```

This writes:

```text
data/team_ratings_external_calibrated_proposed.csv
```

Write allowed calibrated rows explicitly:

```bash
.venv/bin/python scripts/calibrate_ratings_from_external.py --write
```

Default write behavior updates only `generated_from_behavior`, `neutral_placeholder`, and `manual_review_candidate` rows. Use `--include-manual-existing` to update existing manual rows. `manual_reviewed` rows are preserved unless `--include-reviewed` is explicitly passed.

Audit calibration quality:

```bash
.venv/bin/python scripts/audit_external_benchmark_calibration.py
```

Then rerun rating coverage, backtest QA, and the backtest:

```bash
.venv/bin/python scripts/audit_rating_coverage.py --start-date 2026-06-11 --end-date 2026-06-21 --competition "World Cup"
.venv/bin/python scripts/audit_backtest_quality.py --start-date 2026-06-11 --end-date 2026-06-21 --competition "World Cup"
.venv/bin/python scripts/run_asof_backtest.py --start-date 2026-06-11 --end-date 2026-06-21 --competition "World Cup" --save-report
.venv/bin/python scripts/run_blend_sensitivity.py --start-date 2026-06-11 --end-date 2026-06-21 --competition "World Cup" --save-report
```

Calibration warnings flag missing benchmarks, large internal-vs-external disagreements, behavior scores that sit far above external benchmarks, preserved reviewed rows, and delta caps. These warnings are audit signals; they do not trigger staking, trade execution, or automatic strategy changes.

## 12. Post-Mortem Model Training Loop

The post-mortem loop improves the model from completed matches without tuning to one result or copying any external model output. It has two persistent ledgers:

- `data/prediction_ledger.csv`: append-only pre-match prediction snapshots by model version and parameter set.
- `data/results_ledger.csv`: completed match results with outcome, totals, and BTTS flags.
- `data/tournament_learning_ledger.csv`: completed tournament matches that are eligible for future calibration only after their `eligible_after_utc`.

The app uses two learning loops:

- Before-match fast loop: for a selected upcoming match, build a match-specific calibration set from historical senior national-team matches and already-completed World Cup matches available before kickoff. It can recommend a match-specific parameter set, but it does not change the global primary model.
- After-match post-mortem loop: after results are imported, join final scores to saved pre-kickoff predictions, calculate errors, update the candidate leaderboard, and make completed matches available for future calibration only.

Before games, snapshot predictions before kickoff:

```bash
.venv/bin/python scripts/snapshot_upcoming_predictions.py \
  --start-date 2026-06-22 \
  --end-date 2026-06-27 \
  --competition "World Cup"
```

After games finish, run the rolling update:

```bash
.venv/bin/python scripts/update_after_completed_games.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-27 \
  --competition "World Cup"
```

This imports completed results, updates `data/tournament_learning_ledger.csv`, runs post-mortem scoring, refreshes the candidate leaderboard, and writes:

```text
reports/tournament_learning_update_YYYY-MM-DD.md
reports/tournament_learning_update_YYYY-MM-DD.csv
```

Before analyzing one upcoming match, build its match-specific calibration evidence:

```bash
.venv/bin/python scripts/run_match_specific_calibration.py \
  --home Jordan \
  --away Algeria \
  --prediction-date 2026-06-23 \
  --competition "World Cup" \
  --save-report
```

Completed World Cup games are included only when `eligible_after_utc < target_kickoff_utc`. The target match itself is excluded. This means Match A can train Match B after Match A finishes, but Match B cannot use its own result before evaluation.

You can still import final results directly:

```bash
.venv/bin/python scripts/import_completed_results.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup"
```

Run post-mortem analysis:

```bash
.venv/bin/python scripts/run_postmortem.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --save-report
```

The report shows Brier score, log loss, accuracy, actual-result probability, totals/BTTS errors, worst misses, best calls, draw calibration, team-level errors, and model-version comparisons.

Candidate model training uses walk-forward validation:

```bash
.venv/bin/python scripts/train_model_candidates.py \
  --start-date 2026-06-11 \
  --end-date 2026-06-21 \
  --competition "World Cup" \
  --save-report
```

The model does not learn only from World Cup games. It also uses recent qualifiers, continental matches, Nations League-style matches, friendlies, and other senior national-team games before the prediction date. These matches are relevance-weighted:

- recent World Cup and World Cup qualifying matches receive the highest weight;
- recent friendlies are included with moderate weight;
- old friendlies and older-cycle matches receive low weight;
- future matches are never included.

The transparent relevance score is:

```text
final_relevance_weight =
  competition_type_weight
  x recency_multiplier
  x tournament_cycle_multiplier
```

It is clipped between `0.10` and `2.00`. Training reports include total matches used, World Cup matches, qualifiers, friendlies, weighted match count, mean relevance weight, latest match used, and oldest match used.

Candidate parameter sets are stored in `data/model_parameter_sets.csv`. A candidate is promoted only if it improves strict out-of-sample 1X2 Brier score and log loss, does not worsen over 2.5 Brier by more than 5%, has at least 30 evaluated matches, and has no look-ahead violations. If no candidate clears that rule, the external-calibrated baseline remains the primary model.

Small samples should not be overinterpreted. These outputs are evaluation-only and do not provide staking, trade execution, Kelly sizing, or betting advice.

## 13. Weekly Semantic Audit

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

## 14. Validate The Model

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

## 15. Known Limitations

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
- Post-mortem training depends on saved pre-kickoff prediction snapshots; predictions saved after kickoff are excluded from scored post-mortems.
- Rolling tournament learning only uses completed matches after `eligible_after_utc`; a match result is never used to calibrate that same match before evaluation.
- Match-specific calibration recommendations are diagnostic-only and do not change the global primary model.
- Candidate training uses transparent parameter grids and walk-forward validation, not black-box machine learning.
- Candidate promotion requires strict out-of-sample improvement; otherwise the current baseline remains primary.
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
