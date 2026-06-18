AGENTS.md — World Cup Alpha Model

Project purpose

This is a local Streamlit app for FIFA World Cup match probability modelling and Polymarket alpha screening.

The app estimates:

* match outcome probabilities,
* expected goals,
* scoreline probabilities,
* fair odds,
* goal markets,
* environmental adjustments,
* model-vs-market alpha signals.

The app is for statistical alpha generation only. It must not provide staking advice, investment advice, Kelly sizing, or automated trading.

Current architecture

Main files:

app.py
src/model.py
src/config.py
src/data_sources.py
src/weather.py
src/climate.py
src/ratings.py
src/odds.py
src/cache.py
src/polymarket.py
src/market_mapping.py
src/alpha.py
src/sensitivity.py
src/backtesting.py
src/storage.py
data/fixtures.csv
data/team_ratings.csv
data/venues.csv
data/market_odds.csv
data/polymarket_markets.csv
data/market_mappings.csv
data/prediction_log.csv
data/results_log.csv
requirements.txt
README.md

Security rules

Never hard-code API keys.

Never commit .env.

Use environment variables only:

FOOTBALL_API_URL=
FOOTBALL_API_KEY=
WEATHER_API_URL=
WEATHER_API_KEY=
RATINGS_API_URL=
ODDS_API_URL=
ODDS_API_KEY=
POLYMARKET_API_URL=
POLYMARKET_CLOB_API_URL=
POLYMARKET_GAMMA_API_URL=

If an API key is found in source code, remove it and replace it with os.getenv(...).

Design rules

The app must always work without APIs.

Every API function must have a local CSV fallback.

If an API fails, the app should show a warning and continue.

Do not let Streamlit crash because of:

* missing API keys,
* missing columns,
* empty dataframes,
* failed API requests,
* unexpected API schemas,
* missing market prices,
* missing cache files.

Use defensive programming.

Modelling rules

Keep the model transparent.

Do not replace the current model with a black-box machine learning model unless explicitly requested.

Expected goals should remain interpretable and based on:

* Elo or rating strength,
* attack strength,
* opponent defense,
* recent form,
* venue/environment,
* training-climate mismatch.

Scorelines should come from the existing Poisson-style framework unless explicitly changed.

Fair odds:

fair_odds = 1 / probability

Alpha EV:

alpha_EV = model_probability × market_decimal_odds − 1

Polymarket alpha gap:

fair_price_cents = model_probability × 100
alpha_gap_cents = fair_price_cents − polymarket_price_cents

Polymarket rules

Do not execute trades.

Do not add wallet functionality.

Do not add private-key handling.

Do not add order placement.

The app should only display statistical alpha signals.

Polymarket APIs are read-only market-data sources in this project. The presence
of `POLYMARKET_CLOB_API_URL` must not imply wallet authentication, order
placement, signing, cancellation, private-key handling, or any execution flow.

For a market like:

Will England beat Croatia?

Mapping is:

YES = England win
NO = Draw or Croatia win

For totals:

Under 2.5 YES = model under_2_5 probability
Over 2.5 YES = model over_2_5 probability

For BTTS:

BTTS Yes = btts_yes
BTTS No = btts_no

Manual market mappings must override fuzzy mappings.

Prediction and backtesting rules

`data/prediction_log.csv` is append-only prediction history. Normal app use
should append new snapshots, not overwrite prior predictions.

`data/results_log.csv` is manual or validated result history.

Backtesting is evaluation-only. It may report Brier score, log loss, calibration,
mean alpha gap, and hit rates by signal strength. It must not turn those outputs
into staking, sizing, execution, or investment recommendations.

Dashboard rules

The dashboard should remain beginner-friendly.

Each selected match should show:

* Summary,
* Score matrix,
* Markets,
* Polymarket alpha,
* Sensitivity,
* Backtesting,
* Team inputs,
* Venue/environment,
* Model notes.

Always show source diagnostics:

* fixtures source,
* ratings source,
* weather source,
* odds source,
* Polymarket market source,
* last updated timestamp.

Testing rules

Before saying the task is complete, run:

python -m py_compile app.py src/*.py scripts/*.py tests/*.py

If tests exist, run:

pytest

The app should start with:

python -m streamlit run app.py

Git rules

Before making large changes, create a branch.

Do not commit:

* .env,
* .venv,
* cache files,
* API keys,
* private credentials.

Use small commits with clear messages.

Development style

Prefer simple, readable Python.

Avoid unnecessary abstractions.

Avoid paid API requirements.

Keep manual CSV override support.

Do not break the local fallback mode.
