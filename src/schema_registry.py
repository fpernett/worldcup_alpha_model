"""Central schema and semantic-audit file registry.

The weekly semantic audit uses this module as the local source of truth for
expected CSV headers and the files that define model, metric, source, mapping,
dashboard, and documentation behavior.
"""

EXPECTED_CSV_SCHEMAS: dict[str, list[str]] = {
    "data/fixtures.csv": [
        "match_id",
        "date_utc",
        "time_utc",
        "competition",
        "group",
        "home",
        "away",
        "venue",
        "city",
        "country",
    ],
    "data/team_ratings.csv": [
        "team",
        "elo",
        "attack",
        "defense",
        "recent_form",
        "fifa_rank_proxy",
        "training_temp_c",
        "training_humidity_pct",
        "data_quality",
        "last_updated",
        "notes",
    ],
    "data/venues.csv": [
        "venue",
        "city",
        "country",
        "latitude",
        "longitude",
        "altitude_m",
        "temp_c",
        "humidity_pct",
        "wind_kmh",
        "precipitation_mm",
        "roof_expected_closed",
        "neutral_site",
    ],
    "data/market_odds.csv": [
        "match_id",
        "market",
        "selection",
        "odds",
        "source",
        "last_updated",
    ],
    "data/polymarket_markets.csv": [
        "market_id",
        "question",
        "slug",
        "event_title",
        "category",
        "start_date",
        "end_date",
        "active",
        "closed",
        "outcomes",
        "yes_price",
        "no_price",
        "liquidity",
        "volume",
        "source",
        "last_updated",
    ],
    "data/market_mappings.csv": [
        "match_id",
        "market_id",
        "market_type",
        "model_side",
        "polymarket_side",
        "manual_confirmed",
        "notes",
    ],
    "data/prediction_log.csv": [
        "timestamp_utc",
        "match_id",
        "home",
        "away",
        "kickoff_utc",
        "market_id",
        "question",
        "market_type",
        "model_side",
        "polymarket_side",
        "model_probability",
        "fair_price_cents",
        "polymarket_price_cents",
        "alpha_gap_cents",
        "alpha_ev",
        "signal_strength",
        "home_xg",
        "away_xg",
        "model_config_name",
        "mapping_confidence",
        "liquidity",
        "volume",
    ],
    "data/results_log.csv": [
        "match_id",
        "home",
        "away",
        "home_goals",
        "away_goals",
        "result_home_win",
        "result_draw",
        "result_away_win",
        "over_2_5",
        "under_2_5",
        "btts_yes",
        "btts_no",
        "completed",
        "result_source",
        "last_updated",
    ],
}

MODEL_DEFINITION_FILES: set[str] = {
    "src/model.py",
    "src/ratings.py",
    "src/climate.py",
    "src/weather.py",
}

METRIC_DEFINITION_FILES: set[str] = {
    "src/model.py",
    "src/alpha.py",
    "src/sensitivity.py",
    "src/backtesting.py",
    "src/report_metrics.py",
    "src/market_tables.py",
}

API_SOURCE_FILES: set[str] = {
    "src/config.py",
    "src/data_sources.py",
    "src/weather.py",
    "src/odds.py",
    "src/polymarket.py",
    "src/cache.py",
}

POLYMARKET_MAPPING_FILES: set[str] = {
    "src/polymarket.py",
    "src/market_mapping.py",
    "src/alpha.py",
    "data/polymarket_markets.csv",
    "data/market_mappings.csv",
}

DASHBOARD_DEFINITION_FILES: set[str] = {
    "app.py",
    "src/report_charts.py",
    "src/report_metrics.py",
    "src/market_tables.py",
    "src/timeline.py",
}

DOCUMENTATION_FILES: set[str] = {
    "README.md",
    "AGENTS.md",
}

SEMANTIC_LAYER_DOCUMENTS: set[str] = {
    "~/.codex/skills/worldcup-alpha-model-semantic-layer/SKILL.md",
    "~/.codex/skills/worldcup-alpha-model-semantic-layer/references/semantic-layer.md",
    "~/.codex/skills/worldcup-alpha-model-semantic-layer/references/source-inventory.md",
    "~/.codex/skills/worldcup-alpha-model-semantic-layer/references/evidence.md",
}

