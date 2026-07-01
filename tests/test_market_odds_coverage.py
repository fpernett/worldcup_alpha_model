from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR
from src.data_sources import FIXTURE_COLUMNS, filter_future_fixtures
from src.model import ModelConfig, market_probability_map, run_match_model
from src.odds import GENERATED_ODDS_SOURCE, ODDS_COLUMNS, ensure_market_odds_for_fixtures
from src.ratings import TEAM_RATING_COLUMNS
from src.weather import VENUE_COLUMNS


def _disable_api_env(monkeypatch) -> None:
    for name in [
        "FOOTBALL_API_URL",
        "FOOTBALL_API_KEY",
        "WEATHER_API_URL",
        "WEATHER_API_KEY",
        "RATINGS_API_URL",
        "ODDS_API_URL",
        "ODDS_API_KEY",
    ]:
        monkeypatch.setenv(name, "")


def _real_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixtures = pd.read_csv(DATA_DIR / "fixtures.csv", dtype=str)
    teams = pd.read_csv(DATA_DIR / "team_ratings.csv")
    venues = pd.read_csv(DATA_DIR / "venues.csv")

    for col in FIXTURE_COLUMNS:
        assert col in fixtures.columns
    for col in TEAM_RATING_COLUMNS:
        assert col in teams.columns
    for col in VENUE_COLUMNS:
        assert col in venues.columns

    return fixtures, teams, venues


def _future_fixture_window(fixtures: pd.DataFrame) -> pd.DataFrame:
    return filter_future_fixtures(
        fixtures,
        now_utc="2026-07-01T18:49:45Z",
        horizon_hours=24 * 7,
    )


def _expected_markets(fixture: pd.Series) -> set[tuple[str, str]]:
    return set(
        market_probability_map(
            str(fixture["home"]),
            str(fixture["away"]),
            {
                "home_win": 0.4,
                "draw": 0.25,
                "away_win": 0.35,
                "under_2_5": 0.5,
                "over_2_5": 0.5,
                "under_3_5": 0.7,
                "over_3_5": 0.3,
                "btts_yes": 0.5,
                "btts_no": 0.5,
                "home_minus_1_5": 0.2,
                "away_plus_1_5": 0.8,
                "away_minus_1_5": 0.2,
                "home_plus_1_5": 0.8,
                "home_or_draw": 0.65,
                "draw_or_away": 0.6,
            },
        ).keys()
    )


def test_generated_benchmark_odds_cover_many_future_fixtures_without_api(monkeypatch) -> None:
    _disable_api_env(monkeypatch)
    fixtures, teams, venues = _real_inputs()
    selected = _future_fixture_window(fixtures)
    assert len(selected) > 5

    market_odds = ensure_market_odds_for_fixtures(
        selected,
        teams,
        venues,
        pd.DataFrame(columns=ODDS_COLUMNS),
    )

    assert int(market_odds.attrs.get("generated_rows", 0)) >= len(selected) * 15
    assert GENERATED_ODDS_SOURCE in set(market_odds["source"])

    for _, fixture in selected.iterrows():
        fixture_odds = market_odds.loc[market_odds["match_id"].astype(str) == str(fixture["match_id"])]
        actual_markets = set(zip(fixture_odds["market"], fixture_odds["selection"]))

        assert _expected_markets(fixture) <= actual_markets

        result = run_match_model(fixture, teams, venues, market_odds, ModelConfig())
        assert "local benchmark odds available; alpha EV is benchmark-only" in result["confidence"]["reasons"]
        assert "market odds missing; alpha EV left blank" not in result["confidence"]["reasons"]
        assert result["alpha"]["market_odds"].notna().all()
        assert result["alpha"]["alpha_ev"].notna().all()


def test_generated_benchmark_odds_do_not_overwrite_existing_manual_odds(monkeypatch) -> None:
    _disable_api_env(monkeypatch)
    fixtures, teams, venues = _real_inputs()
    fixture = _future_fixture_window(fixtures).iloc[0]
    manual_odds = pd.DataFrame(
        [
            {
                "match_id": fixture["match_id"],
                "market": "1X2",
                "selection": fixture["home"],
                "odds": 9.99,
                "source": "manual_csv",
                "last_updated": "2026-07-01",
            }
        ],
        columns=ODDS_COLUMNS,
    )

    market_odds = ensure_market_odds_for_fixtures(pd.DataFrame([fixture]), teams, venues, manual_odds)
    preserved = market_odds.loc[
        (market_odds["match_id"].astype(str) == str(fixture["match_id"]))
        & (market_odds["market"] == "1X2")
        & (market_odds["selection"] == fixture["home"])
    ].iloc[0]

    assert float(preserved["odds"]) == 9.99
    assert preserved["source"] == "manual_csv"
    assert len(market_odds) == len(_expected_markets(fixture))
    assert int((market_odds["source"] == GENERATED_ODDS_SOURCE).sum()) == len(_expected_markets(fixture)) - 1
