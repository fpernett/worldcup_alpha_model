from __future__ import annotations

import pandas as pd

from src.climate import environment_from_venue_row
from src.model import ModelConfig, environmental_adjustments, fair_odds, outcome_probs, run_match_model, score_matrix


def sample_teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Alpha",
                "elo": 1800,
                "attack": 0.70,
                "defense": 0.68,
                "recent_form": 0.65,
                "fifa_rank_proxy": 20,
                "training_temp_c": 18,
                "training_humidity_pct": 60,
                "data_quality": "manual_csv",
                "last_updated": "2026-06-17",
                "notes": "",
            },
            {
                "team": "Beta",
                "elo": 1720,
                "attack": 0.58,
                "defense": 0.57,
                "recent_form": 0.55,
                "fifa_rank_proxy": 40,
                "training_temp_c": 27,
                "training_humidity_pct": 75,
                "data_quality": "manual_csv",
                "last_updated": "2026-06-17",
                "notes": "",
            },
        ]
    )


def sample_venues(roof_closed: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "venue": "Test Stadium",
                "city": "Test City",
                "country": "USA",
                "altitude_m": 1400,
                "temp_c": 32,
                "humidity_pct": 78,
                "wind_kmh": 24,
                "precipitation_mm": 3,
                "roof_expected_closed": roof_closed,
                "indoor_adjusted_temp_c": 22,
                "environment_source": "venues_csv",
                "last_updated": "2026-06-17",
                "neutral_site": 1,
            }
        ]
    )


def sample_match() -> pd.Series:
    return pd.Series(
        {
            "match_id": "TEST-1",
            "date_utc": "2026-06-17",
            "time_utc": "17:00",
            "competition": "FIFA World Cup",
            "group": "Group X",
            "home": "Alpha",
            "away": "Beta",
            "venue": "Test Stadium",
            "city": "Test City",
            "country": "USA",
        }
    )


def test_outcome_probabilities_sum_to_one() -> None:
    mat = score_matrix(1.4, 1.1)
    probs = outcome_probs(mat)
    assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1.0) < 1e-9


def test_fair_odds_calculation() -> None:
    assert fair_odds(0.5) == 2.0
    assert fair_odds(0.25) == 4.0


def test_score_matrix_sums_to_one() -> None:
    mat = score_matrix(1.6, 0.9)
    assert abs(mat["prob"].sum() - 1.0) < 1e-9


def test_missing_odds_do_not_crash() -> None:
    result = run_match_model(sample_match(), sample_teams(), sample_venues(), pd.DataFrame(), ModelConfig())
    assert "alpha" in result
    assert result["alpha"]["market_odds"].isna().all()


def test_missing_weather_does_not_crash() -> None:
    result = run_match_model(sample_match(), sample_teams(), pd.DataFrame(), pd.DataFrame(), ModelConfig())
    assert result["hxg"] > 0
    assert result["axg"] > 0
    assert result["environment"]["environment_source"] == "neutral_venue_fallback"


def test_roof_closed_reduces_weather_impact() -> None:
    teams = sample_teams()
    open_env = environment_from_venue_row(sample_venues(roof_closed=0).iloc[0])
    roof_env = environment_from_venue_row(sample_venues(roof_closed=1).iloc[0])

    open_adj = environmental_adjustments(teams.iloc[0], teams.iloc[1], open_env, ModelConfig())
    roof_adj = environmental_adjustments(teams.iloc[0], teams.iloc[1], roof_env, ModelConfig())

    assert abs(roof_adj["total_environment_log_adj"]) < abs(open_adj["total_environment_log_adj"])
    assert roof_adj["weather_impact_multiplier"] < open_adj["weather_impact_multiplier"]
