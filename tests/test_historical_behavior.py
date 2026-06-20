from __future__ import annotations

import pandas as pd

from src.environment_response import calculate_environment_response
from src.historical_data import load_historical_matches
from src.recency import calculate_match_weight, calculate_recency_weight, competition_weight
from src.team_behavior import calculate_attack_behavior, calculate_defense_behavior


def sample_historical_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-01-01",
                "competition": "World Cup",
                "competition_type": "world_cup",
                "team": "Alpha",
                "opponent": "Beta",
                "team_goals": 2,
                "opponent_goals": 0,
                "team_xg": 1.8,
                "opponent_xg": 0.7,
                "shots_for": 12,
                "shots_against": 6,
                "shots_on_target_for": 5,
                "shots_on_target_against": 2,
                "opponent_elo": 1800,
                "temperature_c": 31,
                "humidity_pct": 72,
                "altitude_m": 1200,
            },
            {
                "match_id": "m2",
                "date_utc": "2025-06-01",
                "competition": "Friendly",
                "competition_type": "friendly",
                "team": "Alpha",
                "opponent": "Gamma",
                "team_goals": 1,
                "opponent_goals": 1,
                "team_xg": 1.2,
                "opponent_xg": 1.0,
                "shots_for": 9,
                "shots_against": 8,
                "shots_on_target_for": 3,
                "shots_on_target_against": 3,
                "opponent_elo": 1650,
                "temperature_c": 18,
                "humidity_pct": 55,
                "altitude_m": 50,
            },
            {
                "match_id": "m3",
                "date_utc": "2024-06-01",
                "competition": "Other",
                "competition_type": "other",
                "team": "Alpha",
                "opponent": "Delta",
                "team_goals": 0,
                "opponent_goals": 2,
                "team_xg": 0.8,
                "opponent_xg": 1.7,
                "shots_for": 6,
                "shots_against": 14,
                "shots_on_target_for": 1,
                "shots_on_target_against": 6,
                "opponent_elo": 1700,
                "temperature_c": 20,
                "humidity_pct": 60,
                "altitude_m": 100,
            },
        ]
    )


def test_recency_weight_decreases_with_older_matches() -> None:
    reference = "2026-06-19"
    recent = calculate_recency_weight("2026-05-01", reference)
    old = calculate_recency_weight("2024-05-01", reference)

    assert recent > old


def test_competition_weighting_works() -> None:
    assert competition_weight("world_cup") > competition_weight("friendly")
    assert competition_weight("nations_league") == 0.90


def test_match_weight_combines_recency_and_competition() -> None:
    reference = "2026-06-19"
    world_cup = calculate_match_weight(pd.Series({"date_utc": "2026-05-01", "competition_type": "world_cup"}), reference)
    friendly = calculate_match_weight(pd.Series({"date_utc": "2026-05-01", "competition_type": "friendly"}), reference)

    assert world_cup > friendly


def test_attack_index_remains_between_zero_and_one() -> None:
    behavior = calculate_attack_behavior(sample_historical_matches(), "Alpha", "2026-06-19")

    assert 0.0 <= behavior["attack_index"] <= 1.0


def test_defense_index_remains_between_zero_and_one() -> None:
    behavior = calculate_defense_behavior(sample_historical_matches(), "Alpha", "2026-06-19")

    assert 0.0 <= behavior["defense_index"] <= 1.0


def test_low_environment_sample_size_returns_low_confidence() -> None:
    response = calculate_environment_response(sample_historical_matches(), "Alpha", "2026-06-19")

    assert response["environment_data_quality"] == "low"
    assert response["environment_warning"]


def test_missing_historical_data_does_not_crash(tmp_path) -> None:
    missing_path = tmp_path / "historical_matches.csv"
    loaded = load_historical_matches(path=missing_path, use_cache=False)

    assert loaded.empty
    assert "team" in loaded.columns
