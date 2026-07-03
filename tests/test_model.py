from __future__ import annotations

import pandas as pd

from src.climate import environment_from_venue_row
from src.data_sources import filter_future_fixtures
from src.model import ModelConfig, environmental_adjustments, expected_goals, fair_odds, outcome_probs, run_match_model, score_matrix


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


def test_expected_goals_uses_natural_matchup_total_not_fixed_anchor() -> None:
    def team(name: str, attack: float, defense: float) -> pd.Series:
        return pd.Series(
            {
                "team": name,
                "elo": 1700,
                "attack": attack,
                "defense": defense,
                "recent_form": 0.60,
                "training_temp_c": 22,
                "training_humidity_pct": 55,
            }
        )

    env = {
        "venue": "Neutral Test",
        "country": "USA",
        "altitude_m": 0,
        "temp_c": 22,
        "humidity_pct": 55,
        "wind_kmh": 0,
        "precipitation_mm": 0,
        "roof_expected_closed": 0,
        "neutral_site": 1,
    }
    cfg = ModelConfig(base_total_goals=2.50)

    high_hxg, high_axg, high_components = expected_goals(team("Alpha", 0.90, 0.40), team("Beta", 0.90, 0.40), env, cfg)
    low_hxg, low_axg, low_components = expected_goals(team("Gamma", 0.50, 0.80), team("Delta", 0.50, 0.80), env, cfg)

    assert high_hxg + high_axg > low_hxg + low_axg
    assert abs((high_hxg + high_axg) - cfg.base_total_goals) > 0.05
    assert abs((low_hxg + low_axg) - cfg.base_total_goals) > 0.05
    assert high_components["old_base_total_goals_anchored_total_xg"] == cfg.base_total_goals
    assert low_components["old_base_total_goals_anchored_total_xg"] == cfg.base_total_goals
    assert high_components["home_raw_xg_before_adjustments"] > 0
    assert high_components["home_final_xg"] == high_hxg


def test_future_fixture_filter_excludes_past_and_keeps_tomorrow() -> None:
    fixtures = pd.DataFrame(
        [
            {"match_id": "PAST", "date_utc": "2026-06-18", "time_utc": "02:00", "home": "A", "away": "B"},
            {"match_id": "LATER", "date_utc": "2026-06-18", "time_utc": "16:00", "home": "C", "away": "D"},
            {"match_id": "TOMORROW", "date_utc": "2026-06-19", "time_utc": "01:00", "home": "E", "away": "F"},
            {"match_id": "TOO_FAR", "date_utc": "2026-06-21", "time_utc": "12:00", "home": "G", "away": "H"},
        ]
    )

    filtered = filter_future_fixtures(fixtures, now_utc="2026-06-18T13:07:00Z", horizon_hours=48)

    assert filtered["match_id"].tolist() == ["LATER", "TOMORROW"]
    assert "kickoff_utc" in filtered.columns


def test_fixture_filter_can_include_past_for_selected_date_range() -> None:
    fixtures = pd.DataFrame(
        [
            {"match_id": "PAST", "date_utc": "2026-06-18", "time_utc": "02:00", "home": "A", "away": "B"},
            {"match_id": "LATER", "date_utc": "2026-06-18", "time_utc": "16:00", "home": "C", "away": "D"},
            {"match_id": "TOMORROW", "date_utc": "2026-06-19", "time_utc": "01:00", "home": "E", "away": "F"},
        ]
    )

    filtered = filter_future_fixtures(
        fixtures,
        now_utc="2026-06-18T13:07:00Z",
        include_past=True,
    )

    assert filtered["match_id"].tolist() == ["PAST", "LATER", "TOMORROW"]
    assert "kickoff_utc" in filtered.columns


def test_future_fixture_filter_excludes_unresolved_bracket_slots() -> None:
    fixtures = pd.DataFrame(
        [
            {"match_id": "REAL", "date_utc": "2026-07-01", "time_utc": "16:00", "home": "England", "away": "DR Congo"},
            {
                "match_id": "PLACEHOLDER",
                "date_utc": "2026-07-01",
                "time_utc": "20:00",
                "home": "Winner Group K",
                "away": "3rd Group E/I/L",
            },
        ]
    )

    filtered = filter_future_fixtures(
        fixtures,
        now_utc="2026-07-01T12:00:00Z",
        horizon_hours=24,
    )

    assert filtered["match_id"].tolist() == ["REAL"]
    assert filtered.attrs.get("unresolved_fixture_count") == 1
    assert "future fixtures pending prior match results" in filtered.attrs.get("warning", "")


def test_fixture_filter_can_include_unresolved_slots_for_audit_use() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "PLACEHOLDER",
                "date_utc": "2026-07-01",
                "time_utc": "20:00",
                "home": "winner of group K",
                "away": "third of group L",
            },
        ]
    )

    filtered = filter_future_fixtures(
        fixtures,
        now_utc="2026-07-01T12:00:00Z",
        horizon_hours=24,
        include_unresolved=True,
    )

    assert filtered["match_id"].tolist() == ["PLACEHOLDER"]
    assert filtered["has_unresolved_team_slot"].tolist() == [True]


def test_local_fixture_fallback_covers_next_48_hours() -> None:
    fixtures = pd.read_csv("data/fixtures.csv")

    filtered = filter_future_fixtures(
        fixtures,
        now_utc="2026-06-18T23:00:00Z",
        horizon_hours=48,
    )

    labels = set(filtered["home"].astype(str) + " vs " + filtered["away"].astype(str))
    assert len(filtered) >= 7
    assert "Mexico vs South Korea" in labels
    assert "United States vs Australia" in labels
    assert "Brazil vs Haiti" in labels


def test_local_fixture_fallback_covers_june_28_dashboard_default() -> None:
    fixtures = pd.read_csv("data/fixtures.csv")

    filtered = filter_future_fixtures(
        fixtures,
        now_utc="2026-06-28T09:00:00Z",
        horizon_hours=48,
    )

    labels = set(filtered["home"].astype(str) + " vs " + filtered["away"].astype(str))
    assert len(filtered) >= 4
    assert "South Africa vs Canada" in labels
    assert "Brazil vs Japan" in labels
