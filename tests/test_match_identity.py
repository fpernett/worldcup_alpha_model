from __future__ import annotations

import pandas as pd

from src.match_identity import (
    build_match_key,
    build_symmetric_match_key,
    normalize_competition_name,
    normalize_match_date,
    normalize_team_name,
    resolve_prediction_to_result,
    validate_results_ledger_semantics,
)


def test_normalized_identity_keys_are_stable() -> None:
    assert normalize_team_name("Cabo Verde") == "Cape Verde"
    assert normalize_competition_name("FIFA World Cup 2026") == "world cup"
    assert normalize_match_date("2026-06-30T01:00:00+00:00") == "2026-06-30"
    assert build_match_key("Cabo Verde", "Argentina", "2026-07-03T22:00:00+00:00", "FIFA World Cup")[0] == "cape verde"
    assert build_symmetric_match_key("Argentina", "Cape Verde", "2026-07-03", "World Cup") == build_symmetric_match_key("Cape Verde", "Argentina", "2026-07-03", "FIFA World Cup")


def test_resolver_exact_match_id_join() -> None:
    prediction = pd.Series(
        {
            "match_id": "m1",
            "home_team": "Alpha",
            "away_team": "Beta",
            "kickoff_utc": "2026-06-11T18:00:00+00:00",
            "competition": "FIFA World Cup",
        }
    )
    results = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-11",
                "competition": "FIFA World Cup",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 1,
                "away_goals": 0,
                "actual_result": "home_win",
                "result_source": "completed_matches",
            }
        ]
    )

    resolved = resolve_prediction_to_result(prediction, results)

    assert resolved["result_join_method"] == "exact_match_id"
    assert resolved["resolved_result_match_id"] == "m1"


def test_results_semantics_validation_keeps_penalties_separate_from_1x2_draw() -> None:
    results = pd.DataFrame(
        [
            {
                "match_id": "knockout_1",
                "date_utc": "2026-06-29",
                "competition": "FIFA World Cup",
                "home": "Germany",
                "away": "Paraguay",
                "home_goals": 1,
                "away_goals": 1,
                "actual_result": "draw",
                "actual_advancing_team": "Paraguay",
                "result_source": "completed_matches",
                "had_penalties": True,
                "penalties_home": 4,
                "penalties_away": 5,
            }
        ]
    )

    validated = validate_results_ledger_semantics(results)
    row = validated.iloc[0]

    assert row["actual_result"] == "draw"
    assert bool(row["evaluation_eligible_1x2"])
    assert row["actual_advancing_team"] == "Paraguay"
    assert bool(row["had_penalties"])
