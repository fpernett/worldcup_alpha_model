from __future__ import annotations

import pandas as pd

from src.match_identity import (
    align_result_to_prediction,
    build_result_fixture_crosswalk,
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


def _bridge_result(match_id: str, home: str = "Alpha", away: str = "Beta", date_utc: str = "2026-06-11", actual: str = "home_win") -> dict:
    return {
        "match_id": match_id,
        "date_utc": date_utc,
        "competition": "FIFA World Cup",
        "home": home,
        "away": away,
        "home_goals": 2 if actual == "home_win" else 0,
        "away_goals": 0 if actual == "home_win" else 2,
        "actual_result": actual,
        "result_source": "completed_matches",
    }


def _bridge_fixture(match_id: str = "wc2026_1", home: str = "Alpha", away: str = "Beta", date_utc: str = "2026-06-11", time_utc: str = "18:00") -> dict:
    return {
        "match_id": match_id,
        "date_utc": date_utc,
        "time_utc": time_utc,
        "competition": "FIFA World Cup",
        "home": home,
        "away": away,
    }


def test_result_fixture_crosswalk_maps_provider_id_by_team_date() -> None:
    crosswalk = build_result_fixture_crosswalk(
        pd.DataFrame([_bridge_result("provider_99")]),
        pd.DataFrame([_bridge_fixture("wc2026_1")]),
    )

    row = crosswalk.iloc[0]
    assert row["fixture_match_id"] == "wc2026_1"
    assert row["join_method"] == "exact_home_away_same_utc_date"
    assert row["team_order_status"] == "same"


def test_result_fixture_crosswalk_maps_generated_csv_id_by_team_date() -> None:
    crosswalk = build_result_fixture_crosswalk(
        pd.DataFrame([_bridge_result("csv_2026_06_11_alpha_beta_2_0_fifa_world_cup")]),
        pd.DataFrame([_bridge_fixture("wc2026_1")]),
    )

    assert crosswalk.iloc[0]["fixture_match_id"] == "wc2026_1"
    assert crosswalk.iloc[0]["join_method"] == "exact_home_away_same_utc_date"


def test_result_fixture_crosswalk_normalizes_required_team_aliases() -> None:
    results = pd.DataFrame(
        [
            _bridge_result("r_bosnia", home="Canada", away="Bosnia-H."),
            _bridge_result("r_ivory", home="Cote d Ivoire", away="Norway"),
            _bridge_result("r_curacao", home="Germany", away="Curacao"),
        ]
    )
    fixtures = pd.DataFrame(
        [
            _bridge_fixture("f_bosnia", home="Canada", away="Bosnia and Herzegovina"),
            _bridge_fixture("f_ivory", home="Ivory Coast", away="Norway"),
            _bridge_fixture("f_curacao", home="Germany", away="Curaçao"),
        ]
    )

    crosswalk = build_result_fixture_crosswalk(results, fixtures)

    assert set(crosswalk["fixture_match_id"]) == {"f_bosnia", "f_ivory", "f_curacao"}
    by_result = crosswalk.set_index("result_match_id")
    assert by_result.loc["r_bosnia", "join_method"] == "normalized_home_away_same_date"
    assert by_result.loc["r_ivory", "join_method"] == "normalized_home_away_same_date"
    assert by_result.loc["r_curacao", "fixture_match_id"] == "f_curacao"


def test_result_fixture_crosswalk_reversed_order_is_marked_and_aligns_result() -> None:
    crosswalk = build_result_fixture_crosswalk(
        pd.DataFrame([_bridge_result("provider_rev", home="Beta", away="Alpha", actual="home_win")]),
        pd.DataFrame([_bridge_fixture("wc2026_rev", home="Alpha", away="Beta")]),
    )
    result_row = pd.Series(
        {
            "home_goals": 2,
            "away_goals": 0,
            "actual_home_goals_90": 2,
            "actual_away_goals_90": 0,
            "actual_result": "home_win",
            "actual_result_1x2": "home_win",
        }
    )
    aligned = align_result_to_prediction(result_row, "reversed")

    assert crosswalk.iloc[0]["join_method"] == "reversed_home_away_same_utc_date"
    assert crosswalk.iloc[0]["team_order_status"] == "reversed"
    assert aligned["actual_result_1x2"] == "away_win"
    assert aligned["actual_home_goals_90"] == 0


def test_result_fixture_crosswalk_does_not_join_ambiguous_candidates_silently() -> None:
    crosswalk = build_result_fixture_crosswalk(
        pd.DataFrame([_bridge_result("provider_ambiguous")]),
        pd.DataFrame([_bridge_fixture("wc2026_a"), _bridge_fixture("wc2026_b")]),
    )

    assert crosswalk.iloc[0]["join_method"] == "ambiguous_multiple_candidates"
    assert crosswalk.iloc[0]["fixture_match_id"] == ""


def test_result_fixture_crosswalk_within_36h_handles_utc_date_difference() -> None:
    crosswalk = build_result_fixture_crosswalk(
        pd.DataFrame([_bridge_result("provider_late", date_utc="2026-06-29")]),
        pd.DataFrame([_bridge_fixture("wc2026_late", date_utc="2026-06-30", time_utc="01:00")]),
    )

    assert crosswalk.iloc[0]["join_method"] == "exact_home_away_within_36h"
    assert crosswalk.iloc[0]["fixture_match_id"] == "wc2026_late"


def test_result_fixture_crosswalk_uses_high_confidence_fuzzy_team_date() -> None:
    crosswalk = build_result_fixture_crosswalk(
        pd.DataFrame([_bridge_result("provider_fuzzy", home="Bosnia and Herzegovin", away="South Afric")]),
        pd.DataFrame([_bridge_fixture("wc2026_fuzzy", home="Bosnia and Herzegovina", away="South Africa")]),
    )

    assert crosswalk.iloc[0]["join_method"] == "fuzzy_team_match_same_date"
    assert crosswalk.iloc[0]["fixture_match_id"] == "wc2026_fuzzy"


def test_result_fixture_crosswalk_exact_match_id_takes_priority() -> None:
    crosswalk = build_result_fixture_crosswalk(
        pd.DataFrame([_bridge_result("wc2026_exact")]),
        pd.DataFrame(
            [
                _bridge_fixture("wc2026_exact"),
                _bridge_fixture("wc2026_other"),
            ]
        ),
    )

    assert crosswalk.iloc[0]["join_method"] == "exact_match_id"
    assert crosswalk.iloc[0]["fixture_match_id"] == "wc2026_exact"
