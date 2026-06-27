from __future__ import annotations

import pandas as pd

from src.tournament_context import (
    apply_tournament_context_adjustment,
    build_group_standings_asof,
    calculate_team_qualification_need,
    classify_match_context,
    remaining_group_fixtures_asof,
)


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": "H1", "date_utc": "2026-06-21", "time_utc": "16:00", "competition": "FIFA World Cup", "group": "Group H", "home": "Spain", "away": "Saudi Arabia"},
            {"match_id": "H2", "date_utc": "2026-06-21", "time_utc": "22:00", "competition": "FIFA World Cup", "group": "Group H", "home": "Uruguay", "away": "Cape Verde"},
            {"match_id": "H3", "date_utc": "2026-06-26", "time_utc": "23:59", "competition": "FIFA World Cup", "group": "Group H", "home": "Uruguay", "away": "Spain"},
            {"match_id": "H4", "date_utc": "2026-06-26", "time_utc": "23:59", "competition": "FIFA World Cup", "group": "Group H", "home": "Cape Verde", "away": "Saudi Arabia"},
        ]
    )


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": "R0", "date_utc": "2026-06-15", "competition": "FIFA World Cup", "home": "Spain", "away": "Cape Verde", "home_goals": 0, "away_goals": 0},
            {"match_id": "R1", "date_utc": "2026-06-15", "competition": "FIFA World Cup", "home": "Saudi Arabia", "away": "Uruguay", "home_goals": 1, "away_goals": 1},
            {"match_id": "H1", "date_utc": "2026-06-21", "competition": "FIFA World Cup", "home": "Spain", "away": "Saudi Arabia", "home_goals": 2, "away_goals": 0},
            {"match_id": "H2", "date_utc": "2026-06-21", "competition": "FIFA World Cup", "home": "Uruguay", "away": "Cape Verde", "home_goals": 1, "away_goals": 1},
            {"match_id": "H3", "date_utc": "2026-06-26", "competition": "FIFA World Cup", "home": "Uruguay", "away": "Spain", "home_goals": 3, "away_goals": 0},
        ]
    )


def _baseline_probs() -> dict:
    return {
        "home_win": 0.30,
        "draw": 0.28,
        "away_win": 0.42,
        "over_2_5": 0.48,
        "under_2_5": 0.52,
        "btts_yes": 0.50,
        "btts_no": 0.50,
    }


def test_group_standings_are_built_only_before_kickoff() -> None:
    standings = build_group_standings_asof(_results(), _fixtures(), "H", "2026-06-26 23:59 UTC")
    spain = standings.loc[standings["team"] == "Spain"].iloc[0]
    uruguay = standings.loc[standings["team"] == "Uruguay"].iloc[0]

    assert int(spain["points"]) == 4
    assert int(uruguay["points"]) == 2


def test_future_group_matches_are_excluded() -> None:
    standings = build_group_standings_asof(_results(), _fixtures(), "H", "2026-06-22 00:00 UTC")

    assert "H3" not in standings.to_string()
    assert int(standings.loc[standings["team"] == "Spain", "points"].iloc[0]) == 4
    assert int(standings.loc[standings["team"] == "Uruguay", "points"].iloc[0]) == 2


def test_spain_uruguay_classification_from_standings() -> None:
    standings = build_group_standings_asof(_results(), _fixtures(), "H", "2026-06-26 23:59 UTC")
    remaining = remaining_group_fixtures_asof(_fixtures(), "H", "2026-06-26 23:59 UTC")
    match = _fixtures().loc[_fixtures()["match_id"] == "H3"].iloc[0]

    context = classify_match_context("Uruguay", "Spain", match, standings, remaining)

    assert context["away_incentive_label"] == "draw_enough"
    assert context["home_incentive_label"] == "needs_win"
    assert context["context_type"] == "one_needs_win_other_draw_enough"
    assert context["draw_bias"] == "positive"
    assert context["tempo_bias"] == "slower_early"


def test_reversed_fixture_order_still_works() -> None:
    standings = build_group_standings_asof(_results(), _fixtures(), "H", "2026-06-26 23:59 UTC")
    remaining = remaining_group_fixtures_asof(_fixtures(), "H", "2026-06-26 23:59 UTC")
    match = pd.Series({"match_id": "H3", "date_utc": "2026-06-26", "time_utc": "23:59", "competition": "FIFA World Cup", "group": "Group H", "home": "Spain", "away": "Uruguay"})

    context = classify_match_context("Spain", "Uruguay", match, standings, remaining)

    assert context["home_incentive_label"] == "draw_enough"
    assert context["away_incentive_label"] == "needs_win"
    assert context["context_type"] == "one_needs_win_other_draw_enough"


def test_knockout_match_does_not_use_draw_enough_logic() -> None:
    match = pd.Series({"match_id": "K1", "stage": "knockout", "competition": "FIFA World Cup Round of 16", "home": "Spain", "away": "Uruguay"})

    context = classify_match_context("Spain", "Uruguay", match, pd.DataFrame(), pd.DataFrame())

    assert context["context_type"] == "knockout_must_advance"
    assert context["home_incentive_label"] == "must_advance"
    assert context["away_incentive_label"] == "must_advance"


def test_context_adjustment_never_shifts_1x2_more_than_five_pp() -> None:
    standings = build_group_standings_asof(_results(), _fixtures(), "H", "2026-06-26 23:59 UTC")
    remaining = remaining_group_fixtures_asof(_fixtures(), "H", "2026-06-26 23:59 UTC")
    context = classify_match_context("Uruguay", "Spain", _fixtures().loc[_fixtures()["match_id"] == "H3"].iloc[0], standings, remaining)
    adjusted, _markets, diagnostics = apply_tournament_context_adjustment(_baseline_probs(), pd.DataFrame(), context)

    assert diagnostics["max_probability_shift_pp"] <= 5.0
    for key in ["home_win", "draw", "away_win"]:
        assert abs(adjusted[key] - _baseline_probs()[key]) <= 0.050001


def test_context_adjusted_probabilities_remain_normalized() -> None:
    context = {"context_type": "both_draw_ok", "home_incentive_label": "draw_enough", "away_incentive_label": "draw_enough"}
    adjusted, _markets, _diagnostics = apply_tournament_context_adjustment(_baseline_probs(), pd.DataFrame(), context)

    assert abs(sum(adjusted[key] for key in ["home_win", "draw", "away_win"]) - 1.0) < 1e-9


def test_both_need_win_reduces_draw_slightly() -> None:
    context = {"context_type": "both_need_win", "home_incentive_label": "needs_win", "away_incentive_label": "needs_win"}
    adjusted, _markets, _diagnostics = apply_tournament_context_adjustment(_baseline_probs(), pd.DataFrame(), context)

    assert adjusted["draw"] < _baseline_probs()["draw"]


def test_both_draw_ok_increases_draw_slightly() -> None:
    context = {"context_type": "both_draw_ok", "home_incentive_label": "draw_enough", "away_incentive_label": "draw_enough"}
    adjusted, _markets, _diagnostics = apply_tournament_context_adjustment(_baseline_probs(), pd.DataFrame(), context)

    assert adjusted["draw"] > _baseline_probs()["draw"]


def test_no_data_returns_unknown_context_safely() -> None:
    match = pd.Series({"match_id": "X", "home": "A", "away": "B", "competition": "FIFA World Cup", "group": "Group Z"})
    context = classify_match_context("A", "B", match, pd.DataFrame(), pd.DataFrame())

    assert context["context_type"] == "unknown"
    assert context["home_incentive_label"] == "unknown"
    assert context["away_incentive_label"] == "unknown"


def test_team_need_directly_classifies_draw_enough_and_needs_win() -> None:
    standings = build_group_standings_asof(_results(), _fixtures(), "H", "2026-06-26 23:59 UTC")
    remaining = remaining_group_fixtures_asof(_fixtures(), "H", "2026-06-26 23:59 UTC")

    spain = calculate_team_qualification_need("Spain", standings, remaining, "H3")
    uruguay = calculate_team_qualification_need("Uruguay", standings, remaining, "H3")

    assert spain["draw_enough"] is True
    assert uruguay["needs_win"] is True
