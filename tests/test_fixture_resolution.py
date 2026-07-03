from __future__ import annotations

import pandas as pd

from src.fixture_resolution import resolve_fixture_placeholders


def _fixture(match_id: str, home: str, away: str, group: str = "Round of 16") -> dict:
    return {
        "match_id": match_id,
        "date_utc": "2026-07-04",
        "time_utc": "17:00",
        "competition": "FIFA World Cup",
        "group": group,
        "home": home,
        "away": away,
        "venue": "Test Stadium",
        "city": "Test City",
        "country": "USA",
    }


def _result(match_id: str, home: str, away: str, home_goals: int, away_goals: int, advancing: str = "") -> dict:
    return {
        "match_id": match_id,
        "date_utc": "2026-07-01",
        "competition": "FIFA World Cup",
        "home": home,
        "away": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "actual_result": "draw" if home_goals == away_goals else ("home_win" if home_goals > away_goals else "away_win"),
        "actual_advancing_team": advancing,
        "result_semantics": "90-minute regular time",
        "result_source": "test",
        "last_updated": "2026-07-01T23:00:00+00:00",
    }


def test_resolves_knockout_winner_and_loser_from_regular_time_result() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture("wc2026_73", "Alpha", "Beta", group="Round of 32"),
            _fixture("wc2026_90", "Winner Match 73", "Loser Match 73"),
        ]
    )
    results = pd.DataFrame([_result("wc2026_73", "Alpha", "Beta", 2, 0)])

    resolved = resolve_fixture_placeholders(fixtures, results)
    row = resolved.loc[resolved["match_id"] == "wc2026_90"].iloc[0]

    assert row["home"] == "Alpha"
    assert row["away"] == "Beta"
    assert row["fixture_resolution_status"] == "Resolved automatically"


def test_resolves_penalty_advancement_when_regular_time_is_drawn() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture("wc2026_79", "Mexico", "Ecuador", group="Round of 32"),
            _fixture("wc2026_92", "Winner Match 79", "Loser Match 79"),
        ]
    )
    results = pd.DataFrame([_result("wc2026_79", "Mexico", "Ecuador", 1, 1, advancing="Ecuador")])

    resolved = resolve_fixture_placeholders(fixtures, results)
    row = resolved.loc[resolved["match_id"] == "wc2026_92"].iloc[0]

    assert row["home"] == "Ecuador"
    assert row["away"] == "Mexico"
    assert row["fixture_resolution_status"] == "Resolved automatically"


def test_regular_time_draw_without_advancing_team_requires_manual_mapping() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture("wc2026_79", "Mexico", "Ecuador", group="Round of 32"),
            _fixture("wc2026_92", "Winner Match 79", "Winner Match 80"),
            _fixture("wc2026_80", "England", "DR Congo", group="Round of 32"),
        ]
    )
    results = pd.DataFrame(
        [
            _result("wc2026_79", "Mexico", "Ecuador", 1, 1),
            _result("wc2026_80", "England", "DR Congo", 3, 0),
        ]
    )

    resolved = resolve_fixture_placeholders(fixtures, results)
    row = resolved.loc[resolved["match_id"] == "wc2026_92"].iloc[0]

    assert row["home"] == "Winner Match 79"
    assert row["away"] == "England"
    assert row["home_resolution_status"] == "Requires manual mapping"
    assert row["fixture_resolution_status"] == "Requires manual mapping"


def test_unplayed_prior_match_stays_pending_prior_result() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture("wc2026_74", "Germany", "Paraguay", group="Round of 32"),
            _fixture("wc2026_89", "Winner Match 74", "Winner Match 77"),
        ]
    )

    resolved = resolve_fixture_placeholders(fixtures, pd.DataFrame())
    row = resolved.loc[resolved["match_id"] == "wc2026_89"].iloc[0]

    assert row["home"] == "Winner Match 74"
    assert row["fixture_resolution_status"] == "Pending prior result"


def test_resolves_group_winner_and_runner_up_from_completed_group_table() -> None:
    fixtures = pd.DataFrame(
        [
            _fixture("g1", "Alpha", "Beta", group="Group A"),
            _fixture("g2", "Gamma", "Delta", group="Group A"),
            _fixture("g3", "Alpha", "Gamma", group="Group A"),
            _fixture("g4", "Beta", "Delta", group="Group A"),
            _fixture("r16", "Winner Group A", "Runner-up Group A"),
        ]
    )
    results = pd.DataFrame(
        [
            _result("g1", "Alpha", "Beta", 2, 0),
            _result("g2", "Gamma", "Delta", 1, 0),
            _result("g3", "Alpha", "Gamma", 1, 1),
            _result("g4", "Beta", "Delta", 3, 0),
        ]
    )

    resolved = resolve_fixture_placeholders(fixtures, results)
    row = resolved.loc[resolved["match_id"] == "r16"].iloc[0]

    assert row["home"] == "Alpha"
    assert row["away"] == "Gamma"
