from __future__ import annotations

import pandas as pd

from src import data_sources
from src.fixture_diagnostics import audit_fixture_availability
from src.team_names import is_unresolved_team_slot


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "past",
                "date_utc": "2026-06-22",
                "time_utc": "10:00",
                "competition": "World Cup",
                "group": "A",
                "home": "A",
                "away": "B",
                "venue": "V1",
            },
            {
                "match_id": "today",
                "date_utc": "2026-06-23",
                "time_utc": "18:00",
                "competition": "World Cup",
                "group": "A",
                "home": "C",
                "away": "D",
                "venue": "V2",
            },
            {
                "match_id": "future",
                "date_utc": "2026-06-28",
                "time_utc": "18:00",
                "competition": "World Cup",
                "group": "B",
                "home": "E",
                "away": "F",
                "venue": "V3",
            },
        ]
    )


def test_next_7_days_includes_fixtures_beyond_today():
    audit, summary = audit_fixture_availability(
        _fixtures(),
        start_date="2026-06-23",
        end_date="2026-06-30",
        hide_past=True,
        now_utc="2026-06-23T12:00:00Z",
    )
    assert "future" in set(audit.loc[audit["visible_in_app"], "match_id"])
    assert summary["fixtures_visible"] == 2


def test_hide_past_only_hides_past_fixtures():
    audit, _ = audit_fixture_availability(
        _fixtures(),
        start_date="2026-06-22",
        end_date="2026-06-23",
        hide_past=True,
        now_utc="2026-06-23T12:00:00Z",
    )
    past = audit.loc[audit["match_id"] == "past"].iloc[0]
    today = audit.loc[audit["match_id"] == "today"].iloc[0]
    assert past["hidden_by_past_filter"] is True or bool(past["hidden_by_past_filter"])
    assert today["visible_in_app"] is True or bool(today["visible_in_app"])


def test_utc_window_boundaries_work():
    audit, _ = audit_fixture_availability(
        _fixtures(),
        start_date="2026-06-23T18:00:00Z",
        end_date="2026-06-23T18:00:00Z",
        hide_past=False,
        now_utc="2026-06-23T00:00:00Z",
    )
    assert audit.loc[audit["match_id"] == "today", "visible_in_app"].iloc[0]
    assert not audit.loc[audit["match_id"] == "future", "visible_in_app"].iloc[0]


def test_custom_date_range_excluded_reason_is_populated():
    audit, _ = audit_fixture_availability(
        _fixtures(),
        start_date="2026-06-24",
        end_date="2026-06-25",
        hide_past=False,
        now_utc="2026-06-23T00:00:00Z",
    )
    excluded = audit.loc[~audit["visible_in_app"]]
    assert not excluded.empty
    assert excluded["excluded_reason"].str.len().min() > 0


def test_unresolved_bracket_slots_are_hidden_with_specific_reason():
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "resolved",
                "date_utc": "2026-07-01",
                "time_utc": "16:00",
                "competition": "World Cup",
                "group": "Round of 32",
                "home": "England",
                "away": "DR Congo",
                "venue": "V1",
            },
            {
                "match_id": "placeholder",
                "date_utc": "2026-07-01",
                "time_utc": "20:00",
                "competition": "World Cup",
                "group": "Round of 32",
                "home": "winner of group K",
                "away": "third of group L",
                "venue": "V2",
            },
        ]
    )

    audit, summary = audit_fixture_availability(
        fixtures,
        start_date="2026-07-01",
        end_date="2026-07-01",
        hide_past=True,
        now_utc="2026-07-01T12:00:00Z",
    )

    placeholder = audit.loc[audit["match_id"] == "placeholder"].iloc[0]
    assert not placeholder["visible_in_app"]
    assert placeholder["hidden_by_unresolved_slot"]
    assert placeholder["excluded_reason"] == "unresolved_team_slot"
    assert summary["fixtures_visible"] == 1
    assert summary["fixtures_hidden_as_unresolved"] == 1
    assert "unresolved bracket slot" in summary["warning"]


def test_local_round_of_32_fixture_rows_are_resolved():
    fixtures = pd.read_csv("data/fixtures.csv")
    round_of_32 = fixtures.loc[fixtures["group"].astype(str) == "Round of 32"].copy()
    unresolved = round_of_32.loc[
        round_of_32["home"].map(is_unresolved_team_slot)
        | round_of_32["away"].map(is_unresolved_team_slot)
    ]
    labels = set(round_of_32["home"].astype(str) + " vs " + round_of_32["away"].astype(str))

    assert unresolved.empty
    assert "England vs DR Congo" in labels
    assert "Portugal vs Croatia" in labels
    assert "Colombia vs Ghana" in labels


def test_source_end_warning_when_fixture_file_has_no_later_games():
    audit, summary = audit_fixture_availability(
        _fixtures().loc[_fixtures()["date_utc"] <= "2026-06-23"],
        start_date="2026-06-24",
        end_date="2026-06-30",
        hide_past=True,
        now_utc="2026-06-25T12:00:00Z",
    )
    assert summary["fixtures_visible"] == 0
    assert "No fixtures after" in summary["warning"] or "loaded fixture source ends" in summary["warning"]


def test_get_upcoming_fixtures_uses_international_results_as_supplemental_source(tmp_path, monkeypatch):
    data_dir = tmp_path
    pd.DataFrame(
        [
            {
                "match_id": "today_only",
                "date_utc": "2026-06-25",
                "time_utc": "20:00",
                "competition": "FIFA World Cup",
                "group": "",
                "home": "Curacao",
                "away": "Ivory Coast",
                "venue": "Known Stadium",
                "city": "Known City",
                "country": "USA",
            }
        ]
    ).to_csv(data_dir / "fixtures.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-06-25",
                "home_team": "Curaçao",
                "away_team": "Ivory Coast",
                "home_score": pd.NA,
                "away_score": pd.NA,
                "tournament": "FIFA World Cup",
                "city": "Philadelphia",
                "country": "United States",
                "neutral": True,
            },
            {
                "date": "2026-06-26",
                "home_team": "Norway",
                "away_team": "France",
                "home_score": pd.NA,
                "away_score": pd.NA,
                "tournament": "FIFA World Cup",
                "city": "Foxborough",
                "country": "United States",
                "neutral": True,
            },
        ]
    ).to_csv(data_dir / "international_results.csv", index=False)

    class Config:
        football_configured = False

    monkeypatch.setattr(data_sources, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_sources, "get_config", lambda: Config())

    fixtures = data_sources.get_upcoming_fixtures(pd.Timestamp("2026-06-25").date(), pd.Timestamp("2026-06-27").date())

    assert "Norway" in set(fixtures["home"])
    assert len(fixtures.loc[fixtures["away"].astype(str) == "Ivory Coast"]) == 1
    assert fixtures.attrs.get("source_detail") == "data/fixtures.csv + data/international_results.csv"
    assert "date-only kickoff placeholders" in fixtures.attrs.get("warning", "")
