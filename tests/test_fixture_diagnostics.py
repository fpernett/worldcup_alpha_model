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
    assert "future fixtures pending prior match results" in summary["warning"]


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


def test_get_upcoming_fixtures_uses_cached_polymarket_events_for_unresolved_slots(tmp_path, monkeypatch):
    data_dir = tmp_path
    pd.DataFrame(
        [
            {
                "match_id": "wc2026_90",
                "date_utc": "2026-07-04",
                "time_utc": "17:00",
                "competition": "FIFA World Cup",
                "group": "Round of 16",
                "home": "Winner Match 73",
                "away": "Winner Match 75",
                "venue": "NRG Stadium",
                "city": "Houston",
                "country": "USA",
            }
        ]
    ).to_csv(data_dir / "fixtures.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "event_1",
                "event_slug": "fifwc-can-mar-2026-07-04",
                "event_title": "Canada vs. Morocco",
                "event_category": "FIFA World Cup",
                "event_start_date": "2026-07-03T18:00:00Z",
                "event_end_date": "2026-07-04T17:00:00Z",
                "event_active": 1,
                "event_closed": 0,
                "raw_event_json": "",
                "source": "API",
                "last_updated": "2026-07-04T07:00:00+00:00",
            }
        ]
    ).to_csv(data_dir / "polymarket_events_cache.csv", index=False)

    class Config:
        football_configured = False

    monkeypatch.setattr(data_sources, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_sources, "get_config", lambda: Config())

    fixtures = data_sources.get_upcoming_fixtures(pd.Timestamp("2026-07-04").date(), pd.Timestamp("2026-07-04").date())
    row = fixtures.iloc[0]

    assert len(fixtures) == 1
    assert row["match_id"] == "wc2026_90"
    assert row["home"] == "Canada"
    assert row["away"] == "Morocco"
    assert row["venue"] == "NRG Stadium"
    assert "data/polymarket_events_cache.csv" in fixtures.attrs.get("source_detail", "")
    assert "read-only fixture discovery" in fixtures.attrs.get("warning", "")


def test_get_upcoming_fixtures_uses_live_polymarket_events_when_local_window_is_unresolved(tmp_path, monkeypatch):
    data_dir = tmp_path
    pd.DataFrame(
        [
            {
                "match_id": "wc2026_90",
                "date_utc": "2026-07-04",
                "time_utc": "17:00",
                "competition": "FIFA World Cup",
                "group": "Round of 16",
                "home": "Winner Match 73",
                "away": "Winner Match 75",
                "venue": "NRG Stadium",
                "city": "Houston",
                "country": "USA",
            },
            {
                "match_id": "wc2026_89",
                "date_utc": "2026-07-04",
                "time_utc": "21:00",
                "competition": "FIFA World Cup",
                "group": "Round of 16",
                "home": "Winner Match 74",
                "away": "Winner Match 77",
                "venue": "Lincoln Financial Field",
                "city": "Philadelphia",
                "country": "USA",
            },
        ]
    ).to_csv(data_dir / "fixtures.csv", index=False)
    live_events = pd.DataFrame(
        [
            {
                "event_id": "event_corners",
                "event_slug": "fifwc-can-mar-2026-07-04-total-corners",
                "event_title": "Canada vs. Morocco - Total Corners",
                "event_category": "FIFA World Cup",
                "event_start_date": "2026-07-03T18:00:00Z",
                "event_end_date": "2026-07-04T17:00:00Z",
                "event_active": 1,
                "event_closed": 0,
                "raw_event_json": "",
                "source": "API",
                "last_updated": "2026-07-04T07:00:00+00:00",
            },
            {
                "event_id": "event_1",
                "event_slug": "fifwc-can-mar-2026-07-04",
                "event_title": "Canada vs. Morocco",
                "event_category": "FIFA World Cup",
                "event_start_date": "2026-07-03T18:00:00Z",
                "event_end_date": "2026-07-04T17:00:00Z",
                "event_active": 1,
                "event_closed": 0,
                "raw_event_json": "",
                "source": "API",
                "last_updated": "2026-07-04T07:00:00+00:00",
            },
            {
                "event_id": "event_2",
                "event_slug": "fifwc-par-fra-2026-07-04",
                "event_title": "Paraguay vs. France",
                "event_category": "FIFA World Cup",
                "event_start_date": "2026-07-03T18:00:00Z",
                "event_end_date": "2026-07-04T21:00:00Z",
                "event_active": 1,
                "event_closed": 0,
                "raw_event_json": "",
                "source": "API",
                "last_updated": "2026-07-04T07:00:00+00:00",
            },
        ]
    )

    class Config:
        football_configured = False

    monkeypatch.setattr(data_sources, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_sources, "get_config", lambda: Config())
    monkeypatch.setattr(data_sources, "_live_polymarket_event_rows", lambda: live_events)

    fixtures = data_sources.get_upcoming_fixtures(pd.Timestamp("2026-07-04").date(), pd.Timestamp("2026-07-04").date())
    labels = set(fixtures["home"].astype(str) + " vs " + fixtures["away"].astype(str))

    assert labels == {"Canada vs Morocco", "Paraguay vs France"}
    assert set(fixtures["match_id"]) == {"wc2026_90", "wc2026_89"}
    assert "Polymarket Gamma sports events" in fixtures.attrs.get("source_detail", "")
    assert "live Polymarket FIFA World Cup event metadata" in fixtures.attrs.get("warning", "")


def test_get_upcoming_fixtures_falls_through_when_cache_misses_requested_window(tmp_path, monkeypatch):
    data_dir = tmp_path
    pd.DataFrame(
        [
            {
                "match_id": "local_next",
                "date_utc": "2026-07-04",
                "time_utc": "17:00",
                "competition": "FIFA World Cup",
                "group": "Round of 16",
                "home": "Canada",
                "away": "Morocco",
                "venue": "NRG Stadium",
                "city": "Houston",
                "country": "USA",
            }
        ]
    ).to_csv(data_dir / "fixtures.csv", index=False)
    stale_cache = pd.DataFrame(
        [
            {
                "match_id": "old",
                "date_utc": "2026-06-20",
                "time_utc": "17:00",
                "competition": "FIFA World Cup",
                "group": "Group",
                "home": "Old A",
                "away": "Old B",
                "venue": "Old Venue",
                "city": "Old City",
                "country": "USA",
            }
        ]
    )

    class Config:
        football_configured = True

    monkeypatch.setattr(data_sources, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_sources, "get_config", lambda: Config())
    monkeypatch.setattr(data_sources, "read_dataframe_cache", lambda *_args, **_kwargs: stale_cache)
    monkeypatch.setattr(data_sources, "cache_last_updated", lambda *_args, **_kwargs: "2026-07-04T00:00:00+00:00")
    monkeypatch.setattr(data_sources, "_fetch_football_data_fixtures", lambda *_args, **_kwargs: (pd.DataFrame(), "API failed"))

    fixtures = data_sources.get_upcoming_fixtures(pd.Timestamp("2026-07-04").date(), pd.Timestamp("2026-07-04").date())

    assert fixtures["match_id"].tolist() == ["local_next"]
    assert fixtures.attrs.get("source_detail") == "data/fixtures.csv"
    assert "Cached fixtures did not contain the requested date window" in fixtures.attrs.get("warning", "")


def test_get_upcoming_fixtures_supplements_partial_api_schedule_from_local_csv(tmp_path, monkeypatch):
    data_dir = tmp_path
    pd.DataFrame(
        [
            {
                "match_id": "wc2026_101",
                "date_utc": "2026-07-14",
                "time_utc": "19:00",
                "competition": "FIFA World Cup",
                "group": "Semifinal",
                "home": "France",
                "away": "Spain",
                "venue": "Dallas Stadium",
                "city": "Arlington",
                "country": "USA",
            },
            {
                "match_id": "wc2026_102",
                "date_utc": "2026-07-15",
                "time_utc": "19:00",
                "competition": "FIFA World Cup",
                "group": "Semifinal",
                "home": "England",
                "away": "Argentina",
                "venue": "Mercedes-Benz Stadium",
                "city": "Atlanta",
                "country": "USA",
            },
        ]
    ).to_csv(data_dir / "fixtures.csv", index=False)

    class Config:
        football_configured = True

    api_fixture = pd.DataFrame(
        [
            {
                "match_id": "provider_only",
                "date_utc": "2026-07-13",
                "time_utc": "19:00",
                "competition": "FIFA World Cup",
                "group": "Quarterfinal",
                "home": "Other A",
                "away": "Other B",
            }
        ]
    )

    monkeypatch.setattr(data_sources, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_sources, "get_config", lambda: Config())
    monkeypatch.setattr(data_sources, "read_dataframe_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_sources, "write_dataframe_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        data_sources,
        "_fetch_football_data_fixtures",
        lambda *_args, **_kwargs: (api_fixture, None),
    )

    fixtures = data_sources.get_upcoming_fixtures(
        pd.Timestamp("2026-07-13").date(),
        pd.Timestamp("2026-07-15").date(),
    )
    labels = set(fixtures["home"].astype(str) + " vs " + fixtures["away"].astype(str))

    assert {"France vs Spain", "England vs Argentina"}.issubset(labels)
    assert "Other A vs Other B" in labels
    assert fixtures.attrs["source_label"] == "API + local CSV"
    assert "Primary fixture source was incomplete" in fixtures.attrs["warning"]


def test_local_knockout_slots_use_cached_semifinal_team_names():
    fixtures = data_sources._load_local_fixture_pool(
        pd.Timestamp("2026-07-14").date(),
        pd.Timestamp("2026-07-15").date(),
    )
    semifinal = fixtures.loc[fixtures["match_id"].isin(["wc2026_101", "wc2026_102"])]
    labels = set(semifinal["home"].astype(str) + " vs " + semifinal["away"].astype(str))

    assert labels == {"France vs Spain", "England vs Argentina"}


def test_final_and_third_place_fixtures_have_concrete_teams():
    fixtures = pd.read_csv("data/fixtures.csv")
    last_two = fixtures.loc[
        fixtures["match_id"].astype(str).isin(["wc2026_103", "wc2026_104"])
    ].set_index("match_id")

    assert last_two.loc["wc2026_103", "group"] == "Third-place match"
    assert last_two.loc["wc2026_103", "home"] == "England"
    assert last_two.loc["wc2026_103", "away"] == "France"
    assert last_two.loc["wc2026_104", "group"] == "Final"
    assert last_two.loc["wc2026_104", "home"] == "Argentina"
    assert last_two.loc["wc2026_104", "away"] == "Spain"
    assert not last_two["home"].map(is_unresolved_team_slot).any()
    assert not last_two["away"].map(is_unresolved_team_slot).any()


def test_local_resolved_fixture_replaces_cached_placeholder_with_same_match_id(tmp_path, monkeypatch):
    data_dir = tmp_path
    pd.DataFrame(
        [
            {
                "match_id": "wc2026_104",
                "date_utc": "2026-07-19",
                "time_utc": "19:00",
                "competition": "FIFA World Cup",
                "group": "Final",
                "home": "Argentina",
                "away": "Spain",
                "venue": "MetLife Stadium",
                "city": "East Rutherford",
                "country": "USA",
            }
        ]
    ).to_csv(data_dir / "fixtures.csv", index=False)
    cached_placeholder = pd.DataFrame(
        [
            {
                "match_id": "wc2026_104",
                "date_utc": "2026-07-19",
                "time_utc": "19:00",
                "competition": "FIFA World Cup",
                "group": "Final",
                "home": "Winner Match 101",
                "away": "Winner Match 102",
                "venue": "MetLife Stadium",
                "city": "East Rutherford",
                "country": "USA",
            }
        ]
    )

    class Config:
        football_configured = True

    monkeypatch.setattr(data_sources, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_sources, "get_config", lambda: Config())
    monkeypatch.setattr(data_sources, "read_dataframe_cache", lambda *_args, **_kwargs: cached_placeholder)
    monkeypatch.setattr(data_sources, "cache_last_updated", lambda *_args, **_kwargs: "2026-07-16T00:00:00Z")

    fixtures = data_sources.get_upcoming_fixtures(
        pd.Timestamp("2026-07-19").date(),
        pd.Timestamp("2026-07-19").date(),
    )

    assert len(fixtures) == 1
    assert fixtures.iloc[0]["home"] == "Argentina"
    assert fixtures.iloc[0]["away"] == "Spain"
    assert fixtures.attrs["source_label"] == "cache + local CSV"
