from __future__ import annotations

import pandas as pd

from src.historical_binding import (
    audit_historical_binding_for_match,
    audit_historical_file_status,
    canonicalize_team_for_data_sources,
    suggest_close_team_names,
)


def _aliases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"alias": "CDR", "canonical": "DR Congo"},
            {"alias": "COD", "canonical": "DR Congo"},
            {"alias": "USA", "canonical": "United States"},
        ]
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-01",
                "team": "Jordan",
                "opponent": "Algeria",
                "team_goals": 1,
                "opponent_goals": 2,
                "competition": "Friendly",
                "competition_type": "friendly",
                "source": "test",
            },
            {
                "match_id": "m1",
                "date_utc": "2026-06-01",
                "team": "algeria",
                "opponent": "jordan",
                "team_goals": 2,
                "opponent_goals": 1,
                "competition": "Friendly",
                "competition_type": "friendly",
                "source": "test",
            },
            {
                "match_id": "m2",
                "date_utc": "2026-06-25",
                "team": "Jordan",
                "opponent": "Brazil",
                "team_goals": 0,
                "opponent_goals": 3,
                "competition": "World Cup",
                "competition_type": "world_cup",
                "source": "test",
            },
        ]
    )


def test_lowercase_teams_canonicalize():
    assert canonicalize_team_for_data_sources("jordan", _aliases()) == "Jordan"
    assert canonicalize_team_for_data_sources("algeria", _aliases()) == "Algeria"


def test_cdr_alias_maps_to_dr_congo():
    assert canonicalize_team_for_data_sources("CDR", _aliases()) == "DR Congo"


def test_historical_rows_found_when_names_differ_by_case():
    audit = audit_historical_binding_for_match("jordan", "ALGERIA", _history(), aliases_df=_aliases())
    assert audit["home_historical_rows_total"] == 2
    assert audit["away_historical_rows_total"] == 1
    assert audit["data_binding_status"] == "ok"


def test_h2h_count_works():
    audit = audit_historical_binding_for_match("Jordan", "Algeria", _history(), aliases_df=_aliases())
    assert audit["h2h_rows_total"] == 2


def test_asof_filter_excludes_future_rows():
    audit = audit_historical_binding_for_match(
        "Jordan",
        "Algeria",
        _history(),
        aliases_df=_aliases(),
        as_of_date="2026-06-23",
    )
    assert audit["home_historical_rows_total"] == 2
    assert audit["home_historical_rows_before_asof"] == 1


def test_close_name_suggestions_work():
    suggestions = suggest_close_team_names("Algiria", ["Algeria", "Argentina", "Brazil"])
    assert suggestions[0] == "Algeria"


def test_file_status_reports_rows_loaded():
    status = audit_historical_file_status(_history())
    assert status["rows_loaded"] == 3
    assert status["unique_teams"] == 2


def test_report_does_not_mark_missing_when_rows_exist():
    audit = audit_historical_binding_for_match("Jordan", "Algeria", _history(), aliases_df=_aliases())
    assert "No rows found" not in audit["warning"]
