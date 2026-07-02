from __future__ import annotations

import pandas as pd

from src.config import SOURCE_FALLBACK
from src.historical_ingestion import (
    classify_competition_type,
    fetch_historical_matches_for_team,
    football_matches_to_historical_rows,
    merge_historical_matches,
    update_historical_matches,
)
from src.team_names import normalize_team_name


def sample_match(match_id: int = 100) -> dict:
    return {
        "id": match_id,
        "utcDate": "2026-06-01T19:00:00Z",
        "competition": {"name": "FIFA World Cup"},
        "area": {"name": "World"},
        "homeTeam": {"name": "USA", "shortName": "USA"},
        "awayTeam": {"name": "Korea Republic", "shortName": "Korea Republic"},
        "score": {"fullTime": {"home": 2, "away": 1}},
    }


def test_team_name_normalization() -> None:
    assert normalize_team_name("USA") == "United States"
    assert normalize_team_name("Korea Republic") == "South Korea"
    assert normalize_team_name("Czech Republic") == "Czechia"
    assert normalize_team_name("Bosnia-Herzegovina") == "Bosnia and Herzegovina"
    assert normalize_team_name("Congo DR") == "DR Congo"
    assert normalize_team_name("Cabo Verde") == "Cape Verde"
    assert normalize_team_name("Côte d'Ivoire") == "Ivory Coast"


def test_competition_type_classification() -> None:
    assert classify_competition_type("FIFA World Cup") == "world_cup"
    assert classify_competition_type("World Cup Qualification") == "world_cup_qualifier"
    assert classify_competition_type("UEFA European Championship") == "continental_tournament"
    assert classify_competition_type("UEFA Nations League") == "nations_league"
    assert classify_competition_type("Friendly") == "friendly"


def test_long_format_conversion_creates_two_rows_per_match() -> None:
    rows = football_matches_to_historical_rows([sample_match()], teams=["United States"])

    assert len(rows) == 2
    assert set(rows["team"]) == {"United States", "South Korea"}
    assert rows.loc[rows["team"] == "United States", "team_goals"].iloc[0] == 2
    assert rows.loc[rows["team"] == "South Korea", "team_goals"].iloc[0] == 1
    assert rows["competition_type"].eq("world_cup").all()


def test_merge_preserves_manual_rows() -> None:
    existing = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "date_utc": "2026-06-01",
                "team": "United States",
                "opponent": "South Korea",
                "team_goals": 3,
                "opponent_goals": 1,
                "competition": "Manual",
                "competition_type": "other",
                "source": "manual",
                "last_updated": "2026-06-01T00:00:00+00:00",
            }
        ]
    )
    new = pd.DataFrame(
        [
            {
                "match_id": "M1",
                "date_utc": "2026-06-01",
                "team": "United States",
                "opponent": "South Korea",
                "team_goals": 2,
                "opponent_goals": 1,
                "competition": "FIFA World Cup",
                "competition_type": "world_cup",
                "source": "football-data.org",
                "last_updated": "2026-06-02T00:00:00+00:00",
            }
        ]
    )

    merged = merge_historical_matches(existing, new)

    assert len(merged) == 1
    assert merged.iloc[0]["source"] == "manual"
    assert merged.iloc[0]["team_goals"] == 3


def test_deduplication_prefers_newer_non_manual_rows() -> None:
    old = pd.DataFrame(
        [
            {
                "match_id": "M2",
                "date_utc": "2026-06-02",
                "team": "South Korea",
                "opponent": "United States",
                "team_goals": 0,
                "opponent_goals": 0,
                "competition": "Friendly",
                "competition_type": "friendly",
                "source": "cache",
                "last_updated": "2026-06-01T00:00:00+00:00",
            }
        ]
    )
    newer = old.copy()
    newer.loc[0, "team_goals"] = 1
    newer.loc[0, "last_updated"] = "2026-06-03T00:00:00+00:00"

    merged = merge_historical_matches(old, newer)

    assert len(merged) == 1
    assert merged.iloc[0]["team_goals"] == 1


def test_missing_api_key_returns_safe_empty_dataframe(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.historical_ingestion.fetch_finished_matches",
        lambda start_date, end_date, force_refresh=False: ([], SOURCE_FALLBACK, "missing key"),
    )

    rows = fetch_historical_matches_for_team("England", "2026-01-01", "2026-01-31")

    assert rows.empty
    assert "team" in rows.columns
    assert rows.attrs.get("warning") == "missing key"


def test_update_script_path_does_not_crash_with_empty_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "src.historical_ingestion.fetch_finished_matches",
        lambda start_date, end_date, force_refresh=False: ([], SOURCE_FALLBACK, "missing key"),
    )
    output_path = tmp_path / "historical_matches.csv"

    summary = update_historical_matches(
        "2026-01-01",
        "2026-01-31",
        teams=["England"],
        rebuild_behavior=False,
        existing_path=output_path,
        output_path=output_path,
    )

    assert summary["status"] == "success"
    assert summary["historical_rows"] == 0
    assert output_path.exists()
