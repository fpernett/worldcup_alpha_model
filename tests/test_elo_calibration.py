from __future__ import annotations

import pandas as pd

from scripts.audit_elo_calibration import write_elo_calibration_report
from src.elo_calibration import current_elo_table, validate_rolling_elo


def _long_match(
    match_id: str,
    date_utc: str,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    competition_type: str = "world_cup",
) -> list[dict[str, object]]:
    common = {
        "match_id": match_id,
        "date_utc": date_utc,
        "competition": competition_type,
        "competition_type": competition_type,
        "is_neutral": 1,
        "source": "test",
        "last_updated": "2026-06-21",
    }
    return [
        {
            **common,
            "team": home,
            "opponent": away,
            "is_home": 1,
            "team_goals": home_goals,
            "opponent_goals": away_goals,
        },
        {
            **common,
            "team": away,
            "opponent": home,
            "is_home": 0,
            "team_goals": away_goals,
            "opponent_goals": home_goals,
        },
    ]


def _historical_rows() -> pd.DataFrame:
    return pd.DataFrame(
        _long_match("m1", "2024-01-01", "Alpha", "Beta", 3, 0)
        + _long_match("m2", "2024-02-01", "Gamma", "Delta", 2, 0)
        + _long_match("m3", "2024-03-01", "Alpha", "Gamma", 2, 1)
        + _long_match("m4", "2024-04-01", "Beta", "Delta", 1, 1)
    )


def _manual_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "Alpha", "elo": 1900, "attack": 0.9, "defense": 0.8, "recent_form": 0.9},
            {"team": "Beta", "elo": 1500, "attack": 0.5, "defense": 0.5, "recent_form": 0.5},
            {"team": "Gamma", "elo": 1700, "attack": 0.7, "defense": 0.7, "recent_form": 0.7},
            {"team": "Delta", "elo": 1300, "attack": 0.3, "defense": 0.3, "recent_form": 0.3},
        ]
    )


def test_current_elo_table_returns_latest_rating_per_team() -> None:
    current = current_elo_table(_historical_rows())

    assert set(current["team"]) == {"Alpha", "Beta", "Gamma", "Delta"}
    assert current["rolling_elo"].notna().all()
    assert current["matches_count"].sum() == 8


def test_validate_rolling_elo_returns_expected_sections() -> None:
    validation = validate_rolling_elo(_historical_rows(), _manual_ratings())

    assert set(validation) == {
        "top_elo_teams",
        "bottom_elo_teams",
        "model_team_elos",
        "elo_distribution_summary",
        "manual_vs_elo_correlation",
        "teams_with_large_manual_elo_disagreement",
        "warnings",
    }
    assert validation["elo_distribution_summary"]["team_count"] == 4
    assert len(validation["model_team_elos"]) == 4


def test_validate_rolling_elo_warns_on_empty_data() -> None:
    validation = validate_rolling_elo(pd.DataFrame(), _manual_ratings())

    assert "Rolling Elo table is empty" in " ".join(validation["warnings"])


def test_elo_calibration_report_writer_runs_without_crashing(tmp_path) -> None:
    validation = validate_rolling_elo(_historical_rows(), _manual_ratings())
    report_path = write_elo_calibration_report(validation, tmp_path / "elo.md")

    assert report_path.exists()
    assert "Elo Calibration Audit" in report_path.read_text(encoding="utf-8")
