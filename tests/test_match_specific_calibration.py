from __future__ import annotations

import sys

import pandas as pd

from src import match_specific_calibration, tournament_learning
from src.historical_data import HISTORICAL_MATCH_COLUMNS


def _historical() -> pd.DataFrame:
    row = {col: pd.NA for col in HISTORICAL_MATCH_COLUMNS}
    return pd.DataFrame(
        [
            {
                **row,
                "match_id": "hist1",
                "date_utc": "2026-06-01",
                "competition": "Friendly",
                "competition_type": "friendly",
                "team": "Alpha",
                "opponent": "Beta",
                "team_goals": 1,
                "opponent_goals": 0,
                "source": "test",
            },
            {
                **row,
                "match_id": "target",
                "date_utc": "2026-06-23",
                "competition": "FIFA World Cup",
                "competition_type": "world_cup",
                "team": "Alpha",
                "opponent": "Gamma",
                "team_goals": 0,
                "opponent_goals": 0,
                "source": "test",
            },
        ]
    )


def _learning() -> pd.DataFrame:
    results = pd.DataFrame(
        [
            {
                "match_id": "wc_old",
                "date_utc": "2026-06-21",
                "competition": "FIFA World Cup",
                "home": "Alpha",
                "away": "Delta",
                "home_goals": 3,
                "away_goals": 1,
                "actual_result": "home_win",
                "total_goals": 4,
                "btts_actual": 1,
                "result_source": "test",
            },
            {
                "match_id": "wc_future",
                "date_utc": "2026-06-24",
                "competition": "FIFA World Cup",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 1,
                "away_goals": 1,
                "actual_result": "draw",
                "total_goals": 2,
                "btts_actual": 1,
                "result_source": "test",
            },
        ]
    )
    return tournament_learning.update_tournament_learning_ledger(results, pd.DataFrame())


def test_match_specific_calibration_uses_earlier_completed_world_cup_matches() -> None:
    calibration, diagnostics = match_specific_calibration.build_match_specific_calibration_set(
        _historical(),
        _learning(),
        "Alpha",
        "Gamma",
        "2026-06-23T18:00:00+00:00",
        target_match_id="target",
        competition="World Cup",
    )

    assert "wc_old" in set(calibration["match_id"])
    assert "wc_future" not in set(calibration["match_id"])
    assert "target" not in set(calibration["match_id"])
    assert diagnostics["completed_world_cup_matches_used"] == 1
    assert diagnostics["lookahead_safe"] is True


def test_match_specific_calibration_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.run_match_specific_calibration as script

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "load_historical_matches", lambda **kwargs: _historical())
    monkeypatch.setattr(script, "load_tournament_learning_ledger", lambda: _learning())
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-25")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_match_specific_calibration.py",
            "--home",
            "Alpha",
            "--away",
            "Gamma",
            "--prediction-date",
            "2026-06-23",
            "--kickoff-utc",
            "2026-06-23T18:00:00+00:00",
            "--match-id",
            "target",
            "--save-report",
        ],
    )

    script.main()

    output = capsys.readouterr().out
    assert "lookahead safe: True" in output
    assert (tmp_path / "reports" / "match_specific_calibration_2026-06-25.md").exists()
