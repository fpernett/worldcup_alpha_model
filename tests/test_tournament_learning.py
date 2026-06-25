from __future__ import annotations

import sys

import pandas as pd

from src import tournament_learning


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-21",
                "competition": "FIFA World Cup",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 2,
                "away_goals": 1,
                "actual_result": "home_win",
                "total_goals": 3,
                "btts_actual": 1,
                "result_source": "test",
                "last_updated": "2026-06-22T00:00:00+00:00",
            },
            {
                "match_id": "m2",
                "date_utc": "2026-06-24",
                "competition": "FIFA World Cup",
                "home": "Gamma",
                "away": "Delta",
                "home_goals": 0,
                "away_goals": 0,
                "actual_result": "draw",
                "total_goals": 0,
                "btts_actual": 0,
                "result_source": "test",
                "last_updated": "2026-06-24T23:00:00+00:00",
            },
        ]
    )


def test_completed_match_added_to_tournament_learning_ledger() -> None:
    ledger = tournament_learning.update_tournament_learning_ledger(_results().head(1), pd.DataFrame())

    assert len(ledger) == 1
    assert ledger.iloc[0]["match_id"] == "m1"
    assert bool(ledger.iloc[0]["used_for_future_calibration"]) is True
    assert ledger.iloc[0]["eligible_after_utc"] == "2026-06-21T02:00:00+00:00"


def test_duplicate_completed_match_is_not_added_twice() -> None:
    first = tournament_learning.update_tournament_learning_ledger(_results().head(1), pd.DataFrame())
    second = tournament_learning.update_tournament_learning_ledger(_results().head(1), first)

    assert len(second) == 1
    assert second["match_id"].tolist() == ["m1"]


def test_completed_match_before_target_kickoff_is_included() -> None:
    ledger = tournament_learning.update_tournament_learning_ledger(_results(), pd.DataFrame())
    used, diagnostics = tournament_learning.filter_tournament_learning_asof(ledger, "2026-06-23T18:00:00+00:00")

    assert set(used["match_id"]) == {"m1"}
    assert diagnostics["completed_world_cup_matches_used"] == 1
    assert diagnostics["lookahead_safe"] is True


def test_completed_match_after_target_kickoff_is_excluded() -> None:
    ledger = tournament_learning.update_tournament_learning_ledger(_results(), pd.DataFrame())
    used, diagnostics = tournament_learning.filter_tournament_learning_asof(ledger, "2026-06-21T01:00:00+00:00")

    assert used.empty
    assert diagnostics["completed_world_cup_matches_available"] == 2


def test_target_match_itself_is_excluded() -> None:
    ledger = tournament_learning.update_tournament_learning_ledger(_results().head(1), pd.DataFrame())
    used, _ = tournament_learning.filter_tournament_learning_asof(
        ledger,
        "2026-06-23T18:00:00+00:00",
        target_match_id="m1",
    )

    assert used.empty


def test_update_after_completed_games_script_runs_without_promoting(monkeypatch, tmp_path, capsys) -> None:
    import scripts.update_after_completed_games as script

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "RESULTS_LEDGER_PATH", tmp_path / "data" / "results_ledger.csv")
    monkeypatch.setattr(script, "TOURNAMENT_LEARNING_LEDGER_PATH", tmp_path / "data" / "tournament_learning_ledger.csv")
    monkeypatch.setattr(script, "load_completed_matches_for_backtest", lambda **kwargs: _results().head(1))
    monkeypatch.setattr(script, "load_prediction_ledger", lambda: pd.DataFrame())
    monkeypatch.setattr(script, "load_historical_matches", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(script, "read_csv_with_columns", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(script, "run_walk_forward_training", lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-25")
    monkeypatch.setattr(sys, "argv", ["update_after_completed_games.py", "--competition", "World Cup"])

    script.main()

    output = capsys.readouterr().out
    assert "new learning rows added: 1" in output
    assert "promotion" not in output.lower()
    assert (tmp_path / "data" / "tournament_learning_ledger.csv").exists()
