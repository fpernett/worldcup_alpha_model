from __future__ import annotations

import sys

import pandas as pd

from src import prediction_ledger


def test_prediction_ledger_appends_without_overwriting(tmp_path) -> None:
    path = tmp_path / "prediction_ledger.csv"
    row = {col: "" for col in prediction_ledger.PREDICTION_LEDGER_COLUMNS}
    row.update({"prediction_id": "p1", "match_id": "m1", "home_win_prob": 0.5})
    first = pd.DataFrame([row])
    prediction_ledger.append_prediction_snapshots(first, path=path)

    row2 = dict(row)
    row2["prediction_id"] = "p2"
    row2["home_win_prob"] = 0.6
    combined = prediction_ledger.append_prediction_snapshots(pd.DataFrame([row2]), path=path)

    assert len(combined) == 2
    assert list(combined["prediction_id"]) == ["p1", "p2"]


def test_results_import_creates_actual_result_and_market_flags(tmp_path) -> None:
    path = tmp_path / "results_ledger.csv"
    completed = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-11",
                "competition": "World Cup",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 2,
                "away_goals": 1,
            }
        ]
    )

    results = prediction_ledger.import_completed_results(completed, path=path)

    assert len(results) == 1
    assert results.iloc[0]["actual_result"] == "home_win"
    assert int(results.iloc[0]["over_2_5_actual"]) == 1
    assert int(results.iloc[0]["btts_actual"]) == 1


def test_snapshot_predictions_for_fixtures_appends(monkeypatch, tmp_path) -> None:
    path = tmp_path / "prediction_ledger.csv"
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-22",
                "time_utc": "18:00",
                "competition": "World Cup",
                "group": "A",
                "home": "Alpha",
                "away": "Beta",
                "venue": "Test",
            }
        ]
    )
    teams = pd.DataFrame(
        [
            {"team": "Alpha", "elo": 1700, "attack": 0.6, "defense": 0.6, "recent_form": 0.6},
            {"team": "Beta", "elo": 1600, "attack": 0.5, "defense": 0.5, "recent_form": 0.5},
        ]
    )

    def fake_run_match_model(*args, **kwargs):
        return {
            "hxg": 1.5,
            "axg": 0.9,
            "probs": {
                "home_win": 0.5,
                "draw": 0.25,
                "away_win": 0.25,
                "over_2_5": 0.45,
                "over_3_5": 0.20,
                "btts_yes": 0.48,
            },
            "score_matrix": pd.DataFrame(
                [
                    {"home_goals": 0, "away_goals": 0, "prob": 0.2},
                    {"home_goals": 1, "away_goals": 0, "prob": 0.3},
                    {"home_goals": 1, "away_goals": 1, "prob": 0.5},
                ]
            ),
        }

    monkeypatch.setattr(prediction_ledger, "run_match_model", fake_run_match_model)

    snapshots = prediction_ledger.snapshot_predictions_for_fixtures(
        fixtures,
        team_ratings_df=teams,
        venues_df=pd.DataFrame(),
        market_odds_df=pd.DataFrame(),
        path=path,
    )

    assert len(snapshots) == 1
    assert snapshots.iloc[0]["over_0_5_prob"] == 0.8
    assert bool(snapshots.iloc[0]["prediction_before_kickoff"]) is True
    assert path.exists()


def test_import_completed_results_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.import_completed_results as script

    completed = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-11",
                "competition": "World Cup",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 1,
                "away_goals": 0,
            }
        ]
    )
    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "RESULTS_LEDGER_PATH", tmp_path / "data" / "results_ledger.csv")
    monkeypatch.setattr(script, "load_completed_matches_for_backtest", lambda **kwargs: completed)
    monkeypatch.setattr(sys, "argv", ["import_completed_results.py", "--competition", "World Cup"])

    script.main()

    assert "results ledger rows: 1" in capsys.readouterr().out
