from __future__ import annotations

import math
import sys

import pandas as pd

from src import backtest


def _long_historical_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "M1",
                "date_utc": "2026-06-18",
                "competition": "FIFA World Cup",
                "competition_type": "world_cup",
                "team": "Alpha",
                "opponent": "Beta",
                "is_home": 1,
                "is_neutral": 1,
                "venue": "Test Stadium",
                "city": "Test City",
                "country": "USA",
                "team_goals": 2,
                "opponent_goals": 1,
            },
            {
                "match_id": "M1",
                "date_utc": "2026-06-18",
                "competition": "FIFA World Cup",
                "competition_type": "world_cup",
                "team": "Beta",
                "opponent": "Alpha",
                "is_home": 0,
                "is_neutral": 1,
                "venue": "Test Stadium",
                "city": "Test City",
                "country": "USA",
                "team_goals": 1,
                "opponent_goals": 2,
            },
        ]
    )


def test_completed_match_loader_deduplicates_long_format(monkeypatch) -> None:
    monkeypatch.setattr(backtest, "_completed_matches_from_fixtures", lambda: pd.DataFrame(columns=backtest.COMPLETED_MATCH_COLUMNS))
    monkeypatch.setattr(backtest, "load_historical_matches", lambda use_cache=False: _long_historical_rows())

    completed = backtest.load_completed_matches_for_backtest()

    assert len(completed) == 1
    assert completed.iloc[0]["home"] == "Alpha"
    assert completed.iloc[0]["away"] == "Beta"
    assert completed.iloc[0]["home_goals"] == 2
    assert completed.iloc[0]["away_goals"] == 1


def test_actual_result_classification() -> None:
    assert backtest.classify_actual_result(2, 1) == "home_win"
    assert backtest.classify_actual_result(1, 1) == "draw"
    assert backtest.classify_actual_result(0, 3) == "away_win"


def test_brier_score_calculation() -> None:
    predictions = pd.DataFrame(
        [
            {
                "model_mode": "baseline_manual",
                "home_win_prob": 0.7,
                "draw_prob": 0.2,
                "away_win_prob": 0.1,
                "actual_result": "home_win",
                "actual_result_prob": 0.7,
                "most_likely_result": "home_win",
                "over_2_5_prob": 0.4,
                "btts_yes_prob": 0.5,
                "home_xg": 1.5,
                "away_xg": 1.0,
                "home_goals": 2,
                "away_goals": 1,
            }
        ]
    )

    metrics = backtest.calculate_backtest_metrics(predictions)

    expected_brier = (0.7 - 1) ** 2 + 0.2**2 + 0.1**2
    assert math.isclose(metrics.iloc[0]["brier_score_1x2"], expected_brier)
    assert metrics.iloc[0]["most_likely_result_accuracy"] == 1.0


def test_log_loss_clips_probabilities_safely() -> None:
    predictions = pd.DataFrame(
        [
            {
                "model_mode": "baseline_manual",
                "home_win_prob": 0.0,
                "draw_prob": 0.5,
                "away_win_prob": 0.5,
                "actual_result": "home_win",
                "actual_result_prob": 0.0,
                "most_likely_result": "draw",
                "over_2_5_prob": 0.0,
                "btts_yes_prob": 0.0,
                "home_xg": 0.5,
                "away_xg": 0.5,
                "home_goals": 1,
                "away_goals": 0,
            }
        ]
    )

    metrics = backtest.calculate_backtest_metrics(predictions)

    assert math.isfinite(metrics.iloc[0]["log_loss_1x2"])
    assert metrics.iloc[0]["log_loss_1x2"] > 6.0


def test_calibration_bucket_uses_outcome_probabilities() -> None:
    predictions = pd.DataFrame(
        [
            {
                "model_mode": "baseline_manual",
                "home_win_prob": 0.65,
                "draw_prob": 0.25,
                "away_win_prob": 0.10,
                "actual_result": "home_win",
            }
        ]
    )

    calibration = backtest.build_calibration_table(predictions, "baseline_manual")

    assert calibration["n"].sum() == 3
    assert set(calibration["bucket"].astype(str)).issuperset({"0-20%", "20-40%", "60-80%"})


def test_baseline_and_behavior_modes_return_predictions(monkeypatch) -> None:
    teams = pd.DataFrame(
        [
            {"team": "Alpha", "elo": 1800, "attack": 0.70, "defense": 0.68, "recent_form": 0.65},
            {"team": "Beta", "elo": 1700, "attack": 0.55, "defense": 0.55, "recent_form": 0.55},
        ]
    )

    def fake_run_match_model(match_row, teams_df, venues_df, market_odds_df, cfg):
        home_attack = float(teams_df.loc[teams_df["team"] == match_row["home"], "attack"].iloc[0])
        home_prob = min(0.85, max(0.15, home_attack))
        draw_prob = 0.20
        away_prob = 1 - home_prob - draw_prob
        return {
            "hxg": 1.6,
            "axg": 0.9,
            "probs": {
                "home_win": home_prob,
                "draw": draw_prob,
                "away_win": away_prob,
                "over_2_5": 0.48,
                "under_2_5": 0.52,
                "btts_yes": 0.44,
                "btts_no": 0.56,
            },
        }

    monkeypatch.setattr(backtest, "run_match_model", fake_run_match_model)
    match = {
        "match_id": "M1",
        "date_utc": "2026-06-18",
        "home": "Alpha",
        "away": "Beta",
        "home_goals": 2,
        "away_goals": 1,
    }

    baseline = backtest.run_model_for_backtest_match(match, "baseline_manual", teams, pd.DataFrame(), pd.DataFrame())
    behavior = backtest.run_model_for_backtest_match(match, "behavior_adjusted", teams, pd.DataFrame(), pd.DataFrame())

    assert baseline["actual_result"] == "home_win"
    assert behavior["actual_result_prob"] > 0
    assert baseline["model_mode"] == "baseline_manual"
    assert behavior["model_mode"] == "behavior_adjusted"


def test_compare_predictions_marks_behavior_improvement() -> None:
    predictions = pd.DataFrame(
        [
            {
                "model_mode": "baseline_manual",
                "match_id": "M1",
                "date_utc": "2026-06-18",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 2,
                "away_goals": 1,
                "actual_result": "home_win",
                "home_win_prob": 0.45,
                "draw_prob": 0.30,
                "away_win_prob": 0.25,
                "actual_result_prob": 0.45,
                "log_loss_1x2": 0.8,
            },
            {
                "model_mode": "behavior_adjusted",
                "match_id": "M1",
                "date_utc": "2026-06-18",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 2,
                "away_goals": 1,
                "actual_result": "home_win",
                "home_win_prob": 0.55,
                "draw_prob": 0.25,
                "away_win_prob": 0.20,
                "actual_result_prob": 0.55,
                "log_loss_1x2": 0.6,
            },
        ]
    )

    comparison = backtest.compare_backtest_predictions(predictions)

    assert bool(comparison.iloc[0]["improved_by_behavior"]) is True
    assert comparison.iloc[0]["actual_prob_delta"] > 0


def test_run_backtest_script_does_not_crash(monkeypatch, capsys) -> None:
    import scripts.run_backtest as script

    fake_result = {
        "matches": pd.DataFrame([{"match_id": "M1"}]),
        "predictions": pd.DataFrame(),
        "metrics": pd.DataFrame(
            [
                {
                    "model_mode": "baseline_manual",
                    "n_matches": 1,
                    "brier_score_1x2": 0.5,
                }
            ]
        ),
        "comparison": pd.DataFrame(),
        "calibration": pd.DataFrame(),
        "warning": "approximate",
    }
    monkeypatch.setattr(script, "run_backtest", lambda **kwargs: fake_result)
    monkeypatch.setattr(sys, "argv", ["run_backtest.py", "--start-date", "2026-06-18", "--end-date", "2026-06-18"])

    script.main()

    output = capsys.readouterr().out
    assert "completed matches: 1" in output
    assert "Summary metrics" in output
