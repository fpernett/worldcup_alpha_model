from __future__ import annotations

import math
import sys

import pandas as pd

from src import model_training
from src.historical_data import HISTORICAL_MATCH_COLUMNS


def _hist_row(date: str, competition_type: str, team: str = "Alpha", opponent: str = "Beta") -> dict[str, object]:
    row = {col: pd.NA for col in HISTORICAL_MATCH_COLUMNS}
    row.update(
        {
            "match_id": f"{date}_{team}_{opponent}",
            "date_utc": date,
            "competition": competition_type,
            "competition_type": competition_type,
            "team": team,
            "opponent": opponent,
            "is_home": True,
            "is_neutral": True,
            "team_goals": 2,
            "opponent_goals": 1,
            "source": "test",
        }
    )
    return row


def _historical() -> pd.DataFrame:
    rows = [
        _hist_row("2026-05-01", "friendly", "Alpha", "Beta"),
        _hist_row("2026-05-01", "friendly", "Beta", "Alpha"),
        _hist_row("2026-04-01", "world_cup_qualifier", "Alpha", "Beta"),
        _hist_row("2026-04-01", "world_cup_qualifier", "Beta", "Alpha"),
        _hist_row("2020-01-01", "friendly", "Alpha", "Beta"),
        _hist_row("2020-01-01", "friendly", "Beta", "Alpha"),
        _hist_row("2026-06-12", "world_cup", "Alpha", "Beta"),
        _hist_row("2026-06-12", "world_cup", "Beta", "Alpha"),
    ]
    return pd.DataFrame(rows, columns=HISTORICAL_MATCH_COLUMNS)


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "Alpha", "elo": 1700, "attack": 0.60, "defense": 0.58, "recent_form": 0.57},
            {"team": "Beta", "elo": 1600, "attack": 0.52, "defense": 0.50, "recent_form": 0.51},
        ]
    )


def _completed() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-11",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 2,
                "away_goals": 1,
                "competition": "World Cup",
                "group": "",
                "venue": "",
                "city": "",
                "country": "",
                "neutral_site": 1,
            }
        ]
    )


def test_recent_friendly_has_useful_lower_weight_than_qualifier() -> None:
    prediction_date = "2026-06-11"
    friendly = model_training.calculate_match_relevance_weight(_hist_row("2026-05-01", "friendly"), prediction_date)
    qualifier = model_training.calculate_match_relevance_weight(_hist_row("2026-04-01", "world_cup_qualifier"), prediction_date)

    assert friendly > 0.5
    assert qualifier > friendly


def test_old_friendly_receives_low_weight() -> None:
    weight = model_training.calculate_match_relevance_weight(_hist_row("2020-01-01", "friendly"), "2026-06-11")

    assert 0.10 <= weight <= 0.20


def test_world_cup_qualifier_receives_high_weight() -> None:
    weight = model_training.calculate_match_relevance_weight(_hist_row("2026-04-01", "world_cup_qualifier"), "2026-06-11")

    assert weight > 1.4


def test_future_matches_are_not_included() -> None:
    weight = model_training.calculate_match_relevance_weight(_hist_row("2026-06-12", "world_cup"), "2026-06-11")
    diagnostics = model_training.build_training_relevance_diagnostics(_historical(), "2026-06-11")

    assert weight == 0.0
    assert diagnostics["latest_match_used"] == "2026-05-01"


def test_weighted_match_count_is_calculated() -> None:
    diagnostics = model_training.build_training_relevance_diagnostics(_historical(), "2026-06-11")

    assert diagnostics["matches_used_total"] >= 3
    assert diagnostics["weighted_match_count"] > 0
    assert diagnostics["friendly_matches_used"] >= 2
    assert diagnostics["qualifier_matches_used"] >= 1


def test_candidate_parameter_generation_contains_baseline() -> None:
    params = model_training.generate_candidate_parameter_sets()

    assert "baseline_current" in set(params["parameter_set_id"])
    assert len(params) > 5


def test_promotion_rule_blocks_tiny_sample() -> None:
    metrics = pd.DataFrame(
        [
            {"parameter_set_id": "baseline_current", "n_matches": 10, "brier_1x2": 0.60, "log_loss_1x2": 1.0, "over_2_5_brier": 0.20},
            {"parameter_set_id": "candidate", "n_matches": 10, "brier_1x2": 0.55, "log_loss_1x2": 0.9, "over_2_5_brier": 0.20},
        ]
    )

    out = model_training.evaluate_candidate_promotion(metrics)
    candidate = out.loc[out["parameter_set_id"] == "candidate"].iloc[0]

    assert candidate["promotion_status"] == "not_promoted_sample_too_small"


def test_walk_forward_training_uses_relevance_weights() -> None:
    candidates = model_training.generate_candidate_parameter_sets().head(2)
    predictions, metrics = model_training.run_walk_forward_training(
        _historical(),
        _completed(),
        candidates,
        "2026-06-11",
        "2026-06-11",
        team_ratings_df=_ratings(),
    )

    assert set(predictions["parameter_set_id"]) == set(candidates["parameter_set_id"])
    assert "evaluation_relevance_weight" in predictions.columns
    assert predictions["evaluation_relevance_weight"].gt(0).all()
    assert predictions["weighted_match_count"].gt(0).all()
    assert predictions["latest_match_used"].max() < "2026-06-11"
    assert not metrics.empty


def test_probability_adjustment_keeps_probabilities_normalized() -> None:
    pred = {
        "home_win_prob": 0.5,
        "draw_prob": 0.25,
        "away_win_prob": 0.25,
        "actual_result": "home_win",
        "over_2_5_prob": 0.4,
        "btts_yes_prob": 0.5,
    }
    params = {"draw_inflation_factor": 1.2, "favorite_strength_scale": 1.15, "underdog_resistance_scale": 1.0}

    out = model_training.apply_candidate_probability_adjustments(pred, params)

    assert math.isclose(out["home_win_prob"] + out["draw_prob"] + out["away_win_prob"], 1.0)
    assert out["actual_result_prob"] == out["home_win_prob"]


def test_train_model_candidates_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.train_model_candidates as script

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "MODEL_PARAMETER_SETS_PATH", tmp_path / "data" / "model_parameter_sets.csv")
    monkeypatch.setattr(script, "load_completed_matches_for_backtest", lambda **kwargs: _completed())
    monkeypatch.setattr(script, "load_historical_matches", lambda **kwargs: _historical())
    monkeypatch.setattr(script, "read_csv_with_columns", lambda *args, **kwargs: _ratings())
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["train_model_candidates.py", "--save-report", "--competition", "World Cup"])

    script.main()

    output = capsys.readouterr().out
    assert "candidate parameter sets tested" in output
    assert (tmp_path / "reports" / "model_training_report_2026-06-21.md").exists()
