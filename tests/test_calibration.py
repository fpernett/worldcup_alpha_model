from __future__ import annotations

import pandas as pd

from src.calibration import (
    build_calibration_backtest_report,
    build_calibration_evaluation_dataset,
    calibration_metrics,
    market_type_semantics,
)


def _prediction(match_id: str = "m1", kickoff: str = "2026-06-11T18:00:00+00:00", home_prob: float = 0.60) -> dict:
    draw_prob = 0.25
    away_prob = 1.0 - home_prob - draw_prob
    return {
        "prediction_id": f"p_{match_id}",
        "snapshot_utc": "2026-06-10T12:00:00+00:00",
        "match_id": match_id,
        "kickoff_utc": kickoff,
        "competition": "FIFA World Cup",
        "group": "A",
        "home": "Mexico",
        "away": "Ecuador",
        "venue": "Estadio Azteca",
        "primary_model_mode": "baseline_manual",
        "model_version": "v1",
        "parameter_set_id": "baseline",
        "home_win_prob": home_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_prob,
        "home_xg": 1.5,
        "away_xg": 0.9,
        "over_0_5_prob": 0.9,
        "over_1_5_prob": 0.7,
        "over_2_5_prob": 0.45,
        "over_3_5_prob": 0.2,
        "btts_yes_prob": 0.4,
        "model_confidence": "Moderate",
        "behavior_home_prob": 0.48,
        "behavior_draw_prob": 0.28,
        "behavior_away_prob": 0.24,
        "source_status": "test",
        "market_snapshot_available": True,
        "prediction_before_kickoff": True,
        "notes": "",
    }


def _result(match_id: str = "m1", actual: str = "home_win") -> dict:
    score = {"home_win": (2, 0), "draw": (1, 1), "away_win": (0, 1)}[actual]
    return {
        "match_id": match_id,
        "date_utc": "2026-06-11",
        "competition": "FIFA World Cup",
        "home": "Mexico",
        "away": "Ecuador",
        "home_goals": score[0],
        "away_goals": score[1],
        "actual_result": actual,
        "total_goals": sum(score),
        "btts_actual": int(score[0] > 0 and score[1] > 0),
        "over_0_5_actual": 1,
        "over_1_5_actual": int(sum(score) > 1.5),
        "over_2_5_actual": int(sum(score) > 2.5),
        "over_3_5_actual": int(sum(score) > 3.5),
        "actual_advancing_team": "",
        "result_semantics": "90-minute regular time",
        "result_source": "test",
        "last_updated": "2026-06-12T00:00:00+00:00",
    }


def test_build_calibration_evaluation_dataset_joins_results_and_de_vigs_market() -> None:
    predictions = pd.DataFrame([_prediction()])
    results = pd.DataFrame([_result()])
    venues = pd.DataFrame(
        [
            {
                "venue": "Estadio Azteca",
                "country": "Mexico",
                "altitude_m": 2240,
                "temp_c": 20,
                "humidity_pct": 45,
                "wind_kmh": 8,
                "neutral_site": 0,
            }
        ]
    )
    odds = pd.DataFrame(
        [
            {"match_id": "m1", "market": "1X2", "selection": "Mexico", "odds": 2.00},
            {"match_id": "m1", "market": "1X2", "selection": "Draw", "odds": 3.50},
            {"match_id": "m1", "market": "1X2", "selection": "Ecuador", "odds": 4.00},
        ]
    )

    evaluation = build_calibration_evaluation_dataset(predictions, results, venues_df=venues, market_odds_df=odds)

    row = evaluation.iloc[0]
    assert row["actual_home_goals_90"] == 2
    assert row["actual_1x2_result"] == "home_win"
    assert row["evaluation_scope"] == "90-minute"
    assert row["venue_context"] == "home_host"
    assert row["altitude_category"] == "high_altitude"
    assert round(float(row["market_home_prob"]) + float(row["market_draw_prob"]) + float(row["market_away_prob"]), 8) == 1.0
    assert row["behavior_warning"]


def test_market_type_semantics_separates_advancement_from_1x2() -> None:
    assert market_type_semantics("1X2", "90-minute regular time") == "90-minute"
    assert market_type_semantics("to_advance", "penalties included") == "advancement-inclusive"
    assert market_type_semantics("winner", "extra time included") == "extra-time-inclusive"


def test_calibration_metrics_and_walk_forward_variants() -> None:
    predictions = pd.DataFrame(
        [
            _prediction("m1", "2026-06-11T18:00:00+00:00", 0.60),
            _prediction("m2", "2026-06-12T18:00:00+00:00", 0.45),
            _prediction("m3", "2026-06-13T18:00:00+00:00", 0.35),
            _prediction("m4", "2026-06-14T18:00:00+00:00", 0.50),
        ]
    )
    results = pd.DataFrame(
        [
            _result("m1", "home_win"),
            _result("m2", "draw"),
            _result("m3", "away_win"),
            _result("m4", "home_win"),
        ]
    )
    evaluation = build_calibration_evaluation_dataset(predictions, results, latest_snapshot_only=False)
    summary = calibration_metrics(evaluation)
    report = build_calibration_backtest_report(evaluation, min_train=2)

    assert int(summary.iloc[0]["n"]) == 4
    assert set(report["variant_metrics"]["model"]) >= {"baseline_model", "model_plus_shrinkage", "model_plus_market_prior", "full_calibrated_model"}
