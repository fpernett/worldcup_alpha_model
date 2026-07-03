from __future__ import annotations

import math

import pandas as pd

from src.evaluation import (
    build_formal_evaluation_dataset,
    build_prediction_source_dataset,
    build_result_prediction_join_audit,
    calibration_by_bin,
    calibration_summary,
    model_vs_market_benchmark,
    recent_match_postmortem,
    walk_forward_calibration_comparison,
)


def _prediction(
    match_id: str = "m1",
    snapshot: str = "2026-06-10T12:00:00+00:00",
    kickoff: str = "2026-06-11T18:00:00+00:00",
    home: str = "Alpha",
    away: str = "Beta",
    home_prob: float = 0.60,
    draw_prob: float = 0.25,
    away_prob: float = 0.15,
    market: tuple[float, float, float] | None = None,
) -> dict:
    row = {
        "prediction_id": f"p_{match_id}_{snapshot[-14:-6]}",
        "snapshot_utc": snapshot,
        "match_id": match_id,
        "kickoff_utc": kickoff,
        "competition": "FIFA World Cup",
        "group": "Round of 32",
        "home": home,
        "away": away,
        "venue": "Test Venue",
        "model_version": "v1",
        "home_win_prob": home_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_prob,
        "model_confidence": "Moderate",
        "behavior_home_prob": 0.45,
        "behavior_draw_prob": 0.30,
        "behavior_away_prob": 0.25,
        "market_snapshot_available": bool(market),
        "prediction_before_kickoff": True,
    }
    if market:
        row["market_home_prob"], row["market_draw_prob"], row["market_away_prob"] = market
    return row


def _result(
    match_id: str = "m1",
    home: str = "Alpha",
    away: str = "Beta",
    home_goals: int = 1,
    away_goals: int = 1,
    actual: str = "draw",
    advancing: str = "Beta",
    semantics: str = "90-minute regular time",
) -> dict:
    return {
        "match_id": match_id,
        "date_utc": "2026-06-11",
        "competition": "FIFA World Cup",
        "home": home,
        "away": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "actual_result": actual,
        "total_goals": home_goals + away_goals,
        "btts_actual": int(home_goals > 0 and away_goals > 0),
        "over_0_5_actual": 1,
        "over_1_5_actual": int(home_goals + away_goals > 1),
        "over_2_5_actual": int(home_goals + away_goals > 2),
        "over_3_5_actual": int(home_goals + away_goals > 3),
        "actual_advancing_team": advancing,
        "result_semantics": semantics,
        "result_source": "test",
        "last_updated": "2026-06-12T00:00:00+00:00",
    }


def test_formal_evaluation_dataset_scores_90_minute_draw_and_keeps_advancement_separate() -> None:
    evaluation = build_formal_evaluation_dataset(
        pd.DataFrame([_prediction()]),
        pd.DataFrame([_result()]),
    )

    row = evaluation.iloc[0]
    assert row["actual_result_1x2"] == "draw"
    assert row["advancing_team"] == "Beta"
    assert row["probability_assigned_to_actual_raw"] == 0.25
    assert math.isclose(row["brier_raw"], (0.60 - 0) ** 2 + (0.25 - 1) ** 2 + (0.15 - 0) ** 2)


def test_advancement_or_penalty_semantics_are_not_mixed_with_1x2() -> None:
    evaluation = build_formal_evaluation_dataset(
        pd.DataFrame([_prediction()]),
        pd.DataFrame([_result(semantics="penalties included")]),
    )

    assert evaluation.empty


def test_missing_market_and_calibrated_probabilities_remain_unavailable() -> None:
    evaluation = build_formal_evaluation_dataset(
        pd.DataFrame([_prediction(market=None)]),
        pd.DataFrame([_result(actual="home_win", home_goals=2, away_goals=0, advancing="Alpha")]),
    )

    row = evaluation.iloc[0]
    assert row["market_join_status"] == "no_event_resolved"
    assert pd.isna(row["home_win_prob_market"])
    assert pd.isna(row["home_win_prob_calibrated"])
    assert pd.isna(row["brier_market"])


def test_multiple_prediction_snapshots_for_same_match_are_preserved() -> None:
    predictions = pd.DataFrame(
        [
            _prediction(snapshot="2026-06-10T10:00:00+00:00", home_prob=0.55, draw_prob=0.25, away_prob=0.20),
            _prediction(snapshot="2026-06-10T12:00:00+00:00", home_prob=0.65, draw_prob=0.20, away_prob=0.15),
        ]
    )
    evaluation = build_formal_evaluation_dataset(
        predictions,
        pd.DataFrame([_result(actual="home_win", home_goals=2, away_goals=0)]),
        latest_snapshot_only=False,
    )

    assert len(evaluation) == 2
    assert set(evaluation["home_win_prob_raw"]) == {0.55, 0.65}


def test_formal_evaluation_defaults_to_latest_valid_pre_kickoff_snapshot() -> None:
    predictions = pd.DataFrame(
        [
            _prediction(snapshot="2026-06-10T10:00:00+00:00", home_prob=0.55, draw_prob=0.25, away_prob=0.20),
            _prediction(snapshot="2026-06-10T12:00:00+00:00", home_prob=0.65, draw_prob=0.20, away_prob=0.15),
        ]
    )
    evaluation = build_formal_evaluation_dataset(predictions, pd.DataFrame([_result(actual="home_win", home_goals=2, away_goals=0)]))

    assert len(evaluation) == 1
    assert evaluation.iloc[0]["home_win_prob_raw"] == 0.65


def test_calibration_summary_log_loss_ece_and_bins() -> None:
    evaluation = build_formal_evaluation_dataset(
        pd.DataFrame(
            [
                _prediction("m1", home_prob=0.60, draw_prob=0.25, away_prob=0.15),
                _prediction("m2", home_prob=0.30, draw_prob=0.40, away_prob=0.30, kickoff="2026-06-12T18:00:00+00:00"),
                _prediction("m3", home_prob=0.20, draw_prob=0.25, away_prob=0.55, kickoff="2026-06-13T18:00:00+00:00"),
            ]
        ),
        pd.DataFrame(
            [
                _result("m1", actual="home_win", home_goals=1, away_goals=0, advancing="Alpha"),
                _result("m2", actual="draw", home_goals=1, away_goals=1, advancing="Alpha"),
                _result("m3", actual="away_win", home_goals=0, away_goals=2, advancing="Beta"),
            ]
        ),
    )

    summary = calibration_summary(evaluation)
    bins = calibration_by_bin(evaluation)

    raw = summary.loc[summary["model"] == "raw_model"].iloc[0]
    assert int(raw["n"]) == 3
    assert raw["multiclass_log_loss"] > 0
    assert raw["expected_calibration_error"] >= 0
    assert not bins.empty


def test_walk_forward_comparison_uses_only_past_rows_and_promotion_rule() -> None:
    predictions = []
    results = []
    outcomes = ["home_win", "draw", "away_win", "home_win", "draw", "away_win"]
    for idx, actual in enumerate(outcomes):
        match_id = f"m{idx}"
        kickoff = f"2026-06-{11 + idx:02d}T18:00:00+00:00"
        predictions.append(_prediction(match_id, kickoff=kickoff, home_prob=0.70 if actual == "home_win" else 0.30, draw_prob=0.50 if actual == "draw" else 0.15, away_prob=0.55 if actual == "away_win" else 0.15))
        results.append(_result(match_id, actual=actual, home_goals=1 if actual == "home_win" else 0, away_goals=1 if actual == "away_win" else 0))
    evaluation = build_formal_evaluation_dataset(pd.DataFrame(predictions), pd.DataFrame(results))

    comparison, scored = walk_forward_calibration_comparison(evaluation, min_train=3)

    assert not comparison.empty
    assert not scored.empty
    assert set(comparison["model"]) >= {"raw_model", "temperature_scaling", "combined_conservative_calibrated_model"}
    promoted = comparison.loc[comparison["production_status"] == "production_eligible"]
    for _, row in promoted.iterrows():
        assert row["brier_delta_vs_raw"] < 0
        assert row["log_loss_delta_vs_raw"] < 0


def test_model_vs_market_benchmark_marks_small_samples() -> None:
    evaluation = build_formal_evaluation_dataset(
        pd.DataFrame([_prediction(market=(0.50, 0.30, 0.20))]),
        pd.DataFrame([_result(actual="home_win", home_goals=2, away_goals=0, advancing="Alpha")]),
    )

    benchmark = model_vs_market_benchmark(evaluation, min_sample=5)

    assert benchmark.iloc[0]["n"] == 1
    assert benchmark.iloc[0]["status"] == "Market benchmark sample too small for reliable conclusions."


def test_prediction_log_1x2_market_rows_are_devigged_without_advancement_markets() -> None:
    prediction_log = pd.DataFrame(
        [
            {"timestamp_utc": "2026-06-10T12:00:00+00:00", "match_id": "m1", "home": "Alpha", "away": "Beta", "kickoff_utc": "2026-06-11T18:00:00+00:00", "model_side": "home_win", "model_probability": 0.5, "primary_model_probability": 0.5, "polymarket_price_cents": 50, "polymarket_side": "YES", "mapping_confidence": "high"},
            {"timestamp_utc": "2026-06-10T12:00:00+00:00", "match_id": "m1", "home": "Alpha", "away": "Beta", "kickoff_utc": "2026-06-11T18:00:00+00:00", "model_side": "draw", "model_probability": 0.3, "primary_model_probability": 0.3, "polymarket_price_cents": 30, "polymarket_side": "YES", "mapping_confidence": "high"},
            {"timestamp_utc": "2026-06-10T12:00:00+00:00", "match_id": "m1", "home": "Alpha", "away": "Beta", "kickoff_utc": "2026-06-11T18:00:00+00:00", "model_side": "away_win", "model_probability": 0.2, "primary_model_probability": 0.2, "polymarket_price_cents": 20, "polymarket_side": "YES", "mapping_confidence": "high"},
            {"timestamp_utc": "2026-06-10T12:00:00+00:00", "match_id": "m1", "home": "Alpha", "away": "Beta", "kickoff_utc": "2026-06-11T18:00:00+00:00", "model_side": "to_advance", "model_probability": 0.7, "primary_model_probability": 0.7, "polymarket_price_cents": 70, "polymarket_side": "YES", "mapping_confidence": "high"},
        ]
    )
    sources = build_prediction_source_dataset(pd.DataFrame(), prediction_log)

    row = sources.iloc[0]
    assert len(sources) == 1
    assert row["market_join_status"] == "joined_price"
    assert round(row["home_win_prob_market"] + row["draw_prob_market"] + row["away_win_prob_market"], 8) == 1.0


def test_evaluation_uses_fixture_bridge_when_match_id_formats_differ() -> None:
    prediction = _prediction(match_id="wc2026_1", home="Alpha", away="Beta")
    result = _result(match_id="provider_99", home="Alpha", away="Beta")
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "wc2026_1",
                "date_utc": "2026-06-11",
                "time_utc": "18:00",
                "competition": "FIFA World Cup",
                "home": "Alpha",
                "away": "Beta",
            }
        ]
    )

    evaluation = build_formal_evaluation_dataset(pd.DataFrame([prediction]), pd.DataFrame([result]), fixtures_df=fixtures)
    audit = build_result_prediction_join_audit(pd.DataFrame([prediction]), pd.DataFrame([result]), fixtures_df=fixtures)

    assert len(evaluation) == 1
    assert evaluation.iloc[0]["result_match_id_original"] == "provider_99"
    assert evaluation.iloc[0]["fixture_match_id"] == "wc2026_1"
    assert evaluation.iloc[0]["result_join_method"] == "result_fixture_crosswalk"
    assert evaluation.iloc[0]["actual_result_from_prediction_perspective"] == "draw"
    assert audit.iloc[0]["join_status"] == "schedule_bridge_join"


def test_evaluation_uses_normalized_team_date_join_when_ids_are_mixed() -> None:
    prediction = _prediction(match_id="wc2026_alpha_beta", home="Alpha", away="Beta")
    result = _result(match_id="csv_2026_06_11_alpha_beta_1_1", home="Alpha", away="Beta")

    evaluation = build_formal_evaluation_dataset(pd.DataFrame([prediction]), pd.DataFrame([result]))
    audit = build_result_prediction_join_audit(pd.DataFrame([prediction]), pd.DataFrame([result]))

    assert len(evaluation) == 1
    assert audit.iloc[0]["join_status"] == "normalized_team_date_join"


def test_reversed_home_away_result_is_detected_and_aligned() -> None:
    prediction = _prediction(match_id="wc2026_rev", home="Alpha", away="Beta", home_prob=0.20, draw_prob=0.20, away_prob=0.60)
    result = _result(match_id="provider_rev", home="Beta", away="Alpha", home_goals=2, away_goals=0, actual="home_win", advancing="Beta")

    evaluation = build_formal_evaluation_dataset(pd.DataFrame([prediction]), pd.DataFrame([result]))
    audit = build_result_prediction_join_audit(pd.DataFrame([prediction]), pd.DataFrame([result]))

    row = evaluation.iloc[0]
    assert row["actual_result_1x2"] == "away_win"
    assert row["actual_home_goals_90"] == 0
    assert row["actual_away_goals_90"] == 2
    assert audit.iloc[0]["join_status"] == "result_found_but_team_order_mismatch"


def test_join_audit_classifies_invalid_probabilities_and_after_kickoff() -> None:
    invalid = _prediction(match_id="m1", home_prob=1.2, draw_prob=0.2, away_prob=-0.4)
    late = _prediction(match_id="m2", snapshot="2026-06-11T19:00:00+00:00", kickoff="2026-06-11T18:00:00+00:00")
    results = pd.DataFrame([_result("m1"), _result("m2")])

    audit = build_result_prediction_join_audit(pd.DataFrame([invalid, late]), results)

    assert set(audit["join_status"]) == {"prediction_missing_probabilities", "result_found_but_after_kickoff_issue"}


def test_ambiguous_result_semantics_are_excluded_from_evaluation() -> None:
    result = _result(semantics="needs_review")
    evaluation = build_formal_evaluation_dataset(pd.DataFrame([_prediction()]), pd.DataFrame([result]))
    audit = build_result_prediction_join_audit(pd.DataFrame([_prediction()]), pd.DataFrame([result]))

    assert evaluation.empty
    assert audit.iloc[0]["join_status"] == "result_found_but_ambiguous_semantics"


def test_recent_postmortem_includes_mexico_diagnostic_warning() -> None:
    predictions = pd.DataFrame(
        [
            {
                "prediction_id": "p_mex",
                "generated_at_utc": "2026-06-30T19:27:11+00:00",
                "match_id": "wc2026_79",
                "kickoff_utc": "2026-07-01T01:00:00+00:00",
                "home_team": "Mexico",
                "away_team": "Ecuador",
                "home_win_prob_raw": 0.171,
                "draw_prob_raw": 0.242,
                "away_win_prob_raw": 0.588,
                "home_win_prob_market": pd.NA,
                "draw_prob_market": pd.NA,
                "away_win_prob_market": pd.NA,
                "behavior_home_prob": pd.NA,
                "behavior_draw_prob": pd.NA,
                "behavior_away_prob": pd.NA,
                "market_join_status": "no_event_resolved",
            }
        ]
    )

    postmortem = recent_match_postmortem(pd.DataFrame(), predictions)
    mexico = postmortem.loc[postmortem["home_team"] == "Mexico"].iloc[0]

    assert mexico["actual_90_min_result"] == "home_win"
    assert "low-confidence forecast" in mexico["recommended_model_system_response"]
    assert "Behavior-disagreement caution" in mexico["what_warning_should_have_appeared"]
