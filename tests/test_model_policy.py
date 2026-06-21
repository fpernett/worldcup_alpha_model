from __future__ import annotations

import pandas as pd

from src.alpha import calculate_polymarket_alpha
from src.backtest import run_model_for_backtest_match
from src.model_policy import (
    annotate_polymarket_alpha_with_policy,
    behavior_disagreement_warning,
    get_current_model_policy,
)


def _model_result(home_prob: float, away_prob: float = 0.25) -> dict[str, object]:
    draw = 1.0 - home_prob - away_prob
    return {
        "match_id": "M1",
        "home": "Argentina",
        "away": "Norway",
        "hxg": 1.4,
        "axg": 1.0,
        "probs": {
            "home_win": home_prob,
            "draw": draw,
            "away_win": away_prob,
            "over_2_5": 0.45,
            "under_2_5": 0.55,
            "btts_yes": 0.50,
            "btts_no": 0.50,
        },
    }


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "M1",
                "market_id": "PM1",
                "question": "Will Argentina beat Norway?",
                "market_type": "match_winner_home",
                "model_side": "home_win",
                "polymarket_side": "YES",
                "mapping_confidence": "high",
                "manual_confirmed": True,
                "yes_price": 0.50,
                "no_price": 0.50,
                "liquidity": 1000,
                "volume": 1000,
                "closed": 0,
            }
        ]
    )


def test_current_policy_uses_baseline_primary_and_diagnostic_behavior() -> None:
    policy = get_current_model_policy()

    assert policy["primary_model_mode"] == "baseline_manual"
    assert policy["secondary_model_mode"] == "behavior_adjusted_asof"
    assert policy["behavior_status"] == "diagnostic_only"
    assert policy["behavior_default_blend"] == 0.0


def test_behavior_disagreement_warning_triggers_above_threshold() -> None:
    warning = behavior_disagreement_warning(_model_result(0.55), _model_result(0.62), threshold=0.05)

    assert "Behavior disagreement: diagnostic only" in warning
    assert "Argentina win probability" in warning


def test_polymarket_alpha_uses_primary_probability_by_default() -> None:
    alpha = calculate_polymarket_alpha(_model_result(0.55), _mapping())

    row = alpha.iloc[0]
    assert row["edge_source"] == "primary_model"
    assert float(row["model_probability"]) == 0.55
    assert float(row["primary_model_probability"]) == 0.55
    assert row["model_policy"] == "baseline_manual | behavior diagnostic_only"


def test_behavior_only_edge_is_not_labelled_primary_alpha() -> None:
    primary_alpha = calculate_polymarket_alpha(_model_result(0.55), _mapping())
    annotated = annotate_polymarket_alpha_with_policy(primary_alpha, _model_result(0.66), show_behavior_diagnostic=False)

    row = annotated.iloc[0]
    assert row["edge_source"] == "primary_model"
    assert float(row["model_probability"]) == 0.55
    assert round(float(row["behavior_diagnostic_probability"]), 2) == 0.66
    assert round(float(row["behavior_probability_delta"]), 2) == 0.11
    assert round(float(row["alpha_gap_cents"]), 1) == 5.0


def test_diagnostic_toggle_marks_edge_source_without_recomputing_alpha() -> None:
    primary_alpha = calculate_polymarket_alpha(_model_result(0.55), _mapping())
    annotated = annotate_polymarket_alpha_with_policy(primary_alpha, _model_result(0.66), show_behavior_diagnostic=True)

    row = annotated.iloc[0]
    assert row["edge_source"] == "diagnostic_behavior_view"
    assert float(row["model_probability"]) == 0.55
    assert round(float(row["alpha_gap_cents"]), 1) == 5.0


def test_backtest_prediction_rows_include_model_policy_fields(monkeypatch) -> None:
    teams = pd.DataFrame(
        [
            {"team": "Argentina", "elo": 1900, "attack": 0.80, "defense": 0.78, "recent_form": 0.75},
            {"team": "Norway", "elo": 1750, "attack": 0.70, "defense": 0.66, "recent_form": 0.70},
        ]
    )

    def fake_run_match_model(match_row, teams_df, venues_df, market_odds_df, cfg, model_mode=None):
        return _model_result(0.55)

    monkeypatch.setattr("src.backtest.run_match_model", fake_run_match_model)
    prediction = run_model_for_backtest_match(
        {
            "match_id": "M1",
            "date_utc": "2026-06-20",
            "home": "Argentina",
            "away": "Norway",
            "home_goals": 1,
            "away_goals": 0,
        },
        mode="baseline_manual",
        teams_df=teams,
        venues_df=pd.DataFrame(),
        market_odds_df=pd.DataFrame(),
    )

    assert prediction["model_policy"] == "baseline_manual | behavior diagnostic_only"
    assert prediction["primary_model_mode"] == "baseline_manual"
    assert prediction["behavior_status"] == "diagnostic_only"
    assert prediction["behavior_blend_used"] is False
    assert "Strict as-of-date blend sensitivity" in prediction["strict_validation_summary"]
