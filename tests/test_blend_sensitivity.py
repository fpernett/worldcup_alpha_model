from __future__ import annotations

import sys

import pandas as pd

from src import blend_sensitivity
from src.asof_backtest import get_team_ratings_asof, run_asof_backtest
from src.historical_data import HISTORICAL_MATCH_COLUMNS


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Alpha",
                "elo": 1700,
                "attack": 0.55,
                "defense": 0.55,
                "recent_form": 0.55,
                "fifa_rank_proxy": 50,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "manual_csv",
                "last_updated": "2026-06-01",
                "notes": "",
            },
            {
                "team": "Beta",
                "elo": 1500,
                "attack": 0.50,
                "defense": 0.50,
                "recent_form": 0.50,
                "fifa_rank_proxy": 90,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "manual_csv",
                "last_updated": "2026-06-01",
                "notes": "",
            },
        ]
    )


def _historical(n_prior: int = 20) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-05-01")
    for i in range(n_prior):
        rows.extend(_long_match_rows(f"prior_{i}", (start + pd.Timedelta(days=i)).date().isoformat(), 3, 0))
    rows.extend(_long_match_rows("current_match", "2026-06-11", 2, 0))
    rows.extend(_long_match_rows("future_match", "2026-06-12", 0, 3))
    return pd.DataFrame(rows, columns=HISTORICAL_MATCH_COLUMNS)


def _long_match_rows(match_id: str, date_utc: str, alpha_goals: int, beta_goals: int) -> list[dict[str, object]]:
    base = {
        "date_utc": date_utc,
        "competition": "FIFA World Cup",
        "competition_type": "world_cup",
        "is_neutral": True,
        "venue": "",
        "city": "",
        "country": "",
        "team_xg": pd.NA,
        "opponent_xg": pd.NA,
        "shots_for": pd.NA,
        "shots_against": pd.NA,
        "shots_on_target_for": pd.NA,
        "shots_on_target_against": pd.NA,
        "possession_pct": pd.NA,
        "opponent_elo": pd.NA,
        "team_elo_pre": pd.NA,
        "temperature_c": pd.NA,
        "humidity_pct": pd.NA,
        "altitude_m": pd.NA,
        "wind_kmh": pd.NA,
        "precipitation_mm": pd.NA,
        "roof_closed": pd.NA,
        "source": "test",
        "last_updated": "2026-06-21",
    }
    return [
        {
            **base,
            "match_id": match_id,
            "team": "Alpha",
            "opponent": "Beta",
            "is_home": True,
            "team_goals": alpha_goals,
            "opponent_goals": beta_goals,
        },
        {
            **base,
            "match_id": match_id,
            "team": "Beta",
            "opponent": "Alpha",
            "is_home": False,
            "team_goals": beta_goals,
            "opponent_goals": alpha_goals,
        },
    ]


def _completed() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "current_match",
                "date_utc": "2026-06-11",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 2,
                "away_goals": 0,
                "competition": "FIFA World Cup",
                "group": "",
                "venue": "",
                "city": "",
                "country": "",
                "neutral_site": 1,
            }
        ]
    )


def test_multiplier_zero_behaves_like_strict_baseline() -> None:
    sensitivity_predictions, _ = blend_sensitivity.run_asof_blend_sensitivity(_completed(), _ratings(), _historical(), [0.0])
    baseline_predictions = run_asof_backtest(_completed(), _ratings(), _historical(), mode="baseline_manual")

    sensitivity_row = sensitivity_predictions.iloc[0]
    baseline_row = baseline_predictions.iloc[0]
    assert round(sensitivity_row["home_win_prob"], 6) == round(baseline_row["home_win_prob"], 6)
    assert round(sensitivity_row["actual_result_prob"], 6) == round(baseline_row["actual_result_prob"], 6)


def test_lower_multiplier_reduces_behavior_movement() -> None:
    baseline = get_team_ratings_asof(_ratings(), _historical(), "2026-06-11", teams=["Alpha", "Beta"], behavior_blend_multiplier=0.0)
    weak = get_team_ratings_asof(_ratings(), _historical(), "2026-06-11", teams=["Alpha", "Beta"], behavior_blend_multiplier=0.10)
    full = get_team_ratings_asof(_ratings(), _historical(), "2026-06-11", teams=["Alpha", "Beta"], behavior_blend_multiplier=1.0)

    base_attack = float(baseline.loc[baseline["team"] == "Alpha", "attack"].iloc[0])
    weak_attack = float(weak.loc[weak["team"] == "Alpha", "attack"].iloc[0])
    full_attack = float(full.loc[full["team"] == "Alpha", "attack"].iloc[0])
    assert abs(weak_attack - base_attack) < abs(full_attack - base_attack)


def test_sensitivity_runner_returns_all_blends_and_no_lookahead() -> None:
    predictions, metrics = blend_sensitivity.run_asof_blend_sensitivity(_completed(), _ratings(), _historical(), [0.0, 0.10, 1.0])

    assert set(predictions["blend_multiplier"]) == {0.0, 0.10, 1.0}
    assert set(metrics["blend_multiplier"]) == {0.0, 0.10, 1.0}
    assert predictions["lookahead_safe"].all()


def test_metrics_table_ranks_correctly() -> None:
    predictions = pd.DataFrame(
        [
            _prediction("m1", 0.0, 0.40, "home_win", 1, 0),
            _prediction("m1", 0.5, 0.70, "home_win", 1, 0),
            _prediction("m2", 0.0, 0.40, "home_win", 1, 0),
            _prediction("m2", 0.5, 0.70, "home_win", 1, 0),
        ]
    )

    metrics = blend_sensitivity.calculate_blend_sensitivity_metrics(predictions)

    best = metrics.sort_values("rank_by_brier").iloc[0]
    assert best["blend_multiplier"] == 0.5
    assert metrics.loc[metrics["blend_multiplier"] == 0.5, "delta_brier_vs_baseline"].iloc[0] < 0


def test_decision_rule() -> None:
    no_help = pd.DataFrame(
        [
            {"blend_multiplier": 0.0, "delta_brier_vs_baseline": 0.0, "delta_log_loss_vs_baseline": 0.0, "brier_score_1x2": 0.6, "log_loss_1x2": 1.0},
            {"blend_multiplier": 0.1, "delta_brier_vs_baseline": 0.01, "delta_log_loss_vs_baseline": 0.01, "brier_score_1x2": 0.61, "log_loss_1x2": 1.01},
        ]
    )
    weak_help = pd.DataFrame(
        [
            {"blend_multiplier": 0.0, "delta_brier_vs_baseline": 0.0, "delta_log_loss_vs_baseline": 0.0, "brier_score_1x2": 0.6, "log_loss_1x2": 1.0},
            {"blend_multiplier": 0.1, "delta_brier_vs_baseline": -0.01, "delta_log_loss_vs_baseline": -0.01, "brier_score_1x2": 0.59, "log_loss_1x2": 0.99},
        ]
    )

    assert "diagnostic only" in blend_sensitivity.blend_sensitivity_recommendation(no_help)
    assert "reduced behavior blend" in blend_sensitivity.blend_sensitivity_recommendation(weak_help)


def test_team_sensitivity_table() -> None:
    predictions, _ = blend_sensitivity.run_asof_blend_sensitivity(_completed(), _ratings(), _historical(), [0.0, 0.10, 1.0])

    team = blend_sensitivity.build_team_sensitivity_table(predictions)

    assert {"Alpha", "Beta"}.issubset(set(team["team"]))
    assert "best_blend_by_actual_prob" in team.columns


def test_script_runs_on_small_synthetic_dataset(monkeypatch, tmp_path, capsys) -> None:
    import scripts.run_blend_sensitivity as script

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "load_completed_matches_for_backtest", lambda **kwargs: _completed())
    monkeypatch.setattr(script, "read_csv_with_columns", lambda *args, **kwargs: _ratings())
    monkeypatch.setattr(script, "load_historical_matches", lambda **kwargs: _historical())
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["run_blend_sensitivity.py", "--blend-multipliers", "0", "0.1", "--save-report"])

    script.main()

    output = capsys.readouterr().out
    assert "completed matches: 1" in output
    assert (tmp_path / "reports" / "blend_sensitivity_predictions_2026-06-21.csv").exists()
    assert (tmp_path / "reports" / "blend_sensitivity_metrics_2026-06-21.csv").exists()
    assert (tmp_path / "reports" / "blend_sensitivity_team_2026-06-21.csv").exists()
    assert (tmp_path / "reports" / "blend_sensitivity_report_2026-06-21.md").exists()


def _prediction(match_id: str, blend: float, home_prob: float, actual_result: str, home_goals: int, away_goals: int) -> dict[str, object]:
    return {
        "match_id": match_id,
        "date_utc": "2026-06-11",
        "home": "Alpha",
        "away": "Beta",
        "home_goals": home_goals,
        "away_goals": away_goals,
        "actual_result": actual_result,
        "blend_multiplier": blend,
        "home_win_prob": home_prob,
        "draw_prob": (1.0 - home_prob) / 2.0,
        "away_win_prob": (1.0 - home_prob) / 2.0,
        "actual_result_prob": home_prob,
        "home_xg": 1.5,
        "away_xg": 0.8,
        "over_2_5_prob": 0.45,
        "under_2_5_prob": 0.55,
        "btts_yes_prob": 0.40,
        "btts_no_prob": 0.60,
        "most_likely_result": "home_win",
        "log_loss_1x2": 0.0,
        "lookahead_safe": True,
        "warning": "",
    }
