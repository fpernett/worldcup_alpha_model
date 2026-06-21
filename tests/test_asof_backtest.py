from __future__ import annotations

import sys

import pandas as pd
import pytest

from src import asof_backtest
from src.elo import build_elo_asof
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


def _historical(n_prior: int = 10, include_current: bool = True, include_future: bool = True) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-05-01")
    for i in range(n_prior):
        rows.extend(_long_match_rows(f"prior_{i}", (start + pd.Timedelta(days=i)).date().isoformat(), 3, 1))
    if include_current:
        rows.extend(_long_match_rows("current_match", "2026-06-11", 1, 0))
    if include_future:
        rows.extend(_long_match_rows("future_match", "2026-06-12", 0, 4))
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
                "home_goals": 1,
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


def test_filter_historical_matches_asof_excludes_future_and_current_match() -> None:
    filtered = asof_backtest.filter_historical_matches_asof(_historical(), "2026-06-11", exclude_match_id="current_match")

    assert "current_match" not in set(filtered["match_id"])
    assert "future_match" not in set(filtered["match_id"])
    assert filtered["date_utc"].max() < "2026-06-11"


def test_lookahead_violation_raises() -> None:
    with pytest.raises(ValueError, match="on/after 2026-06-11"):
        asof_backtest.assert_no_future_matches(_historical(n_prior=0, include_current=True, include_future=False), "2026-06-11")


def test_build_elo_asof_uses_only_prior_matches() -> None:
    elo = build_elo_asof(_historical(), "2026-06-11")

    alpha = elo.loc[elo["team"] == "Alpha"].iloc[0]
    assert alpha["matches_used_for_elo"] == 10
    assert alpha["latest_match_used_for_elo"] == "2026-05-10"


def test_asof_behavior_uses_only_earlier_matches() -> None:
    behavior = asof_backtest.build_team_behavior_asof(_historical(), _ratings(), "2026-06-11", teams=["Alpha"])

    alpha = behavior.loc[behavior["team"] == "Alpha"].iloc[0]
    assert alpha["latest_match_used"] == "2026-05-10"
    assert bool(alpha["lookahead_safe"]) is True
    assert alpha["matches_used_recent"] == 10


def test_baseline_mode_does_not_use_behavior() -> None:
    ratings = asof_backtest.get_team_ratings_asof(_ratings(), _historical(), "2026-06-11", teams=["Alpha", "Beta"], mode="baseline_manual")

    alpha = ratings.loc[ratings["team"] == "Alpha"].iloc[0]
    assert alpha["attack"] == 0.55
    assert bool(alpha["behavior_blend_used"]) is False
    assert alpha["latest_behavior_match_used"] == ""


def test_behavior_adjusted_asof_uses_asof_behavior() -> None:
    ratings = asof_backtest.get_team_ratings_asof(_ratings(), _historical(), "2026-06-11", teams=["Alpha", "Beta"], mode="behavior_adjusted_asof")

    alpha = ratings.loc[ratings["team"] == "Alpha"].iloc[0]
    assert bool(alpha["behavior_blend_used"]) is True
    assert alpha["latest_behavior_match_used"] == "2026-05-10"
    assert alpha["attack"] >= 0.55


def test_asof_backtest_returns_both_modes() -> None:
    predictions = asof_backtest.run_asof_backtest(_completed(), _ratings(), _historical(), mode="both")

    assert set(predictions["mode"]) == {"baseline_manual", "behavior_adjusted_asof"}
    assert predictions["lookahead_safe"].all()


def test_no_crash_with_insufficient_pre_match_behavior() -> None:
    predictions = asof_backtest.run_asof_backtest(_completed(), _ratings(), _historical(n_prior=1), mode="both")
    behavior = predictions.loc[predictions["mode"] == "behavior_adjusted_asof"].iloc[0]

    assert "insufficient_asof_behavior" in behavior["warning"]


def test_run_asof_backtest_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.run_asof_backtest as script

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "load_completed_matches_for_backtest", lambda **kwargs: _completed())
    monkeypatch.setattr(script, "read_csv_with_columns", lambda *args, **kwargs: _ratings())
    monkeypatch.setattr(script, "load_historical_matches", lambda **kwargs: _historical())
    monkeypatch.setattr(script, "_load_v1_metrics", lambda args: pd.DataFrame())
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["run_asof_backtest.py", "--save-report"])

    script.main()

    output = capsys.readouterr().out
    assert "completed matches used: 1" in output
    assert (tmp_path / "reports" / "asof_backtest_predictions_2026-06-21.csv").exists()
    assert (tmp_path / "reports" / "asof_backtest_report_2026-06-21.md").exists()
