from __future__ import annotations

import math
import sys

import pandas as pd

from src import postmortem


def _prediction() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_id": "p1",
                "snapshot_utc": "2026-06-10T12:00:00+00:00",
                "match_id": "m1",
                "kickoff_utc": "2026-06-11T18:00:00+00:00",
                "competition": "World Cup",
                "group": "A",
                "home": "Alpha",
                "away": "Beta",
                "venue": "Test",
                "primary_model_mode": "baseline_manual",
                "model_version": "v1",
                "parameter_set_id": "baseline_current",
                "home_win_prob": 0.7,
                "draw_prob": 0.2,
                "away_win_prob": 0.1,
                "home_xg": 1.8,
                "away_xg": 0.8,
                "over_0_5_prob": 0.9,
                "over_1_5_prob": 0.7,
                "over_2_5_prob": 0.45,
                "over_3_5_prob": 0.2,
                "btts_yes_prob": 0.4,
                "source_status": "",
                "market_snapshot_available": False,
                "notes": "",
            }
        ]
    )


def _result() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-11",
                "competition": "World Cup",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 2,
                "away_goals": 1,
                "actual_result": "home_win",
                "total_goals": 3,
                "btts_actual": 1,
                "over_0_5_actual": 1,
                "over_1_5_actual": 1,
                "over_2_5_actual": 1,
                "over_3_5_actual": 0,
                "result_source": "test",
                "last_updated": "2026-06-12",
            }
        ]
    )


def test_join_marks_pre_kickoff_prediction_valid() -> None:
    joined = postmortem.join_predictions_to_results(_prediction(), _result())

    assert len(joined) == 1
    assert bool(joined.iloc[0]["prediction_before_kickoff"]) is True


def test_prediction_errors_calculate_brier_log_loss_and_markets() -> None:
    joined = postmortem.join_predictions_to_results(_prediction(), _result())
    errors = postmortem.calculate_prediction_errors(joined)

    row = errors.iloc[0]
    assert math.isclose(row["brier_1x2"], (0.7 - 1) ** 2 + 0.2**2 + 0.1**2)
    assert math.isclose(row["log_loss_1x2"], -math.log(0.7))
    assert math.isclose(row["over_2_5_brier"], (0.45 - 1) ** 2)
    assert math.isclose(row["btts_brier"], (0.4 - 1) ** 2)
    assert bool(row["result_correct"]) is True


def test_after_kickoff_prediction_is_not_scored_by_default() -> None:
    pred = _prediction()
    pred.loc[0, "snapshot_utc"] = "2026-06-11T19:00:00+00:00"

    joined = postmortem.join_predictions_to_results(pred, _result())
    errors = postmortem.calculate_prediction_errors(joined)

    assert errors.empty


def test_build_postmortem_report_sections() -> None:
    joined = postmortem.join_predictions_to_results(_prediction(), _result())
    errors = postmortem.calculate_prediction_errors(joined)
    report = postmortem.build_postmortem_report(errors, joined)

    assert "summary_metrics" in report
    assert "worst_misses" in report
    assert not report["summary_metrics"].empty


def test_postmortem_action_plan_blocks_when_no_scored_predictions() -> None:
    actions = postmortem.build_postmortem_action_plan(
        prediction_rows=1,
        result_rows=1,
        scored_prediction_rows=0,
        leaderboard_df=pd.DataFrame(),
    )

    scored = actions.loc[actions["area"] == "Scored predictions"].iloc[0]
    training = actions.loc[actions["area"] == "Candidate training"].iloc[0]
    assert scored["status"] == "Blocked"
    assert "match IDs" in scored["next_action"]
    assert training["status"] == "Not run"


def test_postmortem_action_plan_marks_promotion_for_manual_review() -> None:
    leaderboard = pd.DataFrame(
        [{"parameter_set_id": "draw_1.10", "promotion_status": "promotion_candidate"}]
    )

    actions = postmortem.build_postmortem_action_plan(
        prediction_rows=40,
        result_rows=40,
        scored_prediction_rows=34,
        leaderboard_df=leaderboard,
    )

    scored = actions.loc[actions["area"] == "Scored predictions"].iloc[0]
    training = actions.loc[actions["area"] == "Candidate training"].iloc[0]
    assert scored["status"] == "Usable"
    assert training["status"] == "Review needed"
    assert "before any model-policy change" in training["next_action"]


def test_run_postmortem_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.run_postmortem as script

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "load_prediction_ledger", lambda: _prediction())
    monkeypatch.setattr(script, "load_results_ledger", lambda: _result())
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["run_postmortem.py", "--save-report"])

    script.main()

    output = capsys.readouterr().out
    assert "joined valid predictions: 1" in output
    assert (tmp_path / "reports" / "postmortem_report_2026-06-21.md").exists()
