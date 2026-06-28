from __future__ import annotations

import sys
from pathlib import Path

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
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-22T12:00:00+00:00")

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


def test_dashboard_snapshot_uses_prediction_ledger_not_prediction_log() -> None:
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "save_prediction_snapshot(" not in app_source
    assert "Selected upcoming matches are auto-saved to data/prediction_ledger.csv" in app_source
    assert "Saved {written} prediction row(s) to data/prediction_ledger.csv." in app_source


def test_selected_future_fixture_writes_one_ledger_row(monkeypatch, tmp_path) -> None:
    path = tmp_path / "prediction_ledger.csv"
    monkeypatch.setattr(prediction_ledger, "run_match_model", _fake_run_match_model)
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-22T12:00:00+00:00")

    snapshots, diagnostics = prediction_ledger.snapshot_predictions_for_fixtures_with_diagnostics(
        _fixtures("2026-06-22", "18:00"),
        team_ratings_df=_teams(),
        venues_df=pd.DataFrame(),
        market_odds_df=pd.DataFrame(),
        path=path,
        selected_match_ids=["m1"],
    )

    assert len(snapshots) == 1
    assert diagnostics["predictions_written"] == 1
    assert diagnostics["output_path"].endswith("prediction_ledger.csv")
    ledger = pd.read_csv(path)
    assert len(ledger) == 1


def test_past_fixture_skipped_without_explicit_include_past(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prediction_ledger, "run_match_model", _fake_run_match_model)
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-22T12:00:00+00:00")

    snapshots, diagnostics = prediction_ledger.snapshot_predictions_for_fixtures_with_diagnostics(
        _fixtures("2026-06-22", "11:00"),
        team_ratings_df=_teams(),
        venues_df=pd.DataFrame(),
        path=tmp_path / "prediction_ledger.csv",
        include_past=False,
    )

    assert snapshots.empty
    assert diagnostics["predictions_written"] == 0
    assert diagnostics["skipped_past_kickoff"] == 1
    assert diagnostics["skip_reasons"][0]["reason_code"] == "past_kickoff"


def test_past_fixture_can_be_written_with_pre_kickoff_false(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prediction_ledger, "run_match_model", _fake_run_match_model)
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-22T12:00:00+00:00")

    snapshots, diagnostics = prediction_ledger.snapshot_predictions_for_fixtures_with_diagnostics(
        _fixtures("2026-06-22", "11:00"),
        team_ratings_df=_teams(),
        venues_df=pd.DataFrame(),
        path=tmp_path / "prediction_ledger.csv",
        include_past=True,
    )

    assert len(snapshots) == 1
    assert diagnostics["predictions_written"] == 1
    assert bool(snapshots.iloc[0]["prediction_before_kickoff"]) is False


def test_no_selected_fixtures_returns_clear_diagnostic(tmp_path) -> None:
    snapshots, diagnostics = prediction_ledger.snapshot_predictions_for_fixtures_with_diagnostics(
        _fixtures("2026-06-22", "18:00"),
        team_ratings_df=_teams(),
        venues_df=pd.DataFrame(),
        path=tmp_path / "prediction_ledger.csv",
        selected_match_ids=[],
    )

    assert snapshots.empty
    assert diagnostics["fixtures_available"] == 1
    assert diagnostics["fixtures_selected"] == 0
    assert diagnostics["warnings"] == ["No fixtures were selected."]


def test_missing_model_result_returns_clear_diagnostic(monkeypatch, tmp_path) -> None:
    def raise_model(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(prediction_ledger, "run_match_model", raise_model)
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-22T12:00:00+00:00")

    snapshots, diagnostics = prediction_ledger.snapshot_predictions_for_fixtures_with_diagnostics(
        _fixtures("2026-06-22", "18:00"),
        team_ratings_df=_teams(),
        venues_df=pd.DataFrame(),
        path=tmp_path / "prediction_ledger.csv",
    )

    assert snapshots.empty
    assert diagnostics["skipped_missing_model_result"] == 1
    assert "model unavailable" in diagnostics["skip_reasons"][0]["reason"]


def test_snapshot_preserves_existing_ledger_rows(monkeypatch, tmp_path) -> None:
    path = tmp_path / "prediction_ledger.csv"
    existing = {col: "" for col in prediction_ledger.PREDICTION_LEDGER_COLUMNS}
    existing.update({"prediction_id": "existing", "match_id": "old"})
    pd.DataFrame([existing]).to_csv(path, index=False)
    monkeypatch.setattr(prediction_ledger, "run_match_model", _fake_run_match_model)
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-22T12:00:00+00:00")

    prediction_ledger.snapshot_predictions_for_fixtures_with_diagnostics(
        _fixtures("2026-06-22", "18:00"),
        team_ratings_df=_teams(),
        venues_df=pd.DataFrame(),
        path=path,
    )

    ledger = pd.read_csv(path)
    assert list(ledger["prediction_id"])[0] == "existing"
    assert len(ledger) == 2


def test_dashboard_parameter_set_id_changes_for_custom_config() -> None:
    assert prediction_ledger.dashboard_parameter_set_id(prediction_ledger.ModelConfig()) == "baseline_current"

    custom = prediction_ledger.ModelConfig(base_total_goals=2.65)
    assert prediction_ledger.dashboard_parameter_set_id(custom).startswith("dashboard_custom_")


def test_auto_snapshot_skips_fresh_duplicate(monkeypatch, tmp_path) -> None:
    path = tmp_path / "prediction_ledger.csv"
    existing = {col: "" for col in prediction_ledger.PREDICTION_LEDGER_COLUMNS}
    existing.update(
        {
            "prediction_id": "existing",
            "snapshot_utc": "2026-06-22T12:00:00+00:00",
            "match_id": "m1",
            "primary_model_mode": "baseline_manual",
            "model_version": "baseline_external_calibrated_v1",
            "parameter_set_id": "baseline_current",
            "prediction_before_kickoff": True,
        }
    )
    pd.DataFrame([existing]).to_csv(path, index=False)
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-22T12:30:00+00:00")

    snapshots, diagnostics = prediction_ledger.snapshot_selected_match_if_needed(
        _fixtures("2026-06-22", "18:00").iloc[0],
        team_ratings_df=_teams(),
        venues_df=pd.DataFrame(),
        path=path,
    )

    assert snapshots.empty
    assert diagnostics["skipped_duplicate_policy"] == 1
    assert len(pd.read_csv(path)) == 1


def test_auto_snapshot_writes_when_no_fresh_duplicate(monkeypatch, tmp_path) -> None:
    path = tmp_path / "prediction_ledger.csv"
    monkeypatch.setattr(prediction_ledger, "run_match_model", _fake_run_match_model)
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-22T12:00:00+00:00")

    snapshots, diagnostics = prediction_ledger.snapshot_selected_match_if_needed(
        _fixtures("2026-06-22", "18:00").iloc[0],
        team_ratings_df=_teams(),
        venues_df=pd.DataFrame(),
        path=path,
    )

    assert len(snapshots) == 1
    assert diagnostics["predictions_written"] == 1
    assert pd.read_csv(path).iloc[0]["notes"] == "Auto-saved by dashboard selected-match analysis"


def test_auto_result_import_waits_until_post_game_delay(tmp_path) -> None:
    results, diagnostics = prediction_ledger.import_completed_result_for_fixture_if_ready(
        _fixtures("2026-06-22", "18:00").iloc[0],
        completed_matches_df=pd.DataFrame(),
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-22T21:59:00+00:00",
    )

    assert results.empty
    assert diagnostics["status"] == "not_ready"
    assert diagnostics["rows_imported"] == 0


def test_auto_result_import_uses_selected_fixture_match_id(tmp_path) -> None:
    completed = pd.DataFrame(
        [
            {
                "match_id": "external_id",
                "date_utc": "2026-06-22",
                "competition": "World Cup",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 2,
                "away_goals": 1,
            }
        ]
    )

    results, diagnostics = prediction_ledger.import_completed_result_for_fixture_if_ready(
        _fixtures("2026-06-22", "18:00").iloc[0],
        completed_matches_df=completed,
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-22T22:01:00+00:00",
    )

    assert diagnostics["status"] == "imported"
    assert diagnostics["rows_imported"] == 1
    assert results.iloc[0]["match_id"] == "m1"
    assert results.iloc[0]["actual_result"] == "home_win"


def test_required_prediction_ledger_columns_exist() -> None:
    required = {
        "prediction_id",
        "snapshot_utc",
        "match_id",
        "kickoff_utc",
        "competition",
        "group",
        "home",
        "away",
        "venue",
        "primary_model_mode",
        "model_version",
        "parameter_set_id",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "home_xg",
        "away_xg",
        "over_0_5_prob",
        "over_1_5_prob",
        "over_2_5_prob",
        "over_3_5_prob",
        "btts_yes_prob",
        "source_status",
        "prediction_before_kickoff",
        "market_snapshot_available",
        "notes",
    }

    assert required.issubset(set(prediction_ledger.PREDICTION_LEDGER_COLUMNS))


def test_snapshot_audit_dry_run_does_not_write(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_prediction_snapshot as script

    path = tmp_path / "prediction_ledger.csv"
    _patch_audit_script(monkeypatch, script, path)
    monkeypatch.setattr(sys, "argv", ["audit_prediction_snapshot.py", "--start-date", "2026-06-22", "--end-date", "2026-06-22", "--competition", "World Cup"])

    script.main()

    assert "predictions_that_would_be_written: 1" in capsys.readouterr().out
    assert not path.exists()


def test_snapshot_audit_write_appends_rows(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_prediction_snapshot as script

    path = tmp_path / "prediction_ledger.csv"
    _patch_audit_script(monkeypatch, script, path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_prediction_snapshot.py", "--start-date", "2026-06-22", "--end-date", "2026-06-22", "--competition", "World Cup", "--write"],
    )

    script.main()

    assert "predictions_written: 1" in capsys.readouterr().out
    assert path.exists()
    assert len(pd.read_csv(path)) == 1


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


def _fixtures(date_utc: str, time_utc: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": date_utc,
                "time_utc": time_utc,
                "competition": "World Cup",
                "group": "A",
                "home": "Alpha",
                "away": "Beta",
                "venue": "Test",
            }
        ]
    )


def _teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "Alpha", "elo": 1700, "attack": 0.6, "defense": 0.6, "recent_form": 0.6},
            {"team": "Beta", "elo": 1600, "attack": 0.5, "defense": 0.5, "recent_form": 0.5},
        ]
    )


def _fake_run_match_model(*_args, **_kwargs):
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


def _patch_audit_script(monkeypatch, script, path) -> None:
    monkeypatch.setattr(script, "PREDICTION_LEDGER_PATH", path)
    monkeypatch.setattr(script, "get_upcoming_fixtures", lambda **_kwargs: _fixtures("2026-06-22", "18:00"))
    monkeypatch.setattr(script, "get_team_ratings", lambda **_kwargs: _teams())
    monkeypatch.setattr(script, "load_venues", lambda: pd.DataFrame())
    monkeypatch.setattr(script, "load_market_odds", lambda: pd.DataFrame())
    monkeypatch.setattr(prediction_ledger, "run_match_model", _fake_run_match_model)
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-22T12:00:00+00:00")
