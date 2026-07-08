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
        now_utc="2026-06-22T18:14:00+00:00",
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


def test_auto_result_import_refreshes_provider_rows_for_brazil_japan_like_fixture(monkeypatch, tmp_path) -> None:
    provider_rows = pd.DataFrame(
        [
            {
                "provider": "football-data.org",
                "provider_match_id": "provider_bra_jpn",
                "provider_kickoff_utc": "2026-06-29T17:00:00+00:00",
                "date_utc": "2026-06-29",
                "competition": "FIFA World Cup",
                "group": "Round of 32",
                "home": "Brazil",
                "away": "Japan",
                "home_score_90": 1,
                "away_score_90": 0,
                "score_source": "score.fullTime",
                "score_semantics": "90-minute regular time",
                "is_completed": 1,
                "source": "football-data.org",
                "last_updated": "2026-06-29T21:15:00+00:00",
            }
        ]
    )
    fixture = _fixtures("2026-06-29", "17:00").iloc[0].copy()
    fixture["match_id"] = "wc2026_76"
    fixture["group"] = "Round of 32"
    fixture["home"] = "Brazil"
    fixture["away"] = "Japan"
    monkeypatch.setattr(
        prediction_ledger,
        "refresh_completed_results_for_fixture",
        lambda *_args, **_kwargs: (
            provider_rows,
            {
                "status": "success",
                "provider_checked": "football-data.org /matches?status=FINISHED",
                "date_window_checked": "2026-06-28 to 2026-06-30",
                "cache_path_checked": "data/cache/completed_results_latest.csv",
                "local_path_checked": "data/completed_results.csv",
                "completed_rows_loaded": 1,
                "last_updated": "2026-06-29T21:15:00+00:00",
            },
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    results, diagnostics = prediction_ledger.import_completed_result_for_fixture_if_ready(
        fixture,
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-29T22:01:00+00:00",
    )

    assert diagnostics["status"] == "imported"
    assert diagnostics["provider_checked"] == "football-data.org /matches?status=FINISHED"
    assert diagnostics["score_semantics"] == "90-minute regular time"
    assert diagnostics["imported_score"] == "1-0"
    row = results.iloc[0]
    assert row["match_id"] == "wc2026_76"
    assert row["actual_result"] == "home_win"


def test_auto_result_import_matches_nearby_reversed_provider_fixture(tmp_path) -> None:
    completed = pd.DataFrame(
        [
            {
                "match_id": "provider_external_id",
                "date_utc": "2026-06-21",
                "competition": "World Cup",
                "home": "Beta",
                "away": "Alpha",
                "home_goals": 0,
                "away_goals": 2,
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
    row = results.iloc[0]
    assert row["match_id"] == "m1"
    assert row["home"] == "Alpha"
    assert row["away"] == "Beta"
    assert int(row["home_goals"]) == 2
    assert int(row["away_goals"]) == 0
    assert row["actual_result"] == "home_win"


def test_auto_result_import_rejects_one_day_mismatch_with_incompatible_competition(tmp_path) -> None:
    completed = pd.DataFrame(
        [
            {
                "match_id": "friendly_external_id",
                "date_utc": "2026-06-21",
                "competition": "Friendly",
                "home": "Beta",
                "away": "Alpha",
                "home_goals": 0,
                "away_goals": 2,
            }
        ]
    )

    results, diagnostics = prediction_ledger.import_completed_result_for_fixture_if_ready(
        _fixtures("2026-06-22", "18:00").iloc[0],
        completed_matches_df=completed,
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-22T22:01:00+00:00",
    )

    assert results.empty
    assert diagnostics["status"] == "completed_rows_exist_no_fixture_match"
    assert diagnostics["completed_rows_loaded"] == 1
    assert diagnostics["nearest_candidate_rows"][0]["competition"] == "Friendly"


def test_results_import_prefers_regular_time_scores_for_polymarket_markets(tmp_path) -> None:
    completed = pd.DataFrame(
        [
            {
                "match_id": "knockout_1",
                "date_utc": "2026-06-22",
                "competition": "World Cup",
                "home": "Alpha",
                "away": "Beta",
                "home_goals_90": 1,
                "away_goals_90": 1,
                "home_goals": 2,
                "away_goals": 1,
            }
        ]
    )

    results = prediction_ledger.import_completed_results(completed, path=tmp_path / "results_ledger.csv")

    row = results.iloc[0]
    assert int(row["home_goals"]) == 1
    assert int(row["away_goals"]) == 1
    assert row["actual_result"] == "draw"


def test_load_manual_missing_results_accepts_score_aliases_and_filters(tmp_path, monkeypatch) -> None:
    path = tmp_path / "manual_missing_results.csv"
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-22",
                "competition": "FIFA World Cup",
                "home": "Alpha",
                "away": "Beta",
                "home_score_90": 2,
                "away_score_90": 1,
            },
            {
                "match_id": "m2",
                "date_utc": "2026-06-23",
                "competition": "FIFA World Cup",
                "home": "Gamma",
                "away": "Delta",
                "home_goals": "",
                "away_goals": 0,
            },
        ]
    ).to_csv(path, index=False)
    monkeypatch.setattr(prediction_ledger, "utc_now_iso", lambda: "2026-06-24T00:00:00+00:00")

    rows = prediction_ledger.load_manual_missing_results(
        start_date="2026-06-22",
        end_date="2026-06-22",
        teams=["Alpha"],
        path=path,
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["provider"] == "manual_missing_results"
    assert row["provider_match_id"] == "m1"
    assert int(row["home_goals"]) == 2
    assert int(row["away_goals_90"]) == 1
    assert row["score_source"] == "data/manual_missing_results.csv"
    assert row["score_semantics"] == "90-minute regular time"
    assert row["result_source"] == "manual_verified_missing_results"
    assert row["last_updated"] == "2026-06-24T00:00:00+00:00"


def test_bulk_sync_uses_manual_missing_results_candidates(monkeypatch, tmp_path) -> None:
    manual_path = tmp_path / "manual_missing_results.csv"
    manual_candidates = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "provider_match_id": "m1",
                "provider": "manual_missing_results",
                "date_utc": "2026-06-22",
                "competition": "FIFA World Cup",
                "group": "",
                "home": "Alpha",
                "away": "Beta",
                "home_goals": 2,
                "away_goals": 1,
                "home_goals_90": 2,
                "away_goals_90": 1,
                "score_source": "data/manual_missing_results.csv",
                "score_semantics": "90-minute regular time",
                "provider_kickoff_utc": "",
                "result_source": "manual_verified_missing_results",
                "last_updated": "2026-06-24T00:00:00+00:00",
            }
        ]
    )
    monkeypatch.setattr(
        prediction_ledger,
        "load_completed_results",
        lambda *_args, **_kwargs: (
            pd.DataFrame(),
            {"status": "completed_source_empty", "completed_rows_loaded": 0},
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())
    monkeypatch.setattr(prediction_ledger, "load_manual_missing_results", lambda **_kwargs: manual_candidates)
    monkeypatch.setattr(prediction_ledger, "MANUAL_MISSING_RESULTS_PATH", manual_path)

    results, diagnostics = prediction_ledger.sync_all_completed_results(
        _bulk_fixtures().head(1),
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-24T12:00:00+00:00",
        result_ready_delay_hours=0,
    )

    assert diagnostics["manual_candidate_rows"] == 1
    assert diagnostics["manual_path_checked"] == str(manual_path)
    assert diagnostics["imported_or_updated_rows"] == 1
    assert len(results) == 1
    assert int(results.iloc[0]["home_goals"]) == 2


def test_auto_result_import_reports_completed_source_empty(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        prediction_ledger,
        "refresh_completed_results_for_fixture",
        lambda *_args, **_kwargs: (
            pd.DataFrame(),
            {
                "status": "completed_source_empty",
                "provider_checked": "football-data.org /matches?status=FINISHED",
                "date_window_checked": "2026-06-21 to 2026-06-23",
                "cache_path_checked": "data/cache/completed_results_latest.csv",
                "local_path_checked": "data/completed_results.csv",
                "completed_rows_loaded": 0,
            },
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    results, diagnostics = prediction_ledger.import_completed_result_for_fixture_if_ready(
        _fixtures("2026-06-22", "18:00").iloc[0],
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-22T22:01:00+00:00",
    )

    assert results.empty
    assert diagnostics["status"] == "completed_source_empty"
    assert diagnostics["provider_checked"] == "football-data.org /matches?status=FINISHED"
    assert diagnostics["cache_path_checked"] == "data/cache/completed_results_latest.csv"


def test_bulk_sync_imports_multiple_finished_fixtures_across_dates(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_load_completed_results(start_date, end_date, **_kwargs):
        captured["window"] = (start_date, end_date)
        return (
            _provider_completed_rows(
                [
                    ("p1", "2026-06-22", "Alpha", "Beta", 2, 1),
                    ("p2", "2026-06-23", "Gamma", "Delta", 0, 0),
                ]
            ),
            {
                "status": "success",
                "provider_checked": "football-data.org /matches?status=FINISHED",
                "date_window_checked": f"{start_date} to {end_date}",
                "cache_path_checked": "data/cache/completed_results_latest.csv",
                "completed_rows_loaded": 2,
                "normalized_rows_fetched": 2,
                "last_updated": "2026-06-24T00:00:00+00:00",
            },
        )

    monkeypatch.setattr(prediction_ledger, "load_completed_results", fake_load_completed_results)
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    results, diagnostics = prediction_ledger.sync_all_completed_results(
        _bulk_fixtures(),
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-24T12:00:00+00:00",
        result_ready_delay_hours=0,
    )

    assert captured["window"] == ("2026-06-22", "2026-06-23")
    assert diagnostics["imported_or_updated_rows"] == 2
    assert diagnostics["unmatched_fixtures"] == 0
    assert set(results["match_id"]) == {"m1", "m2"}


def test_bulk_sync_is_idempotent_and_reports_already_present(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        prediction_ledger,
        "load_completed_results",
        lambda *_args, **_kwargs: (
            _provider_completed_rows([("p1", "2026-06-22", "Alpha", "Beta", 2, 1)]),
            {"status": "success", "normalized_rows_fetched": 1, "completed_rows_loaded": 1},
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())
    path = tmp_path / "results_ledger.csv"

    prediction_ledger.sync_all_completed_results(
        _bulk_fixtures().head(1),
        path=path,
        now_utc="2026-06-24T12:00:00+00:00",
        result_ready_delay_hours=0,
    )
    results, diagnostics = prediction_ledger.sync_all_completed_results(
        _bulk_fixtures().head(1),
        path=path,
        now_utc="2026-06-24T12:00:00+00:00",
        result_ready_delay_hours=0,
    )

    assert len(results) == 1
    assert diagnostics["already_present_rows"] == 1
    assert diagnostics["imported_or_updated_rows"] == 0


def test_bulk_sync_marks_conflicting_existing_score_for_review(monkeypatch, tmp_path) -> None:
    path = tmp_path / "results_ledger.csv"
    existing = {col: "" for col in prediction_ledger.RESULTS_LEDGER_COLUMNS}
    existing.update(
        {
            "match_id": "m1",
            "date_utc": "2026-06-22",
            "competition": "FIFA World Cup",
            "home": "Alpha",
            "away": "Beta",
            "home_goals": 0,
            "away_goals": 0,
            "actual_result": "draw",
        }
    )
    pd.DataFrame([existing]).to_csv(path, index=False)
    monkeypatch.setattr(
        prediction_ledger,
        "load_completed_results",
        lambda *_args, **_kwargs: (
            _provider_completed_rows([("p1", "2026-06-22", "Alpha", "Beta", 2, 1)]),
            {"status": "success", "normalized_rows_fetched": 1, "completed_rows_loaded": 1},
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    results, diagnostics = prediction_ledger.sync_all_completed_results(
        _bulk_fixtures().head(1),
        path=path,
        now_utc="2026-06-24T12:00:00+00:00",
        result_ready_delay_hours=0,
    )

    assert diagnostics["conflict_rows"] == 1
    assert diagnostics["fixture_diagnostics"][0]["import_status"] == "conflict_needs_review"
    assert int(results.iloc[0]["home_goals"]) == 0


def test_bulk_sync_reorients_reversed_provider_team_rows(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        prediction_ledger,
        "load_completed_results",
        lambda *_args, **_kwargs: (
            _provider_completed_rows([("p1", "2026-06-22", "Beta", "Alpha", 0, 2)]),
            {"status": "success", "normalized_rows_fetched": 1, "completed_rows_loaded": 1},
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    results, diagnostics = prediction_ledger.sync_all_completed_results(
        _bulk_fixtures().head(1),
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-24T12:00:00+00:00",
        result_ready_delay_hours=0,
    )

    assert diagnostics["imported_or_updated_rows"] == 1
    assert results.iloc[0]["home"] == "Alpha"
    assert int(results.iloc[0]["home_goals"]) == 2
    assert int(results.iloc[0]["away_goals"]) == 0


def test_bulk_sync_reports_unmatched_fixture_with_nearest_candidates(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        prediction_ledger,
        "load_completed_results",
        lambda *_args, **_kwargs: (
            _provider_completed_rows([("p1", "2026-06-22", "Alpha", "Other", 1, 0)]),
            {"status": "success", "normalized_rows_fetched": 1, "completed_rows_loaded": 1},
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    _results, diagnostics = prediction_ledger.sync_all_completed_results(
        _bulk_fixtures().head(1),
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-24T12:00:00+00:00",
        result_ready_delay_hours=0,
    )

    assert diagnostics["unmatched_fixtures"] == 1
    unmatched = diagnostics["unmatched_rows"][0]
    assert unmatched["fixture_id"] == "m1"
    assert unmatched["nearest_candidates"][0]["home"] == "Alpha"


def test_bulk_sync_reports_completed_source_empty(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        prediction_ledger,
        "load_completed_results",
        lambda *_args, **_kwargs: (
            pd.DataFrame(),
            {
                "status": "completed_source_empty",
                "provider_checked": "football-data.org /matches?status=FINISHED",
                "date_window_checked": "2026-06-22 to 2026-06-22",
                "cache_path_checked": "data/cache/completed_results_latest.csv",
                "completed_rows_loaded": 0,
            },
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    _results, diagnostics = prediction_ledger.sync_all_completed_results(
        _bulk_fixtures().head(1),
        path=tmp_path / "results_ledger.csv",
        now_utc="2026-06-24T12:00:00+00:00",
        result_ready_delay_hours=0,
    )

    assert diagnostics["status"] == "completed_source_empty"
    assert diagnostics["unmatched_fixtures"] == 1
    assert diagnostics["provider_checked"] == "football-data.org /matches?status=FINISHED"


def test_auto_launch_sync_imports_fixture_after_delay(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_load_completed_results(start_date, end_date, **_kwargs):
        captured["window"] = (start_date, end_date)
        return (
            _provider_completed_rows([("p1", "2000-06-22", "Alpha", "Beta", 1, 1)]),
            {"status": "success", "provider_checked": "football-data.org /matches?status=FINISHED", "completed_rows_loaded": 1},
        )

    monkeypatch.setattr(prediction_ledger, "load_completed_results", fake_load_completed_results)
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    results, diagnostics = prediction_ledger.auto_sync_completed_results_on_launch(
        _fixtures("2000-06-22", "18:00").assign(competition="FIFA World Cup"),
        path=tmp_path / "results_ledger.csv",
    )

    assert captured["window"] == ("2000-06-22", "2000-06-22")
    assert diagnostics["auto_sync_ran"] is True
    assert diagnostics["imported_rows"] == 1
    assert len(results) == 1
    assert results.iloc[0]["result_semantics"] == "90-minute regular time"


def test_auto_launch_sync_skips_fixture_before_delay(monkeypatch, tmp_path) -> None:
    called = {"provider": False}

    def fake_load_completed_results(*_args, **_kwargs):
        called["provider"] = True
        return pd.DataFrame(), {}

    monkeypatch.setattr(prediction_ledger, "load_completed_results", fake_load_completed_results)

    results, diagnostics = prediction_ledger.auto_sync_completed_results_on_launch(
        _fixtures("2999-06-22", "18:00").assign(competition="FIFA World Cup"),
        path=tmp_path / "results_ledger.csv",
    )

    assert called["provider"] is False
    assert results.empty
    assert diagnostics["skipped_not_ready_rows"] == 1
    assert diagnostics["skipped_rows"] == 1


def test_auto_launch_sync_same_existing_score_is_not_duplicated(monkeypatch, tmp_path) -> None:
    path = tmp_path / "results_ledger.csv"
    monkeypatch.setattr(
        prediction_ledger,
        "load_completed_results",
        lambda *_args, **_kwargs: (
            _provider_completed_rows([("p1", "2000-06-22", "Alpha", "Beta", 2, 1)]),
            {"status": "success", "completed_rows_loaded": 1},
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    prediction_ledger.auto_sync_completed_results_on_launch(
        _fixtures("2000-06-22", "18:00").assign(competition="FIFA World Cup"),
        path=path,
    )
    results, diagnostics = prediction_ledger.auto_sync_completed_results_on_launch(
        _fixtures("2000-06-22", "18:00").assign(competition="FIFA World Cup"),
        path=path,
    )

    assert len(results) == 1
    assert diagnostics["already_present_rows"] == 1
    assert diagnostics["imported_rows"] == 0


def test_auto_launch_sync_conflict_does_not_overwrite_existing_score(monkeypatch, tmp_path) -> None:
    path = tmp_path / "results_ledger.csv"
    existing = {col: "" for col in prediction_ledger.RESULTS_LEDGER_COLUMNS}
    existing.update(
        {
            "match_id": "m1",
            "date_utc": "2000-06-22",
            "competition": "FIFA World Cup",
            "home": "Alpha",
            "away": "Beta",
            "home_goals": 0,
            "away_goals": 0,
            "actual_result": "draw",
            "result_semantics": "90-minute regular time",
            "evaluation_eligible_1x2": True,
        }
    )
    pd.DataFrame([existing]).to_csv(path, index=False)
    monkeypatch.setattr(
        prediction_ledger,
        "load_completed_results",
        lambda *_args, **_kwargs: (
            _provider_completed_rows([("p1", "2000-06-22", "Alpha", "Beta", 2, 1)]),
            {"status": "success", "completed_rows_loaded": 1},
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    results, diagnostics = prediction_ledger.auto_sync_completed_results_on_launch(
        _fixtures("2000-06-22", "18:00").assign(competition="FIFA World Cup"),
        path=path,
    )

    assert diagnostics["conflict_rows"] == 1
    assert int(results.iloc[0]["home_goals"]) == 0
    assert int(results.iloc[0]["away_goals"]) == 0


def test_auto_launch_sync_handles_missing_provider_rows(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        prediction_ledger,
        "load_completed_results",
        lambda *_args, **_kwargs: (
            pd.DataFrame(),
            {
                "status": "completed_source_empty",
                "provider_checked": "football-data.org /matches?status=FINISHED",
                "completed_rows_loaded": 0,
            },
        ),
    )
    monkeypatch.setattr(prediction_ledger, "load_completed_matches_for_backtest", lambda **_kwargs: pd.DataFrame())

    results, diagnostics = prediction_ledger.auto_sync_completed_results_on_launch(
        _fixtures("2000-06-22", "18:00").assign(competition="FIFA World Cup"),
        path=tmp_path / "results_ledger.csv",
    )

    assert results.empty
    assert diagnostics["status"] == "completed_source_empty"
    assert diagnostics["provider_rows_loaded"] == 0
    for key in ["imported_rows", "already_present_rows", "unmatched_rows_count", "conflict_rows", "skipped_rows"]:
        assert key in diagnostics


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


def test_select_latest_valid_snapshots_keeps_latest_duplicate_match() -> None:
    rows = [
        _ledger_row("m1", "2026-06-22T10:00:00+00:00", home_prob=0.40),
        _ledger_row("m1", "2026-06-22T11:00:00+00:00", home_prob=0.60),
        _ledger_row("m2", "2026-06-22T09:00:00+00:00", home_prob=0.55),
    ]

    selected = prediction_ledger.select_latest_valid_snapshots(pd.DataFrame(rows))

    assert len(selected) == 2
    latest = selected.loc[selected["match_id"].eq("m1")].iloc[0]
    assert latest["prediction_id"] == "m1_2026-06-22T11:00:00+00:00"
    assert latest["home_win_prob"] == 0.60
    assert int(latest["n_snapshots_for_match"]) == 2
    assert int(latest["snapshot_rank_for_match"]) == 2
    assert bool(latest["is_latest_valid_snapshot"]) is True


def test_select_latest_valid_snapshots_excludes_post_kickoff_when_required() -> None:
    df = pd.DataFrame(
        [
            _ledger_row("m1", "2026-06-22T10:00:00+00:00", before_kickoff=True),
            _ledger_row("m2", "2026-06-22T11:00:00+00:00", before_kickoff=False),
        ]
    )

    selected = prediction_ledger.select_latest_valid_snapshots(df, require_pre_kickoff=True)

    assert list(selected["match_id"]) == ["m1"]


def test_select_latest_valid_snapshots_excludes_unresolved_teams_by_default() -> None:
    df = pd.DataFrame(
        [
            _ledger_row("m1", "2026-06-22T10:00:00+00:00", home="Winner Group K", away="Japan"),
            _ledger_row("m2", "2026-06-22T10:00:00+00:00", home="Brazil", away="3rd Group E/I/L"),
            _ledger_row("m3", "2026-06-22T10:00:00+00:00", home="Argentina", away="Austria"),
        ]
    )

    selected = prediction_ledger.select_latest_valid_snapshots(df)
    included = prediction_ledger.select_latest_valid_snapshots(df, exclude_unresolved_teams=False)

    assert list(selected["match_id"]) == ["m3"]
    assert set(included["match_id"]) == {"m1", "m2", "m3"}


def test_select_latest_valid_snapshots_does_not_mutate_append_only_ledger_frame() -> None:
    df = pd.DataFrame(
        [
            _ledger_row("m1", "2026-06-22T10:00:00+00:00"),
            _ledger_row("m1", "2026-06-22T11:00:00+00:00"),
        ]
    )
    original_columns = list(df.columns)
    original_len = len(df)

    selected = prediction_ledger.select_latest_valid_snapshots(df)

    assert len(df) == original_len
    assert list(df.columns) == original_columns
    assert len(selected) == 1


def test_select_latest_valid_snapshots_is_stable_when_snapshot_times_differ() -> None:
    df = pd.DataFrame(
        [
            _ledger_row("m1", "2026-06-22T12:00:00+00:00", home_prob=0.30),
            _ledger_row("m1", "2026-06-22T10:00:00+00:00", home_prob=0.70),
            _ledger_row("m1", "2026-06-22T11:00:00+00:00", home_prob=0.50),
        ]
    )

    selected = prediction_ledger.select_latest_valid_snapshots(df)

    assert len(selected) == 1
    assert selected.iloc[0]["snapshot_utc"] == "2026-06-22T12:00:00+00:00"
    assert selected.iloc[0]["home_win_prob"] == 0.30


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


def test_result_coverage_audit_script_writes_reports(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_result_coverage as script

    predictions_path = tmp_path / "data" / "prediction_ledger.csv"
    results_path = tmp_path / "data" / "results_ledger.csv"
    fixtures_path = tmp_path / "data" / "fixtures.csv"
    reports_dir = tmp_path / "reports"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    prediction = {col: "" for col in prediction_ledger.PREDICTION_LEDGER_COLUMNS}
    prediction.update(
        {
            "prediction_id": "p1",
            "snapshot_utc": "2000-06-22T12:00:00+00:00",
            "match_id": "m1",
            "kickoff_utc": "2000-06-22T18:00:00+00:00",
            "competition": "FIFA World Cup",
            "home": "Alpha",
            "away": "Beta",
            "prediction_before_kickoff": True,
        }
    )
    pd.DataFrame([prediction]).to_csv(predictions_path, index=False)
    _fixtures("2000-06-22", "18:00").assign(competition="FIFA World Cup").to_csv(fixtures_path, index=False)
    pd.DataFrame(columns=prediction_ledger.RESULTS_LEDGER_COLUMNS).to_csv(results_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_result_coverage.py",
            "--predictions",
            str(predictions_path),
            "--results-ledger",
            str(results_path),
            "--fixtures",
            str(fixtures_path),
            "--reports-dir",
            str(reports_dir),
            "--now-utc",
            "2000-06-22T19:00:00+00:00",
        ],
    )

    script.main()

    assert "matches missing local result: 1" in capsys.readouterr().out
    assert (reports_dir / "result_coverage_audit.csv").exists()
    assert (reports_dir / "result_coverage_audit.md").exists()


def _ledger_row(
    match_id: str,
    snapshot_utc: str,
    *,
    home: str = "Alpha",
    away: str = "Beta",
    home_prob: float = 0.50,
    before_kickoff: bool = True,
) -> dict:
    row = {col: "" for col in prediction_ledger.PREDICTION_LEDGER_COLUMNS}
    row.update(
        {
            "prediction_id": f"{match_id}_{snapshot_utc}",
            "snapshot_utc": snapshot_utc,
            "match_id": match_id,
            "kickoff_utc": "2026-06-22T18:00:00+00:00",
            "competition": "FIFA World Cup",
            "home": home,
            "away": away,
            "home_win_prob": home_prob,
            "draw_prob": 0.25,
            "away_win_prob": max(0.0, 0.75 - float(home_prob)),
            "prediction_before_kickoff": before_kickoff,
        }
    )
    return row


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


def _bulk_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": "2026-06-22",
                "time_utc": "18:00",
                "competition": "FIFA World Cup",
                "group": "A",
                "home": "Alpha",
                "away": "Beta",
                "venue": "Test",
            },
            {
                "match_id": "m2",
                "date_utc": "2026-06-23",
                "time_utc": "18:00",
                "competition": "FIFA World Cup",
                "group": "A",
                "home": "Gamma",
                "away": "Delta",
                "venue": "Test",
            },
        ]
    )


def _provider_completed_rows(rows: list[tuple[str, str, str, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "provider": "football-data.org",
                "provider_match_id": provider_id,
                "provider_kickoff_utc": f"{date_utc}T18:00:00+00:00",
                "date_utc": date_utc,
                "competition": "FIFA World Cup",
                "group": "A",
                "home": home,
                "away": away,
                "home_score_90": home_goals,
                "away_score_90": away_goals,
                "score_source": "score.fullTime",
                "score_semantics": "90-minute regular time",
                "is_completed": 1,
                "source": "football-data.org",
                "last_updated": "2026-06-24T00:00:00+00:00",
            }
            for provider_id, date_utc, home, away, home_goals, away_goals in rows
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
