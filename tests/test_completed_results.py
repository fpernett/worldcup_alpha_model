from __future__ import annotations

import pandas as pd

from src import completed_results


def test_completed_results_fetches_and_persists_provider_rows(monkeypatch, tmp_path) -> None:
    raw = [
        {
            "id": 1001,
            "utcDate": "2026-06-29T17:00:00Z",
            "competition": {"name": "FIFA World Cup"},
            "homeTeam": {"name": "Brazil"},
            "awayTeam": {"name": "Japan"},
            "score": {"fullTime": {"home": 1, "away": 0}},
            "status": "FINISHED",
        }
    ]
    monkeypatch.setattr(completed_results, "fetch_finished_matches", lambda *_args, **_kwargs: (raw, "API", ""))
    monkeypatch.setattr(completed_results, "write_dataframe_cache", lambda df, filename, source: tmp_path / filename)

    path = tmp_path / "completed_results.csv"
    rows, diagnostics = completed_results.load_completed_results(
        "2026-06-29",
        "2026-06-29",
        teams=["Brazil", "Japan"],
        force_refresh=True,
        persist=True,
        path=path,
    )

    assert diagnostics["status"] == "success"
    assert diagnostics["completed_rows_loaded"] == 1
    assert path.exists()
    assert rows.iloc[0]["home"] == "Brazil"
    assert int(rows.iloc[0]["home_score_90"]) == 1


def test_completed_results_prefers_regular_time_over_extra_time_and_penalties() -> None:
    rows = completed_results.normalise_provider_completed_results(
        [
            {
                "id": 1002,
                "utcDate": "2026-06-29T20:30:00Z",
                "competition": {"name": "FIFA World Cup"},
                "homeTeam": {"name": "Alpha"},
                "awayTeam": {"name": "Beta"},
                "score": {
                    "regularTime": {"home": 1, "away": 1},
                    "fullTime": {"home": 2, "away": 1},
                    "extraTime": {"home": 1, "away": 0},
                    "penalties": {"home": 4, "away": 3},
                },
            }
        ],
        provider="football-data.org",
    )

    row = rows.iloc[0]
    assert int(row["home_score_90"]) == 1
    assert int(row["away_score_90"]) == 1
    assert row["score_source"] == "score.regularTime"
    assert row["score_semantics"] == "90-minute regular time"


def test_completed_results_empty_diagnostics_when_no_provider_cache_or_csv(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        completed_results,
        "fetch_finished_matches",
        lambda *_args, **_kwargs: ([], "fallback", "FOOTBALL_API_KEY is not configured"),
    )
    monkeypatch.setattr(completed_results, "read_dataframe_cache", lambda *_args, **_kwargs: None)

    rows, diagnostics = completed_results.load_completed_results(
        "2026-06-29",
        "2026-06-29",
        force_refresh=True,
        persist=True,
        path=tmp_path / "completed_results.csv",
    )

    assert rows.empty
    assert diagnostics["status"] == "completed_source_empty"
    assert diagnostics["provider_checked"] == "football-data.org /matches?status=FINISHED"
    assert diagnostics["cache_path_checked"] == "data/cache/completed_results_latest.csv"


def test_completed_results_to_backtest_matches_uses_90_minute_scores() -> None:
    rows = pd.DataFrame(
        [
            {
                "provider": "test",
                "provider_match_id": "p1",
                "provider_kickoff_utc": "2026-06-29T17:00:00+00:00",
                "date_utc": "2026-06-29",
                "competition": "FIFA World Cup",
                "group": "Round of 32",
                "home": "Brazil",
                "away": "Japan",
                "home_score_90": 1,
                "away_score_90": 1,
                "score_source": "score.regularTime",
                "score_semantics": "90-minute regular time",
                "is_completed": 1,
                "source": "test",
                "last_updated": "2026-06-29T20:00:00+00:00",
            }
        ]
    )

    matches = completed_results.completed_results_to_backtest_matches(rows)

    assert int(matches.iloc[0]["home_goals"]) == 1
    assert int(matches.iloc[0]["away_goals"]) == 1
