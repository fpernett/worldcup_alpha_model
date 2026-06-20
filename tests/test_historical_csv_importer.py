from __future__ import annotations

import pandas as pd
import pytest

from src.historical_csv_importer import (
    HistoricalCsvImportError,
    convert_results_csv_to_historical,
    import_historical_csv,
)


def source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "home_team": "USA",
                "away_team": "Korea Republic",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "Austin",
                "country": "United States",
                "neutral": False,
            },
            {
                "date": "2026-06-04",
                "home_team": "England",
                "away_team": "Croatia",
                "home_score": 1,
                "away_score": 1,
                "tournament": "UEFA Nations League",
                "city": "London",
                "country": "England",
                "neutral": False,
            },
        ]
    )


def test_import_creates_two_rows_per_match() -> None:
    converted, diagnostics = convert_results_csv_to_historical(source_rows(), source="unit_test")

    assert len(converted) == 4
    assert set(converted["team"]) >= {"United States", "South Korea"}
    assert diagnostics["unknown_team_names"] == []


def test_optional_columns_can_be_missing() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "home_team": "England",
                "away_team": "Croatia",
                "home_score": 1,
                "away_score": 0,
            }
        ]
    )

    converted, _ = convert_results_csv_to_historical(df, source="unit_test")

    assert len(converted) == 2
    assert converted["competition"].eq("Unknown").all()
    assert "venue" in converted.columns


def test_team_aliases_work() -> None:
    converted, _ = convert_results_csv_to_historical(source_rows().head(1), source="unit_test")

    assert set(converted["team"]) == {"United States", "South Korea"}


def test_duplicate_imports_do_not_duplicate_rows(tmp_path) -> None:
    source_path = tmp_path / "results.csv"
    output_path = tmp_path / "historical_matches.csv"
    source_rows().to_csv(source_path, index=False)

    first = import_historical_csv(
        source_path,
        source="unit_test",
        output_path=output_path,
        existing_path=output_path,
        write_cache=False,
    )
    second = import_historical_csv(
        source_path,
        source="unit_test",
        output_path=output_path,
        existing_path=output_path,
        write_cache=False,
    )

    assert first["final_historical_rows"] == 4
    assert second["final_historical_rows"] == 4
    assert second["rows_added"] == 0
    assert second["rows_deduplicated"] == 4


def test_manual_rows_are_preserved(tmp_path) -> None:
    source_path = tmp_path / "results.csv"
    output_path = tmp_path / "historical_matches.csv"
    source_rows().head(1).to_csv(source_path, index=False)
    manual = pd.DataFrame(
        [
            {
                "match_id": "csv_2026_06_01_united_states_south_korea_2_1_friendly",
                "date_utc": "2026-06-01",
                "competition": "Manual",
                "competition_type": "other",
                "team": "United States",
                "opponent": "South Korea",
                "team_goals": 9,
                "opponent_goals": 1,
                "source": "manual",
                "last_updated": "2026-06-02T00:00:00+00:00",
            }
        ]
    )
    manual.to_csv(output_path, index=False)

    import_historical_csv(
        source_path,
        source="unit_test",
        output_path=output_path,
        existing_path=output_path,
        write_cache=False,
    )
    out = pd.read_csv(output_path)

    manual_row = out.loc[(out["team"] == "United States") & (out["opponent"] == "South Korea")].iloc[0]
    assert manual_row["source"] == "manual"
    assert manual_row["team_goals"] == 9


def test_behavior_rebuild_works(tmp_path) -> None:
    source_path = tmp_path / "results.csv"
    output_path = tmp_path / "historical_matches.csv"
    behavior_path = tmp_path / "team_behavior.csv"
    source_rows().to_csv(source_path, index=False)

    diagnostics = import_historical_csv(
        source_path,
        source="unit_test",
        teams=["United States", "South Korea", "England", "Croatia"],
        rebuild_behavior=True,
        output_path=output_path,
        existing_path=output_path,
        behavior_path=behavior_path,
        write_cache=False,
    )

    assert diagnostics["final_team_behavior_rows"] == 4
    behavior = pd.read_csv(behavior_path)
    assert set(behavior["team"]) == {"United States", "South Korea", "England", "Croatia"}


def test_missing_input_columns_produce_clear_errors() -> None:
    df = pd.DataFrame([{"date": "2026-06-01", "home_team": "England"}])

    with pytest.raises(HistoricalCsvImportError, match="Missing required source column"):
        convert_results_csv_to_historical(df, source="unit_test")
