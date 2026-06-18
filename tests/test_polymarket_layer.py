from __future__ import annotations

import pandas as pd

from src.alpha import calculate_polymarket_alpha
from src.backtesting import load_prediction_log, save_prediction_snapshot
from src.polymarket import POLYMARKET_COLUMNS, get_polymarket_markets, normalize_price
from src.sensitivity import run_sensitivity_analysis
from src.model import ModelConfig


def sample_model_result() -> dict:
    return {
        "match_id": "M1",
        "home": "England",
        "away": "Croatia",
        "hxg": 1.4,
        "axg": 1.0,
        "probs": {
            "home_win": 0.491,
            "draw": 0.287,
            "away_win": 0.222,
            "over_2_5": 0.46,
            "under_2_5": 0.54,
            "btts_yes": 0.50,
            "btts_no": 0.50,
        },
    }


def sample_mapping(polymarket_side: str = "YES") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "M1",
                "home": "England",
                "away": "Croatia",
                "market_id": "PM1",
                "question": "Will England beat Croatia?",
                "market_type": "match_winner_home",
                "model_side": "home_win",
                "polymarket_side": polymarket_side,
                "mapping_confidence": "high",
                "mapping_reason": "test",
                "manual_confirmed": True,
                "yes_price": 0.42,
                "no_price": 0.58,
                "liquidity": 500,
                "volume": 1000,
                "closed": 0,
            }
        ]
    )


def sample_match() -> pd.Series:
    return pd.Series(
        {
            "match_id": "M1",
            "date_utc": "2026-06-17",
            "time_utc": "20:00",
            "home": "England",
            "away": "Croatia",
            "venue": "Dallas Stadium",
        }
    )


def sample_teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "England", "elo": 1980, "attack": 0.82, "defense": 0.80, "recent_form": 0.78, "training_temp_c": 15, "training_humidity_pct": 70},
            {"team": "Croatia", "elo": 1855, "attack": 0.70, "defense": 0.74, "recent_form": 0.67, "training_temp_c": 18, "training_humidity_pct": 65},
        ]
    )


def sample_venues() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"venue": "Dallas Stadium", "city": "Arlington", "country": "USA", "latitude": 32.7473, "longitude": -97.0945, "altitude_m": 190, "temp_c": 22, "humidity_pct": 45, "wind_kmh": 0, "precipitation_mm": 0, "roof_expected_closed": 1, "neutral_site": 1}
        ]
    )


def test_polymarket_price_normalization() -> None:
    assert normalize_price(42) == 0.42
    assert normalize_price(0.42) == 0.42
    assert pd.isna(normalize_price(142))


def test_yes_no_mapping_probability() -> None:
    yes_alpha = calculate_polymarket_alpha(sample_model_result(), sample_mapping("YES"))
    no_alpha = calculate_polymarket_alpha(sample_model_result(), sample_mapping("NO"))
    assert abs(float(yes_alpha.iloc[0]["model_probability"]) - 0.491) < 1e-9
    assert abs(float(no_alpha.iloc[0]["model_probability"]) - 0.509) < 1e-9


def test_alpha_gap_calculation() -> None:
    alpha = calculate_polymarket_alpha(sample_model_result(), sample_mapping("YES"))
    assert round(float(alpha.iloc[0]["fair_price_cents"]), 1) == 49.1
    assert round(float(alpha.iloc[0]["polymarket_price_cents"]), 1) == 42.0
    assert round(float(alpha.iloc[0]["alpha_gap_cents"]), 1) == 7.1


def test_sensitivity_output_shape() -> None:
    out = run_sensitivity_analysis(sample_match(), sample_teams(), sample_venues(), pd.DataFrame(), {"Default": ModelConfig()})
    assert len(out) == 1
    for col in ["scenario", "home_win", "draw", "away_win", "over_2_5", "under_2_5", "btts_yes", "btts_no", "home_xg", "away_xg"]:
        assert col in out.columns


def test_prediction_log_append(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.storage.DATA_DIR", tmp_path)
    alpha = calculate_polymarket_alpha(sample_model_result(), sample_mapping("YES"))
    save_prediction_snapshot(sample_match(), sample_model_result(), alpha)
    log = load_prediction_log()
    assert len(log) == 1
    assert log.iloc[0]["market_id"] == "PM1"


def test_missing_polymarket_api_falls_back_to_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("POLYMARKET_API_URL", raising=False)
    monkeypatch.delenv("POLYMARKET_GAMMA_API_URL", raising=False)
    monkeypatch.delenv("POLYMARKET_CLOB_API_URL", raising=False)
    monkeypatch.setattr("src.storage.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.polymarket.DATA_DIR", tmp_path)
    pd.DataFrame(columns=POLYMARKET_COLUMNS).to_csv(tmp_path / "polymarket_markets.csv", index=False)
    markets = get_polymarket_markets()
    assert list(markets.columns) == POLYMARKET_COLUMNS
    assert markets.attrs.get("source_label") == "local CSV"
