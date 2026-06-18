from src.schema_registry import EXPECTED_CSV_SCHEMAS


def test_expected_schema_registry_contains_core_csvs():
    assert "data/fixtures.csv" in EXPECTED_CSV_SCHEMAS
    assert "data/team_ratings.csv" in EXPECTED_CSV_SCHEMAS
    assert "data/venues.csv" in EXPECTED_CSV_SCHEMAS
    assert "data/polymarket_markets.csv" in EXPECTED_CSV_SCHEMAS


def test_expected_schema_registry_preserves_polymarket_price_columns():
    columns = EXPECTED_CSV_SCHEMAS["data/polymarket_markets.csv"]

    assert "yes_price" in columns
    assert "no_price" in columns
    assert "liquidity" in columns
    assert "last_updated" in columns

