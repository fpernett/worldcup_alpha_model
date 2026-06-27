from __future__ import annotations

import pandas as pd

from src.polymarket_event_registry import load_polymarket_event_registry, match_fixture_to_polymarket_registry


def test_registry_contains_colombia_portugal_slug() -> None:
    registry = load_polymarket_event_registry()
    rows = registry.loc[registry["event_slug"].astype(str) == "fifwc-col-prt-2026-06-27"]

    assert not rows.empty
    assert rows.iloc[0]["home"] == "Colombia"
    assert rows.iloc[0]["away"] == "Portugal"


def test_registry_resolves_colombia_portugal_reversed_fixture_order() -> None:
    registry = load_polymarket_event_registry()
    resolved = match_fixture_to_polymarket_registry("Portugal", "Colombia", "2026-06-27", "World Cup", registry)

    assert resolved["resolved_slug"] == "fifwc-col-prt-2026-06-27"
    assert resolved["resolution_status"] == "resolved"
    assert resolved["confidence"] == "high"
    assert resolved["team_order"] == "away-home"


def test_registry_returns_unresolved_diagnostics_when_empty() -> None:
    resolved = match_fixture_to_polymarket_registry(
        "Colombia",
        "Portugal",
        "2026-06-27",
        "World Cup",
        pd.DataFrame(),
    )

    assert resolved["resolution_status"] == "unresolved"
    assert resolved["warning"]
