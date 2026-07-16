from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.alpha import model_probability_for_side
from src.model import ModelConfig, run_match_model


SENSITIVITY_COLUMNS = [
    "scenario",
    "home_win",
    "draw",
    "away_win",
    "home_advance",
    "away_advance",
    "over_2_5",
    "under_2_5",
    "btts_yes",
    "btts_no",
    "home_xg",
    "away_xg",
]


def default_sensitivity_configs() -> dict[str, ModelConfig]:
    return {
        "Conservative": ModelConfig(
            base_total_goals=2.40,
            elo_xg_weight=0.0028,
            attack_weight=0.50,
            defense_weight=0.50,
            form_weight=0.15,
            environment_weight=0.008,
        ),
        "Default": ModelConfig(),
        "Aggressive rating-based": ModelConfig(
            base_total_goals=2.60,
            elo_xg_weight=0.0040,
            attack_weight=0.60,
            defense_weight=0.40,
            form_weight=0.20,
            environment_weight=0.010,
        ),
    }


def run_sensitivity_analysis(
    match_row,
    teams: pd.DataFrame,
    venues: pd.DataFrame,
    market_odds: pd.DataFrame,
    model_configs: dict[str, ModelConfig] | Iterable[tuple[str, ModelConfig]] | None = None,
) -> pd.DataFrame:
    configs = model_configs or default_sensitivity_configs()
    if isinstance(configs, dict):
        items = configs.items()
    else:
        items = configs

    rows = []
    for scenario, cfg in items:
        result = run_match_model(match_row, teams, venues, market_odds, cfg)
        probs = result["probs"]
        rows.append(
            {
                "scenario": scenario,
                "home_win": probs.get("home_win", pd.NA),
                "draw": probs.get("draw", pd.NA),
                "away_win": probs.get("away_win", pd.NA),
                "home_advance": probs.get("home_advance", pd.NA),
                "away_advance": probs.get("away_advance", pd.NA),
                "over_2_5": probs.get("over_2_5", pd.NA),
                "under_2_5": probs.get("under_2_5", pd.NA),
                "btts_yes": probs.get("btts_yes", pd.NA),
                "btts_no": probs.get("btts_no", pd.NA),
                "home_xg": result.get("hxg", pd.NA),
                "away_xg": result.get("axg", pd.NA),
            }
        )
    return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)


def assess_alpha_robustness(alpha_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    if alpha_df is None or alpha_df.empty:
        return pd.DataFrame(columns=["market_id", "robustness", "supporting_scenarios"])
    if sensitivity_df is None or sensitivity_df.empty:
        out = alpha_df[["market_id"]].copy()
        out["robustness"] = "unstable"
        out["supporting_scenarios"] = ""
        return out

    rows = []
    scenario_records = sensitivity_df.to_dict("records")
    for _, alpha_row in alpha_df.iterrows():
        supporting = []
        market_price = alpha_row.get("polymarket_price_cents", pd.NA)
        if pd.isna(market_price):
            rows.append({"market_id": alpha_row.get("market_id", ""), "robustness": "unstable", "supporting_scenarios": ""})
            continue

        for scenario in scenario_records:
            prob = model_probability_for_side(scenario, alpha_row.get("model_side", ""), alpha_row.get("market_type", ""))
            if pd.isna(prob):
                continue
            if str(alpha_row.get("polymarket_side", "YES")).upper() == "NO":
                prob = 1.0 - float(prob)
            fair_cents = float(prob) * 100.0
            if fair_cents - float(market_price) > 0:
                supporting.append(str(scenario.get("scenario", "")))
        rows.append(
            {
                "market_id": alpha_row.get("market_id", ""),
                "robustness": "robust" if len(supporting) >= 2 else "unstable",
                "supporting_scenarios": ", ".join([item for item in supporting if item]),
            }
        )
    return pd.DataFrame(rows)
