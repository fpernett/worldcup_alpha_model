from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from src.alpha import calculate_polymarket_alpha
from src.backtesting import evaluate_predictions, load_prediction_log, load_results_log, save_prediction_snapshot
from src.climate import get_team_training_climate, get_venue_environment, load_venues
from src.config import api_summary
from src.data_sources import get_upcoming_fixtures, update_all_sources
from src.market_mapping import map_match_to_polymarket_markets, mapping_status
from src.model import ModelConfig, fair_odds, run_match_model
from src.odds import load_market_odds
from src.polymarket import get_polymarket_markets, update_polymarket_markets
from src.ratings import get_team_ratings
from src.sensitivity import assess_alpha_robustness, run_sensitivity_analysis


st.set_page_config(page_title="World Cup Alpha Model", layout="wide")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def odds_fmt(value: float) -> str:
    if pd.isna(value):
        return ""
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def alpha_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_prob"] = out["model_prob"].map(pct)
    out["fair_odds"] = out["fair_odds"].map(odds_fmt)
    out["market_odds"] = out["market_odds"].map(odds_fmt)
    out["alpha_ev"] = out["alpha_ev"].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
    return out


def polymarket_alpha_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["model_probability"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
    for col in ["fair_price_cents", "polymarket_price_cents", "alpha_gap_cents"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    if "alpha_ev" in out.columns:
        out["alpha_ev"] = out["alpha_ev"].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
    return out

def build_source_status(fixtures, teams, venues, odds) -> pd.DataFrame:
    apis = api_summary()

    def source_label(df, fallback):
        try:
            return df.attrs.get("source_label", fallback)
        except Exception:
            return fallback

    def last_updated(df):
        try:
            return df.attrs.get("last_updated", "")
        except Exception:
            return ""

    def warning(df):
        try:
            return df.attrs.get("warning", "")
        except Exception:
            return ""

    def safe_len(df):
        try:
            return len(df) if df is not None else 0
        except Exception:
            return 0

    def missing_columns(df, required_cols):
        if df is None:
            return ", ".join(required_cols)
        try:
            return ", ".join([c for c in required_cols if c not in df.columns])
        except Exception:
            return ", ".join(required_cols)

    diagnostics = [
        {
            "input": "fixtures",
            "source": source_label(fixtures, "local CSV"),
            "rows": safe_len(fixtures),
            "api_configured": apis.get("fixtures", False),
            "last_updated": last_updated(fixtures),
            "warning": warning(fixtures),
            "missing_columns": missing_columns(
                fixtures,
                [
                    "match_id",
                    "date_utc",
                    "time_utc",
                    "competition",
                    "group",
                    "home",
                    "away",
                    "venue",
                    "city",
                    "country",
                ],
            ),
        },
        {
            "input": "team_ratings",
            "source": source_label(teams, "local CSV"),
            "rows": safe_len(teams),
            "api_configured": apis.get("ratings", False),
            "last_updated": last_updated(teams),
            "warning": warning(teams),
            "missing_columns": missing_columns(
                teams,
                [
                    "team",
                    "elo",
                    "attack",
                    "defense",
                    "recent_form",
                    "fifa_rank_proxy",
                    "training_temp_c",
                    "training_humidity_pct",
                    "data_quality",
                    "last_updated",
                    "notes",
                ],
            ),
        },
        {
            "input": "venues",
            "source": source_label(venues, "local CSV"),
            "rows": safe_len(venues),
            "api_configured": apis.get("weather", False),
            "last_updated": last_updated(venues),
            "warning": warning(venues),
            "missing_columns": missing_columns(
                venues,
                [
                    "venue",
                    "city",
                    "country",
                    "latitude",
                    "longitude",
                    "altitude_m",
                    "temp_c",
                    "humidity_pct",
                    "wind_kmh",
                    "precipitation_mm",
                    "roof_expected_closed",
                    "neutral_site",
                ],
            ),
        },
        {
            "input": "market_odds",
            "source": source_label(odds, "local CSV"),
            "rows": safe_len(odds),
            "api_configured": apis.get("odds", False),
            "last_updated": last_updated(odds),
            "warning": warning(odds),
            "missing_columns": missing_columns(
                odds,
                [
                    "match_id",
                    "market",
                    "selection",
                    "odds",
                    "source",
                    "last_updated",
                ],
            ),
        },
    ]

    for row in diagnostics:
        if row["missing_columns"]:
            row["warning"] = "Missing required columns"

    return pd.DataFrame(diagnostics)

@st.cache_data(ttl=600)
def load_inputs(start_date: date, end_date: date, refresh_counter: int):
    fixtures = get_upcoming_fixtures(start_date, end_date)
    teams = get_team_ratings()
    venues = load_venues()
    odds = load_market_odds()
    status = build_source_status(fixtures, teams, venues, odds)
    return fixtures, teams, venues, odds, status


@st.cache_data(ttl=600)
def load_polymarket_inputs(query: str, use_cache: bool, refresh_counter: int):
    return get_polymarket_markets(query=query or None, use_cache=use_cache)


st.title("FIFA World Cup Match Alpha Model")
st.caption(
    "Transparent local model for statistical alpha estimates. It estimates probabilities, fair odds, scorelines, "
    "environmental effects, and alpha EV versus optional market odds. It does not provide staking or investment advice."
)

if "refresh_counter" not in st.session_state:
    st.session_state["refresh_counter"] = 0
if "polymarket_refresh_counter" not in st.session_state:
    st.session_state["polymarket_refresh_counter"] = 0

with st.sidebar:
    st.header("Fixture Window")
    anchor_date = st.date_input("Date selector", value=date.today())
    window = st.radio("Quick range", ["Today", "Tomorrow", "Date range"], index=0)
    if window == "Today":
        start_date = anchor_date
        end_date = anchor_date
    elif window == "Tomorrow":
        start_date = anchor_date + timedelta(days=1)
        end_date = start_date
    else:
        selected_range = st.date_input("Custom range", value=(anchor_date, anchor_date + timedelta(days=3)))
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date = anchor_date
            end_date = anchor_date

    st.header("Data")
    if st.button("Update API data", width="stretch"):
        with st.spinner("Refreshing configured APIs and local caches..."):
            summary = update_all_sources(start_date, end_date)
        st.session_state["refresh_counter"] += 1
        st.cache_data.clear()
        st.session_state["last_update_summary"] = summary

    if "last_update_summary" in st.session_state:
        st.caption(f"Last update: {st.session_state['last_update_summary'].get('updated_at', '')}")
        for step in st.session_state["last_update_summary"].get("steps", []):
            status = step.get("status", "")
            name = step.get("name", "source")
            message = step.get("message") or f"{step.get('rows', 0)} row(s); source: {step.get('source', '')}"
            if status == "success":
                st.success(f"{name}: {message}")
            elif status in {"partial", "skipped"}:
                st.warning(f"{name}: {message}")
            elif status == "failed":
                st.error(f"{name}: {message}")
            if step.get("warning"):
                st.warning(f"{name}: {step['warning']}")
            for error in step.get("errors", [])[:3]:
                st.warning(f"{name}: {error}")

    st.header("Polymarket")
    polymarket_query = st.text_input("Market search query", value="World Cup")
    use_cached_polymarket = st.checkbox("Use cached markets", value=True)
    min_liquidity = st.number_input("Minimum liquidity", min_value=0.0, value=0.0, step=25.0)
    min_alpha_gap = st.slider("Minimum alpha gap (cents)", 0.0, 20.0, 0.0, 0.5)
    show_only_mapped = st.toggle("Show only mapped markets", value=True)

    if st.button("Update Polymarket markets", width="stretch"):
        with st.spinner("Refreshing Polymarket markets if an API is configured..."):
            pm_summary = update_polymarket_markets(polymarket_query or None)
        st.session_state["polymarket_refresh_counter"] += 1
        st.cache_data.clear()
        st.session_state["last_polymarket_update"] = pm_summary

    if "last_polymarket_update" in st.session_state:
        pm_summary = st.session_state["last_polymarket_update"]
        message = pm_summary.get("message", "")
        if pm_summary.get("status") == "success":
            st.success(message)
        else:
            st.warning(message)
        if pm_summary.get("warning"):
            st.warning(pm_summary["warning"])

    st.header("Model Parameters")
    base_total = st.slider("Base total goals", 1.8, 3.2, 2.50, 0.05)
    elo_weight = st.slider(
    "Elo xG weight",
    0.0010,
    0.0060,
    0.0032,
    0.0001,
    format="%.4f",
)
    attack_weight = st.slider("Attack weight", 0.20, 0.90, 0.55, 0.05)
    defense_weight = st.slider("Defense weight", 0.20, 0.90, 0.45, 0.05)
    form_weight = st.slider("Recent form weight", 0.00, 0.40, 0.20, 0.02)
    env_weight = st.slider("Environment weight", 0.000, 0.030, 0.010, 0.001)

fixtures, teams, venues, market_odds, source_status = load_inputs(
    start_date, end_date, st.session_state["refresh_counter"]
)
polymarket_markets = load_polymarket_inputs(
    polymarket_query,
    use_cached_polymarket,
    st.session_state["polymarket_refresh_counter"],
)

with st.sidebar:
    st.header("Source Diagnostics")
    st.dataframe(
        source_status[["input", "source", "rows", "api_configured", "last_updated", "warning"]],
        hide_index=True,
        width="stretch",
    )
    st.caption("Source labels are API, cache, or local CSV. API keys are optional; failed requests fall back safely.")
    for _, source_row in source_status.iterrows():
        if source_row.get("missing_columns"):
            st.warning(f"{source_row['input']} missing columns: {source_row['missing_columns']}")
    st.caption(
        "Polymarket markets: "
        f"{len(polymarket_markets)} rows from {polymarket_markets.attrs.get('source_label', 'unknown source')}."
    )
    if polymarket_markets.attrs.get("warning"):
        st.warning(polymarket_markets.attrs["warning"])

st.subheader("Available Matches")
if fixtures.empty:
    st.warning("No fixtures found for the selected window. Add rows to `data/fixtures.csv` or configure a fixture API.")
    st.stop()

fixture_view = fixtures.copy()
fixture_view["match_label"] = (
    fixture_view["date_utc"].astype(str)
    + " "
    + fixture_view["time_utc"].astype(str)
    + " UTC - "
    + fixture_view["home"].astype(str)
    + " vs "
    + fixture_view["away"].astype(str)
)

st.dataframe(
    fixture_view[["match_id", "date_utc", "time_utc", "competition", "group", "home", "away", "venue", "city"]],
    width="stretch",
    hide_index=True,
)

selected_labels = st.multiselect(
    "Select one or more games to model",
    options=fixture_view["match_label"].tolist(),
    default=[],
)

if not selected_labels:
    st.info("Select a match above to run the model.")
    st.stop()

cfg = ModelConfig(
    base_total_goals=base_total,
    environment_weight=env_weight,
    elo_xg_weight=elo_weight,
    attack_weight=attack_weight,
    defense_weight=defense_weight,
    form_weight=form_weight,
)

for label in selected_labels:
    match = fixture_view.loc[fixture_view["match_label"] == label].iloc[0]
    result = run_match_model(match, teams, venues, market_odds, cfg)

    probs = result["probs"]
    alpha = result["alpha"].copy()

    if "odds_source" not in alpha.columns:
        alpha["odds_source"] = ""

    if "odds_last_updated" not in alpha.columns:
        alpha["odds_last_updated"] = ""

    confidence = result["confidence"]
    mapped_markets = map_match_to_polymarket_markets(match, polymarket_markets)
    polymarket_alpha = calculate_polymarket_alpha(result, mapped_markets, min_liquidity=min_liquidity)
    sensitivity_df = run_sensitivity_analysis(match, teams, venues, market_odds)
    robustness_df = assess_alpha_robustness(polymarket_alpha, sensitivity_df)
    if not polymarket_alpha.empty and not robustness_df.empty:
        polymarket_alpha = polymarket_alpha.merge(robustness_df, on="market_id", how="left")

    filtered_polymarket_alpha = polymarket_alpha.copy()
    if min_liquidity > 0 and not filtered_polymarket_alpha.empty:
        filtered_polymarket_alpha = filtered_polymarket_alpha.loc[
            pd.to_numeric(filtered_polymarket_alpha["liquidity"], errors="coerce").fillna(-1) >= min_liquidity
        ]
    if min_alpha_gap > 0 and not filtered_polymarket_alpha.empty:
        filtered_polymarket_alpha = filtered_polymarket_alpha.loc[
            pd.to_numeric(filtered_polymarket_alpha["alpha_gap_cents"], errors="coerce").fillna(-999) >= min_alpha_gap
        ]

    st.divider()
    st.header(f"{result['home']} vs {result['away']}")
    st.caption(
        f"{match['competition']} | {match['group']} | "
        f"{match['date_utc']} {match['time_utc']} UTC | {match['venue']}"
    )

    tabs = st.tabs(
        [
            "Summary",
            "Score matrix",
            "Markets",
            "Polymarket alpha",
            "Sensitivity",
            "Backtesting",
            "Team inputs",
            "Venue/environment",
            "Model notes",
        ]
    )

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{result['home']} win", pct(probs["home_win"]), f"fair {fair_odds(probs['home_win']):.2f}")
        c2.metric("Draw", pct(probs["draw"]), f"fair {fair_odds(probs['draw']):.2f}")
        c3.metric(f"{result['away']} win", pct(probs["away_win"]), f"fair {fair_odds(probs['away_win']):.2f}")
        c4.metric("Expected goals", f"{result['hxg']:.2f} - {result['axg']:.2f}")

        st.subheader("Model Confidence")
        st.write(f"**{confidence['label']} confidence**: {', '.join(confidence['reasons'])}.")

        st.subheader("Top Alpha Signals")
        positive_alpha = alpha.loc[alpha["alpha_ev"].notna() & (alpha["alpha_ev"] > 0)].head(5)
        if positive_alpha.empty:
            st.caption("No positive alpha EV rows are available from the current market odds file.")
        else:
            st.dataframe(
                alpha_display(positive_alpha)[["market", "selection", "model_prob", "fair_odds", "market_odds", "alpha_ev"]],
                hide_index=True,
                width="stretch",
            )
        st.caption("Alpha EV = model_probability x market_decimal_odds - 1. This is a statistical estimate, not a staking instruction.")

    with tabs[1]:
        mat = result["score_matrix"].copy()
        pivot = mat.pivot(index="home_goals", columns="away_goals", values="prob") * 100
        fig = px.imshow(
            pivot,
            text_auto=".1f",
            labels={"x": f"{result['away']} goals", "y": f"{result['home']} goals", "color": "Probability %"},
            aspect="auto",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, width="stretch")

        scores = result["top_scores"].copy()
        scores["probability"] = scores["prob"].map(pct)
        st.dataframe(scores[["label", "probability"]], hide_index=True, width="stretch")

    with tabs[2]:
        st.subheader("1X2, Totals, BTTS, Handicap, and Market Alpha")
        st.dataframe(
            alpha_display(alpha)[
                ["market", "selection", "model_prob", "fair_odds", "market_odds", "alpha_ev", "odds_source", "odds_last_updated"]
            ],
            hide_index=True,
            width="stretch",
        )

    with tabs[3]:
        st.subheader("Candidate Matched Markets")
        st.caption(f"Mapping status: {mapping_status(mapped_markets)}")
        if mapped_markets.empty:
            st.warning("No Polymarket mappings found for this selected match. Add rows to `data/market_mappings.csv` or `data/polymarket_markets.csv`.")
        else:
            candidate_display = mapped_markets.copy()
            for col in ["yes_price", "no_price"]:
                if col in candidate_display.columns:
                    candidate_display[col] = candidate_display[col].map(lambda x: "" if pd.isna(x) else f"{100*float(x):.1f}")
            st.dataframe(
                candidate_display[
                    [
                        "market_id",
                        "question",
                        "market_type",
                        "model_side",
                        "polymarket_side",
                        "mapping_confidence",
                        "mapping_reason",
                        "manual_confirmed",
                        "yes_price",
                        "no_price",
                        "liquidity",
                        "volume",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        st.subheader("Polymarket Alpha")
        if filtered_polymarket_alpha.empty:
            st.info("No alpha rows match the current mapping, liquidity, and alpha-gap filters.")
        else:
            display_pm = polymarket_alpha_display(filtered_polymarket_alpha)
            st.dataframe(
                display_pm[
                    [
                        "market_id",
                        "question",
                        "market_type",
                        "polymarket_side",
                        "model_probability",
                        "fair_price_cents",
                        "polymarket_price_cents",
                        "alpha_gap_cents",
                        "alpha_ev",
                        "liquidity",
                        "volume",
                        "mapping_confidence",
                        "signal_strength",
                        "robustness",
                        "warning",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption("Model fair price and alpha gap are shown in cents. This is a statistical screen, not betting advice.")

        if not show_only_mapped:
            st.subheader("Loaded Polymarket Markets")
            st.dataframe(
                polymarket_markets.head(50)[
                    [
                        "market_id",
                        "question",
                        "slug",
                        "event_title",
                        "yes_price",
                        "no_price",
                        "liquidity",
                        "volume",
                        "source",
                        "last_updated",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        if st.button("Save prediction snapshot", key=f"save_snapshot_{match['match_id']}"):
            save_prediction_snapshot(match, result, polymarket_alpha)
            st.success(f"Saved {len(polymarket_alpha)} prediction row(s) to data/prediction_log.csv.")

    with tabs[4]:
        st.subheader("Sensitivity Analysis")
        sens_display = sensitivity_df.copy()
        for col in ["home_win", "draw", "away_win", "over_2_5", "under_2_5", "btts_yes", "btts_no"]:
            sens_display[col] = sens_display[col].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
        for col in ["home_xg", "away_xg"]:
            sens_display[col] = sens_display[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        st.dataframe(sens_display, hide_index=True, width="stretch")

        st.subheader("Alpha Robustness")
        if polymarket_alpha.empty:
            st.info("No Polymarket alpha rows to test for robustness.")
        else:
            robustness_display = polymarket_alpha[
                [
                    "market_id",
                    "question",
                    "market_type",
                    "polymarket_side",
                    "alpha_gap_cents",
                    "signal_strength",
                    "robustness",
                    "supporting_scenarios",
                ]
            ].copy()
            robustness_display["alpha_gap_cents"] = robustness_display["alpha_gap_cents"].map(
                lambda x: "" if pd.isna(x) else f"{x:.1f}"
            )
            st.dataframe(robustness_display, hide_index=True, width="stretch")
            st.caption("A signal is robust when the same Polymarket side remains positive alpha in at least 2 of 3 scenarios.")

    with tabs[5]:
        st.subheader("Backtesting")
        prediction_log = load_prediction_log()
        results_log = load_results_log()
        evaluation = evaluate_predictions(prediction_log, results_log)

        c1, c2, c3 = st.columns(3)
        c1.metric("Saved predictions", f"{len(prediction_log):,}")
        if not evaluation.empty:
            eval_row = evaluation.iloc[0]
            c2.metric("Brier score", "" if pd.isna(eval_row["brier_score"]) else f"{eval_row['brier_score']:.4f}")
            c3.metric("Log loss", "" if pd.isna(eval_row["log_loss"]) else f"{eval_row['log_loss']:.4f}")
            st.dataframe(evaluation, hide_index=True, width="stretch")
        if results_log.empty:
            st.info("Add completed match rows to `data/results_log.csv` to evaluate saved predictions.")

    with tabs[6]:
        st.subheader("Team Inputs")
        team_inputs = pd.DataFrame([result["home_inputs"], result["away_inputs"]])
        st.dataframe(
            team_inputs[
                [
                    "team",
                    "elo",
                    "attack",
                    "defense",
                    "recent_form",
                    "fifa_rank_proxy",
                    "training_temp_c",
                    "training_humidity_pct",
                    "data_quality",
                    "last_updated",
                    "notes",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        st.subheader("Training Climate")
        climates = pd.DataFrame(
            [
                {"team": result["home"], **get_team_training_climate(result["home"])},
                {"team": result["away"], **get_team_training_climate(result["away"])},
            ]
        )
        st.dataframe(climates, hide_index=True, width="stretch")

    with tabs[7]:
        st.subheader("Venue And Environment")
        env = get_venue_environment(match["venue"], match["date_utc"], match["time_utc"])
        env_df = pd.DataFrame([env])
        st.dataframe(
            env_df[
                [
                    "venue",
                    "city",
                    "country",
                    "altitude_m",
                    "temp_c",
                    "humidity_pct",
                    "wind_kmh",
                    "precipitation_mm",
                    "roof_expected_closed",
                    "effective_temp_c",
                    "effective_humidity_pct",
                    "effective_wind_kmh",
                    "effective_precipitation_mm",
                    "source_label",
                    "environment_source",
                    "last_updated",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        comp = result["components"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Home env log adj", f"{comp['home_environment_log_adj']:.3f}")
        c2.metric("Away env log adj", f"{comp['away_environment_log_adj']:.3f}")
        c3.metric("Total-goals env adj", f"{comp['total_environment_log_adj']:.3f}")
        st.write(comp["environment_note"])
        st.caption(
            "Roof-closed venues dampen outdoor weather by 85%. Altitude mainly matters above about 1,200 m. "
            "Wind and precipitation mostly reduce total-goals quality, while heat/humidity mismatch affects each team slightly."
        )

    with tabs[8]:
        st.markdown(
            """
            **Assumptions**

            - Expected goals combine Elo, attack, opponent defense, recent form, and conservative environmental adjustments.
            - Scorelines come from an independent Poisson goal model and are normalized over the displayed score grid.
            - Fair odds are calculated as `1 / probability`.
            - Alpha EV is calculated as `model_probability x market_decimal_odds - 1`.
            - Manual CSV inputs remain valid overrides. Optional APIs only refresh cache files when configured.

            **Limitations**

            - This is not a black-box machine learning model and does not account for every lineup, tactical, injury, or motivation factor.
            - Missing market odds leave alpha EV blank.
            - Weather and training climates are approximate when API or recent match data are unavailable.
            - Outputs are statistical estimates only and are not staking, bet sizing, or investment recommendations.
            """
        )
