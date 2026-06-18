from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_goal_distribution_chart(goal_distribution: pd.DataFrame, home: str, away: str) -> go.Figure:
    df = goal_distribution.copy() if goal_distribution is not None else pd.DataFrame()
    for col in ["goals", "home_probability", "away_probability"]:
        if col not in df.columns:
            df[col] = 0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["goals"],
            y=100 * pd.to_numeric(df["home_probability"], errors="coerce").fillna(0.0),
            mode="lines+markers",
            name=home,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["goals"],
            y=100 * pd.to_numeric(df["away_probability"], errors="coerce").fillna(0.0),
            mode="lines+markers",
            name=away,
        )
    )
    fig.update_layout(
        title="Goal Distribution",
        xaxis_title="Goals",
        yaxis_title="Probability %",
        legend_title="Team",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_match_outcome_donut(probs: dict[str, Any], home: str, away: str) -> go.Figure:
    values = [
        _prob(probs.get("home_win")),
        _prob(probs.get("draw")),
        _prob(probs.get("away_win")),
    ]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[f"{home} win", "Draw", f"{away} win"],
                values=values,
                hole=0.55,
                textinfo="label+percent",
                sort=False,
            )
        ]
    )
    fig.update_layout(title="Match Outcome", margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    return fig


def create_score_timeline_chart(timeline_df: pd.DataFrame) -> go.Figure:
    df = timeline_df.copy() if timeline_df is not None else pd.DataFrame()
    for col in ["minute", "home_goal_probability", "away_goal_probability", "any_goal_probability"]:
        if col not in df.columns:
            df[col] = 0
    fig = go.Figure()
    for col, label in [
        ("home_goal_probability", "Home goal by minute"),
        ("away_goal_probability", "Away goal by minute"),
        ("any_goal_probability", "Any goal by minute"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df["minute"],
                y=100 * pd.to_numeric(df[col], errors="coerce").fillna(0.0),
                mode="lines",
                name=label,
            )
        )
    fig.add_vline(x=45, line_dash="dash", line_color="gray", annotation_text="HT")
    fig.update_layout(
        title="Score Timeline",
        xaxis_title="Minute",
        yaxis_title="Cumulative probability %",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_compact_score_matrix(score_matrix: pd.DataFrame, home: str, away: str, max_goal: int = 4) -> go.Figure:
    matrix = score_matrix.copy() if score_matrix is not None else pd.DataFrame()
    for col in ["home_goals", "away_goals", "prob"]:
        if col not in matrix.columns:
            matrix[col] = 0
    matrix["home_goals"] = pd.to_numeric(matrix["home_goals"], errors="coerce")
    matrix["away_goals"] = pd.to_numeric(matrix["away_goals"], errors="coerce")
    matrix["prob"] = pd.to_numeric(matrix["prob"], errors="coerce").fillna(0.0)
    matrix = matrix.loc[(matrix["home_goals"] <= max_goal) & (matrix["away_goals"] <= max_goal)].copy()
    pivot = matrix.pivot_table(index="home_goals", columns="away_goals", values="prob", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(index=range(max_goal + 1), columns=range(max_goal + 1), fill_value=0.0)
    fig = px.imshow(
        pivot * 100,
        text_auto=".1f",
        labels={"x": f"{away} goals", "y": f"{home} goals", "color": "Probability %"},
        aspect="auto",
        color_continuous_scale="Blues",
    )
    fig.update_layout(title=f"Compact Score Matrix 0-{max_goal}", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_expected_goals_chart(expected_goals_df: pd.DataFrame) -> go.Figure:
    df = expected_goals_df.copy() if expected_goals_df is not None else pd.DataFrame()
    for col in ["team", "adjusted_xg", "lower", "upper"]:
        if col not in df.columns:
            df[col] = 0 if col != "team" else ""
    adjusted = pd.to_numeric(df["adjusted_xg"], errors="coerce").fillna(0.0)
    lower = pd.to_numeric(df["lower"], errors="coerce").fillna(adjusted)
    upper = pd.to_numeric(df["upper"], errors="coerce").fillna(adjusted)
    fig = go.Figure(
        go.Scatter(
            x=df["team"],
            y=adjusted,
            mode="markers",
            marker=dict(size=14),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(upper - adjusted).clip(lower=0),
                arrayminus=(adjusted - lower).clip(lower=0),
            ),
            name="Adjusted xG",
        )
    )
    fig.update_layout(title="Expected Goals", xaxis_title="", yaxis_title="xG", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_team_ratings_chart(ratings_df: pd.DataFrame) -> go.Figure:
    df = ratings_df.copy() if ratings_df is not None else pd.DataFrame()
    for col in ["team", "attack", "defense"]:
        if col not in df.columns:
            df[col] = 0 if col != "team" else ""
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["team"], x=df["attack"], name="Attack", orientation="h"))
    fig.add_trace(go.Bar(y=df["team"], x=df["defense"], name="Defense", orientation="h"))
    fig.update_layout(
        title="Team Ratings",
        xaxis_title="Strength 0-1",
        yaxis_title="",
        barmode="group",
        xaxis_range=[0, 1],
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _prob(value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(value_f):
        return 0.0
    return max(0.0, min(1.0, value_f))
