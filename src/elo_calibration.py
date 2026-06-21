from __future__ import annotations

from typing import Any

import pandas as pd

from src.elo import calculate_rolling_elo
from src.utils import coerce_float


def validate_rolling_elo(historical_matches_df: pd.DataFrame | None, team_ratings_df: pd.DataFrame | None) -> dict[str, Any]:
    current = current_elo_table(historical_matches_df)
    manual = _normalise_manual_ratings(team_ratings_df)
    comparison = _manual_vs_elo(current, manual)
    warnings: list[str] = []
    if current.empty:
        warnings.append("Rolling Elo table is empty; historical rows may be missing Elo inputs.")
    if comparison.empty:
        warnings.append("No overlap between rolling Elo teams and manual team ratings.")
    correlation = _correlation(comparison)
    if pd.isna(correlation):
        warnings.append("Manual-vs-Elo correlation unavailable.")
    elif correlation < 0.35:
        warnings.append(f"Manual-vs-Elo correlation is low ({correlation:.3f}); review Elo calibration.")

    disagreements = (
        comparison.loc[comparison["manual_elo_delta"].abs() >= 175].copy()
        if not comparison.empty
        else comparison.iloc[0:0].copy()
    )
    if not disagreements.empty:
        warnings.append(f"{len(disagreements)} model teams differ from rolling Elo by at least 175 points.")

    return {
        "top_elo_teams": _records(current.sort_values("rolling_elo", ascending=False).head(30)),
        "bottom_elo_teams": _records(current.sort_values("rolling_elo", ascending=True).head(30)),
        "model_team_elos": _records(comparison.sort_values("team")),
        "elo_distribution_summary": _distribution_summary(current),
        "manual_vs_elo_correlation": None if pd.isna(correlation) else round(float(correlation), 3),
        "teams_with_large_manual_elo_disagreement": _records(disagreements.sort_values("manual_elo_delta", key=lambda s: s.abs(), ascending=False)),
        "warnings": warnings,
    }


def current_elo_table(historical_matches_df: pd.DataFrame | None) -> pd.DataFrame:
    elo = calculate_rolling_elo(historical_matches_df)
    if elo.empty:
        return pd.DataFrame(columns=["team", "rolling_elo", "latest_match_date", "matches_count"])
    elo["date_utc"] = pd.to_datetime(elo["date_utc"], errors="coerce")
    elo = elo.dropna(subset=["date_utc", "team", "team_elo_post"]).copy()
    if elo.empty:
        return pd.DataFrame(columns=["team", "rolling_elo", "latest_match_date", "matches_count"])
    latest = elo.sort_values(["team", "date_utc"]).groupby("team").tail(1)
    counts = elo.groupby("team").size().rename("matches_count")
    out = latest[["team", "team_elo_post", "date_utc"]].rename(
        columns={"team_elo_post": "rolling_elo", "date_utc": "latest_match_date"}
    )
    out = out.merge(counts, on="team", how="left")
    out["latest_match_date"] = pd.to_datetime(out["latest_match_date"], errors="coerce").dt.date.astype(str)
    out["rolling_elo"] = pd.to_numeric(out["rolling_elo"], errors="coerce").round(1)
    return out.sort_values("rolling_elo", ascending=False).reset_index(drop=True)


def _normalise_manual_ratings(team_ratings_df: pd.DataFrame | None) -> pd.DataFrame:
    if team_ratings_df is None or team_ratings_df.empty or "team" not in team_ratings_df.columns:
        return pd.DataFrame(columns=["team", "manual_elo", "manual_attack", "manual_defense", "manual_recent_form"])
    out = team_ratings_df.copy()
    for col in ["elo", "attack", "defense", "recent_form"]:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[["team", "elo", "attack", "defense", "recent_form"]].copy()
    out = out.rename(
        columns={
            "elo": "manual_elo",
            "attack": "manual_attack",
            "defense": "manual_defense",
            "recent_form": "manual_recent_form",
        }
    )
    for col in ["manual_elo", "manual_attack", "manual_defense", "manual_recent_form"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["team"] = out["team"].astype(str).str.strip()
    return out.dropna(subset=["team"]).reset_index(drop=True)


def _manual_vs_elo(current: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "team",
        "manual_elo",
        "manual_attack",
        "manual_defense",
        "manual_recent_form",
        "rolling_elo",
        "latest_match_date",
        "matches_count",
        "manual_elo_delta",
    ]
    if current.empty or manual.empty:
        return pd.DataFrame(columns=columns)
    comparison = manual.merge(current, on="team", how="left")
    comparison["manual_elo_delta"] = comparison["rolling_elo"] - comparison["manual_elo"]
    for col in columns:
        if col not in comparison.columns:
            comparison[col] = pd.NA
    return comparison[columns].copy()


def _correlation(comparison: pd.DataFrame) -> float:
    if comparison.empty or comparison[["manual_elo", "rolling_elo"]].dropna().shape[0] < 3:
        return float("nan")
    return float(comparison["manual_elo"].corr(comparison["rolling_elo"]))


def _distribution_summary(current: pd.DataFrame) -> dict[str, float | int | None]:
    if current.empty:
        return {"team_count": 0}
    values = pd.to_numeric(current["rolling_elo"], errors="coerce").dropna()
    if values.empty:
        return {"team_count": len(current)}
    return {
        "team_count": int(len(values)),
        "mean": round(float(values.mean()), 3),
        "median": round(float(values.median()), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
        "std": round(float(values.std(ddof=0)), 3),
    }


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: None if pd.isna(value) else round(coerce_float(value), 3))
    return out.where(pd.notna(out), None).to_dict("records")
