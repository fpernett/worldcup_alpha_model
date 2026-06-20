from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.utils import clamp, coerce_float, read_csv_with_columns


@dataclass(frozen=True)
class BehaviorConfig:
    lookback_years: int = 4
    max_matches: int = 40
    min_matches: int = 10
    half_life_days: int = 365
    friendly_weight: float = 0.45
    nations_league_weight: float = 0.90
    qualifier_weight: float = 1.15
    continental_weight: float = 1.25
    world_cup_weight: float = 1.50
    opponent_adjustment_strength: float = 0.25
    goal_contribution_cap: float = 4.0
    strong_opponent_elo: float = 1700.0
    weak_opponent_elo: float = 1350.0
    min_opponent_elo_coverage: float = 0.65
    max_behavior_blend: float = 0.30
    max_form_blend: float = 0.50

    @property
    def name(self) -> str:
        return (
            f"calibrated_{self.lookback_years}y_"
            f"{self.max_matches}m_hl{self.half_life_days}"
        )


DEFAULT_BEHAVIOR_CONFIG = BehaviorConfig()


def calculate_opponent_quality_modifier(
    opponent_elo: Any,
    baseline_elo: float = 1500,
    strength: float = 0.25,
) -> float:
    elo = coerce_float(opponent_elo, baseline_elo)
    raw = 1.0 + ((elo - baseline_elo) / 400.0) * float(strength)
    return clamp(raw, 0.75, 1.25)


def calculate_schedule_strength_metrics(
    view: pd.DataFrame | None,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> dict[str, Any]:
    if view is None or view.empty or "opponent_elo_resolved" not in view.columns:
        return _empty_schedule_strength()

    elo = pd.to_numeric(view["opponent_elo_resolved"], errors="coerce")
    coverage = float(elo.notna().mean()) if len(elo) else 0.0
    valid = elo.dropna()
    if valid.empty:
        metrics = _empty_schedule_strength()
        metrics["schedule_strength_warning"] = "Opponent Elo missing for recent matches; schedule strength unknown."
        return metrics

    strong_count = int((valid >= config.strong_opponent_elo).sum())
    weak_count = int((valid <= config.weak_opponent_elo).sum())
    mean_elo = float(valid.mean())
    label = schedule_strength_label(mean_elo, strong_count, weak_count, len(valid), coverage, config)
    warnings: list[str] = []
    if coverage < config.min_opponent_elo_coverage:
        warnings.append(f"Opponent Elo coverage is {coverage:.0%}; schedule strength is uncertain.")
    if label == "weak":
        warnings.append("Recent schedule skews toward weak opponents.")
    elif label == "strong":
        warnings.append("Recent schedule skews toward strong opponents.")
    elif label == "mixed":
        warnings.append("Recent schedule includes both strong and weak opponents.")

    return {
        "mean_opponent_elo_recent": round(mean_elo, 3),
        "median_opponent_elo_recent": round(float(valid.median()), 3),
        "min_opponent_elo_recent": round(float(valid.min()), 3),
        "max_opponent_elo_recent": round(float(valid.max()), 3),
        "opponent_elo_coverage_recent": round(coverage, 3),
        "strong_opponent_match_count": strong_count,
        "weak_opponent_match_count": weak_count,
        "schedule_strength_label": label,
        "schedule_strength_warning": " ".join(warnings),
    }


def schedule_strength_label(
    mean_elo: float,
    strong_count: int,
    weak_count: int,
    n_matches: int,
    coverage: float,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> str:
    if n_matches <= 0 or coverage < config.min_opponent_elo_coverage:
        return "unknown"
    strong_share = strong_count / max(n_matches, 1)
    weak_share = weak_count / max(n_matches, 1)
    if strong_share >= 0.25 and weak_share >= 0.25:
        return "mixed"
    if mean_elo >= 1625 or strong_share >= 0.35:
        return "strong"
    if mean_elo <= 1450 or weak_share >= 0.35:
        return "weak"
    return "average"


def competition_weight_for_type(competition_type: str, config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG) -> float:
    key = str(competition_type or "other").strip().lower().replace("-", "_")
    key = " ".join(key.replace("_", " ").split())
    if key in {"world cup", "fifa world cup", "world_cup"}:
        return config.world_cup_weight
    if key in {"world cup qualifier", "world cup qualification", "qualifier", "world_cup_qualifier"}:
        return config.qualifier_weight
    if key in {
        "continental tournament",
        "continental_tournament",
        "euro",
        "copa america",
        "afcon",
        "asian cup",
        "gold cup",
    }:
        return config.continental_weight
    if key in {"nations league", "nations_league"}:
        return config.nations_league_weight
    if key == "friendly":
        return config.friendly_weight
    return 0.80


def filter_recent_team_matches(
    matches_df: pd.DataFrame | None,
    team: str,
    reference_date: Any,
    config: BehaviorConfig = DEFAULT_BEHAVIOR_CONFIG,
) -> pd.DataFrame:
    if matches_df is None or matches_df.empty or "team" not in matches_df.columns:
        return pd.DataFrame()
    out = matches_df.loc[matches_df["team"].astype(str).str.lower() == str(team).lower()].copy()
    if out.empty:
        return out

    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce")
    ref = pd.Timestamp(reference_date)
    if ref.tzinfo is not None:
        ref = ref.tz_convert("UTC").tz_localize(None)
    window_start = ref.normalize() - pd.DateOffset(years=int(config.lookback_years))
    out = out.loc[(out["date_utc"].notna()) & (out["date_utc"] >= window_start) & (out["date_utc"] <= ref.normalize())]
    if out.empty:
        return out
    return out.sort_values("date_utc", ascending=False).head(int(config.max_matches)).reset_index(drop=True)


def reliability_grade(n_recent: int, latest_match: Any, reference_date: Any) -> tuple[str, str, str]:
    latest = pd.Timestamp(latest_match) if latest_match not in {"", None} and not pd.isna(latest_match) else pd.NaT
    ref = pd.Timestamp(reference_date)
    if ref.tzinfo is not None:
        ref = ref.tz_convert("UTC").tz_localize(None)
    if pd.isna(latest):
        staleness_days = None
    else:
        if latest.tzinfo is not None:
            latest = latest.tz_convert("UTC").tz_localize(None)
        staleness_days = max((ref.normalize() - latest.normalize()).days, 0)

    sample_warning = "" if n_recent >= 8 else "Fewer than 8 recent matches; behavior is insufficient."
    staleness_warning = ""
    if staleness_days is None:
        staleness_warning = "No recent match date available."
    elif staleness_days > 365:
        staleness_warning = f"Latest match is {staleness_days} days before reference date."

    if n_recent >= 25 and staleness_days is not None and staleness_days <= 180:
        return "high", sample_warning, staleness_warning
    if n_recent >= 15 and staleness_days is not None and staleness_days <= 365:
        return "moderate", sample_warning, staleness_warning
    if n_recent >= 8:
        return "low", sample_warning, staleness_warning
    return "insufficient", sample_warning, staleness_warning


def build_opponent_quality_map(
    matches_df: pd.DataFrame | None,
    ratings_df: pd.DataFrame | None = None,
    baseline_elo: float = 1500,
) -> dict[str, float]:
    quality: dict[str, float] = {}
    ratings = ratings_df if ratings_df is not None else read_csv_with_columns(DATA_DIR / "team_ratings.csv", ["team", "elo"])
    if ratings is not None and not ratings.empty and {"team", "elo"}.issubset(ratings.columns):
        for _, row in ratings.iterrows():
            team = str(row.get("team", "") or "").strip()
            if team:
                quality[team.lower()] = coerce_float(row.get("elo"), baseline_elo)

    if matches_df is None or matches_df.empty or "team" not in matches_df.columns:
        return quality

    df = matches_df.copy()
    if "date_utc" in df.columns:
        df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
        latest = df["date_utc"].max()
        if not pd.isna(latest):
            df = df.loc[df["date_utc"] >= latest - pd.DateOffset(years=4)].copy()
    for col in ["team_goals", "opponent_goals"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["team", "team_goals", "opponent_goals"])
    if df.empty:
        return quality

    grouped = df.groupby(df["team"].astype(str))
    for team, rows in grouped:
        key = team.lower()
        if key in quality:
            continue
        n = len(rows)
        if n < 5:
            continue
        wins = (rows["team_goals"] > rows["opponent_goals"]).mean()
        draws = (rows["team_goals"] == rows["opponent_goals"]).mean()
        ppg = wins * 3.0 + draws
        gd = (rows["team_goals"] - rows["opponent_goals"]).mean()
        estimated = baseline_elo + (ppg - 1.25) * 240.0 + gd * 80.0
        quality[key] = clamp(estimated, 1150.0, 2050.0)
    return quality


def warning_text(*warnings: str) -> str:
    return "; ".join([warning for warning in warnings if str(warning or "").strip()])


def _empty_schedule_strength() -> dict[str, Any]:
    return {
        "mean_opponent_elo_recent": float("nan"),
        "median_opponent_elo_recent": float("nan"),
        "min_opponent_elo_recent": float("nan"),
        "max_opponent_elo_recent": float("nan"),
        "opponent_elo_coverage_recent": 0.0,
        "strong_opponent_match_count": 0,
        "weak_opponent_match_count": 0,
        "schedule_strength_label": "unknown",
        "schedule_strength_warning": "Opponent Elo missing for recent matches; schedule strength unknown.",
    }
