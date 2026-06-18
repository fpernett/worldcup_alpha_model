from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.utils import coerce_float


def calculate_score_timeline(
    home_xg: float,
    away_xg: float,
    minutes: Iterable[int] = range(0, 91, 5),
) -> pd.DataFrame:
    rows = []
    hxg = max(0.0, coerce_float(home_xg, 0.0))
    axg = max(0.0, coerce_float(away_xg, 0.0))
    for minute in minutes:
        minute_i = int(minute)
        time_fraction = min(max(minute_i / 90.0, 0.0), 1.0)
        late_weighted_fraction = 0.85 * time_fraction + 0.15 * time_fraction**2
        rows.append(
            {
                "minute": minute_i,
                "home_goal_probability": float(1 - np.exp(-hxg * late_weighted_fraction)),
                "away_goal_probability": float(1 - np.exp(-axg * late_weighted_fraction)),
                "any_goal_probability": float(1 - np.exp(-(hxg + axg) * late_weighted_fraction)),
            }
        )
    return pd.DataFrame(rows)
