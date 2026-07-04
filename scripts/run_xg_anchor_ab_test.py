from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import OUTCOMES, brier_1x2, safe_log_loss  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.match_identity import build_match_key, normalize_match_date, normalize_team_name  # noqa: E402
from src.model import ModelConfig, expected_goals, outcome_probs, score_matrix  # noqa: E402
from src.ratings import neutral_team_rating, rating_row_for_team  # noqa: E402
from src.team_names import is_unresolved_team_slot  # noqa: E402
from src.weather import environment_from_venue_row  # noqa: E402


ANCHOR_MODES = ["unanchored_current", "anchored_base_total"]
SPECIAL_CASES = [
    ("Mexico", "Ecuador"),
    ("Germany", "Paraguay"),
    ("Netherlands", "Morocco"),
    ("Argentina", "Cape Verde"),
    ("Canada", "Morocco"),
]


def main() -> None:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fixtures = _read_csv(DATA_DIR / "fixtures.csv")
    ratings = _read_csv(DATA_DIR / "team_ratings.csv")
    venues = _read_csv(DATA_DIR / "venues.csv")
    predictions = _load_predictions()
    results = _load_results()

    match_inputs = _build_match_inputs(fixtures, predictions)
    result_index = _ResultIndex(results)

    rows: list[dict[str, Any]] = []
    for _, match in match_inputs.iterrows():
        result = result_index.match_result(match)
        latest_prediction = _latest_prediction_for_match(predictions, match)
        for mode in ANCHOR_MODES:
            rows.append(_evaluate_match_mode(match, ratings, venues, result, latest_prediction, mode))

    detail = pd.DataFrame(rows)
    summary = _summary_table(detail)
    special = _special_table(detail)

    csv_path = reports_dir / "xg_anchor_ab_test.csv"
    md_path = reports_dir / "xg_anchor_ab_test.md"
    detail.to_csv(csv_path, index=False)
    md_path.write_text(_render_report(detail, summary, special), encoding="utf-8")

    print(f"matches evaluated or diagnosed: {detail['match_id'].nunique() if not detail.empty else 0:,}")
    print(f"scored match-mode rows: {int(detail['has_result'].sum()) if 'has_result' in detail else 0:,}")
    print(f"csv: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {md_path.relative_to(PROJECT_ROOT)}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_predictions() -> pd.DataFrame:
    frames = []
    for name in ["prediction_ledger_enriched.csv", "prediction_ledger.csv"]:
        path = DATA_DIR / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "prediction_id" in out.columns:
        out = out.drop_duplicates(subset=["prediction_id"], keep="first")
    return out.reset_index(drop=True)


def _load_results() -> pd.DataFrame:
    frames = []
    ledger = _read_csv(DATA_DIR / "results_ledger.csv")
    if not ledger.empty:
        frames.append(_normalise_results_ledger(ledger))
    completed = _read_csv(DATA_DIR / "completed_results.csv")
    if not completed.empty:
        frames.append(_normalise_completed_results(completed))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.loc[out["evaluation_eligible_1x2"].map(_bool_value)].copy()
    out = out.loc[out["result_semantics"].astype(str).str.lower().str.contains("90-minute|regular time", regex=True, na=False)]
    return out.drop_duplicates(subset=["match_id", "date_utc", "home", "away"], keep="first").reset_index(drop=True)


def _normalise_results_ledger(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        home_goals = _numeric(row.get("home_goals"))
        away_goals = _numeric(row.get("away_goals"))
        rows.append(
            {
                "match_id": row.get("match_id", ""),
                "date_utc": row.get("date_utc", ""),
                "competition": row.get("competition", ""),
                "home": row.get("home", ""),
                "away": row.get("away", ""),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "actual_result": _text(row.get("actual_result")) or _actual_result(home_goals, away_goals),
                "result_semantics": row.get("result_semantics", "90-minute regular time"),
                "evaluation_eligible_1x2": row.get("evaluation_eligible_1x2", True),
                "result_source": row.get("result_source", row.get("source", "results_ledger")),
            }
        )
    return pd.DataFrame(rows)


def _normalise_completed_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        home_goals = _numeric(row.get("home_score_90"))
        away_goals = _numeric(row.get("away_score_90"))
        is_completed = _bool_value(row.get("is_completed", True))
        rows.append(
            {
                "match_id": row.get("provider_match_id", ""),
                "date_utc": row.get("date_utc", ""),
                "competition": row.get("competition", ""),
                "home": row.get("home", ""),
                "away": row.get("away", ""),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "actual_result": _actual_result(home_goals, away_goals),
                "result_semantics": row.get("score_semantics", "90-minute regular time"),
                "evaluation_eligible_1x2": is_completed,
                "result_source": row.get("source", "completed_results"),
            }
        )
    return pd.DataFrame(rows)


def _build_match_inputs(fixtures: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    if not fixtures.empty:
        for _, row in fixtures.iterrows():
            if is_unresolved_team_slot(row.get("home")) or is_unresolved_team_slot(row.get("away")):
                continue
            match = _match_from_source(row, "fixture_csv")
            key = _match_key(match)
            rows.append(match)
            seen_keys.add(key)

    for home, away in SPECIAL_CASES:
        special_key_prefix = (normalize_team_name(home), normalize_team_name(away))
        existing = any(key[0] == special_key_prefix[0] and key[1] == special_key_prefix[1] for key in seen_keys)
        if existing:
            continue
        prediction = _latest_prediction_for_pair(predictions, home, away)
        if prediction is None:
            continue
        match = _match_from_source(prediction, "prediction_snapshot")
        rows.append(match)
        seen_keys.add(_match_key(match))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["match_id", "home", "away"], keep="first").reset_index(drop=True)


def _match_from_source(row: pd.Series, source: str) -> dict[str, Any]:
    kickoff = _kickoff_from_row(row)
    date_utc = str(row.get("date_utc", "") or normalize_match_date(kickoff))
    time_utc = str(row.get("time_utc", "") or _kickoff_time(kickoff))
    return {
        "match_id": str(row.get("match_id", "") or ""),
        "kickoff_utc": kickoff,
        "date_utc": date_utc,
        "time_utc": time_utc,
        "competition": row.get("competition", ""),
        "group": row.get("group", ""),
        "home": normalize_team_name(row.get("home", "")),
        "away": normalize_team_name(row.get("away", "")),
        "venue": row.get("venue", ""),
        "match_input_source": source,
    }


def _kickoff_from_row(row: pd.Series) -> str:
    explicit = str(row.get("kickoff_utc", "") or "").strip()
    if explicit:
        return explicit
    date = str(row.get("date_utc", "") or "").strip()
    time = str(row.get("time_utc", "") or "").strip()
    if date and time:
        return f"{date}T{time}:00+00:00" if len(time) == 5 else f"{date}T{time}+00:00"
    return date


def _kickoff_time(kickoff: Any) -> str:
    parsed = pd.to_datetime(kickoff, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%H:%M")


class _ResultIndex:
    def __init__(self, results: pd.DataFrame):
        self.by_id: dict[str, pd.Series] = {}
        self.by_key: dict[tuple[str, str, str, str], pd.Series] = {}
        if results.empty:
            return
        for _, row in results.iterrows():
            match_id = str(row.get("match_id", "") or "")
            if match_id and match_id not in self.by_id:
                self.by_id[match_id] = row
            key = build_match_key(row.get("home", ""), row.get("away", ""), row.get("date_utc", ""), row.get("competition", ""))
            if key not in self.by_key:
                self.by_key[key] = row

    def match_result(self, match: pd.Series) -> dict[str, Any] | None:
        match_id = str(match.get("match_id", "") or "")
        if match_id in self.by_id:
            return _align_result_to_match(match, self.by_id[match_id], "exact_match_id")

        key = _match_key(match)
        if key in self.by_key:
            return _align_result_to_match(match, self.by_key[key], "home_away_date")

        reversed_key = build_match_key(match.get("away", ""), match.get("home", ""), match.get("kickoff_utc", ""), match.get("competition", ""))
        if reversed_key in self.by_key:
            return _align_result_to_match(match, self.by_key[reversed_key], "reversed_home_away_date")
        return None


def _align_result_to_match(match: pd.Series, result: pd.Series, join_method: str) -> dict[str, Any]:
    same_order = (
        normalize_team_name(result.get("home", "")) == normalize_team_name(match.get("home", ""))
        and normalize_team_name(result.get("away", "")) == normalize_team_name(match.get("away", ""))
    )
    reversed_order = (
        normalize_team_name(result.get("home", "")) == normalize_team_name(match.get("away", ""))
        and normalize_team_name(result.get("away", "")) == normalize_team_name(match.get("home", ""))
    )
    home_goals = _numeric(result.get("home_goals"))
    away_goals = _numeric(result.get("away_goals"))
    if reversed_order and not same_order:
        home_goals, away_goals = away_goals, home_goals
    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "actual_result": _actual_result(home_goals, away_goals),
        "result_source": result.get("result_source", ""),
        "result_semantics": result.get("result_semantics", ""),
        "result_join_method": join_method,
    }


def _evaluate_match_mode(
    match: pd.Series,
    ratings: pd.DataFrame,
    venues: pd.DataFrame,
    result: dict[str, Any] | None,
    latest_prediction: pd.Series | None,
    mode: str,
) -> dict[str, Any]:
    home_name = str(match.get("home", ""))
    away_name = str(match.get("away", ""))
    env = _local_environment(match, venues)
    home = rating_row_for_team(ratings, home_name)
    away = rating_row_for_team(ratings, away_name)
    if home.empty:
        home = neutral_team_rating(home_name)
    if away.empty:
        away = neutral_team_rating(away_name)

    cfg = ModelConfig(xg_total_anchor_mode=mode)
    hxg, axg, components = expected_goals(home, away, env, cfg)
    probs = outcome_probs(score_matrix(hxg, axg, cfg.max_goals))
    actual = result.get("actual_result") if result else ""
    actual_prob = _prob_for_actual(probs, actual)
    top_pick = max(OUTCOMES, key=lambda outcome: probs.get(outcome, 0.0))

    return {
        "row_type": "special_case" if _is_special_case(home_name, away_name) else "evaluation",
        "match_id": match.get("match_id", ""),
        "kickoff_utc": match.get("kickoff_utc", ""),
        "home": home_name,
        "away": away_name,
        "venue": match.get("venue", ""),
        "match_input_source": match.get("match_input_source", ""),
        "xg_total_anchor_mode": mode,
        "home_xg": hxg,
        "away_xg": axg,
        "total_xg": hxg + axg,
        "anchored_home_xg": components.get("anchored_home_xg"),
        "anchored_away_xg": components.get("anchored_away_xg"),
        "anchored_total_xg": components.get("anchored_total_xg"),
        "unanchored_home_xg": components.get("unanchored_home_xg"),
        "unanchored_away_xg": components.get("unanchored_away_xg"),
        "unanchored_total_xg": components.get("unanchored_total_xg"),
        "weather_xg_multiplier": components.get("weather_xg_multiplier"),
        "base_total_goals": components.get("base_total_goals"),
        "weather_adjusted_base_total": components.get("old_base_total_goals_adjusted_total"),
        "home_win_prob": probs["home_win"],
        "draw_prob": probs["draw"],
        "away_win_prob": probs["away_win"],
        "favorite_probability": max(probs["home_win"], probs["away_win"]),
        "draw_probability": probs["draw"],
        "top_pick": top_pick,
        "has_result": result is not None and actual in OUTCOMES,
        "actual_result": actual,
        "actual_home_goals_90": result.get("home_goals") if result else pd.NA,
        "actual_away_goals_90": result.get("away_goals") if result else pd.NA,
        "probability_assigned_to_actual": actual_prob,
        "brier_score": brier_1x2([probs["home_win"], probs["draw"], probs["away_win"]], actual) if actual in OUTCOMES else pd.NA,
        "log_loss": safe_log_loss(actual_prob) if pd.notna(actual_prob) else pd.NA,
        "top_pick_correct": bool(top_pick == actual) if actual in OUTCOMES else pd.NA,
        "result_source": result.get("result_source") if result else "",
        "result_join_method": result.get("result_join_method") if result else "",
        "has_prediction_snapshot": latest_prediction is not None,
        "latest_prediction_id": latest_prediction.get("prediction_id", "") if latest_prediction is not None else "",
        "latest_snapshot_utc": latest_prediction.get("snapshot_utc", "") if latest_prediction is not None else "",
        "latest_snapshot_home_prob": latest_prediction.get("home_win_prob", pd.NA) if latest_prediction is not None else pd.NA,
        "latest_snapshot_draw_prob": latest_prediction.get("draw_prob", pd.NA) if latest_prediction is not None else pd.NA,
        "latest_snapshot_away_prob": latest_prediction.get("away_win_prob", pd.NA) if latest_prediction is not None else pd.NA,
    }


def _local_environment(match: pd.Series, venues: pd.DataFrame) -> dict[str, Any]:
    venue_name = str(match.get("venue", "") or "")
    if not venues.empty and "venue" in venues.columns:
        rows = venues.loc[venues["venue"].astype(str).str.lower() == venue_name.lower()]
        if not rows.empty:
            return environment_from_venue_row(rows.iloc[0], match.get("date_utc"), match.get("time_utc"))
    return environment_from_venue_row(
        {
            "venue": venue_name,
            "city": "",
            "country": "",
            "latitude": 0.0,
            "longitude": 0.0,
            "altitude_m": 0.0,
            "temp_c": 22.0,
            "humidity_pct": 55.0,
            "wind_kmh": 0.0,
            "precipitation_mm": 0.0,
            "roof_expected_closed": 0,
            "neutral_site": 1,
        },
        match.get("date_utc"),
        match.get("time_utc"),
        environment_source="neutral_venue_fallback",
    )


def _latest_prediction_for_match(predictions: pd.DataFrame, match: pd.Series) -> pd.Series | None:
    if predictions.empty:
        return None
    match_id = str(match.get("match_id", "") or "")
    if match_id and "match_id" in predictions.columns:
        by_id = predictions.loc[predictions["match_id"].astype(str) == match_id]
        if not by_id.empty:
            return _latest_prediction(by_id)
    return _latest_prediction_for_pair(predictions, str(match.get("home", "")), str(match.get("away", "")))


def _latest_prediction_for_pair(predictions: pd.DataFrame, home: str, away: str) -> pd.Series | None:
    if predictions.empty or not {"home", "away"}.issubset(predictions.columns):
        return None
    home_key = normalize_team_name(home)
    away_key = normalize_team_name(away)
    mask = predictions["home"].map(normalize_team_name).eq(home_key) & predictions["away"].map(normalize_team_name).eq(away_key)
    rows = predictions.loc[mask]
    if rows.empty:
        return None
    return _latest_prediction(rows)


def _latest_prediction(rows: pd.DataFrame) -> pd.Series:
    out = rows.copy()
    out["_snapshot_ts"] = pd.to_datetime(out.get("snapshot_utc"), errors="coerce", utc=True)
    out = out.sort_values("_snapshot_ts", na_position="first")
    return out.iloc[-1]


def _summary_table(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    scored = detail.loc[detail["has_result"] == True].copy()  # noqa: E712
    if scored.empty:
        return pd.DataFrame(
            [
                {
                    "xg_total_anchor_mode": mode,
                    "scored_rows": 0,
                    "brier_score": pd.NA,
                    "log_loss": pd.NA,
                    "top_pick_accuracy": pd.NA,
                    "mean_favorite_probability": pd.NA,
                    "mean_draw_probability": pd.NA,
                    "mean_probability_assigned_to_actual": pd.NA,
                }
                for mode in ANCHOR_MODES
            ]
        )
    grouped = scored.groupby("xg_total_anchor_mode", dropna=False)
    return grouped.agg(
        scored_rows=("match_id", "count"),
        brier_score=("brier_score", "mean"),
        log_loss=("log_loss", "mean"),
        top_pick_accuracy=("top_pick_correct", "mean"),
        mean_favorite_probability=("favorite_probability", "mean"),
        mean_draw_probability=("draw_probability", "mean"),
        mean_probability_assigned_to_actual=("probability_assigned_to_actual", "mean"),
    ).reset_index()


def _special_table(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    cols = [
        "match_id",
        "home",
        "away",
        "xg_total_anchor_mode",
        "home_xg",
        "away_xg",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "has_result",
        "actual_result",
        "probability_assigned_to_actual",
        "brier_score",
        "has_prediction_snapshot",
        "latest_snapshot_utc",
    ]
    return detail.loc[detail["row_type"] == "special_case", cols].copy()


def _render_report(detail: pd.DataFrame, summary: pd.DataFrame, special: pd.DataFrame) -> str:
    scored_rows = int(detail["has_result"].sum()) if not detail.empty and "has_result" in detail else 0
    unique_matches = int(detail["match_id"].nunique()) if not detail.empty and "match_id" in detail else 0
    interpretation = _interpret_summary(summary)
    return "\n\n".join(
        [
            "# xG Anchor A/B Diagnostic",
            "This report is diagnostic-only. It does not change production defaults, staking, wallet, trade execution, or betting behavior.",
            f"Match inputs evaluated or diagnosed: `{unique_matches}`",
            f"Scored match-mode rows with local 90-minute results: `{scored_rows}`",
            "## Aggregate Metrics",
            _to_markdown(_rounded(summary)),
            "## Interpretation",
            interpretation,
            "## Special Diagnostic Matches",
            _to_markdown(_rounded(special)),
            "## Data Notes",
            (
                "Rows without `has_result` are included for diagnostic visibility but excluded from aggregate scoring. "
                "The script uses local CSV fixtures, ratings, venues, prediction ledgers, and result ledgers only."
            ),
        ]
    ) + "\n"


def _interpret_summary(summary: pd.DataFrame) -> str:
    if summary.empty or "xg_total_anchor_mode" not in summary.columns:
        return "_No scored rows available._"
    indexed = summary.set_index("xg_total_anchor_mode")
    if not {"unanchored_current", "anchored_base_total"}.issubset(indexed.index):
        return "_Both modes were not available for comparison._"
    unanchored = indexed.loc["unanchored_current"]
    anchored = indexed.loc["anchored_base_total"]

    fav_delta = _metric_delta(anchored, unanchored, "mean_favorite_probability")
    draw_delta = _metric_delta(anchored, unanchored, "mean_draw_probability")
    brier_delta = _metric_delta(anchored, unanchored, "brier_score")
    log_delta = _metric_delta(anchored, unanchored, "log_loss")

    lines = [
        f"- Favorite probability delta, anchored minus unanchored: `{_fmt(fav_delta)}`.",
        f"- Draw probability delta, anchored minus unanchored: `{_fmt(draw_delta)}`.",
        f"- Brier delta, anchored minus unanchored: `{_fmt(brier_delta)}`.",
        f"- Log-loss delta, anchored minus unanchored: `{_fmt(log_delta)}`.",
    ]
    if pd.notna(fav_delta) and fav_delta < 0:
        lines.append("- Anchored xG reduced favorite extremity on the scored sample.")
    if pd.notna(draw_delta) and draw_delta > 0:
        lines.append("- Anchored xG increased average draw probability on the scored sample.")
    return "\n".join(lines)


def _metric_delta(left: pd.Series, right: pd.Series, column: str) -> float:
    left_value = _numeric(left.get(column))
    right_value = _numeric(right.get(column))
    if pd.isna(left_value) or pd.isna(right_value):
        return pd.NA
    return float(left_value - right_value)


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else round(float(value), 4))
    return out


def _match_key(row: pd.Series | dict[str, Any]) -> tuple[str, str, str, str]:
    return build_match_key(row.get("home", ""), row.get("away", ""), row.get("kickoff_utc", row.get("date_utc", "")), row.get("competition", ""))


def _is_special_case(home: str, away: str) -> bool:
    pair = (normalize_team_name(home), normalize_team_name(away))
    return pair in {(normalize_team_name(h), normalize_team_name(a)) for h, a in SPECIAL_CASES}


def _prob_for_actual(probs: dict[str, float], actual: Any) -> float:
    actual_text = str(actual or "")
    if actual_text not in OUTCOMES:
        return pd.NA
    return float(probs[actual_text])


def _actual_result(home_goals: Any, away_goals: Any) -> str:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return ""
    if float(home_goals) > float(away_goals):
        return "home_win"
    if float(home_goals) < float(away_goals):
        return "away_win"
    return "draw"


def _bool_value(value: Any, default: bool = True) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _numeric(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return pd.NA
    return float(parsed)


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
