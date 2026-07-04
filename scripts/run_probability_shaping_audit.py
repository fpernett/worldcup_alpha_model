from __future__ import annotations

import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import OUTCOMES, brier_1x2, safe_log_loss  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.evaluation import build_formal_evaluation_dataset  # noqa: E402
from src.model import ModelConfig, run_match_model  # noqa: E402
from src.team_names import is_unresolved_team_slot  # noqa: E402


REPORTS_DIR = ROOT / "reports"
CSV_PATH = REPORTS_DIR / "probability_shaping_audit.csv"
MD_PATH = REPORTS_DIR / "probability_shaping_audit.md"

DRAW_INFLATION_MULTIPLIER = 1.15
FAVORITE_SHRINKAGE = 0.10
PROBABILITY_EPSILON = 1e-12

OUTPUT_COLUMNS = [
    "match_id",
    "date_utc",
    "home_team",
    "away_team",
    "actual_90_min_result",
    "variant",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "top_pick",
    "top_pick_correct",
    "probability_assigned_to_actual",
    "brier_score",
    "log_loss",
    "favorite_probability",
    "draw_probability",
    "top_pick_changed_vs_production",
    "notes",
]


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation = _load_evaluation_rows()
    rows, notes = _build_audit_rows(evaluation)
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(CSV_PATH, index=False)
    MD_PATH.write_text(_render_markdown(out, notes), encoding="utf-8")
    print(f"wrote {CSV_PATH}")
    print(f"wrote {MD_PATH}")


def _load_evaluation_rows() -> pd.DataFrame:
    prediction_rows = build_formal_evaluation_dataset(
        _read_csv(DATA_DIR / "prediction_ledger.csv"),
        _read_csv(DATA_DIR / "results_ledger.csv"),
        completed_results_df=_read_csv(DATA_DIR / "completed_results.csv"),
        fixtures_df=_read_csv(DATA_DIR / "fixtures.csv"),
        result_fixture_crosswalk_df=_read_csv(DATA_DIR / "result_fixture_crosswalk.csv"),
        prediction_log_df=_read_csv(DATA_DIR / "prediction_log.csv"),
        market_odds_df=_read_csv(DATA_DIR / "market_odds.csv"),
        latest_snapshot_only=True,
    )
    fixture_rows = _load_completed_fixture_rows()
    frames = [frame for frame in [prediction_rows, fixture_rows] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["_source_priority"] = out["notes"].astype(str).str.contains("source=fixture_csv").map({True: 1, False: 0})
    out["_row_key"] = out.apply(_row_key, axis=1)
    out = (
        out.sort_values(["_row_key", "_source_priority"])
        .drop_duplicates(subset=["_row_key"], keep="first")
        .drop(columns=["_source_priority", "_row_key"])
        .reset_index(drop=True)
    )
    return out


def _load_completed_fixture_rows() -> pd.DataFrame:
    fixtures = _read_csv(DATA_DIR / "fixtures.csv")
    crosswalk = _read_csv(DATA_DIR / "result_fixture_crosswalk.csv")
    teams = _read_csv(DATA_DIR / "team_ratings.csv")
    venues = _read_csv(DATA_DIR / "venues.csv")
    odds = _read_csv(DATA_DIR / "market_odds.csv")
    if fixtures.empty or crosswalk.empty or teams.empty:
        return pd.DataFrame()

    selected_results = _selected_fixture_results(crosswalk)
    rows: list[dict[str, Any]] = []
    for _, fixture in fixtures.iterrows():
        match_id = str(fixture.get("match_id", "") or "").strip()
        result = selected_results.get(match_id)
        if result is None:
            continue
        home = str(fixture.get("home", "") or "")
        away = str(fixture.get("away", "") or "")
        if is_unresolved_team_slot(home) or is_unresolved_team_slot(away):
            continue
        actual = str(result.get("result_actual_result", "") or "")
        if actual not in OUTCOMES:
            continue
        try:
            modeled = run_match_model(fixture, teams, venues, odds, ModelConfig())
        except Exception as exc:
            print(f"warning: skipped fixture {match_id}: {exc}")
            continue
        probs = _normalise(
            [
                modeled.get("probs", {}).get("home_win"),
                modeled.get("probs", {}).get("draw"),
                modeled.get("probs", {}).get("away_win"),
            ]
        )
        if probs is None:
            continue
        rows.append(
            {
                "match_id": match_id,
                "kickoff_utc": _fixture_kickoff(fixture),
                "generated_at_utc": "",
                "home_team": home,
                "away_team": away,
                "actual_result_1x2": actual,
                "home_win_prob_raw": probs[0],
                "draw_prob_raw": probs[1],
                "away_win_prob_raw": probs[2],
                "notes": (
                    "source=fixture_csv; diagnostic production rerun; "
                    f"result_join_method={result.get('join_method', '')}; "
                    f"result_source_match_id={result.get('result_match_id', '')}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _selected_fixture_results(crosswalk: pd.DataFrame) -> dict[str, pd.Series]:
    df = crosswalk.copy()
    required = {"fixture_match_id", "result_actual_result", "evaluation_eligible_1x2"}
    if df.empty or not required.issubset(df.columns):
        return {}
    df = df.loc[
        df["fixture_match_id"].astype(str).str.strip().ne("")
        & df["result_actual_result"].astype(str).isin(OUTCOMES)
        & df["evaluation_eligible_1x2"].astype(str).str.lower().isin({"true", "1", "yes"})
    ].copy()
    if df.empty:
        return {}
    df["_confidence"] = pd.to_numeric(df.get("join_confidence"), errors="coerce").fillna(0.0)
    df["_exact_id"] = df.get("join_method", "").astype(str).eq("exact_match_id").astype(int)
    df = df.sort_values(["fixture_match_id", "_exact_id", "_confidence"], ascending=[True, False, False])
    return {str(row.get("fixture_match_id", "") or ""): row for _, row in df.drop_duplicates("fixture_match_id", keep="first").iterrows()}


def _fixture_kickoff(fixture: pd.Series) -> str:
    date = str(fixture.get("date_utc", "") or "").strip()
    time = str(fixture.get("time_utc", "") or "").strip()
    if not date:
        return ""
    if not time:
        return date
    return f"{date}T{time}:00+00:00" if len(time) == 5 else f"{date}T{time}+00:00"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        print(f"warning: failed to read {path}: {exc}")
        return pd.DataFrame()


def _build_audit_rows(evaluation: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    notes: list[str] = [
        "Diagnostic-only audit; production defaults, Polymarket joins, staking, trading, and confidence scoring are unchanged.",
        f"draw_inflated uses a fixed {DRAW_INFLATION_MULTIPLIER:.2f}x draw multiplier.",
        f"favorite_shrunk uses a fixed {FAVORITE_SHRINKAGE:.0%} relative shrink on the larger of home_win/away_win.",
    ]
    if evaluation.empty:
        notes.append("No local completed 90-minute rows matched usable pre-kickoff prediction rows.")
        return [], notes

    rows: list[dict[str, Any]] = []
    scored = evaluation.loc[evaluation["actual_result_1x2"].astype(str).isin(OUTCOMES)].copy()
    production_by_match: dict[str, str] = {}

    for _, row in scored.iterrows():
        base = _normalise(
            [
                row.get("home_win_prob_raw"),
                row.get("draw_prob_raw"),
                row.get("away_win_prob_raw"),
            ]
        )
        if base is None:
            continue
        production_top = _top_pick(base)
        production_by_match[_row_key(row)] = production_top
        for variant, probs, variant_note in [
            ("production_current", base, "current production raw 1X2 probabilities"),
            (
                "draw_inflated",
                _inflate_draw(base),
                f"draw probability multiplied by {DRAW_INFLATION_MULTIPLIER:.2f} and renormalized",
            ),
            (
                "favorite_shrunk",
                _shrink_favorite(base),
                f"larger of home/away win probabilities shrunk by {FAVORITE_SHRINKAGE:.0%} and renormalized",
            ),
            (
                "draw_inflated_plus_favorite_shrunk",
                _shrink_favorite(_inflate_draw(base)),
                f"draw inflation then {FAVORITE_SHRINKAGE:.0%} favorite shrinkage, both renormalized",
            ),
        ]:
            rows.append(_audit_row(row, variant, probs, production_top, variant_note))

    anchored_note = _anchored_base_total_note()
    notes.append(anchored_note)
    if "available" in anchored_note:
        rows.extend(_anchored_base_total_rows(scored, production_by_match))

    return rows, notes


def _anchored_base_total_note() -> str:
    cfg_fields = {field.name for field in fields(ModelConfig)}
    if "xg_total_anchor_mode" not in cfg_fields:
        return "anchored_base_total skipped: ModelConfig has no xg_total_anchor_mode field in this checkout."
    return "anchored_base_total available: evaluated with ModelConfig(xg_total_anchor_mode='anchored_base_total')."


def _anchored_base_total_rows(scored: pd.DataFrame, production_by_match: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    teams = _read_csv(DATA_DIR / "team_ratings.csv")
    venues = _read_csv(DATA_DIR / "venues.csv")
    fixtures = _read_csv(DATA_DIR / "fixtures.csv")
    odds = _read_csv(DATA_DIR / "market_odds.csv")
    if teams.empty or fixtures.empty:
        return rows
    cfg = ModelConfig(xg_total_anchor_mode="anchored_base_total")  # type: ignore[call-arg]
    for _, row in scored.iterrows():
        fixture = _fixture_for_row(row, fixtures)
        if fixture is None:
            continue
        try:
            result = run_match_model(fixture, teams, venues, odds, cfg)
        except Exception as exc:
            probs = _normalise([row.get("home_win_prob_raw"), row.get("draw_prob_raw"), row.get("away_win_prob_raw")])
            if probs is None:
                continue
            rows.append(
                _audit_row(
                    row,
                    "anchored_base_total",
                    probs,
                    production_by_match.get(_row_key(row), _top_pick(probs)),
                    f"skipped for row: anchored model run failed: {exc}",
                )
            )
            continue
        probs = _normalise(
            [
                result.get("probs", {}).get("home_win"),
                result.get("probs", {}).get("draw"),
                result.get("probs", {}).get("away_win"),
            ]
        )
        if probs is None:
            continue
        rows.append(
            _audit_row(
                row,
                "anchored_base_total",
                probs,
                production_by_match.get(_row_key(row), _top_pick(probs)),
                "ModelConfig(xg_total_anchor_mode='anchored_base_total')",
            )
        )
    return rows


def _fixture_for_row(row: pd.Series, fixtures: pd.DataFrame) -> pd.Series | None:
    if fixtures.empty:
        return None
    match_id = str(row.get("match_id", "") or "").strip()
    if match_id and "match_id" in fixtures.columns:
        matched = fixtures.loc[fixtures["match_id"].astype(str).eq(match_id)]
        if not matched.empty:
            return matched.iloc[0]
    home = str(row.get("home_team", "") or "").strip().lower()
    away = str(row.get("away_team", "") or "").strip().lower()
    if {"home", "away"}.issubset(fixtures.columns):
        matched = fixtures.loc[
            fixtures["home"].astype(str).str.strip().str.lower().eq(home)
            & fixtures["away"].astype(str).str.strip().str.lower().eq(away)
        ]
        if not matched.empty:
            return matched.iloc[0]
    return None


def _audit_row(row: pd.Series, variant: str, probs: list[float], production_top_pick: str, note: str) -> dict[str, Any]:
    actual = str(row.get("actual_result_1x2", "") or "")
    top_pick = _top_pick(probs)
    actual_prob = _prob_for_actual(probs, actual)
    return {
        "match_id": row.get("match_id", ""),
        "date_utc": _date_utc(row),
        "home_team": row.get("home_team", ""),
        "away_team": row.get("away_team", ""),
        "actual_90_min_result": actual,
        "variant": variant,
        "home_win_prob": probs[0],
        "draw_prob": probs[1],
        "away_win_prob": probs[2],
        "top_pick": top_pick,
        "top_pick_correct": top_pick == actual,
        "probability_assigned_to_actual": actual_prob,
        "brier_score": brier_1x2(probs, actual),
        "log_loss": safe_log_loss(actual_prob),
        "favorite_probability": max(probs[0], probs[2]),
        "draw_probability": probs[1],
        "top_pick_changed_vs_production": top_pick != production_top_pick,
        "notes": note,
    }


def _date_utc(row: pd.Series) -> str:
    for col in ["kickoff_utc", "date_utc", "generated_at_utc"]:
        value = row.get(col, "")
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(ts):
            return ts.date().isoformat()
    return ""


def _inflate_draw(probs: list[float]) -> list[float]:
    return _normalise([probs[0], probs[1] * DRAW_INFLATION_MULTIPLIER, probs[2]]) or probs


def _shrink_favorite(probs: list[float]) -> list[float]:
    shaped = list(probs)
    favorite_idx = 0 if shaped[0] >= shaped[2] else 2
    shaped[favorite_idx] *= 1.0 - FAVORITE_SHRINKAGE
    return _normalise(shaped) or probs


def _normalise(values: list[Any]) -> list[float] | None:
    probs = [_coerce_float(value) for value in values]
    if any(value is None or not math.isfinite(value) or value < 0 for value in probs):
        return None
    total = sum(float(value) for value in probs)
    if total <= 0:
        return None
    return [float(value) / total for value in probs]


def _coerce_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _top_pick(probs: list[float]) -> str:
    return OUTCOMES[int(max(range(len(probs)), key=lambda idx: probs[idx]))]


def _prob_for_actual(probs: list[float], actual: str) -> float:
    if actual not in OUTCOMES:
        return PROBABILITY_EPSILON
    return max(min(probs[OUTCOMES.index(actual)], 1.0 - PROBABILITY_EPSILON), PROBABILITY_EPSILON)


def _row_key(row: pd.Series) -> str:
    return "|".join(str(row.get(col, "") or "") for col in ["match_id", "home_team", "away_team", "kickoff_utc"])


def _render_markdown(out: pd.DataFrame, notes: list[str]) -> str:
    summary = _summary_by_variant(out)
    lines = [
        "# Probability Shaping Audit",
        "",
        "This is diagnostic-only. It does not change app defaults, `ModelConfig` defaults, Polymarket join logic, staking, trading, or confidence scoring.",
        "",
        "## Variant Summary",
        "",
        _to_markdown(summary),
        "",
        "## Diagnostic Counts",
        "",
    ]
    lines.extend(_diagnostic_lines(out))
    lines.extend(
        [
            "",
            "## Safer Than Production?",
            "",
            _safer_than_production(summary),
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend([f"- {note}" for note in notes])
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{CSV_PATH.relative_to(ROOT)}`",
            f"- `{MD_PATH.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _summary_by_variant(out: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variant",
        "rows_evaluated",
        "brier_score_mean",
        "log_loss_mean",
        "top_pick_accuracy",
        "average_favorite_probability",
        "average_draw_probability",
        "top_pick_changes_vs_production",
        "actual_draws_correctly_picked",
        "missed_actual_draws",
        "production_wrong_variant_improved_actual_probability",
        "variant_worse_actual_probability_vs_production",
    ]
    if out.empty:
        return pd.DataFrame(columns=columns)

    production = out.loc[out["variant"].eq("production_current"), ["match_id", "home_team", "away_team", "actual_90_min_result", "probability_assigned_to_actual", "top_pick_correct"]].copy()
    production = production.rename(
        columns={
            "probability_assigned_to_actual": "production_actual_probability",
            "top_pick_correct": "production_top_pick_correct",
        }
    )
    merged = out.merge(production, on=["match_id", "home_team", "away_team", "actual_90_min_result"], how="left")

    rows = []
    for variant, group in merged.groupby("variant", sort=False):
        actual_draws = group["actual_90_min_result"].astype(str).eq("draw")
        rows.append(
            {
                "variant": variant,
                "rows_evaluated": int(len(group)),
                "brier_score_mean": _mean(group["brier_score"]),
                "log_loss_mean": _mean(group["log_loss"]),
                "top_pick_accuracy": _mean(group["top_pick_correct"].astype(float)),
                "average_favorite_probability": _mean(group["favorite_probability"]),
                "average_draw_probability": _mean(group["draw_probability"]),
                "top_pick_changes_vs_production": int(group["top_pick_changed_vs_production"].astype(bool).sum()),
                "actual_draws_correctly_picked": int((actual_draws & group["top_pick_correct"].astype(bool)).sum()),
                "missed_actual_draws": int((actual_draws & ~group["top_pick_correct"].astype(bool)).sum()),
                "production_wrong_variant_improved_actual_probability": int(
                    (
                        group["variant"].ne("production_current")
                        & ~group["production_top_pick_correct"].fillna(False).astype(bool)
                        & (pd.to_numeric(group["probability_assigned_to_actual"], errors="coerce") > pd.to_numeric(group["production_actual_probability"], errors="coerce"))
                    ).sum()
                ),
                "variant_worse_actual_probability_vs_production": int(
                    (
                        group["variant"].ne("production_current")
                        & (pd.to_numeric(group["probability_assigned_to_actual"], errors="coerce") < pd.to_numeric(group["production_actual_probability"], errors="coerce"))
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _mean(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").mean()
    return float(value) if pd.notna(value) else float("nan")


def _diagnostic_lines(out: pd.DataFrame) -> list[str]:
    if out.empty:
        return ["No rows were evaluated."]
    summary = _summary_by_variant(out)
    lines: list[str] = []
    for _, row in summary.iterrows():
        lines.append(
            "- "
            f"{row['variant']}: rows={int(row['rows_evaluated'])}, "
            f"top-pick changes={int(row['top_pick_changes_vs_production'])}, "
            f"draws picked={int(row['actual_draws_correctly_picked'])}, "
            f"missed draws={int(row['missed_actual_draws'])}, "
            f"production-wrong improved rows={int(row['production_wrong_variant_improved_actual_probability'])}, "
            f"worse actual-prob rows={int(row['variant_worse_actual_probability_vs_production'])}"
        )
    return lines


def _safer_than_production(summary: pd.DataFrame) -> str:
    if summary.empty or "production_current" not in set(summary["variant"]):
        return "No: there are no production rows to compare against."
    prod = summary.loc[summary["variant"].eq("production_current")].iloc[0]
    candidates = summary.loc[summary["variant"].ne("production_current")].copy()
    if candidates.empty:
        return "No diagnostic variant was evaluated against production_current."
    better = candidates.loc[
        (candidates["brier_score_mean"] < prod["brier_score_mean"])
        & (candidates["log_loss_mean"] < prod["log_loss_mean"])
        & (candidates["variant_worse_actual_probability_vs_production"] <= candidates["production_wrong_variant_improved_actual_probability"])
    ]
    if better.empty:
        return "No. On this local sample, no probability-shaping variant clearly looks safer than production_current across both Brier score and log loss."
    names = ", ".join(str(value) for value in better["variant"].tolist())
    return f"Yes, cautiously: {names} improved both mean Brier score and mean log loss on this local diagnostic sample. This is not a promotion decision."


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_markdown_cell(row.get(col)) for col in columns) + " |")
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
