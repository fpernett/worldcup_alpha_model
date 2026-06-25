from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting import PREDICTION_LOG_COLUMNS  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.prediction_log_repair import repair_prediction_log_by_quarantine  # noqa: E402


def main() -> None:
    result = repair_prediction_log_by_quarantine(DATA_DIR / "prediction_log.csv", PREDICTION_LOG_COLUMNS)
    print("Prediction log repair")
    print(f"path: {result['path']}")
    print(f"data_rows: {result['data_rows']}")
    print(f"malformed_rows: {result['malformed_rows']}")
    print(f"is_corrupt: {result['is_corrupt']}")
    print(f"repaired: {result['repaired']}")
    print(f"rows_quarantined: {result.get('rows_quarantined', 0)}")
    if result.get("backup_path"):
        print(f"backup_path: {result['backup_path']}")
    if result.get("warning"):
        print(f"warning: {result['warning']}")


if __name__ == "__main__":
    main()
