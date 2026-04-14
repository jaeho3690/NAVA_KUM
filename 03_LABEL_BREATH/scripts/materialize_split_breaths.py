from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_app_for_breath_filtering.utils import (
    get_bb_breath_path,
    rebuild_bb_from_logs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild BB_patient_NN_clustered_breaths.pkl from saved split/peak logs."
    )
    parser.add_argument(
        "--patient-dir",
        required=True,
        help="Patient directory relative to 03_breath_detect, e.g. 20260301/patient_03",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patient_dir = str(Path(args.patient_dir))
    out_path = rebuild_bb_from_logs(patient_dir)
    print(f"Saved: {get_bb_breath_path(patient_dir)}")
    print(f"Resolved path: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
