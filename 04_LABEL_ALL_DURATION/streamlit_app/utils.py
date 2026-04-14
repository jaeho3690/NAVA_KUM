import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd


BASE_DIR = Path("/home/jhkim/NAVA/04_LABEL_ALL_DURATION")
DETECTED_DIR = BASE_DIR / "stored_results" / "01_all_duration_segments"
LABELED_DIR = BASE_DIR / "stored_results" / "02_labeled_segments"

ANNOTATORS = ["이주영", "이지선", "조한나", "김재호", "김시현", "오창준", "Test"]
SEGMENT_REVIEW_LABELS = ["breath", "apnea", "not_known", "NotSure"]
LABEL_SCHEMA = [
    "label_id",
    "timestamp",
    "annotator",
    "patient_id",
    "type",
    "item_id",
    "label",
    "comment",
    "start_ts",
    "end_ts",
    "predicted_label",
]


def list_detected_versions() -> List[str]:
    if not DETECTED_DIR.exists():
        return []

    versions: List[str] = []
    for p in DETECTED_DIR.iterdir():
        if p.is_dir() and list(p.glob("patient_*")):
            versions.append(p.name)

    return sorted(set(versions), reverse=True)


def scan_patient_dirs(version_tag: Optional[str] = None) -> List[str]:
    if not DETECTED_DIR.exists():
        return []

    version = str(version_tag).strip() if version_tag else ""
    if version and version.upper() != "ALL":
        version_dir = DETECTED_DIR / version
        if not version_dir.exists():
            return []
        candidates = sorted(p for p in version_dir.iterdir() if p.is_dir() and p.name.startswith("patient_"))
    else:
        candidates = sorted(DETECTED_DIR.glob("*/patient_*"))

    return [str(p.relative_to(DETECTED_DIR)) for p in candidates]


def extract_patient_id(patient_dir: str) -> str:
    name = Path(patient_dir).name
    if name.startswith("patient_"):
        return name[len("patient_") :]
    return name


def extract_version_tag(patient_dir: str) -> str:
    parts = Path(patient_dir).parts
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


def _read_dataframe(preferred_path: Path, fallback_path: Path) -> pd.DataFrame:
    if preferred_path.exists():
        return pd.read_pickle(preferred_path)
    if fallback_path.exists():
        return pd.read_csv(fallback_path)
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {preferred_path} 또는 {fallback_path}")


@lru_cache(maxsize=32)
def _load_patient_outputs_cached(patient_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = DETECTED_DIR / patient_dir
    patient_id = extract_patient_id(patient_dir)

    sample_df = _read_dataframe(
        root / f"patient_{patient_id}_sample_level_signal.pkl",
        root / f"patient_{patient_id}_sample_level_signal.csv",
    )
    segment_df = _read_dataframe(
        root / f"patient_{patient_id}_segment_catalog.pkl",
        root / f"patient_{patient_id}_segment_catalog.csv",
    )
    run_meta_df = _read_dataframe(
        root / f"patient_{patient_id}_run_meta.pkl",
        root / f"patient_{patient_id}_run_meta.csv",
    ) if (root / f"patient_{patient_id}_run_meta.pkl").exists() or (root / f"patient_{patient_id}_run_meta.csv").exists() else pd.DataFrame()

    sample_df = sample_df.copy()
    segment_df = segment_df.copy()

    numeric_cols_sample = [
        "timestamp",
        "time_sec_from_start",
        "edi_raw",
        "edi_detrended",
        "edi_smooth_for_detection",
        "breath_id",
    ]
    for col in numeric_cols_sample:
        if col in sample_df.columns:
            sample_df[col] = pd.to_numeric(sample_df[col], errors="coerce")

    bool_cols_sample = [
        "analysis_window_mask",
        "apnea_mask",
        "candidate_peak_mask",
        "merged_peak_mask",
        "is_breath",
        "is_apnea",
    ]
    for col in bool_cols_sample:
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].fillna(False).astype(bool)

    numeric_cols_segment = [
        "start_idx",
        "end_idx",
        "start_time",
        "end_time",
        "start_time_sec_from_start",
        "end_time_sec_from_start",
        "duration_sec",
        "n_samples",
        "n_peaks",
    ]
    for col in numeric_cols_segment:
        if col in segment_df.columns:
            segment_df[col] = pd.to_numeric(segment_df[col], errors="coerce")

    if "segment_type" in segment_df.columns:
        segment_df["segment_type"] = segment_df["segment_type"].astype(str)
    if "segment_id" in segment_df.columns:
        segment_df["segment_id"] = segment_df["segment_id"].astype(str)
    if "segment_label" in segment_df.columns:
        segment_df["segment_label"] = segment_df["segment_label"].astype(str)

    sample_df = sample_df.sort_values("sample_idx").reset_index(drop=True)
    segment_df = segment_df.sort_values("start_idx").reset_index(drop=True)
    return sample_df, segment_df, run_meta_df


def load_patient_outputs(patient_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_df, segment_df, run_meta_df = _load_patient_outputs_cached(patient_dir)
    return sample_df.copy(), segment_df.copy(), run_meta_df.copy()


def build_segment_candidates(
    segment_df: pd.DataFrame,
    patient_id: str,
    segment_type_filter: str = "ALL",
) -> List[Dict]:
    work = segment_df.copy()
    if segment_type_filter != "ALL":
        work = work[work["segment_type"] == segment_type_filter].copy()

    candidates: List[Dict] = []
    for row in work.itertuples(index=False):
        candidates.append(
            {
                "type": "segment",
                "patient_id": patient_id,
                "item_id": str(row.segment_id),
                "segment_id": str(row.segment_id),
                "segment_label": str(row.segment_label),
                "predicted_label": str(row.segment_type),
                "start_idx": int(row.start_idx),
                "end_idx": int(row.end_idx),
                "start_ts": float(row.start_time),
                "end_ts": float(row.end_time),
                "duration_sec": float(row.duration_sec),
                "n_peaks": int(row.n_peaks),
            }
        )
    return candidates


def _labels_path(annotator: str, version_tag: str) -> Path:
    return LABELED_DIR / version_tag / f"labels_{annotator}.parquet"


def load_labels(annotator: str, version_tag: str) -> pd.DataFrame:
    path = _labels_path(annotator, version_tag)
    if not path.exists():
        return pd.DataFrame(columns=LABEL_SCHEMA)

    df = pd.read_parquet(path)
    for col in LABEL_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan if col in {"start_ts", "end_ts"} else None
    df = df[LABEL_SCHEMA].copy()
    for col in ("start_ts", "end_ts"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def append_label(annotator: str, row: Dict, version_tag: str) -> None:
    path = _labels_path(annotator, version_tag)
    path.parent.mkdir(parents=True, exist_ok=True)

    old_df = load_labels(annotator, version_tag)
    new_df = pd.concat([old_df, pd.DataFrame([row], columns=LABEL_SCHEMA)], ignore_index=True)

    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid4().hex}")
    new_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def undo_last_label(annotator: str, version_tag: str) -> bool:
    path = _labels_path(annotator, version_tag)
    if not path.exists():
        return False

    df = load_labels(annotator, version_tag)
    if df.empty:
        return False

    new_df = df.iloc[:-1].copy()
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid4().hex}")
    new_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    return True


def make_label_row(
    annotator: str,
    patient_id: str,
    item_id: str,
    label: str,
    predicted_label: str,
    comment: str = "",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> Dict:
    return {
        "label_id": str(uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "annotator": annotator,
        "patient_id": patient_id,
        "type": "segment",
        "item_id": item_id,
        "label": label,
        "comment": comment,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "predicted_label": predicted_label,
    }


def get_labeled_item_ids(labels_df: pd.DataFrame, patient_id: str, label_type: str = "segment") -> Set[str]:
    if labels_df.empty:
        return set()
    filtered = labels_df[(labels_df["patient_id"] == patient_id) & (labels_df["type"] == label_type)]
    return set(filtered["item_id"].dropna().astype(str).unique().tolist())


def get_latest_labels_map(labels_df: pd.DataFrame, patient_id: str, label_type: str = "segment") -> Dict[str, str]:
    if labels_df.empty:
        return {}
    filtered = labels_df[(labels_df["patient_id"] == patient_id) & (labels_df["type"] == label_type)]
    if filtered.empty:
        return {}
    if "timestamp" in filtered.columns:
        filtered = filtered.assign(__ts=pd.to_datetime(filtered["timestamp"], errors="coerce")).sort_values("__ts")
    last_rows = filtered.groupby("item_id", sort=False).tail(1)
    return dict(zip(last_rows["item_id"].astype(str), last_rows["label"].astype(str)))


def get_latest_comments_map(labels_df: pd.DataFrame, patient_id: str, label_type: str = "segment") -> Dict[str, str]:
    if labels_df.empty:
        return {}
    filtered = labels_df[(labels_df["patient_id"] == patient_id) & (labels_df["type"] == label_type)].copy()
    if filtered.empty:
        return {}
    filtered["item_id"] = filtered["item_id"].astype(str)
    filtered["comment"] = filtered["comment"].fillna("").astype(str).str.strip()
    if "timestamp" in filtered.columns:
        filtered["__ts"] = pd.to_datetime(filtered["timestamp"], errors="coerce")
        filtered = filtered.sort_values("__ts")

    out: Dict[str, str] = {}
    for row in filtered.itertuples(index=False):
        item_id = str(row.item_id)
        comment = str(row.comment).strip()
        if comment:
            out[item_id] = comment
        elif item_id not in out:
            out[item_id] = ""
    return out


def build_patient_status_snapshot(
    annotator: str,
    version_tag: Optional[str] = None,
    segment_type_filter: str = "ALL",
) -> pd.DataFrame:
    patient_dirs = scan_patient_dirs(version_tag=version_tag)
    labels_by_version: Dict[str, pd.DataFrame] = {}
    rows: List[Dict] = []

    for patient_dir in patient_dirs:
        current_version = extract_version_tag(patient_dir)
        if current_version not in labels_by_version:
            labels_by_version[current_version] = load_labels(annotator, current_version)
        labels_df = labels_by_version[current_version]

        patient_id = extract_patient_id(patient_dir)
        _, segment_df, _ = load_patient_outputs(patient_dir)
        candidates = build_segment_candidates(segment_df, patient_id, segment_type_filter=segment_type_filter)
        labeled_ids = get_labeled_item_ids(labels_df, patient_id, "segment")
        latest_labels = get_latest_labels_map(labels_df, patient_id, "segment")

        candidate_ids = {c["item_id"] for c in candidates}
        labeled_count = len(candidate_ids & labeled_ids)
        total_count = len(candidates)

        if total_count == 0 or labeled_count >= total_count:
            status = "DONE"
        elif labeled_count > 0:
            status = "IN_PROGRESS"
        else:
            status = "NOT_STARTED"

        agreed_count = 0
        for c in candidates:
            latest = latest_labels.get(c["item_id"])
            if latest == c["predicted_label"]:
                agreed_count += 1

        counts = (
            segment_df["segment_type"].value_counts().reindex(["breath", "apnea", "not_known"]).fillna(0).astype(int)
            if not segment_df.empty
            else pd.Series({"breath": 0, "apnea": 0, "not_known": 0})
        )
        rows.append(
            {
                "version": current_version,
                "patient_id": patient_id,
                "segments_total": int(total_count),
                "labeled": int(labeled_count),
                "remaining": int(max(0, total_count - labeled_count)),
                "agreed_with_predicted": int(agreed_count),
                "breath_segments": int(counts.get("breath", 0)),
                "apnea_segments": int(counts.get("apnea", 0)),
                "not_known_segments": int(counts.get("not_known", 0)),
                "status": status,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "version",
                "patient_id",
                "segments_total",
                "labeled",
                "remaining",
                "agreed_with_predicted",
                "breath_segments",
                "apnea_segments",
                "not_known_segments",
                "status",
            ]
        )
    return pd.DataFrame(rows).sort_values(["version", "patient_id"]).reset_index(drop=True)
