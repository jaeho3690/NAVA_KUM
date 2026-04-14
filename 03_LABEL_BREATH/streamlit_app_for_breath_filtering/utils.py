from __future__ import annotations

import ast
import hashlib
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
BREATH_OUTPUT_DIR = BASE_DIR / "notebooks" / "outputs" / "03_breath_detect"
LABELED_DIR = BASE_DIR / "stored_results" / "04_breath_labels"
CACHE_DIR = APP_DIR / ".cache_parquet"

ANNOTATORS = ["이주영", "박지선", "조한나", "김재호", "김시현", "오창준", "Test"]
BREATH_LABELS = ["Normal", "Sigh", "Apnea", "Hiccup", "NotSure"]
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
]
SPLIT_ACTION_SCHEMA = [
    "action_id",
    "timestamp",
    "annotator",
    "patient_id",
    "target_breath_id",
    "target_original_breath_id",
    "split_group_id",
    "split_index",
    "split_breath_id",
    "start_idx",
    "end_idx",
    "start_ts",
    "end_ts",
    "n_splits",
    "comment",
]
PEAK_ACTION_SCHEMA = [
    "action_id",
    "timestamp",
    "annotator",
    "patient_id",
    "target_breath_id",
    "peak_sample_idx",
]
FILTER_REVIEW_SCHEMA = [
    "action_id",
    "timestamp",
    "annotator",
    "patient_id",
    "target_breath_id",
    "target_original_breath_id",
    "decision",
    "comment",
]


def _breath_file_candidates(patient_dir: str) -> List[Path]:
    base_path = BREATH_OUTPUT_DIR / patient_dir
    patient_name = Path(patient_dir).name
    patient_id = patient_name.replace("patient_", "")
    return [
        base_path / f"BB_patient_{patient_id}_clustered_breaths.pkl",
        base_path / f"AA_patient_{patient_id}_clustered_breaths.pkl",
        base_path / f"{patient_name}_clustered_breaths.pkl",
        base_path / f"{patient_name}_clustered_breaths_with_anomaly.pkl",
    ]


def _current_breath_path(patient_dir: str) -> Optional[Path]:
    for path in _breath_file_candidates(patient_dir):
        if path.exists():
            return path
    return None


def get_bb_breath_path(patient_dir: str) -> Path:
    base_path = BREATH_OUTPUT_DIR / patient_dir
    patient_id = Path(patient_dir).name.replace("patient_", "")
    return base_path / f"BB_patient_{patient_id}_clustered_breaths.pkl"


def get_aa_breath_path(patient_dir: str) -> Path:
    base_path = BREATH_OUTPUT_DIR / patient_dir
    patient_id = Path(patient_dir).name.replace("patient_", "")
    return base_path / f"AA_patient_{patient_id}_clustered_breaths.pkl"


def list_detected_versions() -> List[str]:
    if not BREATH_OUTPUT_DIR.exists():
        return []

    versions = sorted(
        [
            p.name
            for p in BREATH_OUTPUT_DIR.iterdir()
            if p.is_dir() and any((_current_breath_path(str(x.relative_to(BREATH_OUTPUT_DIR))) is not None) for x in p.glob("patient_*"))
        ],
        reverse=True,
    )
    return versions


def scan_patient_dirs(version_tag: Optional[str] = None) -> List[str]:
    if not BREATH_OUTPUT_DIR.exists():
        return []

    if version_tag and str(version_tag).strip().upper() != "ALL":
        version_dir = BREATH_OUTPUT_DIR / str(version_tag).strip()
        if not version_dir.exists():
            return []
        return sorted(
            [
                str(p.relative_to(BREATH_OUTPUT_DIR))
                for p in version_dir.glob("patient_*")
                if _current_breath_path(str(p.relative_to(BREATH_OUTPUT_DIR))) is not None
            ]
        )

    return sorted(
        [
            str(p.relative_to(BREATH_OUTPUT_DIR))
            for p in BREATH_OUTPUT_DIR.glob("*/patient_*")
            if _current_breath_path(str(p.relative_to(BREATH_OUTPUT_DIR))) is not None
        ]
    )


def extract_patient_id(patient_dir: str) -> str:
    name = Path(patient_dir).name
    if name.startswith("patient_"):
        return name[len("patient_") :]
    return name


def extract_version_tag(patient_dir: str) -> str:
    parts = Path(patient_dir).parts
    return parts[0] if parts else "unknown"


def _coerce_bool(series: pd.Series) -> pd.Series:
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        text = str(value).strip().upper()
        return text in {"TRUE", "T", "1", "YES", "Y"}

    return series.map(_to_bool)


def _safe_literal_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return np.asarray([], dtype=float)
        if isinstance(parsed, list):
            return np.asarray(parsed, dtype=float)
    return np.asarray([], dtype=float)


def _safe_literal_int_list(value) -> List[int]:
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist() if pd.notna(x)]
    if isinstance(value, list):
        return [int(x) for x in value if pd.notna(x)]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [int(x) for x in parsed if pd.notna(x)]
    return []


def _split_actions_path(version_tag: str) -> Path:
    return BASE_DIR / "stored_results" / "03_breath_splits" / version_tag / "breath_splits.parquet"


def _peak_actions_path(version_tag: str) -> Path:
    return BASE_DIR / "stored_results" / "03_breath_peaks" / version_tag / "breath_peaks.parquet"


def _filter_review_path(version_tag: str) -> Path:
    return BASE_DIR / "stored_results" / "03_breath_filter_reviews" / version_tag / "breath_filter_reviews.parquet"


def _load_action_log(path: Path, schema: List[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=schema)
    df = pd.read_parquet(path)
    if schema == SPLIT_ACTION_SCHEMA:
        if "action_id" not in df.columns:
            if "split_group_id" in df.columns:
                df["action_id"] = df["split_group_id"]
            elif "split_row_id" in df.columns:
                df["action_id"] = df["split_row_id"]
        if "target_breath_id" not in df.columns:
            if "original_breath_id" in df.columns:
                df["target_breath_id"] = df["original_breath_id"]
        if "target_original_breath_id" not in df.columns and "original_breath_id" in df.columns:
            df["target_original_breath_id"] = df["original_breath_id"]
    for col in schema:
        if col not in df.columns:
            df[col] = None
    df = df[schema].copy()
    return _coerce_action_log_types(df, schema)


def _coerce_action_log_types(df: pd.DataFrame, schema: List[str]) -> pd.DataFrame:
    df = df.copy()

    string_columns_by_schema = {
        tuple(SPLIT_ACTION_SCHEMA): {
            "action_id",
            "timestamp",
            "annotator",
            "patient_id",
            "target_breath_id",
            "target_original_breath_id",
            "split_group_id",
            "split_breath_id",
            "comment",
        },
        tuple(PEAK_ACTION_SCHEMA): {
            "action_id",
            "timestamp",
            "annotator",
            "patient_id",
            "target_breath_id",
        },
        tuple(FILTER_REVIEW_SCHEMA): {
            "action_id",
            "timestamp",
            "annotator",
            "patient_id",
            "target_breath_id",
            "target_original_breath_id",
            "decision",
            "comment",
        },
    }
    int_columns_by_schema = {
        tuple(SPLIT_ACTION_SCHEMA): {"split_index", "start_idx", "end_idx", "n_splits"},
        tuple(PEAK_ACTION_SCHEMA): {"peak_sample_idx"},
    }
    float_columns_by_schema = {
        tuple(SPLIT_ACTION_SCHEMA): {"start_ts", "end_ts"},
    }

    schema_key = tuple(schema)
    for col in string_columns_by_schema.get(schema_key, set()):
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), None)
            df[col] = df[col].map(lambda x: None if x is None else str(x))

    for col in int_columns_by_schema.get(schema_key, set()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_columns_by_schema.get(schema_key, set()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _append_action_rows(path: Path, rows: List[Dict], schema: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_df = _load_action_log(path, schema)
    new_df = pd.concat([old_df, pd.DataFrame(rows, columns=schema)], ignore_index=True)
    new_df = _coerce_action_log_types(new_df, schema)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid4().hex}")
    new_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def load_patient_outputs(patient_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_df = _load_signal_df(patient_dir)
    breath_df = _load_breath_df(patient_dir)
    run_meta_df = _load_run_meta_df(patient_dir)
    return signal_df.copy(), breath_df.copy(), run_meta_df.copy()


@lru_cache(maxsize=32)
def _load_signal_df(patient_dir: str) -> pd.DataFrame:
    base_path = BREATH_OUTPUT_DIR / patient_dir
    patient_name = Path(patient_dir).name
    csv_path = base_path / f"{patient_name}_edi_filtered_signal.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"신호 CSV 파일을 찾을 수 없습니다: {csv_path}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(patient_dir.encode("utf-8")).hexdigest()[:12]
    cache_path = CACHE_DIR / f"{patient_name}__{cache_key}__signal.parquet"

    use_cache = cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime
    if use_cache:
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            use_cache = False

    if not use_cache:
        df = pd.read_csv(csv_path)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass

    required_cols = {"timestamp", "time_sec_from_start", "edi_raw", "edi_smooth_for_detection", "breath_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"signal CSV 필수 컬럼 누락: {sorted(missing)}")

    if "sample_idx" in df.columns:
        df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["time_sec_from_start"] = pd.to_numeric(df["time_sec_from_start"], errors="coerce")
    df["edi_raw"] = pd.to_numeric(df["edi_raw"], errors="coerce")
    df["edi_smooth_for_detection"] = pd.to_numeric(df["edi_smooth_for_detection"], errors="coerce")
    df["merged_peak_mask"] = _coerce_bool(df["merged_peak_mask"]) if "merged_peak_mask" in df.columns else False
    df["breath_mask"] = _coerce_bool(df["breath_mask"]) if "breath_mask" in df.columns else False
    if "breath_label" not in df.columns:
        df["breath_label"] = ""
    df = df.dropna(subset=["timestamp", "edi_smooth_for_detection"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=32)
def _load_breath_df(patient_dir: str) -> pd.DataFrame:
    pkl_path = _current_breath_path(patient_dir)
    if pkl_path is None:
        raise FileNotFoundError(f"breath pickle 파일을 찾을 수 없습니다: {patient_dir}")

    df = pd.read_pickle(pkl_path)
    return normalize_breath_df(df)


def normalize_breath_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = {"breath_id", "v_start_idx", "v_end_idx", "v_start_t", "v_end_t"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"breath pickle 필수 컬럼 누락: {sorted(missing)}")

    if "AE_abnormal" not in df.columns:
        if "is_anomaly" in df.columns:
            df["AE_abnormal"] = df["is_anomaly"]
        else:
            df["AE_abnormal"] = True
    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = np.nan
    if "major_cluster" not in df.columns:
        df["major_cluster"] = False
    if "is_split" not in df.columns:
        df["is_split"] = False
    if "original_breath_id" not in df.columns:
        df["original_breath_id"] = df["breath_id"]
    if "split_group_id" not in df.columns:
        df["split_group_id"] = None
    if "split_index" not in df.columns:
        df["split_index"] = pd.NA
    if "split_comment" not in df.columns:
        df["split_comment"] = ""
    if "peak_indices" not in df.columns:
        df["peak_indices"] = [[] for _ in range(len(df))]
    if "peak_rel_indices" not in df.columns:
        df["peak_rel_indices"] = [[] for _ in range(len(df))]
    if "manual_peak_indices" not in df.columns:
        df["manual_peak_indices"] = [[] for _ in range(len(df))]

    df["breath_id"] = df["breath_id"].astype(str)
    df["original_breath_id"] = df["original_breath_id"].astype(str)
    df["v_start_idx"] = pd.to_numeric(df["v_start_idx"], errors="coerce").astype("Int64")
    df["v_end_idx"] = pd.to_numeric(df["v_end_idx"], errors="coerce").astype("Int64")
    df["v_start_t"] = pd.to_numeric(df["v_start_t"], errors="coerce")
    df["v_end_t"] = pd.to_numeric(df["v_end_t"], errors="coerce")
    df["breath_duration_sec"] = pd.to_numeric(df.get("breath_duration_sec"), errors="coerce")
    df["cluster_label"] = pd.to_numeric(df.get("cluster_label"), errors="coerce").astype("Int64")
    df["AE_abnormal"] = _coerce_bool(df["AE_abnormal"])
    df["major_cluster"] = _coerce_bool(df["major_cluster"])
    df["is_split"] = _coerce_bool(df["is_split"])
    df["split_index"] = pd.to_numeric(df["split_index"], errors="coerce").astype("Int64")
    df["filtered_edi_signal_arr"] = df.get("filtered_edi_signal", pd.Series([[]] * len(df))).map(_safe_literal_array)
    df["peak_indices"] = df["peak_indices"].map(_safe_literal_int_list)
    df["peak_rel_indices"] = df["peak_rel_indices"].map(_safe_literal_int_list)
    df["manual_peak_indices"] = df["manual_peak_indices"].map(_safe_literal_int_list)
    if "breath_label" not in df.columns:
        df["breath_label"] = df["breath_id"].map(lambda x: f"breath_{x}")
    df = df.dropna(subset=["breath_id", "v_start_idx", "v_end_idx", "v_start_t", "v_end_t"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=32)
def _load_run_meta_df(patient_dir: str) -> pd.DataFrame:
    base_path = BREATH_OUTPUT_DIR / patient_dir
    patient_name = Path(patient_dir).name
    csv_path = base_path / f"{patient_name}_run_meta.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def clear_patient_caches(patient_dir: str) -> None:
    _load_breath_df.cache_clear()
    _load_signal_df.cache_clear()
    _load_run_meta_df.cache_clear()


def build_breath_candidates(breath_df: pd.DataFrame, patient_id: str, anomaly_only: bool = True) -> List[Dict]:
    work = breath_df.copy()
    if anomaly_only and "AE_abnormal" in work.columns:
        work = work[work["AE_abnormal"] == True].copy()  # noqa: E712
    work = work.sort_values(["v_start_t", "v_end_t", "breath_id"]).reset_index(drop=True)

    candidates: List[Dict] = []
    for row in work.itertuples(index=False):
        start_ts = float(row.v_start_t)
        end_ts = float(row.v_end_t)
        breath_id = str(row.breath_id)
        candidates.append(
            {
                "type": "breath_label",
                "patient_id": patient_id,
                "breath_id": breath_id,
                "original_breath_id": str(getattr(row, "original_breath_id", breath_id)),
                "item_id": f"{patient_id}|breath|{breath_id}",
                "original_item_id": f"{patient_id}|breath|{str(getattr(row, 'original_breath_id', breath_id))}",
                "start_idx": int(row.v_start_idx),
                "end_idx": int(row.v_end_idx),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_sec": float(row.breath_duration_sec) if not pd.isna(row.breath_duration_sec) else np.nan,
                "cluster_label": None if pd.isna(row.cluster_label) else int(row.cluster_label),
                "major_cluster": bool(row.major_cluster),
                "anomaly_score": float(row.anomaly_score) if not pd.isna(row.anomaly_score) else np.nan,
                "is_abnormal": bool(row.AE_abnormal),
                "is_anomaly": bool(row.AE_abnormal),
                "breath_label": str(getattr(row, "breath_label", f"breath_{breath_id}")),
                "is_split": bool(getattr(row, "is_split", False)),
                "split_index": None if pd.isna(getattr(row, "split_index", pd.NA)) else int(getattr(row, "split_index")),
                "split_group_id": getattr(row, "split_group_id", None),
                "split_comment": str(getattr(row, "split_comment", "") or ""),
                "manual_peak_indices": list(getattr(row, "manual_peak_indices", []) or []),
            }
        )
    return candidates


def save_split_breath_segments(
    patient_dir: str,
    breath_df: pd.DataFrame,
    target_breath_id: str,
    segments: List[Dict],
    annotator: str,
    comment: str = "",
) -> Path:
    work = breath_df.copy()
    work["breath_id"] = work["breath_id"].astype(str)
    target_mask = work["breath_id"] == str(target_breath_id)
    if int(target_mask.sum()) != 1:
        raise ValueError(f"split 대상 breath를 정확히 하나 찾지 못했습니다: {target_breath_id}")

    original_row = work.loc[target_mask].iloc[0].copy()
    remaining = work.loc[~target_mask].copy()
    split_group_id = uuid4().hex[:8]
    original_breath_id = str(original_row.get("original_breath_id", original_row["breath_id"]))
    timestamp = datetime.now().isoformat(timespec="seconds")
    action_id = str(uuid4())

    new_rows = []
    action_rows = []
    for idx, segment in enumerate(segments, start=1):
        row = original_row.copy()
        new_breath_id = f"{original_breath_id}_split_{split_group_id}_{idx}"
        row["breath_id"] = new_breath_id
        row["original_breath_id"] = original_breath_id
        row["breath_label"] = f"breath_{new_breath_id}"
        row["v_start_idx"] = int(segment["start_idx"])
        row["v_end_idx"] = int(segment["end_idx"])
        row["v_start_t"] = float(segment["start_ts"])
        row["v_end_t"] = float(segment["end_ts"])
        row["breath_duration_sec"] = (float(segment["end_ts"]) - float(segment["start_ts"])) / 1000.0
        row["is_split"] = True
        row["split_group_id"] = split_group_id
        row["split_index"] = idx
        row["split_comment"] = comment
        if "annotator" in row.index:
            row["annotator"] = annotator
        new_rows.append(row)
        action_rows.append(
            {
                "action_id": action_id,
                "timestamp": timestamp,
                "annotator": annotator,
                "patient_id": extract_patient_id(patient_dir),
                "target_breath_id": str(target_breath_id),
                "target_original_breath_id": original_breath_id,
                "split_group_id": split_group_id,
                "split_index": idx,
                "split_breath_id": new_breath_id,
                "start_idx": int(segment["start_idx"]),
                "end_idx": int(segment["end_idx"]),
                "start_ts": float(segment["start_ts"]),
                "end_ts": float(segment["end_ts"]),
                "n_splits": len(segments),
                "comment": comment,
            }
        )

    out_df = pd.concat([remaining, pd.DataFrame(new_rows)], ignore_index=True)
    out_df = out_df.sort_values(["v_start_t", "v_end_t", "breath_id"]).reset_index(drop=True)
    out_path = get_bb_breath_path(patient_dir)
    out_df.to_pickle(out_path)
    _append_action_rows(
        _split_actions_path(extract_version_tag(patient_dir)),
        action_rows,
        SPLIT_ACTION_SCHEMA,
    )
    clear_patient_caches(patient_dir)
    return out_path


def save_manual_peak(
    patient_dir: str,
    breath_df: pd.DataFrame,
    target_breath_id: str,
    peak_sample_idx: int,
    annotator: str = "",
) -> Path:
    work = breath_df.copy()
    work["breath_id"] = work["breath_id"].astype(str)
    target_mask = work["breath_id"] == str(target_breath_id)
    if int(target_mask.sum()) != 1:
        raise ValueError(f"peak 추가 대상 breath를 정확히 하나 찾지 못했습니다: {target_breath_id}")

    row_idx = work.index[target_mask][0]
    row = work.loc[row_idx].copy()
    start_idx = int(row["v_start_idx"])
    end_idx = int(row["v_end_idx"])
    peak_sample_idx = int(peak_sample_idx)
    if peak_sample_idx < start_idx or peak_sample_idx > end_idx:
        raise ValueError("peak sample_idx는 breath 내부에 있어야 합니다.")

    manual_peaks = list(row.get("manual_peak_indices", []) or [])
    if peak_sample_idx not in manual_peaks:
        manual_peaks.append(peak_sample_idx)
    manual_peaks = sorted(set(int(x) for x in manual_peaks))

    peak_indices = list(row.get("peak_indices", []) or [])
    peak_indices.append(peak_sample_idx)
    peak_indices = sorted(set(int(x) for x in peak_indices))
    peak_rel_indices = [int(x - start_idx) for x in peak_indices if start_idx <= int(x) <= end_idx]

    work.at[row_idx, "manual_peak_indices"] = manual_peaks
    work.at[row_idx, "peak_indices"] = peak_indices
    work.at[row_idx, "peak_rel_indices"] = peak_rel_indices
    if "n_peaks" in work.columns:
        work.at[row_idx, "n_peaks"] = len(peak_indices)

    out_path = get_bb_breath_path(patient_dir)
    work.to_pickle(out_path)
    _append_action_rows(
        _peak_actions_path(extract_version_tag(patient_dir)),
        [
            {
                "action_id": str(uuid4()),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "annotator": annotator,
                "patient_id": extract_patient_id(patient_dir),
                "target_breath_id": str(target_breath_id),
                "peak_sample_idx": peak_sample_idx,
            }
        ],
        PEAK_ACTION_SCHEMA,
    )
    clear_patient_caches(patient_dir)
    return out_path


def save_remove_breath(
    patient_dir: str,
    breath_df: pd.DataFrame,
    target_breath_id: str,
    annotator: str,
    comment: str = "",
) -> Path:
    work = breath_df.copy()
    work["breath_id"] = work["breath_id"].astype(str)
    target_mask = work["breath_id"] == str(target_breath_id)
    if int(target_mask.sum()) != 1:
        raise ValueError(f"remove 대상 breath를 정확히 하나 찾지 못했습니다: {target_breath_id}")

    target_row = work.loc[target_mask].iloc[0].copy()
    save_filter_review(
        patient_dir=patient_dir,
        annotator=annotator,
        target_breath_id=str(target_breath_id),
        target_original_breath_id=str(target_row.get("original_breath_id", target_breath_id)),
        decision="remove",
        comment=comment,
    )
    return get_bb_breath_path(patient_dir)


def load_filter_reviews(version_tag: str, patient_id: Optional[str] = None) -> pd.DataFrame:
    df = _load_action_log(_filter_review_path(version_tag), FILTER_REVIEW_SCHEMA)
    if patient_id is not None and not df.empty:
        df = df[df["patient_id"].astype(str) == str(patient_id)].copy()
    return df.reset_index(drop=True)


def save_filter_review(
    patient_dir: str,
    annotator: str,
    target_breath_id: str,
    target_original_breath_id: str,
    decision: str = "keep",
    comment: str = "",
) -> None:
    version_tag = extract_version_tag(patient_dir)
    patient_id = extract_patient_id(patient_dir)
    existing = load_filter_reviews(version_tag, patient_id=patient_id)
    if not existing.empty:
        same_mask = (
            (existing["annotator"].fillna("").astype(str) == str(annotator))
            & (existing["decision"].fillna("").astype(str) == str(decision))
        )
        if str(decision) == "remove":
            same_mask = same_mask & (existing["target_breath_id"].fillna("").astype(str) == str(target_breath_id))
        else:
            same_mask = same_mask & (
                existing["target_original_breath_id"].fillna("").astype(str) == str(target_original_breath_id)
            )
        same = existing[same_mask]
        if not same.empty:
            return

    _append_action_rows(
        _filter_review_path(version_tag),
        [
            {
                "action_id": str(uuid4()),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "annotator": annotator,
                "patient_id": patient_id,
                "target_breath_id": str(target_breath_id),
                "target_original_breath_id": str(target_original_breath_id),
                "decision": decision,
                "comment": comment,
            }
        ],
        FILTER_REVIEW_SCHEMA,
    )


def _labels_path(annotator: str, version_tag: str) -> Path:
    return LABELED_DIR / version_tag / f"labels_{annotator}.parquet"


def load_labels(annotator: str, version_tag: str) -> pd.DataFrame:
    path = _labels_path(annotator, version_tag)
    if not path.exists():
        return pd.DataFrame(columns=LABEL_SCHEMA)

    df = pd.read_parquet(path)
    needs_migration = False
    for col in LABEL_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan if col in {"start_ts", "end_ts"} else None
            needs_migration = True
    df = df[LABEL_SCHEMA].copy()
    for col in ("start_ts", "end_ts"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if needs_migration:
        tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid4().hex}")
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
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
    label_type: str,
    item_id: str,
    label: str,
    comment: str = "",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> Dict:
    return {
        "label_id": str(uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "annotator": annotator,
        "patient_id": patient_id,
        "type": label_type,
        "item_id": item_id,
        "label": label,
        "comment": comment,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }


def get_labeled_item_ids(labels_df: pd.DataFrame, patient_id: str, label_type: str) -> Set[str]:
    if labels_df.empty:
        return set()
    filtered = labels_df[(labels_df["patient_id"] == patient_id) & (labels_df["type"] == label_type)]
    return set(filtered["item_id"].dropna().astype(str).unique().tolist())


def get_latest_labels_map(labels_df: pd.DataFrame, patient_id: str, label_type: str) -> Dict[str, str]:
    if labels_df.empty:
        return {}
    filtered = labels_df[(labels_df["patient_id"] == patient_id) & (labels_df["type"] == label_type)]
    if filtered.empty:
        return {}
    last_rows = filtered.groupby("item_id", sort=False).tail(1)
    return dict(zip(last_rows["item_id"].astype(str), last_rows["label"].astype(str)))


def get_latest_comments_map(labels_df: pd.DataFrame, patient_id: str, label_type: str) -> Dict[str, str]:
    if labels_df.empty:
        return {}
    filtered = labels_df[
        (labels_df["patient_id"] == patient_id) & (labels_df["type"] == label_type)
    ].copy()
    if filtered.empty:
        return {}
    filtered["item_id"] = filtered["item_id"].astype(str)
    filtered["comment"] = filtered["comment"].fillna("").astype(str)
    if "timestamp" in filtered.columns:
        filtered["__ts"] = pd.to_datetime(filtered["timestamp"], errors="coerce")
        filtered = filtered.sort_values("__ts")
    return dict(zip(filtered["item_id"], filtered["comment"]))


def build_patient_status_snapshot(annotator: str, version_tag: Optional[str] = None) -> pd.DataFrame:
    patient_dirs = scan_patient_dirs(version_tag=version_tag)
    rows = []
    for patient_dir in patient_dirs:
        pid = extract_patient_id(patient_dir)
        ver = extract_version_tag(patient_dir)
        try:
            _, breath_df, _ = load_patient_outputs(patient_dir)
            candidates = build_breath_candidates(breath_df, pid, anomaly_only=True)
            labels_df = load_labels(annotator, ver)
            labeled_ids = get_labeled_item_ids(labels_df, pid, "breath_label")
            total = len(candidates)
            labeled = len({c["item_id"] for c in candidates} & labeled_ids)
            rows.append(
                {
                    "version": ver,
                    "patient_id": pid,
                    "total_candidates": total,
                    "labeled": labeled,
                    "remaining": max(0, total - labeled),
                    "status": "DONE" if total > 0 and labeled >= total else ("EMPTY" if total == 0 else "TODO"),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "version": ver,
                    "patient_id": pid,
                    "total_candidates": 0,
                    "labeled": 0,
                    "remaining": 0,
                    "status": f"ERROR: {e}",
                }
            )
    return pd.DataFrame(rows).sort_values(["version", "patient_id"], ascending=[False, True]).reset_index(drop=True)


def build_filtering_status_snapshot(
    annotator: Optional[str] = None,
    version_tag: Optional[str] = None,
    abnormal_only: bool = True,
) -> pd.DataFrame:
    patient_dirs = scan_patient_dirs(version_tag=version_tag)
    rows = []
    for patient_dir in patient_dirs:
        pid = extract_patient_id(patient_dir)
        ver = extract_version_tag(patient_dir)
        try:
            _, breath_df, _ = load_patient_outputs(patient_dir)
            candidates = build_breath_candidates(breath_df, pid, anomaly_only=abnormal_only)
            original_ids = {str(c["original_breath_id"]) for c in candidates}
            final_breath_count = len(candidates)
            split_saved_ids = set()
            reviewed_keep_ids = set()
            removed_ids = set()
            if original_ids:
                split_actions_df = load_split_actions(ver, patient_id=pid)
                if annotator and str(annotator).strip().upper() != "ALL" and not split_actions_df.empty:
                    split_actions_df = split_actions_df[
                        split_actions_df["annotator"].fillna("").astype(str) == str(annotator)
                    ].copy()
                if not split_actions_df.empty:
                    split_saved_ids = set(
                        split_actions_df["target_original_breath_id"].dropna().astype(str).tolist()
                    ) & original_ids
                review_df = load_filter_reviews(ver, patient_id=pid)
                if annotator and str(annotator).strip().upper() != "ALL" and not review_df.empty:
                    review_df = review_df[
                        review_df["annotator"].fillna("").astype(str) == str(annotator)
                    ].copy()
                if not review_df.empty:
                    keep_df = review_df[review_df["decision"].fillna("").astype(str) == "keep"].copy()
                    remove_df = review_df[review_df["decision"].fillna("").astype(str) == "remove"].copy()
                    reviewed_keep_ids = set(
                        keep_df["target_original_breath_id"].dropna().astype(str).tolist()
                    ) & original_ids
                    removed_ids = set(
                        remove_df["target_original_breath_id"].dropna().astype(str).tolist()
                    ) & original_ids
            processed_ids = split_saved_ids | reviewed_keep_ids | removed_ids
            total = len(original_ids)
            split_saved = len(split_saved_ids)
            reviewed_keep = len(reviewed_keep_ids)
            removed = len(removed_ids)
            processed = len(processed_ids)
            rows.append(
                {
                    "version": ver,
                    "patient_id": pid,
                    "total_candidates": total,
                    "final_breath_count": final_breath_count,
                    "split_saved": split_saved,
                    "kept_as_is": reviewed_keep,
                    "removed": removed,
                    "processed": processed,
                    "remaining": max(0, total - processed),
                    "progress_ratio": 0.0 if total == 0 else processed / total,
                    "status": "DONE" if total > 0 and processed >= total else ("EMPTY" if total == 0 else "TODO"),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "version": ver,
                    "patient_id": pid,
                    "total_candidates": 0,
                    "final_breath_count": 0,
                    "split_saved": 0,
                    "kept_as_is": 0,
                    "removed": 0,
                    "processed": 0,
                    "remaining": 0,
                    "progress_ratio": 0.0,
                    "status": f"ERROR: {e}",
                }
            )
    return pd.DataFrame(rows).sort_values(["version", "patient_id"], ascending=[False, True]).reset_index(drop=True)


def load_split_actions(version_tag: str, patient_id: Optional[str] = None) -> pd.DataFrame:
    df = _load_action_log(_split_actions_path(version_tag), SPLIT_ACTION_SCHEMA)
    if patient_id is not None and not df.empty:
        df = df[df["patient_id"].astype(str) == str(patient_id)].copy()
    return df.reset_index(drop=True)


def load_peak_actions(version_tag: str, patient_id: Optional[str] = None) -> pd.DataFrame:
    df = _load_action_log(_peak_actions_path(version_tag), PEAK_ACTION_SCHEMA)
    if patient_id is not None and not df.empty:
        df = df[df["patient_id"].astype(str) == str(patient_id)].copy()
    return df.reset_index(drop=True)


def rebuild_bb_from_logs(patient_dir: str) -> Path:
    version_tag = extract_version_tag(patient_dir)
    patient_id = extract_patient_id(patient_dir)
    aa_path = get_aa_breath_path(patient_dir)
    if not aa_path.exists():
        raise FileNotFoundError(f"AA breath pickle 파일을 찾을 수 없습니다: {aa_path}")

    work = normalize_breath_df(pd.read_pickle(aa_path))
    split_df = load_split_actions(version_tag, patient_id=patient_id)
    peak_df = load_peak_actions(version_tag, patient_id=patient_id)
    review_df = load_filter_reviews(version_tag, patient_id=patient_id)

    if not split_df.empty:
        split_df["__ts"] = pd.to_datetime(split_df["timestamp"], errors="coerce")
        action_order = (
            split_df.groupby("action_id", dropna=False)["__ts"]
            .min()
            .reset_index()
            .sort_values(["__ts", "action_id"], na_position="last")
        )
        for action_id in action_order["action_id"].astype(str):
            action_rows = split_df[split_df["action_id"].astype(str) == action_id].copy()
            if action_rows.empty:
                continue
            action_rows = action_rows.sort_values("split_index")
            target_breath_id = str(action_rows["target_breath_id"].iloc[0])
            target_mask = work["breath_id"].astype(str) == target_breath_id
            if int(target_mask.sum()) != 1:
                continue

            source_row = work.loc[target_mask].iloc[0].copy()
            remaining = work.loc[~target_mask].copy()
            new_rows = []
            for row in action_rows.itertuples(index=False):
                new_row = source_row.copy()
                new_row["breath_id"] = str(row.split_breath_id)
                new_row["original_breath_id"] = str(row.target_original_breath_id)
                new_row["breath_label"] = f"breath_{row.split_breath_id}"
                new_row["v_start_idx"] = int(row.start_idx)
                new_row["v_end_idx"] = int(row.end_idx)
                new_row["v_start_t"] = float(row.start_ts)
                new_row["v_end_t"] = float(row.end_ts)
                new_row["breath_duration_sec"] = (float(row.end_ts) - float(row.start_ts)) / 1000.0
                new_row["is_split"] = True
                new_row["split_group_id"] = str(row.split_group_id)
                new_row["split_index"] = int(row.split_index)
                new_row["split_comment"] = str(row.comment or "")
                new_rows.append(new_row)
            work = pd.concat([remaining, pd.DataFrame(new_rows)], ignore_index=True)
            work = normalize_breath_df(work)

    if not review_df.empty:
        review_df["__ts"] = pd.to_datetime(review_df["timestamp"], errors="coerce")
        review_df = review_df.sort_values(["__ts", "action_id"], na_position="last")
        for row in review_df.itertuples(index=False):
            if str(row.decision) != "remove":
                continue
            target_breath_id = str(row.target_breath_id)
            target_original_breath_id = str(row.target_original_breath_id)
            if (work["breath_id"].astype(str) == target_breath_id).any():
                work = work[work["breath_id"].astype(str) != target_breath_id].copy()
            else:
                work = work[work["original_breath_id"].astype(str) != target_original_breath_id].copy()
            work = normalize_breath_df(work) if not work.empty else work.copy()

    if not peak_df.empty:
        peak_df["__ts"] = pd.to_datetime(peak_df["timestamp"], errors="coerce")
        peak_df = peak_df.sort_values(["__ts", "action_id"], na_position="last")
        for row in peak_df.itertuples(index=False):
            target_mask = work["breath_id"].astype(str) == str(row.target_breath_id)
            if int(target_mask.sum()) != 1:
                continue
            row_idx = work.index[target_mask][0]
            current_row = work.loc[row_idx].copy()
            start_idx = int(current_row["v_start_idx"])
            end_idx = int(current_row["v_end_idx"])
            peak_sample_idx = int(row.peak_sample_idx)
            if peak_sample_idx < start_idx or peak_sample_idx > end_idx:
                continue

            manual_peaks = list(current_row.get("manual_peak_indices", []) or [])
            peak_indices = list(current_row.get("peak_indices", []) or [])
            if peak_sample_idx not in manual_peaks:
                manual_peaks.append(peak_sample_idx)
            if peak_sample_idx not in peak_indices:
                peak_indices.append(peak_sample_idx)
            manual_peaks = sorted(set(int(x) for x in manual_peaks))
            peak_indices = sorted(set(int(x) for x in peak_indices))
            peak_rel_indices = [int(x - start_idx) for x in peak_indices if start_idx <= int(x) <= end_idx]

            work.at[row_idx, "manual_peak_indices"] = manual_peaks
            work.at[row_idx, "peak_indices"] = peak_indices
            work.at[row_idx, "peak_rel_indices"] = peak_rel_indices
            if "n_peaks" in work.columns:
                work.at[row_idx, "n_peaks"] = len(peak_indices)

    work = work.sort_values(["v_start_t", "v_end_t", "breath_id"]).reset_index(drop=True)
    out_path = get_bb_breath_path(patient_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work.to_pickle(out_path)
    clear_patient_caches(patient_dir)
    return out_path
