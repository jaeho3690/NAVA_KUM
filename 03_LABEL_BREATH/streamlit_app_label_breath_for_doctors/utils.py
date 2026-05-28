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

ANNOTATORS = ["이주영", "박지선", "조한나", "박규현", "김재호", "김시현", "오창준", "Test"]
ANNOTATOR_COLORS: dict[str, str] = {
    "이주영": "#1e40af",
    "박지선": "#15803d",
    "조한나": "#7c3aed",
    "박규현": "#db2777",
    "김재호": "#b91c1c",
    "김시현": "#0f766e",
    "오창준": "#c2410c",
    "Test":   "#4b5563",
}
BREATH_LABELS = ["Phasic", "Sigh", "Apnea", "Hiccup", "Tonic Burst", "Crying", "NeedSplit", "NotSure"]
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


def _sorted_status_df(rows: List[Dict], columns: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    return df.sort_values(["version", "patient_id"], ascending=[False, True]).reset_index(drop=True)


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
    return _sorted_status_df(
        rows,
        ["version", "patient_id", "total_candidates", "labeled", "remaining", "status"],
    )
