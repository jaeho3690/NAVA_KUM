import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4
import hashlib

import numpy as np
import pandas as pd


BASE_DIR = Path("/home/jhkim/NAVA/03_LABEL")
DETECTED_DIR = BASE_DIR / "stored_results" / "00_detected"
LABELED_DIR = BASE_DIR / "stored_results" / "01_labeled"
CACHE_DIR = BASE_DIR / "streamlit_app" / ".cache_parquet"

ANNOTATORS = ["이주영", "이지선", "조한나", "김재호", "김시현", "오창준", "Test"]
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


def list_detected_versions() -> List[str]:
    if not DETECTED_DIR.exists():
        return []

    versions: List[str] = []
    for p in DETECTED_DIR.iterdir():
        if p.is_dir() and list(p.rglob("*.xlsx")):
            versions.append(p.name)

    if list(DETECTED_DIR.glob("*.xlsx")):
        versions.append("legacy")

    versions = sorted(set(versions))
    date_versions = sorted([v for v in versions if v.isdigit() and len(v) == 8], reverse=True)
    other_versions = sorted([v for v in versions if (v not in date_versions and v != "legacy")])
    out = date_versions + other_versions
    if "legacy" in versions:
        out.append("legacy")
    return out


def scan_xlsx_files(version_tag: Optional[str] = None) -> List[str]:
    if not DETECTED_DIR.exists():
        return []

    if version_tag and str(version_tag).strip().upper() != "ALL":
        version = str(version_tag).strip()
        if version == "legacy":
            candidates = sorted(DETECTED_DIR.glob("*.xlsx"))
        else:
            version_dir = DETECTED_DIR / version
            if not version_dir.exists():
                return []
            candidates = sorted(version_dir.rglob("*.xlsx"))
    else:
        candidates = sorted(DETECTED_DIR.rglob("*.xlsx"))

    files = [str(p.relative_to(DETECTED_DIR)) for p in candidates]
    return files


def extract_patient_id(patient_file: str) -> str:
    stem = Path(patient_file).stem
    prefix = "movingwinddetected_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def extract_version_tag(patient_file: str) -> str:
    parts = Path(patient_file).parts
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8:
        return parts[0]
    return "legacy"


def normalize_bool_columns(data_df: pd.DataFrame) -> pd.DataFrame:
    df = data_df.copy()
    has_detected_apnea = "detected_apnea" in df.columns
    has_legacy_apnea = "apnea" in df.columns
    bool_cols = ["gt_peak", "detected_peak", "detected_apnea", "apnea", "detected_sigh"]

    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        text = str(value).strip().upper()
        if text in {"TRUE", "T", "1", "YES", "Y"}:
            return True
        if text in {"FALSE", "F", "0", "NO", "N", ""}:
            return False
        return False

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(_to_bool)
        else:
            df[col] = False

    # Keep backward/forward compatibility between old (`apnea`) and new (`detected_apnea`) schemas.
    if has_detected_apnea and not has_legacy_apnea:
        df["apnea"] = df["detected_apnea"]
    elif has_legacy_apnea and not has_detected_apnea:
        df["detected_apnea"] = df["apnea"]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if "edi" in df.columns:
        df["edi"] = pd.to_numeric(df["edi"], errors="coerce")

    df = df.dropna(subset=["timestamp", "edi"]).copy()
    return df


def load_xlsx(patient_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_df, params_df = _load_xlsx_cached(patient_file)
    return data_df.copy(), params_df.copy()


@lru_cache(maxsize=32)
def _load_xlsx_cached(patient_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = DETECTED_DIR / patient_file
    if not path.exists():
        raise FileNotFoundError(f"XLSX 파일을 찾을 수 없습니다: {path}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(patient_file.encode("utf-8")).hexdigest()[:12]
    cache_stem = f"{Path(patient_file).stem}__{cache_key}"
    data_cache = CACHE_DIR / f"{cache_stem}__data.parquet"
    params_cache = CACHE_DIR / f"{cache_stem}__params.parquet"

    use_cache = (
        data_cache.exists()
        and params_cache.exists()
        and data_cache.stat().st_mtime >= path.stat().st_mtime
        and params_cache.stat().st_mtime >= path.stat().st_mtime
    )
    if use_cache:
        try:
            data_df = pd.read_parquet(data_cache)
            params_df = pd.read_parquet(params_cache)
        except Exception:
            use_cache = False

    if not use_cache:
        data_df = pd.read_excel(path, sheet_name="data")
        params_df = pd.read_excel(path, sheet_name="params")

        required_cols = {"timestamp", "edi", "gt_peak", "detected_peak"}
        missing = required_cols - set(data_df.columns)
        if missing:
            raise ValueError(f"data 시트 필수 컬럼 누락: {sorted(missing)}")
        if ("detected_apnea" not in data_df.columns) and ("apnea" not in data_df.columns):
            raise ValueError("data 시트 필수 컬럼 누락: ['detected_apnea' 또는 'apnea']")

        params_df = params_df.copy()
        for col in params_df.columns:
            if params_df[col].dtype == "object":
                params_df[col] = params_df[col].astype(str)
        try:
            data_df.to_parquet(data_cache, index=False)
            params_df.to_parquet(params_cache, index=False)
        except Exception:
            pass

    data_df = normalize_bool_columns(data_df)
    data_df = data_df.sort_values("timestamp").reset_index(drop=True)

    if "timestamp" not in data_df.columns or "edi" not in data_df.columns:
        raise ValueError("캐시된 data 형식이 올바르지 않습니다. 캐시 파일을 삭제 후 재시도하세요.")
    return data_df, params_df


def build_peak_candidates(
    data_df: pd.DataFrame,
    patient_id: str,
    gt_match_tolerance_ms: int = 1500,
) -> List[Dict]:
    peaks = data_df.loc[data_df["detected_peak"] == True, ["timestamp"]].copy()  # noqa: E712
    peaks = peaks.sort_values("timestamp")
    gt_peaks = np.sort(
        data_df.loc[data_df["gt_peak"] == True, "timestamp"].astype(int).to_numpy()  # noqa: E712
    )

    candidates: List[Dict] = []
    for ts in peaks["timestamp"].tolist():
        ts_i = int(ts)
        if _has_gt_match_within_tolerance(ts_i, gt_peaks, gt_match_tolerance_ms):
            continue
        candidates.append(
            {
                "type": "peak",
                "patient_id": patient_id,
                "timestamp": ts_i,
                "item_id": f"{patient_id}|peak|{ts_i}",
            }
        )
    return candidates


def build_sigh_candidates(
    data_df: pd.DataFrame,
    patient_id: str,
) -> List[Dict]:
    sighs = data_df.loc[data_df["detected_sigh"] == True, ["timestamp"]].copy()  # noqa: E712
    sighs = sighs.sort_values("timestamp")

    candidates: List[Dict] = []
    for ts in sighs["timestamp"].tolist():
        ts_i = int(ts)
        candidates.append(
            {
                "type": "sigh",
                "patient_id": patient_id,
                "timestamp": ts_i,
                "item_id": f"{patient_id}|sigh|{ts_i}",
            }
        )
    return candidates


def _has_gt_match_within_tolerance(
    detected_ts: int,
    gt_timestamps_sorted: np.ndarray,
    tolerance_ms: int,
) -> bool:
    if gt_timestamps_sorted.size == 0:
        return False

    idx = int(np.searchsorted(gt_timestamps_sorted, detected_ts))
    nearest = []
    if idx < gt_timestamps_sorted.size:
        nearest.append(abs(int(gt_timestamps_sorted[idx]) - detected_ts))
    if idx > 0:
        nearest.append(abs(int(gt_timestamps_sorted[idx - 1]) - detected_ts))
    if not nearest:
        return False
    return min(nearest) <= tolerance_ms


def build_apnea_segments(data_df: pd.DataFrame, patient_id: str) -> List[Dict]:
    apnea_df = data_df[["timestamp", "detected_apnea"]].sort_values("timestamp").reset_index(drop=True)
    segments: List[Dict] = []

    in_run = False
    run_start = None
    run_end = None

    for row in apnea_df.itertuples(index=False):
        ts = int(row.timestamp)
        is_apnea = bool(row.detected_apnea)

        if is_apnea and not in_run:
            in_run = True
            run_start = ts
            run_end = ts
        elif is_apnea and in_run:
            run_end = ts
        elif (not is_apnea) and in_run:
            segments.append(
                {
                    "type": "apnea",
                    "patient_id": patient_id,
                    "start": run_start,
                    "end": run_end,
                    "start_ts": run_start,
                    "end_ts": run_end,
                    "item_id": f"{patient_id}|apnea|{run_start}-{run_end}",
                }
            )
            in_run = False
            run_start = None
            run_end = None

    if in_run and run_start is not None and run_end is not None:
        segments.append(
            {
                "type": "apnea",
                "patient_id": patient_id,
                "start": run_start,
                "end": run_end,
                "start_ts": run_start,
                "end_ts": run_end,
                "item_id": f"{patient_id}|apnea|{run_start}-{run_end}",
            }
        )

    return segments


def _labels_path(annotator: str, version_tag: str = "legacy") -> Path:
    safe_name = annotator
    if version_tag and version_tag != "legacy":
        return LABELED_DIR / version_tag / f"labels_{safe_name}.parquet"
    return LABELED_DIR / f"labels_{safe_name}.parquet"


def load_labels(annotator: str, version_tag: str = "legacy") -> pd.DataFrame:
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


def append_label(annotator: str, row: Dict, version_tag: str = "legacy") -> None:
    path = _labels_path(annotator, version_tag)
    path.parent.mkdir(parents=True, exist_ok=True)

    old_df = load_labels(annotator, version_tag)
    new_df = pd.concat([old_df, pd.DataFrame([row], columns=LABEL_SCHEMA)], ignore_index=True)

    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid4().hex}")
    new_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def undo_last_label(annotator: str, version_tag: str = "legacy") -> bool:
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


def get_labeled_item_ids(
    labels_df: pd.DataFrame,
    patient_id: str,
    label_type: str,
) -> Set[str]:
    if labels_df.empty:
        return set()

    filtered = labels_df[
        (labels_df["patient_id"] == patient_id) & (labels_df["type"] == label_type)
    ]
    return set(filtered["item_id"].dropna().astype(str).unique().tolist())


def get_latest_labels_map(
    labels_df: pd.DataFrame,
    patient_id: str,
    label_type: str,
) -> Dict[str, str]:
    if labels_df.empty:
        return {}
    filtered = labels_df[
        (labels_df["patient_id"] == patient_id) & (labels_df["type"] == label_type)
    ]
    if filtered.empty:
        return {}
    last_rows = filtered.groupby("item_id", sort=False).tail(1)
    keys = last_rows["item_id"].astype(str).tolist()
    vals = last_rows["label"].astype(str).tolist()
    return dict(zip(keys, vals))


def get_latest_comments_map(
    labels_df: pd.DataFrame,
    patient_id: str,
    label_type: str,
) -> Dict[str, str]:
    if labels_df.empty:
        return {}
    filtered = labels_df[
        (labels_df["patient_id"] == patient_id) & (labels_df["type"] == label_type)
    ]
    if filtered.empty:
        return {}
    if "comment" not in filtered.columns:
        return {}
    work = filtered.copy()
    work["item_id"] = work["item_id"].astype(str)
    work["comment"] = work["comment"].fillna("").astype(str).str.strip()
    if "timestamp" in work.columns:
        work["__ts"] = pd.to_datetime(work["timestamp"], errors="coerce")
        work = work.sort_values("__ts")

    out: Dict[str, str] = {}
    for row in work.itertuples(index=False):
        item_id = str(row.item_id)
        comment = str(row.comment).strip()
        if comment:
            out[item_id] = comment
        elif item_id not in out:
            out[item_id] = ""
    return out


def ts_to_index(timestamp_array: np.ndarray, ts_value: float) -> int:
    if timestamp_array.size == 0:
        return 0

    idx = int(np.searchsorted(timestamp_array, ts_value))
    if idx <= 0:
        return 0
    if idx >= timestamp_array.size:
        return int(timestamp_array.size - 1)

    prev_idx = idx - 1
    next_idx = idx
    prev_diff = abs(float(timestamp_array[prev_idx]) - float(ts_value))
    next_diff = abs(float(timestamp_array[next_idx]) - float(ts_value))
    return int(prev_idx if prev_diff <= next_diff else next_idx)


def clamp_interval(
    ts_start: float,
    ts_end: float,
    ts_min: float,
    ts_max: float,
) -> Tuple[float, float]:
    start = max(float(ts_min), min(float(ts_start), float(ts_max)))
    end = max(float(ts_min), min(float(ts_end), float(ts_max)))
    if start > end:
        start, end = end, start
    return start, end


def _format_ts_value(ts_value: float) -> str:
    ts_float = float(ts_value)
    if ts_float.is_integer():
        return str(int(ts_float))
    return f"{ts_float:.6f}".rstrip("0").rstrip(".")


def make_apnea_adjusted_item_id(patient_id: str, start_ts: float, end_ts: float) -> str:
    start_str = _format_ts_value(start_ts)
    end_str = _format_ts_value(end_ts)
    return f"{patient_id}|apnea|{start_str}-{end_str}"


def parse_apnea_item_id_interval(item_id: str) -> Optional[Tuple[float, float]]:
    try:
        parts = str(item_id).split("|", 2)
        if len(parts) != 3 or parts[1] != "apnea":
            return None
        start_str, end_str = parts[2].split("-", 1)
        start_ts = float(start_str)
        end_ts = float(end_str)
        return start_ts, end_ts
    except Exception:
        return None


def get_latest_apnea_adjustments_map(
    labels_df: pd.DataFrame,
    patient_id: str,
) -> Dict[str, Tuple[float, float, str]]:
    if labels_df.empty:
        return {}

    filtered = labels_df[
        (labels_df["patient_id"] == patient_id) & (labels_df["type"] == "apnea")
    ].copy()
    if filtered.empty:
        return {}

    for col in ("start_ts", "end_ts"):
        if col not in filtered.columns:
            filtered[col] = np.nan
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

    last_rows = filtered.groupby("item_id", sort=False).tail(1)
    out: Dict[str, Tuple[float, float, str]] = {}
    for row in last_rows.itertuples(index=False):
        item_id = str(row.item_id)
        label = str(row.label)
        start_ts = float(row.start_ts) if pd.notna(row.start_ts) else np.nan
        end_ts = float(row.end_ts) if pd.notna(row.end_ts) else np.nan

        if pd.isna(start_ts) or pd.isna(end_ts):
            parsed = parse_apnea_item_id_interval(item_id)
            if parsed is not None:
                start_ts, end_ts = parsed
            else:
                continue
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts
        out[item_id] = (start_ts, end_ts, label)
    return out


def build_apnea_label_mask(
    data_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    patient_id: str,
    annotator: str,
) -> np.ndarray:
    mask = np.zeros(len(data_df), dtype=bool)
    if data_df.empty or labels_df.empty:
        return mask

    ts_array = data_df["timestamp"].to_numpy()
    if ts_array.size == 0:
        return mask

    filtered = labels_df[
        (labels_df["annotator"] == annotator)
        & (labels_df["patient_id"] == patient_id)
        & (labels_df["type"] == "apnea")
    ].copy()
    if filtered.empty:
        return mask

    latest_map = get_latest_apnea_adjustments_map(filtered, patient_id)
    ts_min = float(np.min(ts_array))
    ts_max = float(np.max(ts_array))

    for _, (start_ts, end_ts, label) in latest_map.items():
        if label != "O":
            continue
        start_ts, end_ts = clamp_interval(start_ts, end_ts, ts_min, ts_max)
        start_idx = ts_to_index(ts_array, start_ts)
        end_idx = ts_to_index(ts_array, end_ts)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        mask[start_idx : end_idx + 1] = True
    return mask


def get_next_candidate(candidates: List[Dict], labeled_item_ids: Set[str]) -> Optional[Dict]:
    for c in candidates:
        if c["item_id"] not in labeled_item_ids:
            return c
    return None


def build_patient_status_snapshot(annotator: str, version_tag: Optional[str] = None) -> pd.DataFrame:
    files = scan_xlsx_files(version_tag=version_tag)
    labels_by_version: Dict[str, pd.DataFrame] = {}
    rows: List[Dict] = []

    def _status_parts(labeled: int, total: int) -> Tuple[str, str]:
        if total == 0 or labeled >= total:
            return "✅ DONE", "done"
        if labeled > 0:
            return "🟡 IN_PROGRESS", "in_progress"
        return "🔴 NOT_STARTED", "not_started"

    for patient_file in files:
        version_tag = extract_version_tag(patient_file)
        if version_tag not in labels_by_version:
            labels_by_version[version_tag] = load_labels(annotator, version_tag)
        labels_df = labels_by_version[version_tag]
        patient_id = extract_patient_id(patient_file)
        data_df, _ = load_xlsx(patient_file)
        peak_candidates = build_peak_candidates(data_df, patient_id)
        apnea_candidates = build_apnea_segments(data_df, patient_id)
        sigh_candidates = build_sigh_candidates(data_df, patient_id)

        labeled_peak_ids = get_labeled_item_ids(labels_df, patient_id, "peak")
        labeled_apnea_ids = get_labeled_item_ids(labels_df, patient_id, "apnea")
        labeled_sigh_ids = get_labeled_item_ids(labels_df, patient_id, "sigh")

        peak_total = len(peak_candidates)
        apnea_total = len(apnea_candidates)
        sigh_total = len(sigh_candidates)
        peak_labeled = len(set(c["item_id"] for c in peak_candidates) & labeled_peak_ids)
        apnea_labeled = len(set(c["item_id"] for c in apnea_candidates) & labeled_apnea_ids)
        sigh_labeled = len(set(c["item_id"] for c in sigh_candidates) & labeled_sigh_ids)

        peak_status, peak_status_key = _status_parts(peak_labeled, peak_total)
        apnea_status, apnea_status_key = _status_parts(apnea_labeled, apnea_total)
        sigh_status, _ = _status_parts(sigh_labeled, sigh_total)
        rows.append(
            {
                "version": version_tag,
                "patient_id": patient_id,
                "peak_labeled": peak_labeled,
                "peak_total": peak_total,
                "peak_status": peak_status,
                "apnea_labeled": apnea_labeled,
                "apnea_total": apnea_total,
                "apnea_status": apnea_status,
                "sigh_labeled": sigh_labeled,
                "sigh_total": sigh_total,
                "sigh_status": sigh_status,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "version",
                "patient_id",
                "peak_labeled",
                "peak_total",
                "peak_status",
                "apnea_labeled",
                "apnea_total",
                "apnea_status",
                "sigh_labeled",
                "sigh_total",
                "sigh_status",
            ]
        )

    out = pd.DataFrame(rows).sort_values(["version", "patient_id"]).reset_index(drop=True)
    return out
