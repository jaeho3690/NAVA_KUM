from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


BASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BASE_DIR.parent
DEFAULT_RAW_DIR = REPO_DIR / '02_YOLO' / 'data' / '1_Refined_Raw'
DEFAULT_PEAK_DIR = REPO_DIR / '02_YOLO' / 'data' / '2_Refined_Peaks'
DEFAULT_OUTPUT_DIR = BASE_DIR / 'stored_results' / '00_detected'

# Detection parameters (same logic as notebook)
FS = 100  # 신호 샘플링 주파수(Hz)
IGNORE_HEAD_SEC = 10.0  # 각 파일 시작부에서 검출을 무시할 구간 길이(초)
IGNORE_TAIL_SEC = 10.0  # 각 파일 끝부분에서 검출을 무시할 구간 길이(초)

# Peak parameters
# Resp 신호를 baseline 제거 + smoothing 후, prominence 기반 find_peaks로 peak를 찾습니다.
# 이후 apnea 구간 peak 제거 및 호흡 간격(min/max breath) 조건으로 peak를 한 번 더 정제합니다.
BASELINE_WIN_SEC = 20.0  # 호흡 신호 baseline(중앙값) 추정 윈도우 길이(초)
SMOOTH_WIN_SEC = 0.4  # detrend 후 호흡 신호 smoothing 평균 윈도우 길이(초)
MIN_PEAK_DIST_SEC = 1.0  # peak 검출 시 최소 peak 간격(초)
MIN_BREATH_SEC = 1.0  # 유효 호흡 간격 최소값(초)
MAX_BREATH_SEC = 10.0  # 유효 호흡 간격 최대값(초)
PROM_FACTOR = 0.8  # peak prominence 최소값 산출 계수(비-apnea std 중앙값 배수)

# Apnea parameters
# Resp의 local std가 낮은 구간을 후보로 보고, 일정 길이 이상 연속될 때 apnea로 표시합니다.
# 임계치는 local std 분포의 하위 분위수(LOW_STD_QUANTILE)로 동적으로 결정합니다.
LOCAL_STD_WIN_SEC = 3.0  # apnea 탐지를 위한 local std 계산 윈도우 길이(초)
APNEA_MIN_SEC = 8.0  # apnea로 인정할 최소 연속 길이(초)
LOW_STD_QUANTILE = 0.10  # apnea 저변동 임계치 산출에 쓰는 local std 하위 분위수

# Sigh parameters
# Peak-to-peak 호흡 구간마다 Edi 최대값을 계산하고, apnea 겹침 호흡은 제외합니다.
# 직전 N호흡 대비 상대적 급증(rel rule) 또는 전체 분포 기반 절대 임계(abs rule) 중 하나를 만족하면 sigh 후보로 잡습니다.
SIGH_N_BREATHS = 20  # sigh 상대 규칙 기준으로 보는 직전 호흡 개수
SIGH_REL_FACTOR = 1.8  # sigh 상대 규칙 계수(현재 edi_peak >= 계수 * 과거 중앙값)
SIGH_ABS_QUANTILE = 0.95  # sigh 절대 규칙 분위수(abs threshold 계산용)
SIGH_MIN_BREATHS_FOR_RULE = 5  # 상대 규칙 계산에 필요한 최소 유효 과거 호흡 수
SIGH_MERGE_SEC = 0.5  # 이 시간 이내로 가까운 sigh 후보들은 하나의 sigh로 병합

# Evaluation parameters
DELTA_EVAL_MS = 140  # GT peak 매칭 평가 허용 오차(ms)
DELTA_EVAL_SEC = DELTA_EVAL_MS / 1000.0  # GT peak 매칭 평가 허용 오차(초)


def normalize_patient_id(v: str) -> str:
    s = str(v).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"Invalid patient id: {v!r}")
    return f"{int(m.group(1)):02d}"


def infer_time_seconds(ts: np.ndarray, fs: float) -> np.ndarray:
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 3:
        return np.arange(len(ts), dtype=float) / fs

    d = np.diff(ts)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return np.arange(len(ts), dtype=float) / fs

    dt = float(np.median(d))
    if 0.005 <= dt <= 0.02:
        return ts
    if 5.0 <= dt <= 20.0:
        return ts / 1000.0
    if 0.5 <= dt <= 2.0:
        return ts / fs
    return np.arange(len(ts), dtype=float) / fs


def rolling_centered_median(x: np.ndarray, win: int) -> np.ndarray:
    return pd.Series(x).rolling(window=win, center=True, min_periods=1).median().to_numpy()


def rolling_centered_mean(x: np.ndarray, win: int) -> np.ndarray:
    return pd.Series(x).rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def rolling_centered_std(x: np.ndarray, win: int) -> np.ndarray:
    return pd.Series(x).rolling(window=win, center=True, min_periods=1).std(ddof=0).to_numpy()


def mark_long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_len:
            out[i:j] = True
        i = j
    return out


def resolve_columns(df: pd.DataFrame) -> Tuple[str, str, str, pd.DataFrame]:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    ts_candidates = ["timestamp", "time", "t", "ts"]
    resp_candidates = ["resp", "respiratory", "semg", "semg_resp", "resp_signal", "breath"]
    edi_candidates = ["edi", "edi_signal"]

    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    resp_col = next((c for c in resp_candidates if c in df.columns), None)
    edi_col = next((c for c in edi_candidates if c in df.columns), None)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if ts_col is None:
        if not num_cols:
            raise ValueError("No numeric timestamp-like column found.")
        ts_col = num_cols[0]

    signal_cols = [c for c in num_cols if c != ts_col]
    if resp_col is None and edi_col is not None:
        resp_col = edi_col
    elif edi_col is None and resp_col is not None:
        edi_col = resp_col
    elif resp_col is None and edi_col is None:
        if len(signal_cols) >= 2:
            resp_col, edi_col = signal_cols[0], signal_cols[1]
        elif len(signal_cols) == 1:
            resp_col = signal_cols[0]
            edi_col = signal_cols[0]
        else:
            raise ValueError("Could not resolve respiratory/edi columns.")

    return ts_col, resp_col, edi_col, df


def find_peak_file(peak_dir: Path, pid: str) -> Path | None:
    candidates = [
        peak_dir / f"{pid}.xlsx",
        peak_dir / f"patient_{pid}.xlsx",
        peak_dir / f"refined_peaks_patient_{pid}.xlsx",
    ]
    for c in candidates:
        if c.exists():
            return c

    glob_hit = sorted(peak_dir.glob(f"*{pid}*.xlsx"))
    if len(glob_hit) == 1:
        return glob_hit[0]
    return None


def load_gt_peak_indices(peak_file: Path | None, raw_ts: np.ndarray) -> np.ndarray:
    if peak_file is None:
        return np.array([], dtype=int)

    peaks = pd.read_excel(peak_file)
    peaks.columns = [str(c).strip().lower() for c in peaks.columns]

    ts_col = next((c for c in ["timestamp", "time", "t", "ts"] if c in peaks.columns), None)
    if ts_col is None:
        num_cols = [c for c in peaks.columns if pd.api.types.is_numeric_dtype(peaks[c])]
        if not num_cols:
            return np.array([], dtype=int)
        ts_col = num_cols[0]

    peak_ts = pd.to_numeric(peaks[ts_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(peak_ts) == 0:
        return np.array([], dtype=int)

    peak_ts = np.unique(peak_ts)
    raw_ts = np.asarray(raw_ts, dtype=float)

    ins = np.searchsorted(raw_ts, peak_ts)
    ins = np.clip(ins, 1, len(raw_ts) - 1)

    left = ins - 1
    right = ins
    choose_left = np.abs(peak_ts - raw_ts[left]) <= np.abs(raw_ts[right] - peak_ts)
    nearest = np.where(choose_left, left, right)
    nearest = nearest[(nearest >= 0) & (nearest < len(raw_ts))]
    return np.unique(nearest.astype(int))


def evaluate_peak_detection(detected_idx: np.ndarray, gt_idx: np.ndarray, t_sec: np.ndarray, delta_sec: float) -> Dict[str, float]:
    detected_idx = np.sort(np.asarray(detected_idx, dtype=int))
    gt_idx = np.sort(np.asarray(gt_idx, dtype=int))

    det_t = t_sec[detected_idx] if len(detected_idx) > 0 else np.array([], dtype=float)
    gt_t = t_sec[gt_idx] if len(gt_idx) > 0 else np.array([], dtype=float)

    i, j = 0, 0
    tp = 0
    while i < len(det_t) and j < len(gt_t):
        dt = det_t[i] - gt_t[j]
        if abs(dt) <= delta_sec:
            tp += 1
            i += 1
            j += 1
        elif det_t[i] < gt_t[j] - delta_sec:
            i += 1
        else:
            j += 1

    fp = len(det_t) - tp
    fn = len(gt_t) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "eval_detected_count": int(len(det_t)),
        "eval_gt_count": int(len(gt_t)),
        "eval_tp": int(tp),
        "eval_fp": int(fp),
        "eval_fn": int(fn),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1_score": float(f1),
    }


def compute_breath_intervals_from_peaks(peak_idx: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(peak_idx, dtype=int)
    if p.size < 2:
        return np.empty((0, 2), dtype=int)

    if not np.all(np.diff(p) > 0):
        p = np.unique(p)
    if p.size < 2:
        return np.empty((0, 2), dtype=int)

    p = np.clip(p, 0, max(0, int(n)))
    starts = p[:-1]
    ends = p[1:]
    valid = ends > starts
    if not np.any(valid):
        return np.empty((0, 2), dtype=int)
    return np.column_stack([starts[valid], ends[valid]]).astype(int)


def compute_per_breath_edi_peak(edi: np.ndarray, breath_intervals: np.ndarray) -> np.ndarray:
    edi = np.asarray(edi, dtype=float)
    intervals = np.asarray(breath_intervals, dtype=int)
    out = np.full(len(intervals), np.nan, dtype=float)
    for i, (s, e) in enumerate(intervals):
        if s < 0 or e > len(edi) or e <= s:
            continue
        seg = edi[s:e]
        if seg.size == 0:
            continue
        if np.any(np.isfinite(seg)):
            out[i] = float(np.nanmax(seg))
    return out


def detect_sigh_candidates(
    edi_raw: np.ndarray,
    peak_idx: np.ndarray,
    apnea_mask: np.ndarray,
    fs: int,
    n_breaths: int,
    rel_factor: float,
    abs_quantile: float,
    merge_sec: float = SIGH_MERGE_SEC,
    edi_baseline_win_sec: float | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    edi_raw = np.asarray(edi_raw, dtype=float)
    apnea_mask = np.asarray(apnea_mask, dtype=bool)
    n = len(edi_raw)

    if len(apnea_mask) != n:
        raise ValueError("apnea_mask length must match edi_raw length.")

    if edi_baseline_win_sec is not None:
        win = max(1, int(edi_baseline_win_sec * fs))
        edi_base = rolling_centered_median(edi_raw, win)
        edi = edi_raw - edi_base
    else:
        edi = edi_raw

    breath_intervals = compute_breath_intervals_from_peaks(peak_idx, n)
    b = len(breath_intervals)
    sigh_mask = np.zeros(n, dtype=bool)
    if b == 0:
        return sigh_mask, {
            "sigh_n_breaths": int(n_breaths),
            "sigh_rel_factor": float(rel_factor),
            "sigh_abs_quantile": float(abs_quantile),
            "sigh_abs_threshold": float("nan"),
            "n_sigh_candidates": 0,
            "sigh_candidate_ratio": 0.0,
            "sigh_merge_sec": float(merge_sec),
        }

    edi_peak = compute_per_breath_edi_peak(edi, breath_intervals)
    overlap_apnea = np.zeros(b, dtype=bool)
    for i, (s, e) in enumerate(breath_intervals):
        if s < 0 or e <= s or e > n:
            overlap_apnea[i] = True
            edi_peak[i] = np.nan
            continue
        if apnea_mask[s:e].any():
            overlap_apnea[i] = True
            edi_peak[i] = np.nan

    finite_peak = np.isfinite(edi_peak)
    q = float(np.clip(abs_quantile, 0.0, 1.0))
    abs_thr = float(np.quantile(edi_peak[finite_peak], q)) if np.any(finite_peak) else float("nan")
    c_abs = finite_peak & np.isfinite(abs_thr) & (edi_peak >= abs_thr)

    c_rel = np.zeros(b, dtype=bool)
    for i in range(b):
        if not finite_peak[i] or overlap_apnea[i]:
            continue
        ref = edi_peak[max(0, i - int(n_breaths)) : i]
        ref = ref[np.isfinite(ref)]
        if ref.size < SIGH_MIN_BREATHS_FOR_RULE:
            continue
        m = float(np.median(ref))
        if np.isfinite(m):
            c_rel[i] = edi_peak[i] >= float(rel_factor) * m

    sigh_breath = (~overlap_apnea) & (c_rel | c_abs)

    # Label a sigh breath by a single representative sample: argmax Edi within the breath.
    raw_sigh_samples: list[int] = []
    for i, flag in enumerate(sigh_breath):
        if not flag:
            continue
        s, e = breath_intervals[i]
        if e <= s:
            continue
        seg = edi[s:e]
        if seg.size == 0 or np.all(~np.isfinite(seg)):
            continue
        t_local = int(np.nanargmax(seg))
        raw_sigh_samples.append(int(s + t_local))

    # Merge nearby sigh candidates within merge_sec into one event (keep highest-edi representative).
    merge_samples = max(0, int(round(float(merge_sec) * fs)))
    if raw_sigh_samples:
        raw_sigh_samples = sorted(set(raw_sigh_samples))
        merged_samples: list[int] = []
        cluster: list[int] = [raw_sigh_samples[0]]
        for t in raw_sigh_samples[1:]:
            if (t - cluster[-1]) <= merge_samples:
                cluster.append(t)
            else:
                best = max(cluster, key=lambda x: edi[x] if np.isfinite(edi[x]) else -np.inf)
                merged_samples.append(int(best))
                cluster = [t]
        best = max(cluster, key=lambda x: edi[x] if np.isfinite(edi[x]) else -np.inf)
        merged_samples.append(int(best))
        sigh_mask[np.asarray(merged_samples, dtype=int)] = True

    n_sigh = int(np.sum(sigh_mask))
    params = {
        "sigh_n_breaths": int(n_breaths),
        "sigh_rel_factor": float(rel_factor),
        "sigh_abs_quantile": float(q),
        "sigh_abs_threshold": float(abs_thr),
        "sigh_merge_sec": float(merge_sec),
        "n_sigh_candidates": n_sigh,
        "sigh_candidate_ratio": float(n_sigh / b) if b > 0 else 0.0,
        "sigh_breath_count": int(b),
        "sigh_eligible_breath_count": int(np.sum(~overlap_apnea)),
        "n_sigh_raw_before_merge": int(len(raw_sigh_samples)),
    }
    return sigh_mask, params


def detect_for_patient(
    raw_file: Path,
    peak_dir: Path,
    ignore_head_sec: float = IGNORE_HEAD_SEC,
    ignore_tail_sec: float = IGNORE_TAIL_SEC,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pid = normalize_patient_id(raw_file.stem)
    raw = pd.read_excel(raw_file)
    ts_col, resp_col, edi_col, raw = resolve_columns(raw)

    df = raw[[ts_col, resp_col, edi_col]].copy().dropna().reset_index(drop=True)
    df.columns = ["timestamp", "resp", "edi"]
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["resp"] = pd.to_numeric(df["resp"], errors="coerce")
    df["edi"] = pd.to_numeric(df["edi"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    raw_ts = df["timestamp"].to_numpy(dtype=float)
    t_sec = infer_time_seconds(raw_ts, FS)
    resp_raw = df["resp"].to_numpy(dtype=float)
    ignore_head_sec = float(max(0.0, ignore_head_sec))
    ignore_tail_sec = float(max(0.0, ignore_tail_sec))
    t0 = float(t_sec[0]) if len(t_sec) > 0 else 0.0
    t1 = float(t_sec[-1]) if len(t_sec) > 0 else 0.0
    valid_mask = (t_sec >= (t0 + ignore_head_sec)) & (t_sec <= (t1 - ignore_tail_sec))

    baseline = rolling_centered_median(resp_raw, int(BASELINE_WIN_SEC * FS))
    resp_detrended = resp_raw - baseline
    resp_proc = rolling_centered_mean(resp_detrended, int(SMOOTH_WIN_SEC * FS))

    local_std = rolling_centered_std(resp_proc, int(LOCAL_STD_WIN_SEC * FS))
    std_ref = local_std[np.isfinite(local_std) & valid_mask]
    if len(std_ref) == 0:
        std_ref = local_std[np.isfinite(local_std)]
    low_std_thr = float(np.quantile(std_ref, LOW_STD_QUANTILE))
    apnea_mask = mark_long_runs(local_std <= low_std_thr, int(APNEA_MIN_SEC * FS))
    apnea_mask &= valid_mask

    non_apnea_std = local_std[~apnea_mask]
    non_apnea_std = non_apnea_std[np.isfinite(non_apnea_std) & valid_mask[~apnea_mask]]
    if len(non_apnea_std) == 0:
        peak_idx = np.array([], dtype=int)
        prominence_min = float("nan")
    else:
        prominence_min = float(np.median(non_apnea_std) * PROM_FACTOR)
        peak_idx, _ = find_peaks(
            resp_proc,
            distance=int(MIN_PEAK_DIST_SEC * FS),
            prominence=prominence_min,
        )
        peak_idx = peak_idx[~apnea_mask[peak_idx]]
        peak_idx = peak_idx[valid_mask[peak_idx]]
        if len(peak_idx) >= 2:
            d = np.diff(peak_idx)
            keep = np.ones(len(peak_idx), dtype=bool)
            keep[1:] &= (d >= int(MIN_BREATH_SEC * FS)) & (d <= int(MAX_BREATH_SEC * FS))
            peak_idx = peak_idx[keep]

    peak_file = find_peak_file(peak_dir, pid)
    gt_idx = load_gt_peak_indices(peak_file, raw_ts)
    eval_result = evaluate_peak_detection(peak_idx, gt_idx, t_sec, DELTA_EVAL_SEC)

    gt_peak_mask = np.zeros(len(df), dtype=bool)
    detected_peak_mask = np.zeros(len(df), dtype=bool)

    if len(gt_idx) > 0:
        gt_peak_mask[gt_idx] = True
    if len(peak_idx) > 0:
        detected_peak_mask[peak_idx] = True

    edi_raw = df["edi"].to_numpy(dtype=float)
    sigh_mask, sigh_params = detect_sigh_candidates(
        edi_raw=edi_raw,
        peak_idx=peak_idx,
        apnea_mask=apnea_mask,
        fs=FS,
        n_breaths=SIGH_N_BREATHS,
        rel_factor=SIGH_REL_FACTOR,
        abs_quantile=SIGH_ABS_QUANTILE,
        merge_sec=SIGH_MERGE_SEC,
        edi_baseline_win_sec=None,
    )

    result = pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(),
            "edi": df["edi"].to_numpy(),
            "gt_peak": gt_peak_mask,
            "detected_peak": detected_peak_mask,
            "detected_apnea": apnea_mask.astype(bool),
            "detected_sigh": sigh_mask.astype(bool),
        }
    )

    params = {
        "patient_id": pid,
        "raw_file": raw_file.name,
        "peak_file": peak_file.name if peak_file is not None else "",
        "timestamp_col": ts_col,
        "resp_col_for_detection": resp_col,
        "edi_col_for_output": edi_col,
        "fs_hz": FS,
        "baseline_win_sec": BASELINE_WIN_SEC,
        "smooth_win_sec": SMOOTH_WIN_SEC,
        "local_std_win_sec": LOCAL_STD_WIN_SEC,
        "apnea_min_sec": APNEA_MIN_SEC,
        "min_peak_dist_sec": MIN_PEAK_DIST_SEC,
        "min_breath_sec": MIN_BREATH_SEC,
        "max_breath_sec": MAX_BREATH_SEC,
        "low_std_quantile": LOW_STD_QUANTILE,
        "prom_factor": PROM_FACTOR,
        "sigh_n_breaths": SIGH_N_BREATHS,
        "sigh_rel_factor": SIGH_REL_FACTOR,
        "sigh_abs_quantile": SIGH_ABS_QUANTILE,
        "sigh_min_breaths_for_rule": SIGH_MIN_BREATHS_FOR_RULE,
        "sigh_merge_sec": SIGH_MERGE_SEC,
        "ignore_head_sec": ignore_head_sec,
        "ignore_tail_sec": ignore_tail_sec,
        "delta_eval_ms": DELTA_EVAL_MS,
        "low_std_threshold": low_std_thr,
        "prominence_min": prominence_min,
        "n_samples": int(len(result)),
        "n_detected_peaks": int(detected_peak_mask.sum()),
        "n_gt_peaks": int(gt_peak_mask.sum()),
        "apnea_ratio": float(apnea_mask.mean()),
        "duration_sec": float(t_sec[-1] - t_sec[0]) if len(t_sec) > 1 else 0.0,
    }
    params.update(eval_result)
    params.update(sigh_params)
    return result, params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moving-window peak detection for all patients")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--peak-dir", type=Path, default=DEFAULT_PEAK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--patient-id",
        action="append",
        default=None,
        help="Optional patient id filter. Can be passed multiple times (e.g. --patient-id 03 --patient-id 04).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y%m%d"),
        help="Output subdirectory date in YYYYMMDD format. Default: today.",
    )
    parser.add_argument(
        "--ignore-head-sec",
        type=float,
        default=IGNORE_HEAD_SEC,
        help=f"Ignore detection in the first N seconds of each file (default: {IGNORE_HEAD_SEC}).",
    )
    parser.add_argument(
        "--ignore-tail-sec",
        type=float,
        default=IGNORE_TAIL_SEC,
        help=f"Ignore detection in the last N seconds of each file (default: {IGNORE_TAIL_SEC}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"\d{8}", args.date):
        raise ValueError("--date must be YYYYMMDD (e.g. 20260226)")

    dated_output_dir = args.output_dir / args.date
    dated_output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(args.raw_dir.glob("*.xlsx"), key=lambda p: normalize_patient_id(p.stem))
    if args.patient_id:
        allowed = {normalize_patient_id(v) for v in args.patient_id}
        raw_files = [p for p in raw_files if normalize_patient_id(p.stem) in allowed]

    if not raw_files:
        raise FileNotFoundError(f"No patient raw files found in: {args.raw_dir}")

    print(f"Processing {len(raw_files)} patient(s)...")
    for raw_file in raw_files:
        pid = normalize_patient_id(raw_file.stem)
        result_df, params = detect_for_patient(
            raw_file,
            args.peak_dir,
            ignore_head_sec=args.ignore_head_sec,
            ignore_tail_sec=args.ignore_tail_sec,
        )

        out_path = dated_output_dir / f"movingwinddetected_patient_{pid}.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="data")
            pd.DataFrame({"parameter": list(params.keys()), "value": list(params.values())}).to_excel(
                writer, index=False, sheet_name="params"
            )

        print(
            f"[OK] patient={pid} rows={len(result_df)} detected={int(result_df['detected_peak'].sum())} "
            f"gt={int(result_df['gt_peak'].sum())} apnea_ratio={result_df['detected_apnea'].mean():.4f} -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
