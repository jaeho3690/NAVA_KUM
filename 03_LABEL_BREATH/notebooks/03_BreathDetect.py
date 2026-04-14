from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances


# =========================
# User Arguments (edit here)
# =========================
NOTEBOOKS_DIR = Path(__file__).resolve().parent
BASE_DIR = NOTEBOOKS_DIR.parent
PARENT_DIR = BASE_DIR / "stored_results" / "02_summarized"
ANALYSIS_FOLDER = "20260301"

TIMESTAMP_COL = "timestamp"
EDI_COL = "edi"
PEAK_COLS = ["detected_peak", "gt_peak"]  # OR
PEAK_MERGE_SEC = 0.25  # if two peaks are closer than this, keep the higher y

APNEA_LABEL_STRATEGY = "any"  # "any" | "all" | "specific"
SPECIFIC_APNEA_LABELERS = ["apnea_label_김재호", "apnea_label_오창준"]

BASELINE_MEDIAN_SEC = 20.0
SMOOTHING_SEC = 0.4
SAMPLE_RATE_HZ = None  # None => infer from timestamp

MIN_BREATH_SEC = 1.0
MAX_BREATH_SEC = 10.0
EDGE_EXCLUDE_SEC = 10.0

CLUSTER_K_MIN = 2
CLUSTER_K_MAX = 9

BEST_K = 7
CLUSTER_LINKAGE = "average"
CLUSTER_DISTANCE_METRIC = "mse"  # "mse" | "manhattan" | "mae"

# Output root requested by user
OUT_ROOT = NOTEBOOKS_DIR / "outputs" / "03_breath_detect"


@dataclass
class PatientRunResult:
    patient_id: str
    n_breaths: int
    n_clustered_breaths: int
    best_k: int | None
    best_silhouette: float | None
    status: str
    message: str = ""


def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) > 0
    text = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y"}
    return text.isin(true_set)


def infer_sample_rate_hz(ts: np.ndarray, sample_rate_override: float | None) -> float:
    if sample_rate_override is not None:
        return float(sample_rate_override)
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Cannot infer sample rate from timestamp.")
    med_dt = float(np.median(dt))
    return 1000.0 / med_dt if med_dt > 1.5 else 1.0 / med_dt


def rolling_median_np(arr: np.ndarray, win: int) -> np.ndarray:
    return pd.Series(arr).rolling(window=win, center=True, min_periods=1).median().to_numpy()


def rolling_mean_np(arr: np.ndarray, win: int) -> np.ndarray:
    return pd.Series(arr).rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def get_patient_id_from_path(path: Path) -> str:
    m = re.search(r"patient_(\d+)\.xlsx$", path.name)
    if not m:
        return path.stem
    return m.group(1)


def build_apnea_mask(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    apnea_label_cols = [c for c in df.columns if c.startswith("apnea_label_")]
    if not apnea_label_cols:
        raise KeyError("No apnea_label_* columns found.")

    if APNEA_LABEL_STRATEGY == "specific":
        use_cols = [c for c in SPECIFIC_APNEA_LABELERS if c in apnea_label_cols]
        if not use_cols:
            raise KeyError("No SPECIFIC_APNEA_LABELERS columns found.")
    else:
        use_cols = apnea_label_cols

    votes = np.column_stack([to_bool_series(df[c]).to_numpy() for c in use_cols])
    if APNEA_LABEL_STRATEGY == "all":
        mask = votes.all(axis=1)
    else:
        mask = votes.any(axis=1)
    return mask.astype(bool), use_cols


def merge_close_peaks(candidate_idx: np.ndarray, time_sec: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(candidate_idx) == 0:
        return np.array([], dtype=int)

    merged: list[int] = []
    cluster = [int(candidate_idx[0])]
    for idx in candidate_idx[1:]:
        idx = int(idx)
        if (time_sec[idx] - time_sec[cluster[-1]]) <= PEAK_MERGE_SEC:
            cluster.append(idx)
        else:
            merged.append(max(cluster, key=lambda j: y[j]))
            cluster = [idx]
    merged.append(max(cluster, key=lambda j: y[j]))
    return np.array(sorted(set(merged)), dtype=int)


def build_breaths_all(
    ts: np.ndarray,
    edi_raw: np.ndarray,
    edi_detrended: np.ndarray,
    edi_smooth: np.ndarray,
    peak_idx: np.ndarray,
    analysis_window_mask: np.ndarray,
    is_ms: bool,
    patient_id: str,
) -> pd.DataFrame:
    if len(peak_idx) < 2:
        return pd.DataFrame()

    valleys: list[int] = []
    for i in range(len(peak_idx) - 1):
        left = int(peak_idx[i])
        right = int(peak_idx[i + 1])
        if right <= left:
            continue
        v_rel = np.argmin(edi_smooth[left : right + 1])
        valleys.append(left + int(v_rel))
    valleys = sorted(set(valleys))

    records = []
    breath_id = 0
    for i in range(len(valleys) - 1):
        v_start = valleys[i]
        v_end = valleys[i + 1]
        if v_end <= v_start:
            continue

        if not (edi_detrended[v_start] < 0 and edi_detrended[v_end] < 0):
            continue
        if not (analysis_window_mask[v_start] and analysis_window_mask[v_end]):
            continue

        dur = (ts[v_end] - ts[v_start]) / (1000.0 if is_ms else 1.0)
        if dur < MIN_BREATH_SEC or dur > MAX_BREATH_SEC:
            continue

        in_seg = peak_idx[(peak_idx >= v_start) & (peak_idx <= v_end)]
        rel_peaks = (in_seg - v_start).astype(int)
        filtered_sig = edi_smooth[v_start : v_end + 1]
        original_sig = edi_raw[v_start : v_end + 1]

        if len(rel_peaks) == 0:
            anchor_rel = int(np.argmax(filtered_sig))
        else:
            cand = rel_peaks[(rel_peaks >= 0) & (rel_peaks < len(filtered_sig))]
            if len(cand) == 0:
                anchor_rel = int(np.argmax(filtered_sig))
            else:
                anchor_rel = int(cand[np.argmax(filtered_sig[cand])])

        records.append(
            {
                "patient_id": patient_id,
                "breath_id": breath_id,
                "breath_label": f"breath_{breath_id}",
                "v_start_idx": int(v_start),
                "v_end_idx": int(v_end),
                "v_start_t": float(ts[v_start]),
                "v_end_t": float(ts[v_end]),
                "breath_duration_sec": float(dur),
                "n_peaks": int(len(in_seg)),
                "peak_indices": in_seg.astype(int).tolist(),
                "peak_rel_indices": rel_peaks.tolist(),
                "anchor_peak_rel_idx": int(anchor_rel),
                "signal_len": int(len(filtered_sig)),
                "filtered_edi_signal": filtered_sig.astype(float).tolist(),
                "original_edi_signal": original_sig.astype(float).tolist(),
            }
        )
        breath_id += 1

    return pd.DataFrame(records)


def align_by_anchor_peak(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    out = df.copy()
    peak_rel = out["anchor_peak_rel_idx"].astype(int).to_numpy()
    sig_len = out["signal_len"].astype(int).to_numpy()

    left_ctx = peak_rel
    right_ctx = sig_len - 1 - peak_rel
    aligned_peak_idx = int(left_ctx.max())
    target_len = int(aligned_peak_idx + 1 + right_ctx.max())

    filtered_aligned_signals = []
    original_aligned_signals = []
    zero_pad_mask = []
    left_pad_len = []
    right_pad_len = []
    for _, row in out.iterrows():
        filtered_sig = np.asarray(row["filtered_edi_signal"], dtype=float)
        original_sig = np.asarray(row["original_edi_signal"], dtype=float)
        p = int(row["anchor_peak_rel_idx"])
        L = len(filtered_sig)
        lp = aligned_peak_idx - p
        rp = target_len - (lp + L)
        if lp < 0 or rp < 0:
            raise RuntimeError("Negative pad length in alignment.")

        padded_filtered = np.pad(filtered_sig, (lp, rp), mode="constant", constant_values=0.0)
        padded_original = np.pad(original_sig, (lp, rp), mode="constant", constant_values=0.0)
        mask = np.zeros(target_len, dtype=bool)
        if lp > 0:
            mask[:lp] = True
        if rp > 0:
            mask[lp + L :] = True

        filtered_aligned_signals.append(padded_filtered.tolist())
        original_aligned_signals.append(padded_original.tolist())
        zero_pad_mask.append(mask.tolist())
        left_pad_len.append(int(lp))
        right_pad_len.append(int(rp))

    out["aligned_peak_idx"] = aligned_peak_idx
    out["filtered_signal_aligned"] = filtered_aligned_signals
    out["original_signal_aligned"] = original_aligned_signals
    out["zero_pad_mask_aligned"] = zero_pad_mask
    out["left_pad_len"] = left_pad_len
    out["right_pad_len"] = right_pad_len
    out["target_len"] = target_len
    return out


def build_distance_matrix(X: np.ndarray, metric_name: str) -> np.ndarray:
    m = metric_name.lower().strip()
    if m == "manhattan":
        return pairwise_distances(X, metric="manhattan")
    if m == "mae":
        return pairwise_distances(X, metric="manhattan") / X.shape[1]
    if m == "mse":
        return pairwise_distances(X, metric="sqeuclidean") / X.shape[1]
    raise ValueError("CLUSTER_DISTANCE_METRIC must be one of: 'mse', 'manhattan', 'mae'.")


def run_clustering(df_aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, float]:
    X = np.vstack(df_aligned["filtered_signal_aligned"].apply(lambda v: np.asarray(v, dtype=float)).to_numpy())
    D = build_distance_matrix(X, CLUSTER_DISTANCE_METRIC)

    n_samples = X.shape[0]
    k_min = max(2, int(CLUSTER_K_MIN))
    k_max = min(int(CLUSTER_K_MAX), n_samples - 1)
    if k_max < k_min:
        raise ValueError(f"Not enough samples ({n_samples}) for cluster range {k_min}..{CLUSTER_K_MAX}.")

    rows = []
    for k in range(k_min, k_max + 1):
        try:
            model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=CLUSTER_LINKAGE)
        except TypeError:
            model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage=CLUSTER_LINKAGE)
        labels = model.fit_predict(D)
        if len(np.unique(labels)) < 2:
            continue
        sil = float(silhouette_score(D, labels, metric="precomputed"))
        rows.append({"k": int(k), "silhouette": sil})

    sil_df = pd.DataFrame(rows)
    if sil_df.empty:
        raise RuntimeError("Silhouette computation failed for all k.")

    best_k = int(BEST_K)
    if best_k < k_min or best_k > k_max:
        raise ValueError(f"BEST_K={best_k} is out of valid range [{k_min}, {k_max}] for n_samples={n_samples}.")
    best_row = sil_df[sil_df["k"] == best_k]
    best_sil = float(best_row["silhouette"].iloc[0]) if len(best_row) else float("nan")

    try:
        best_model = AgglomerativeClustering(n_clusters=best_k, metric="precomputed", linkage=CLUSTER_LINKAGE)
    except TypeError:
        best_model = AgglomerativeClustering(n_clusters=best_k, affinity="precomputed", linkage=CLUSTER_LINKAGE)

    out = df_aligned.copy()
    out["cluster_label"] = best_model.fit_predict(D)

    # Keep original cluster labels, then collapse into major vs non-major:
    # major_cluster=True only for the most populous original cluster.
    label_counts = out["cluster_label"].value_counts().sort_values(ascending=False)
    major_label = int(label_counts.index[0])
    out["major_cluster"] = (out["cluster_label"].astype(int) == major_label)
    out["major_cluster"] = out["major_cluster"].astype(bool)
    return out, sil_df, best_k, best_sil


def compute_cluster_breath_stats(cluster_df: pd.DataFrame, fs: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _features(row: pd.Series) -> pd.Series:
        sig = np.asarray(row["filtered_edi_signal"], dtype=float)
        L = len(sig)
        if L == 0:
            return pd.Series(
                {
                    "signal_len": 0,
                    "max_edi": np.nan,
                    "min_edi": np.nan,
                    "mean_edi": np.nan,
                    "std_edi": np.nan,
                    "ptp_edi": np.nan,
                    "auc": np.nan,
                    "auc_abs": np.nan,
                    "auc_pos": np.nan,
                    "peak_edi": np.nan,
                    "rise_time_sec": np.nan,
                    "fall_time_sec": np.nan,
                    "rise_slope": np.nan,
                    "fall_slope": np.nan,
                }
            )

        p = int(row["anchor_peak_rel_idx"])
        p = max(0, min(p, L - 1))

        max_edi = float(np.max(sig))
        min_edi = float(np.min(sig))
        mean_edi = float(np.mean(sig))
        std_edi = float(np.std(sig))
        ptp_edi = max_edi - min_edi

        auc = float(np.trapz(sig, dx=1.0 / fs))
        auc_abs = float(np.trapz(np.abs(sig), dx=1.0 / fs))
        auc_pos = float(np.trapz(np.clip(sig, 0, None), dx=1.0 / fs))

        peak_edi = float(sig[p])
        rise_time_sec = float(p / fs)
        fall_time_sec = float((L - 1 - p) / fs)
        rise_slope = (peak_edi - float(sig[0])) / max(rise_time_sec, 1.0 / fs)
        fall_slope = (peak_edi - float(sig[-1])) / max(fall_time_sec, 1.0 / fs)

        return pd.Series(
            {
                "signal_len": int(L),
                "max_edi": max_edi,
                "min_edi": min_edi,
                "mean_edi": mean_edi,
                "std_edi": std_edi,
                "ptp_edi": ptp_edi,
                "auc": auc,
                "auc_abs": auc_abs,
                "auc_pos": auc_pos,
                "peak_edi": peak_edi,
                "rise_time_sec": rise_time_sec,
                "fall_time_sec": fall_time_sec,
                "rise_slope": float(rise_slope),
                "fall_slope": float(fall_slope),
            }
        )

    feat_df = cluster_df.apply(_features, axis=1)
    stats_df = pd.concat([cluster_df, feat_df], axis=1)

    breath_cols = [
        "patient_id",
        "breath_id",
        "breath_label",
        "cluster_label",
        "major_cluster",
        "breath_duration_sec",
        "signal_len",
        "max_edi",
        "min_edi",
        "mean_edi",
        "std_edi",
        "ptp_edi",
        "auc",
        "auc_abs",
        "auc_pos",
        "peak_edi",
        "rise_time_sec",
        "fall_time_sec",
        "rise_slope",
        "fall_slope",
    ]
    breath_stats = stats_df[breath_cols].copy()

    summary_cols = [
        "breath_duration_sec",
        "signal_len",
        "max_edi",
        "min_edi",
        "mean_edi",
        "std_edi",
        "ptp_edi",
        "auc",
        "auc_abs",
        "auc_pos",
        "peak_edi",
        "rise_time_sec",
        "fall_time_sec",
        "rise_slope",
        "fall_slope",
    ]
    summary = breath_stats.groupby("cluster_label")[summary_cols].agg(["count", "mean", "std", "median"])
    summary = summary.reset_index()
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in summary.columns.to_flat_index()
    ]
    return breath_stats, summary


def plot_overlay_all_breath_signals(cluster_df: pd.DataFrame, patient_id: str, out_path: Path) -> None:
    if len(cluster_df) == 0:
        return
    labels_sorted = sorted(cluster_df["cluster_label"].unique().tolist())
    n_clusters = len(labels_sorted)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, max(2 * n_clusters, 3)), sharex=True, sharey=True)
    if n_clusters == 1:
        axes = [axes]

    X = np.vstack(cluster_df["filtered_signal_aligned"].apply(lambda v: np.asarray(v, dtype=float)).to_numpy())
    y = cluster_df["cluster_label"].to_numpy()
    for ax, k in zip(axes, labels_sorted):
        sigs = X[y == k]
        for sig in sigs:
            ax.plot(sig, alpha=0.1, linewidth=0.8)
        ax.set_title(f"Cluster {k} (n={len(sigs)})")
        ax.grid(alpha=0.3)

    plt.xlabel("Aligned time index")
    plt.suptitle(f"Overlaid aligned ALL-breath signals by cluster (patient {patient_id})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_silhouette_curve(sil_df: pd.DataFrame, patient_id: str, out_path: Path) -> None:
    if len(sil_df) == 0:
        return
    fig = plt.figure(figsize=(7, 4))
    plt.plot(sil_df["k"], sil_df["silhouette"], marker="o")
    plt.xticks(sil_df["k"].astype(int).tolist())
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title(
        f"Agglomerative silhouette sweep (patient {patient_id}) | metric={CLUSTER_DISTANCE_METRIC}"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def iter_patient_files(folder: Path) -> Iterable[Path]:
    return sorted(folder.glob("movingwinddetected_patient_*.xlsx"))


def run_one_patient(xlsx_path: Path, out_dir: Path) -> PatientRunResult:
    patient_id = get_patient_id_from_path(xlsx_path)
    print(f"[START] patient {patient_id} | {xlsx_path.name}")

    try:
        df = pd.read_excel(xlsx_path)
        required = [TIMESTAMP_COL, EDI_COL] + PEAK_COLS
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        ts = pd.to_numeric(df[TIMESTAMP_COL], errors="coerce").to_numpy()
        edi_raw = pd.to_numeric(df[EDI_COL], errors="coerce").to_numpy()
        peak_masks_raw = {c: to_bool_series(df[c]).to_numpy() for c in PEAK_COLS}

        valid = np.isfinite(ts) & np.isfinite(edi_raw)
        if valid.sum() < 10:
            raise ValueError("Too few valid samples.")
        df = df.loc[valid].reset_index(drop=True)
        ts = ts[valid]
        edi_raw = edi_raw[valid]
        peak_masks_raw = {k: v[valid] for k, v in peak_masks_raw.items()}

        fs = infer_sample_rate_hz(ts, SAMPLE_RATE_HZ)
        is_ms = np.median(np.diff(ts)) > 1.5
        time_sec = (ts - ts[0]) / (1000.0 if is_ms else 1.0)

        baseline_win = max(3, int(round(BASELINE_MEDIAN_SEC * fs)))
        smooth_win = max(3, int(round(SMOOTHING_SEC * fs)))
        baseline = rolling_median_np(edi_raw, baseline_win)
        edi_detrended = edi_raw - baseline
        edi_smooth = rolling_mean_np(edi_detrended, smooth_win)

        apnea_mask, _ = build_apnea_mask(df)
        analysis_window_mask = (time_sec >= EDGE_EXCLUDE_SEC) & (time_sec <= (time_sec[-1] - EDGE_EXCLUDE_SEC))

        cand_mask = np.zeros(len(df), dtype=bool)
        for c in PEAK_COLS:
            cand_mask |= peak_masks_raw[c]
        cand_mask &= (~apnea_mask) & analysis_window_mask
        candidate_idx = np.flatnonzero(cand_mask)
        peak_idx = merge_close_peaks(candidate_idx, time_sec, edi_smooth)

        breaths_all_df = build_breaths_all(
            ts=ts,
            edi_raw=edi_raw,
            edi_detrended=edi_detrended,
            edi_smooth=edi_smooth,
            peak_idx=peak_idx,
            analysis_window_mask=analysis_window_mask,
            is_ms=is_ms,
            patient_id=patient_id,
        )
        n_breaths = len(breaths_all_df)
        if n_breaths == 0:
            msg = "No breaths extracted after filtering."
            print(f"[SKIP] patient {patient_id} | {msg}")
            return PatientRunResult(patient_id, 0, 0, None, None, "skip", msg)

        cluster_input_df = align_by_anchor_peak(breaths_all_df)
        clustered_df, sil_df, best_k, best_sil = run_clustering(cluster_input_df)
        breath_stats_df, cluster_summary_df = compute_cluster_breath_stats(clustered_df, fs=fs)

        patient_out = out_dir / f"patient_{patient_id}"
        patient_out.mkdir(parents=True, exist_ok=True)

        merged_peak_mask = np.zeros(len(df), dtype=bool)
        merged_peak_mask[peak_idx] = True
        breath_mask = np.zeros(len(df), dtype=bool)
        breath_id_per_sample = np.full(len(df), pd.NA, dtype=object)
        breath_label_per_sample = np.full(len(df), "", dtype=object)
        for _, breath_row in breaths_all_df.iterrows():
            start_idx = int(breath_row["v_start_idx"])
            end_idx = int(breath_row["v_end_idx"])
            breath_id = int(breath_row["breath_id"])
            breath_label = str(breath_row["breath_label"])
            breath_mask[start_idx : end_idx + 1] = True
            breath_id_per_sample[start_idx : end_idx + 1] = breath_id
            breath_label_per_sample[start_idx : end_idx + 1] = breath_label
        edi_filtered_df = pd.DataFrame(
            {
                "patient_id": patient_id,
                "sample_idx": np.arange(len(df), dtype=int),
                "timestamp": ts.astype(float),
                "time_sec_from_start": time_sec.astype(float),
                "edi_raw": edi_raw.astype(float),
                "edi_baseline_median": baseline.astype(float),
                "edi_detrended": edi_detrended.astype(float),
                "edi_smooth_for_detection": edi_smooth.astype(float),
                "analysis_window_mask": analysis_window_mask.astype(bool),
                "apnea_mask": apnea_mask.astype(bool),
                "candidate_peak_mask": cand_mask.astype(bool),
                "merged_peak_mask": merged_peak_mask.astype(bool),
                "breath_mask": breath_mask.astype(bool),
                "breath_id": pd.array(breath_id_per_sample, dtype="Int64"),
                "breath_label": breath_label_per_sample,
            }
        )

        edi_filtered_df.to_pickle(patient_out / f"patient_{patient_id}_edi_filtered_signal.pkl")
        edi_filtered_df.to_csv(patient_out / f"patient_{patient_id}_edi_filtered_signal.csv", index=False)
        clustered_df.to_pickle(patient_out / f"patient_{patient_id}_clustered_breaths.pkl")
        clustered_df.to_csv(patient_out / f"patient_{patient_id}_clustered_breaths.csv", index=False)
        breath_stats_df.to_csv(patient_out / f"patient_{patient_id}_cluster_breath_stats.csv", index=False)
        cluster_summary_df.to_csv(patient_out / f"patient_{patient_id}_cluster_summary.csv", index=False)
        sil_df.to_csv(patient_out / f"patient_{patient_id}_silhouette.csv", index=False)

        plot_overlay_all_breath_signals(
            clustered_df,
            patient_id=patient_id,
            out_path=patient_out / f"patient_{patient_id}_overlay_all_breath_signals_by_cluster.png",
        )
        plot_silhouette_curve(
            sil_df,
            patient_id=patient_id,
            out_path=patient_out / f"patient_{patient_id}_silhouette_curve.png",
        )

        run_meta = pd.DataFrame(
            [
                {
                    "patient_id": patient_id,
                    "n_rows_raw": int(len(df)),
                    "n_candidate_peaks": int(len(candidate_idx)),
                    "n_peaks_merged": int(len(peak_idx)),
                    "n_breaths": int(len(breaths_all_df)),
                    "n_clustered_breaths": int(len(clustered_df)),
                    "major_cluster_label": int(clustered_df["cluster_label"][clustered_df["major_cluster"]].iloc[0]),
                    "major_cluster_count": int(clustered_df["major_cluster"].sum()),
                    "best_k": int(best_k),
                    "best_silhouette": float(best_sil),
                    "cluster_metric": CLUSTER_DISTANCE_METRIC,
                    "cluster_linkage": CLUSTER_LINKAGE,
                }
            ]
        )
        run_meta.to_csv(patient_out / f"patient_{patient_id}_run_meta.csv", index=False)

        print(
            f"[DONE] patient {patient_id} | breaths={len(clustered_df)} | best_k={best_k} | silhouette={best_sil:.4f}"
        )
        return PatientRunResult(
            patient_id=patient_id,
            n_breaths=n_breaths,
            n_clustered_breaths=len(clustered_df),
            best_k=best_k,
            best_silhouette=best_sil,
            status="ok",
            message="",
        )

    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] patient {patient_id} | {e}")
        return PatientRunResult(patient_id, 0, 0, None, None, "error", str(e))


def main() -> None:
    in_dir = PARENT_DIR / ANALYSIS_FOLDER
    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    out_dir = OUT_ROOT / ANALYSIS_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_files = list(iter_patient_files(in_dir))
    if not patient_files:
        raise FileNotFoundError(f"No patient files found in {in_dir}")

    results: list[PatientRunResult] = []
    all_patient_cluster_summary = []
    all_patient_breath_stats = []

    for fp in patient_files:
        res = run_one_patient(fp, out_dir=out_dir)
        results.append(res)

        if res.status != "ok":
            continue
        pid = res.patient_id
        patient_out = out_dir / f"patient_{pid}"

        summary_fp = patient_out / f"patient_{pid}_cluster_summary.csv"
        breath_stats_fp = patient_out / f"patient_{pid}_cluster_breath_stats.csv"
        if summary_fp.exists():
            sdf = pd.read_csv(summary_fp)
            sdf.insert(0, "patient_id", pid)
            all_patient_cluster_summary.append(sdf)
        if breath_stats_fp.exists():
            bdf = pd.read_csv(breath_stats_fp)
            all_patient_breath_stats.append(bdf)

    run_summary_df = pd.DataFrame([r.__dict__ for r in results])
    run_summary_df.to_csv(out_dir / "all_patients_run_summary.csv", index=False)

    if all_patient_cluster_summary:
        all_cluster_df = pd.concat(all_patient_cluster_summary, ignore_index=True)
        all_cluster_df.to_csv(out_dir / "all_patients_cluster_statistics.csv", index=False)

    if all_patient_breath_stats:
        all_breath_df = pd.concat(all_patient_breath_stats, ignore_index=True)
        all_breath_df.to_csv(out_dir / "all_patients_cluster_breath_stats.csv", index=False)

        patient_overview = (
            all_breath_df.groupby("patient_id")
            .agg(
                n_breaths=("breath_id", "count"),
                n_clusters=("cluster_label", "nunique"),
                mean_duration_sec=("breath_duration_sec", "mean"),
                mean_max_edi=("max_edi", "mean"),
                mean_auc_abs=("auc_abs", "mean"),
            )
            .reset_index()
        )
        patient_overview.to_csv(out_dir / "all_patients_overview.csv", index=False)

    print("\n=== Completed ===")
    print(f"Input:  {in_dir}")
    print(f"Output: {out_dir}")
    print("Saved:")
    print("- patient-wise cluster outputs under patient_<id>/")
    print("- all_patients_run_summary.csv")
    print("- all_patients_cluster_statistics.csv (if available)")
    print("- all_patients_cluster_breath_stats.csv (if available)")
    print("- all_patients_overview.csv (if available)")


if __name__ == "__main__":
    main()
