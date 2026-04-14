#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_PEAK = 0
CLASS_NORMAL = 1
CLASS_ABNORMAL = 2
BASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BASE_DIR.parent


def to_bool(text: str) -> bool:
    s = str(text).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


@dataclass
class BreathRow:
    v_start_idx: int
    v_end_idx: int
    major_cluster: bool


def parse_patient_id(path: Path) -> str:
    name = path.name
    if name.startswith("patient_"):
        return name.replace("patient_", "")
    return name


def load_signal(signal_csv: Path, signal_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys: list[float] = []
    peak_idx: list[int] = []
    ts: list[float] = []
    with signal_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if signal_col not in cols:
            raise KeyError(f"Missing signal column '{signal_col}' in {signal_csv}")
        if "sample_idx" not in cols:
            raise KeyError(f"Missing sample_idx column in {signal_csv}")
        if "timestamp" not in cols:
            raise KeyError(f"Missing timestamp column in {signal_csv}")

        use_peak_mask = "merged_peak_mask" in cols
        for row in reader:
            sidx = int(float(row["sample_idx"]))
            ys.append(float(row[signal_col]))
            ts.append(float(row["timestamp"]))
            if use_peak_mask and to_bool(row["merged_peak_mask"]):
                peak_idx.append(sidx)

    return (
        np.asarray(ys, dtype=float),
        np.asarray(sorted(set(peak_idx)), dtype=int),
        np.asarray(ts, dtype=float),
    )


def load_breaths(breath_csv: Path) -> list[BreathRow]:
    raw_rows: list[dict] = []
    with breath_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        needed = {"v_start_idx", "v_end_idx", "cluster_label"}
        missing = [c for c in needed if c not in cols]
        if missing:
            raise KeyError(f"Missing columns in {breath_csv}: {missing}")
        for row in reader:
            raw_rows.append(row)

    if not raw_rows:
        return []

    has_major = "major_cluster" in raw_rows[0]
    if has_major:
        return [
            BreathRow(
                v_start_idx=int(float(r["v_start_idx"])),
                v_end_idx=int(float(r["v_end_idx"])),
                major_cluster=to_bool(r["major_cluster"]),
            )
            for r in raw_rows
        ]

    # Backward compatibility for old files without major_cluster.
    counts = Counter(int(float(r["cluster_label"])) for r in raw_rows)
    major_label = counts.most_common(1)[0][0]
    return [
        BreathRow(
            v_start_idx=int(float(r["v_start_idx"])),
            v_end_idx=int(float(r["v_end_idx"])),
            major_cluster=(int(float(r["cluster_label"])) == major_label),
        )
        for r in raw_rows
    ]


def build_windows(n: int, win: int, stride: int) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    if n <= win:
        return [(0, n)]

    starts = list(range(0, n - win + 1, stride))
    if starts[-1] != (n - win):
        starts.append(n - win)
    return [(s, s + win) for s in starts]


def y_to_norm(y: float, y_min: float, y_max: float) -> float:
    if y_max <= y_min:
        return 0.5
    return clamp01((y_max - y) / (y_max - y_min))


def infer_fs_hz(ts: np.ndarray) -> float:
    if len(ts) < 2:
        return 100.0
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 100.0
    med_dt = float(np.median(dt))
    return 1000.0 / med_dt if med_dt > 1.5 else 1.0 / med_dt


def render_window_image_cj(
    sig: np.ndarray,
    x: np.ndarray,
    out_path: Path,
    width: int,
    height: int,
    dpi: int,
    padding_y: float,
) -> tuple[float, float, tuple[float, float, float, float]]:
    # Match 02_CJ_Final_Data_Process style:
    # - dynamic y-range with padding
    # - axis off
    # - tight save on white background
    if len(sig) == 0:
        raise ValueError("Empty signal window.")
    data_min = float(np.min(sig))
    data_max = float(np.max(sig))
    y_min = data_min - float(padding_y)
    y_max = data_max + float(padding_y)
    if y_max <= y_min:
        y_min -= 1.0
        y_max += 1.0

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(x, sig, color="black", linewidth=1.5)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    pos = ax.get_position()
    x_img_left = float(pos.x0)
    x_img_right = float(pos.x1)
    y_img_top = float(1.0 - pos.y1)
    y_img_bottom = float(1.0 - pos.y0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi, facecolor="white")
    plt.close(fig)
    return y_min, y_max, (x_img_left, x_img_right, y_img_top, y_img_bottom)


def render_window_image_grid(
    sig: np.ndarray,
    fs_hz: float,
    out_path: Path,
    width: int,
    height: int,
    dpi: int,
    y_min: float,
    y_max: float,
    x_grid_sec: float,
) -> tuple[float, float, tuple[float, float, float, float]]:
    if len(sig) == 0:
        raise ValueError("Empty signal window.")
    x_sec = np.arange(len(sig), dtype=float) / max(fs_hz, 1e-9)

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(x_sec, sig, color="black", linewidth=1.2)
    ax.set_xlim(0.0, float(x_sec[-1]) if len(x_sec) > 1 else 1.0)
    ax.set_ylim(float(y_min), float(y_max))

    major = max(float(x_grid_sec), 0.1)
    max_sec = float(x_sec[-1]) if len(x_sec) > 1 else 1.0
    ax.set_xticks(np.arange(0.0, max_sec + major, major))
    ax.grid(True, which="major", axis="both", alpha=0.45, linestyle="-", linewidth=0.8)
    plt.tight_layout()
    pos = ax.get_position()
    x_img_left = float(pos.x0)
    x_img_right = float(pos.x1)
    y_img_top = float(1.0 - pos.y1)
    y_img_bottom = float(1.0 - pos.y0)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return float(y_min), float(y_max), (x_img_left, x_img_right, y_img_top, y_img_bottom)


def interval_label(
    cls_id: int,
    left_idx: int,
    right_idx: int,
    window_start: int,
    window_size: int,
    y_low: float,
    y_high: float,
    y_min: float,
    y_max: float,
    min_w: float,
    min_h: float,
    x_img_left: float = 0.0,
    x_img_right: float = 1.0,
    y_img_top: float = 0.0,
    y_img_bottom: float = 1.0,
) -> str:
    l = max(left_idx, window_start)
    r = min(right_idx, window_start + window_size - 1)
    if r < l:
        return ""

    w_axis = max(min_w, (r - l + 1) / window_size)
    xc_axis = (((l + r) * 0.5) - window_start + 0.5) / window_size
    y1 = y_to_norm(y_high, y_min, y_max)
    y2 = y_to_norm(y_low, y_min, y_max)
    yc_axis = clamp01((y1 + y2) * 0.5)
    h_axis = max(min_h, abs(y2 - y1))

    x_scale = max(1e-9, x_img_right - x_img_left)
    y_scale = max(1e-9, y_img_bottom - y_img_top)
    xc = x_img_left + (xc_axis * x_scale)
    w = w_axis * x_scale
    yc = y_img_top + (yc_axis * y_scale)
    h = h_axis * y_scale
    return f"{cls_id} {clamp01(xc):.6f} {yc:.6f} {clamp01(w):.6f} {clamp01(h):.6f}"


def peak_label(
    peak_idx: int,
    peak_y: float,
    window_start: int,
    window_size: int,
    y_min: float,
    y_max: float,
    peak_box_w_samples: int,
    peak_box_h: float,
    min_w: float,
    min_h: float,
    x_img_left: float = 0.0,
    x_img_right: float = 1.0,
    y_img_top: float = 0.0,
    y_img_bottom: float = 1.0,
) -> str:
    if peak_idx < window_start or peak_idx >= (window_start + window_size):
        return ""

    xc_axis = (peak_idx - window_start + 0.5) / window_size
    yc_axis = y_to_norm(peak_y, y_min, y_max)
    w_axis = max(min_w, peak_box_w_samples / window_size)
    h_axis = max(min_h, peak_box_h)

    x_scale = max(1e-9, x_img_right - x_img_left)
    y_scale = max(1e-9, y_img_bottom - y_img_top)
    xc = x_img_left + (xc_axis * x_scale)
    yc = y_img_top + (yc_axis * y_scale)
    w = w_axis * x_scale
    h = h_axis * y_scale
    return f"{CLASS_PEAK} {clamp01(xc):.6f} {clamp01(yc):.6f} {clamp01(w):.6f} {clamp01(h):.6f}"


def process_patient(
    patient_dir: Path,
    out_root: Path,
    render_mode: str,
    signal_col: str,
    window_size: int,
    stride: int,
    img_width: int,
    img_height: int,
    img_dpi: int,
    peak_box_w_samples: int,
    peak_box_h: float,
    breath_box_margin_ratio: float,
    min_box_w: float,
    min_box_h: float,
    cj_padding_y: float,
    fixed_y_min: float,
    fixed_y_max: float,
    x_grid_sec: float,
) -> dict:
    pid = parse_patient_id(patient_dir)
    breath_csv = patient_dir / f"patient_{pid}_clustered_breaths.csv"
    signal_csv = patient_dir / f"patient_{pid}_edi_filtered_signal.csv"

    if not breath_csv.exists():
        return {"patient_id": pid, "status": "skip", "reason": "missing_clustered_breaths"}
    if not signal_csv.exists():
        return {"patient_id": pid, "status": "skip", "reason": "missing_edi_filtered_signal"}

    breaths = load_breaths(breath_csv)
    signal, peak_idx, ts = load_signal(signal_csv, signal_col)
    if len(signal) == 0:
        return {"patient_id": pid, "status": "skip", "reason": "empty_signal"}

    windows = build_windows(len(signal), window_size, stride)
    patient_out = out_root / pid
    image_dir = patient_out / "images"
    label_dir = patient_out / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    class_counter = Counter()
    n_images = 0
    n_labels = 0
    fs_hz = infer_fs_hz(ts)

    for w_idx, (ws, we) in enumerate(windows):
        seg = signal[ws:we]
        if len(seg) == 0:
            continue

        stem = f"p{pid}_w{w_idx:04d}"
        img_path = image_dir / f"{stem}.png"
        lbl_path = label_dir / f"{stem}.txt"
        seg_x = ts[ws:we]
        if len(seg_x) != len(seg):
            seg_x = np.arange(len(seg), dtype=float)

        if render_mode == "nogrid_relative":
            y_min, y_max, plot_bbox = render_window_image_cj(
                sig=seg,
                x=seg_x,
                out_path=img_path,
                width=img_width,
                height=img_height,
                dpi=img_dpi,
                padding_y=cj_padding_y,
            )
        elif render_mode in {"grid_relative", "grid_absolute"}:
            if render_mode == "grid_relative":
                seg_min = float(np.min(seg))
                seg_max = float(np.max(seg))
                y_min_use = seg_min - float(cj_padding_y)
                y_max_use = seg_max + float(cj_padding_y)
                if y_max_use <= y_min_use:
                    y_min_use -= 1.0
                    y_max_use += 1.0
            else:
                y_min_use = float(fixed_y_min)
                y_max_use = float(fixed_y_max)

            y_min, y_max, plot_bbox = render_window_image_grid(
                sig=seg,
                fs_hz=fs_hz,
                out_path=img_path,
                width=img_width,
                height=img_height,
                dpi=img_dpi,
                y_min=y_min_use,
                y_max=y_max_use,
                x_grid_sec=x_grid_sec,
            )
        else:
            raise ValueError(f"Unknown render_mode: {render_mode}")
        x_img_left, x_img_right, y_img_top, y_img_bottom = plot_bbox

        label_lines: list[str] = []
        for br in breaths:
            if br.v_end_idx < ws or br.v_start_idx >= we:
                continue
            li = max(br.v_start_idx, ws)
            ri = min(br.v_end_idx, we - 1)
            if ri < li:
                continue
            local = signal[li : ri + 1]
            if len(local) == 0:
                continue
            local_min = float(np.min(local))
            local_max = float(np.max(local))
            pad = float(breath_box_margin_ratio) * max(y_max - y_min, 1e-9)
            cls = CLASS_NORMAL if br.major_cluster else CLASS_ABNORMAL
            line = interval_label(
                cls_id=cls,
                left_idx=br.v_start_idx,
                right_idx=br.v_end_idx,
                window_start=ws,
                window_size=window_size,
                y_low=local_min - pad,
                y_high=local_max + pad,
                y_min=y_min,
                y_max=y_max,
                min_w=min_box_w,
                min_h=min_box_h,
                x_img_left=x_img_left,
                x_img_right=x_img_right,
                y_img_top=y_img_top,
                y_img_bottom=y_img_bottom,
            )
            if line:
                label_lines.append(line)
                class_counter[cls] += 1

        if len(peak_idx) > 0:
            in_win = peak_idx[(peak_idx >= ws) & (peak_idx < we)]
            for p in in_win:
                p = int(p)
                line = peak_label(
                    peak_idx=p,
                    peak_y=float(signal[p]),
                    window_start=ws,
                    window_size=window_size,
                    y_min=y_min,
                    y_max=y_max,
                    peak_box_w_samples=peak_box_w_samples,
                    peak_box_h=peak_box_h,
                    min_w=min_box_w,
                    min_h=min_box_h,
                    x_img_left=x_img_left,
                    x_img_right=x_img_right,
                    y_img_top=y_img_top,
                    y_img_bottom=y_img_bottom,
                )
                if line:
                    label_lines.append(line)
                    class_counter[CLASS_PEAK] += 1

        lbl_path.write_text("\n".join(label_lines), encoding="utf-8")
        n_images += 1
        n_labels += len(label_lines)

    return {
        "patient_id": pid,
        "render_mode": render_mode,
        "status": "ok",
        "n_signal_samples": int(len(signal)),
        "n_breaths": int(len(breaths)),
        "n_peaks": int(len(peak_idx)),
        "n_windows": int(n_images),
        "n_labels": int(n_labels),
        "n_peak_labels": int(class_counter[CLASS_PEAK]),
        "n_normal_labels": int(class_counter[CLASS_NORMAL]),
        "n_abnormal_labels": int(class_counter[CLASS_ABNORMAL]),
    }


def write_dataset_yaml(out_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {out_dir}",
            "names:",
            "  0: peak",
            "  1: normal_breath",
            "  2: abnormal_breath",
            "",
        ]
    )
    (out_dir / "dataset_3class.yaml").write_text(content, encoding="utf-8")


def run_generation(
    in_dir: Path,
    out_dir: Path,
    render_mode: str,
    signal_col: str,
    window_size: int,
    stride: int,
    img_width: int,
    img_height: int,
    img_dpi: int,
    peak_box_width_samples: int,
    peak_box_height: float,
    breath_box_margin_ratio: float,
    min_box_width: float,
    min_box_height: float,
    cj_padding_y: float,
    fixed_y_min: float,
    fixed_y_max: float,
    x_grid_sec: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_dirs = sorted([p for p in in_dir.iterdir() if p.is_dir() and p.name.startswith("patient_")])
    if not patient_dirs:
        raise FileNotFoundError(f"No patient_* folders found in {in_dir}")

    results = []
    totals = defaultdict(int)
    for pdir in patient_dirs:
        res = process_patient(
            patient_dir=pdir,
            out_root=out_dir,
            render_mode=render_mode,
            signal_col=signal_col,
            window_size=window_size,
            stride=stride,
            img_width=img_width,
            img_height=img_height,
            img_dpi=img_dpi,
            peak_box_w_samples=peak_box_width_samples,
            peak_box_h=peak_box_height,
            breath_box_margin_ratio=breath_box_margin_ratio,
            min_box_w=min_box_width,
            min_box_h=min_box_height,
            cj_padding_y=cj_padding_y,
            fixed_y_min=fixed_y_min,
            fixed_y_max=fixed_y_max,
            x_grid_sec=x_grid_sec,
        )
        results.append(res)
        if res.get("status") == "ok":
            for k in [
                "n_windows",
                "n_labels",
                "n_peak_labels",
                "n_normal_labels",
                "n_abnormal_labels",
            ]:
                totals[k] += int(res.get(k, 0))
        print(f"[{res['status'].upper()}][{render_mode}] patient={res['patient_id']} {res}")

    write_dataset_yaml(out_dir)
    summary = {
        "render_mode": render_mode,
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "window_size": int(window_size),
        "stride": int(stride),
        "signal_col": signal_col,
        "totals": dict(totals),
        "per_patient": results,
    }
    (out_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3-class YOLO dataset from BreathDetect outputs.")
    parser.add_argument(
        "--breath-output-root",
        type=Path,
        default=BASE_DIR / "notebooks" / "outputs" / "03_breath_detect",
    )
    parser.add_argument("--analysis-folder", type=str, default="20260301")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_DIR / "02_YOLO" / "data" / "4_BreathDetect_YOLO",
    )
    parser.add_argument("--signal-col", type=str, default="edi_smooth_for_detection")
    parser.add_argument("--window-size", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--img-width", type=int, default=1200)
    parser.add_argument("--img-height", type=int, default=400)
    parser.add_argument("--img-dpi", type=int, default=100)
    parser.add_argument("--peak-box-width-samples", type=int, default=20)
    parser.add_argument("--peak-box-height", type=float, default=0.06)
    parser.add_argument("--breath-box-margin-ratio", type=float, default=0.03)
    parser.add_argument("--min-box-width", type=float, default=0.006)
    parser.add_argument("--min-box-height", type=float, default=0.02)
    parser.add_argument(
        "--mode",
        type=str,
        default="all3",
        choices=["all3", "nogrid_relative", "grid_relative", "grid_absolute"],
    )
    parser.add_argument("--cj-padding-y", type=float, default=2.0)
    parser.add_argument("--fixed-y-min", type=float, default=-20.0)
    parser.add_argument("--fixed-y-max", type=float, default=70.0)
    parser.add_argument("--x-grid-sec", type=float, default=5.0)
    args = parser.parse_args()

    in_dir = args.breath_output_root / args.analysis_folder
    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    base_name = f"{args.analysis_folder}_win{args.window_size}_stride{args.stride}"
    targets: list[tuple[str, Path]] = []
    if args.mode == "all3":
        targets = [
            ("nogrid_relative", args.out_root / f"{base_name}_nogrid_relative"),
            ("grid_relative", args.out_root / f"{base_name}_grid_relative"),
            ("grid_absolute", args.out_root / f"{base_name}_grid_absolute"),
        ]
    elif args.mode == "nogrid_relative":
        targets = [("nogrid_relative", args.out_root / f"{base_name}_nogrid_relative")]
    elif args.mode == "grid_relative":
        targets = [("grid_relative", args.out_root / f"{base_name}_grid_relative")]
    else:
        targets = [("grid_absolute", args.out_root / f"{base_name}_grid_absolute")]

    print(f"Input: {in_dir}")
    for mode, out_dir in targets:
        print(f"\n[START] mode={mode} -> {out_dir}")
        summary = run_generation(
            in_dir=in_dir,
            out_dir=out_dir,
            render_mode=mode,
            signal_col=args.signal_col,
            window_size=args.window_size,
            stride=args.stride,
            img_width=args.img_width,
            img_height=args.img_height,
            img_dpi=args.img_dpi,
            peak_box_width_samples=args.peak_box_width_samples,
            peak_box_height=args.peak_box_height,
            breath_box_margin_ratio=args.breath_box_margin_ratio,
            min_box_width=args.min_box_width,
            min_box_height=args.min_box_height,
            cj_padding_y=args.cj_padding_y,
            fixed_y_min=args.fixed_y_min,
            fixed_y_max=args.fixed_y_max,
            x_grid_sec=args.x_grid_sec,
        )
        print(f"[DONE] mode={mode} totals={summary['totals']}")

    print("\n=== Completed ===")
    print("Saved:")
    for _, out_dir in targets:
        print(f"- {out_dir}")


if __name__ == "__main__":
    main()
