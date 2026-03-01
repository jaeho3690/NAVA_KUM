#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path("/home/jhkim/NAVA/03_LABEL")
DEFAULT_DETECTED_ROOT = BASE_DIR / "stored_results" / "00_detected"
DEFAULT_LABELS_DIR = BASE_DIR / "stored_results" / "01_labeled"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "stored_results" / "02_summarized"


def extract_patient_id(patient_file: str) -> str:
    stem = Path(patient_file).stem
    prefix = "movingwinddetected_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def scan_xlsx_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.xlsx"))


def pick_latest_version_dir(detected_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for p in detected_root.iterdir():
        if p.is_dir() and list(p.glob("*.xlsx")):
            candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.name)[-1]


def parse_item_id(item_id: str) -> Tuple[str, str, str]:
    parts = str(item_id).split("|")
    if len(parts) < 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def extract_version_tag_from_xlsx(detected_root: Path, xlsx_path: Path) -> str:
    try:
        rel = xlsx_path.relative_to(detected_root)
        if len(rel.parts) >= 2:
            first = rel.parts[0]
            if first.isdigit() and len(first) == 8:
                return first
    except Exception:
        pass

    root_name = detected_root.name
    if root_name.isdigit() and len(root_name) == 8:
        return root_name
    return "legacy"


def parse_apnea_interval_from_item_id(item_id: str) -> Optional[Tuple[int, int]]:
    _, typ, payload = parse_item_id(item_id)
    if typ != "apnea" or "-" not in payload:
        return None
    try:
        start_s, end_s = payload.split("-", 1)
        start_ts = int(float(start_s))
        end_ts = int(float(end_s))
    except ValueError:
        return None
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts
    return start_ts, end_ts


def latest_rows_per_item(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_ts"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.sort_values("_ts")
    return work.groupby("item_id", sort=False).tail(1).drop(columns=["_ts"])


def load_annotator_labels(
    labels_dir: Path,
    annotators: Optional[Iterable[str]],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    files = sorted(labels_dir.rglob("labels_*.parquet"))
    for path in files:
        name = path.stem.replace("labels_", "", 1)
        if annotators is not None and name not in annotators:
            continue
        try:
            parent_rel = path.parent.relative_to(labels_dir)
            if len(parent_rel.parts) == 0:
                version_tag = "legacy"
            else:
                version_tag = parent_rel.parts[0]
        except Exception:
            version_tag = "legacy"

        df = pd.read_parquet(path)
        for col in [
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
        ]:
            if col not in df.columns:
                df[col] = "" if col not in {"start_ts", "end_ts"} else pd.NA
        out.setdefault(name, {})[version_tag] = df
    return out


def apply_peak_labels(data_df: pd.DataFrame, labels_df: pd.DataFrame, annotator: str, patient_id: str) -> pd.DataFrame:
    out = data_df.copy()
    if labels_df.empty:
        out[f"peak_label_{annotator}"] = ""
        return out

    filtered = labels_df[(labels_df["patient_id"] == patient_id) & (labels_df["type"] == "peak")]
    if filtered.empty:
        out[f"peak_label_{annotator}"] = ""
        return out

    latest = latest_rows_per_item(filtered)
    mapping_label: Dict[int, str] = {}
    for row in latest.itertuples(index=False):
        _, typ, payload = parse_item_id(row.item_id)
        if typ != "peak":
            continue
        try:
            ts = int(payload)
        except ValueError:
            continue
        mapping_label[ts] = str(row.label)

    out[f"peak_label_{annotator}"] = out["timestamp"].astype(int).map(mapping_label).fillna("")
    return out


def apply_apnea_labels(data_df: pd.DataFrame, labels_df: pd.DataFrame, annotator: str, patient_id: str) -> pd.DataFrame:
    out = data_df.copy()
    label_col = f"apnea_label_{annotator}"
    comment_col = f"apnea_comment_{annotator}"
    out[label_col] = False
    out[comment_col] = ""

    if labels_df.empty:
        return out

    filtered = labels_df[(labels_df["patient_id"] == patient_id) & (labels_df["type"] == "apnea")]
    if filtered.empty:
        return out

    latest = latest_rows_per_item(filtered)
    for row in latest.itertuples(index=False):
        # Prefer adjusted interval (start_ts/end_ts) when available.
        start_ts = pd.to_numeric(getattr(row, "start_ts", None), errors="coerce")
        end_ts = pd.to_numeric(getattr(row, "end_ts", None), errors="coerce")
        if pd.notna(start_ts) and pd.notna(end_ts):
            start_ts = int(start_ts)
            end_ts = int(end_ts)
            if start_ts > end_ts:
                start_ts, end_ts = end_ts, start_ts
        else:
            parsed = parse_apnea_interval_from_item_id(row.item_id)
            if parsed is None:
                continue
            start_ts, end_ts = parsed

        mask = (out["timestamp"].astype(int) >= start_ts) & (out["timestamp"].astype(int) <= end_ts)
        label_val = str(row.label)
        if label_val == "O":
            out.loc[mask, label_col] = True
            out.loc[mask, comment_col] = str(row.comment) if pd.notna(row.comment) else ""
        else:
            out.loc[mask, label_col] = False

    return out


def summarize_one_file(
    xlsx_path: Path,
    detected_root: Path,
    output_dir: Path,
    labels_by_annotator: Dict[str, Dict[str, pd.DataFrame]],
) -> Path:
    data_df = pd.read_excel(xlsx_path, sheet_name="data")
    params_df = pd.read_excel(xlsx_path, sheet_name="params")

    data_df["timestamp"] = pd.to_numeric(data_df["timestamp"], errors="coerce")
    data_df = data_df.dropna(subset=["timestamp"]).copy()
    data_df["timestamp"] = data_df["timestamp"].astype(int)

    patient_id = extract_patient_id(xlsx_path.name)
    version_tag = extract_version_tag_from_xlsx(detected_root, xlsx_path)
    empty_labels = pd.DataFrame(
        columns=["label_id", "timestamp", "annotator", "patient_id", "type", "item_id", "label", "comment", "start_ts", "end_ts"]
    )

    out_df = data_df.copy()
    for annotator, labels_by_version in labels_by_annotator.items():
        labels_df = labels_by_version.get(version_tag)
        if labels_df is None and version_tag != "legacy":
            labels_df = labels_by_version.get("legacy")
        if labels_df is None:
            labels_df = empty_labels
        out_df = apply_peak_labels(out_df, labels_df, annotator, patient_id)
        out_df = apply_apnea_labels(out_df, labels_df, annotator, patient_id)

    rel_path = xlsx_path.relative_to(detected_root)
    out_path = output_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="data", index=False)
        params_df.to_excel(writer, sheet_name="params", index=False)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge label parquet logs into detected XLSX files.")
    parser.add_argument("--detected-root", type=Path, default=DEFAULT_DETECTED_ROOT)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--version-dir",
        type=str,
        default="",
        help="Subdirectory under detected root to process (e.g. 2026-02-26).",
    )
    parser.add_argument(
        "--latest-version",
        action="store_true",
        help="Process only the latest version subdirectory under detected root.",
    )
    parser.add_argument(
        "--annotators",
        type=str,
        default="",
        help="Comma-separated annotators. Default: all labels_*.parquet files.",
    )
    parser.add_argument(
        "--file-substr",
        type=str,
        default="",
        help="Only process files whose relative path contains this substring.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most N files (0 means all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detected_root = args.detected_root
    if args.version_dir:
        detected_root = detected_root / args.version_dir
    elif args.latest_version:
        latest = pick_latest_version_dir(detected_root)
        if latest is None:
            raise SystemExit("No version subdirectory with xlsx files found under detected root.")
        detected_root = latest

    if not detected_root.exists():
        raise SystemExit(f"Detected root not found: {detected_root}")

    annotators = None
    if args.annotators.strip():
        annotators = [x.strip() for x in args.annotators.split(",") if x.strip()]

    labels_by_annotator = load_annotator_labels(args.labels_dir, annotators)
    if not labels_by_annotator:
        raise SystemExit(f"No label parquet files found in {args.labels_dir}")

    xlsx_files = scan_xlsx_files(detected_root)
    if args.file_substr:
        xlsx_files = [p for p in xlsx_files if args.file_substr in str(p.relative_to(detected_root))]
    if args.max_files and args.max_files > 0:
        xlsx_files = xlsx_files[: args.max_files]
    if not xlsx_files:
        raise SystemExit(f"No xlsx files found in {detected_root}")

    output_dir = args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected root: {detected_root}")
    print(f"Output dir: {output_dir}")
    print(f"Annotators: {', '.join(sorted(labels_by_annotator.keys()))}")

    written: List[Path] = []
    for xlsx_path in xlsx_files:
        out_path = summarize_one_file(
            xlsx_path=xlsx_path,
            detected_root=detected_root,
            output_dir=output_dir,
            labels_by_annotator=labels_by_annotator,
        )
        written.append(out_path)
        print(f"Wrote: {out_path}")

    print(f"Done. {len(written)} files written.")


if __name__ == "__main__":
    main()
