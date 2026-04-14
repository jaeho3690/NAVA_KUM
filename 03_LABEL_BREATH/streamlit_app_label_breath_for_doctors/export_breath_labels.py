#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from utils import BASE_DIR, LABELED_DIR, extract_patient_id, extract_version_tag, load_labels, scan_patient_dirs


DEFAULT_OUTPUT_ROOT = BASE_DIR / "stored_results" / "05_breath_label_exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy source breath tables and append latest doctor breath-label results as new columns."
        )
    )
    parser.add_argument(
        "--patient-dir",
        default="",
        help="Patient directory relative to 03_breath_detect, e.g. 20260414/patient_03.",
    )
    parser.add_argument(
        "--version-tag",
        default="ALL",
        help="Version directory under 03_breath_detect. Ignored when --patient-dir is provided.",
    )
    parser.add_argument(
        "--annotators",
        default="",
        help="Comma-separated annotators. Default: all labels_<annotator>.parquet files in scope.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to write copied/exported files.",
    )
    return parser.parse_args()


def _breath_file_candidates(patient_dir: str) -> List[Path]:
    base_path = BASE_DIR / "notebooks" / "outputs" / "03_breath_detect" / patient_dir
    patient_name = Path(patient_dir).name
    patient_id = extract_patient_id(patient_dir)
    return [
        base_path / f"BB_patient_{patient_id}_clustered_breaths.pkl",
        base_path / f"AA_patient_{patient_id}_clustered_breaths.pkl",
        base_path / f"{patient_name}_clustered_breaths.pkl",
        base_path / f"{patient_name}_clustered_breaths_with_anomaly.pkl",
    ]


def resolve_breath_source_path(patient_dir: str) -> Path:
    for path in _breath_file_candidates(patient_dir):
        if path.exists():
            return path
    raise FileNotFoundError(f"breath pickle 파일을 찾을 수 없습니다: {patient_dir}")


def normalize_patient_dirs(patient_dir: str, version_tag: str) -> List[str]:
    if patient_dir.strip():
        return [str(Path(patient_dir.strip()))]
    return scan_patient_dirs(version_tag=version_tag)


def list_annotators_for_scope(patient_dirs: Iterable[str]) -> List[str]:
    version_tags = {extract_version_tag(patient_dir) for patient_dir in patient_dirs}
    annotators = set()
    if not LABELED_DIR.exists():
        return []

    for path in sorted(LABELED_DIR.rglob("labels_*.parquet")):
        version = path.parent.name
        if version not in version_tags:
            continue
        annotators.add(path.stem.replace("labels_", "", 1))
    return sorted(annotators)


def parse_requested_annotators(raw: str, patient_dirs: Iterable[str]) -> List[str]:
    if raw.strip():
        return sorted({name.strip() for name in raw.split(",") if name.strip()})
    return list_annotators_for_scope(patient_dirs)


def latest_rows_per_item(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.copy().reset_index(names="__row_order")
    work["__ts"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.sort_values(["__ts", "__row_order"], na_position="last")
    return work.groupby("item_id", sort=False).tail(1).drop(columns=["__row_order", "__ts"])


def extract_breath_labels(labels_df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    if labels_df.empty:
        return pd.DataFrame(columns=["breath_id", "item_id", "label", "comment", "timestamp"])

    filtered = labels_df[
        (labels_df["patient_id"].astype(str) == str(patient_id)) & (labels_df["type"] == "breath_label")
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["breath_id", "item_id", "label", "comment", "timestamp"])

    latest = latest_rows_per_item(filtered)
    item_parts = latest["item_id"].astype(str).str.split("|")
    latest["breath_id"] = item_parts.str[2].where(item_parts.str.len() >= 3)
    latest = latest.dropna(subset=["breath_id"]).copy()
    latest["breath_id"] = latest["breath_id"].astype(str)
    latest["comment"] = latest["comment"].fillna("").astype(str)
    latest["label"] = latest["label"].fillna("").astype(str)
    latest["timestamp"] = latest["timestamp"].fillna("").astype(str)
    return latest[["breath_id", "item_id", "label", "comment", "timestamp"]]


def attach_annotator_columns(breath_df: pd.DataFrame, annotator: str, labels_df: pd.DataFrame) -> pd.DataFrame:
    prefix = f"doctor__{annotator}"
    rename_map = {
        "item_id": f"{prefix}__item_id",
        "label": f"{prefix}__label",
        "comment": f"{prefix}__comment",
        "timestamp": f"{prefix}__labeled_at",
    }

    merged = labels_df.rename(columns=rename_map).copy()
    out = breath_df.copy()
    out["__breath_id_key"] = out["breath_id"].astype(str)
    if merged.empty:
        for col_name in rename_map.values():
            out[col_name] = pd.NA
        return out.drop(columns=["__breath_id_key"])

    merged = merged.rename(columns={"breath_id": "__breath_id_key"})
    out = out.merge(merged, on="__breath_id_key", how="left")
    return out.drop(columns=["__breath_id_key"])


def export_one_patient(patient_dir: str, annotators: Iterable[str], output_root: Path) -> tuple[Path, Path]:
    source_path = resolve_breath_source_path(patient_dir)
    source_df = pd.read_pickle(source_path)
    patient_id = extract_patient_id(patient_dir)
    version_tag = extract_version_tag(patient_dir)

    out_df = source_df.copy()
    for annotator in annotators:
        labels_df = load_labels(annotator, version_tag)
        latest_labels = extract_breath_labels(labels_df, patient_id)
        out_df = attach_annotator_columns(out_df, annotator, latest_labels)

    patient_out_dir = output_root / patient_dir
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{source_path.stem}__doctor_labels"
    pkl_out = patient_out_dir / f"{stem}.pkl"
    csv_out = patient_out_dir / f"{stem}.csv"
    out_df.to_pickle(pkl_out)
    out_df.to_csv(csv_out, index=False)
    return pkl_out, csv_out


def main() -> int:
    args = parse_args()
    patient_dirs = normalize_patient_dirs(args.patient_dir, args.version_tag)
    if not patient_dirs:
        raise SystemExit("export 대상 patient_dir를 찾지 못했습니다.")

    annotators = parse_requested_annotators(args.annotators, patient_dirs)
    if not annotators:
        raise SystemExit("선택한 범위에서 labels_<annotator>.parquet 파일을 찾지 못했습니다.")

    print(f"Patients: {len(patient_dirs)}")
    print(f"Annotators: {', '.join(annotators)}")
    print(f"Output root: {args.output_root}")

    written = []
    for patient_dir in patient_dirs:
        pkl_out, csv_out = export_one_patient(patient_dir, annotators, args.output_root)
        written.append((pkl_out, csv_out))
        print(f"[OK] {patient_dir}")
        print(f"  PKL: {pkl_out}")
        print(f"  CSV: {csv_out}")

    print(f"Done. Exported {len(written)} patient file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
