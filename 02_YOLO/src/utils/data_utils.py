import csv
import os
import re
from pathlib import Path

import yaml
from omegaconf import DictConfig


WINDOW_PATTERN = re.compile(r"_w(\d+)\.")


def prepare_dataset_yaml(cfg: DictConfig) -> str:
    """
    Build a YOLO dataset from a configured split setting and write dataset.yaml.
    """
    dataset_root = resolve_dataset_root(cfg)
    work_dir = Path(os.getcwd())  # Hydra changes cwd
    prepared_data_dir = work_dir / "data"

    clear_prepared_dataset(prepared_data_dir)
    split_manifest = build_split_manifest(cfg, dataset_root)
    materialize_split_manifest(split_manifest, prepared_data_dir)
    write_split_manifest_csv(split_manifest, work_dir / "split_manifest.csv")

    dataset_config = {
        "path": str(prepared_data_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(cfg.data.classes)},
    }
    if split_manifest["test"]:
        dataset_config["test"] = "test/images"

    dataset_yaml_path = work_dir / "dataset.yaml"
    with open(dataset_yaml_path, "w", encoding="utf-8") as handle:
        yaml.dump(dataset_config, handle, default_flow_style=False, sort_keys=False)

    print_split_counts(prepared_data_dir)
    print(f"Dataset YAML created at: {dataset_yaml_path}")
    return str(dataset_yaml_path)


def resolve_dataset_root(cfg: DictConfig) -> Path:
    data_root = Path(cfg.data.data_root)
    if cfg.data.dataset_variant:
        data_root = data_root / cfg.data.dataset_variant
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    return data_root


def clear_prepared_dataset(prepared_data_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        cache_file = prepared_data_dir / split / "labels.cache"
        if cache_file.exists():
            cache_file.unlink()
        for subdir in ["images", "labels"]:
            target_dir = prepared_data_dir / split / subdir
            if target_dir.exists():
                for item in target_dir.iterdir():
                    item.unlink()
            target_dir.mkdir(parents=True, exist_ok=True)


def build_split_manifest(cfg: DictConfig, dataset_root: Path) -> dict[str, list[dict[str, str]]]:
    split_setting = cfg.data.split_setting

    if split_setting == "within_patient_60_20_20":
        return build_within_patient_manifest(cfg, dataset_root)
    if split_setting == "across_patient_10_3_3":
        return build_across_patient_manifest(cfg, dataset_root)

    raise ValueError(f"Unsupported split setting: {split_setting}")


def build_within_patient_manifest(cfg: DictConfig, dataset_root: Path) -> dict[str, list[dict[str, str]]]:
    patient_id = str(cfg.data.target_patient)
    patient_dir = dataset_root / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    samples = sorted(load_patient_samples(patient_dir, patient_id), key=lambda item: item["order_key"])
    if not samples:
        raise ValueError(f"No valid samples found for patient {patient_id} in {patient_dir}")

    total = len(samples)
    train_end = int(total * cfg.data.split_ratio.train)
    val_end = train_end + int(total * cfg.data.split_ratio.val)
    val_end = min(val_end, total)

    manifest = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    validate_manifest(manifest)
    return manifest


def build_across_patient_manifest(cfg: DictConfig, dataset_root: Path) -> dict[str, list[dict[str, str]]]:
    manifest = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        patient_ids = [str(patient_id) for patient_id in cfg.data.patient_splits.get(split, [])]
        for patient_id in patient_ids:
            patient_dir = dataset_root / patient_id
            if not patient_dir.exists():
                raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
            manifest[split].extend(load_patient_samples(patient_dir, patient_id))

    validate_manifest(manifest)
    return manifest


def load_patient_samples(patient_dir: Path, patient_id: str) -> list[dict[str, str]]:
    images_dir = patient_dir / "images"
    labels_dir = patient_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing images/labels directories in {patient_dir}")

    samples = []
    for image_path in sorted(images_dir.glob("*.png")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for image {image_path}")

        samples.append(
            {
                "patient_id": patient_id,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "filename": image_path.name,
                "order_key": extract_order_key(image_path.name),
            }
        )

    return samples


def extract_order_key(filename: str) -> int:
    match = WINDOW_PATTERN.search(filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract temporal order from filename: {filename}")


def validate_manifest(manifest: dict[str, list[dict[str, str]]]) -> None:
    for split in ["train", "val", "test"]:
        if not manifest[split]:
            raise ValueError(f"Split '{split}' is empty. Check the configured split setting.")


def materialize_split_manifest(
    manifest: dict[str, list[dict[str, str]]],
    prepared_data_dir: Path,
) -> None:
    for split, samples in manifest.items():
        for sample in samples:
            image_dst = prepared_data_dir / split / "images" / Path(sample["image_path"]).name
            label_dst = prepared_data_dir / split / "labels" / Path(sample["label_path"]).name

            if not image_dst.exists():
                image_dst.symlink_to(sample["image_path"])
            if not label_dst.exists():
                label_dst.symlink_to(sample["label_path"])


def write_split_manifest_csv(
    manifest: dict[str, list[dict[str, str]]],
    output_path: Path,
) -> None:
    fieldnames = ["split", "patient_id", "filename", "order_key", "image_path", "label_path"]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for split, samples in manifest.items():
            for sample in samples:
                writer.writerow(
                    {
                        "split": split,
                        "patient_id": sample["patient_id"],
                        "filename": sample["filename"],
                        "order_key": sample["order_key"],
                        "image_path": sample["image_path"],
                        "label_path": sample["label_path"],
                    }
                )


def print_split_counts(prepared_data_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        image_count = len(list((prepared_data_dir / split / "images").glob("*.png")))
        label_count = len(list((prepared_data_dir / split / "labels").glob("*.txt")))
        print(f"{split}: {image_count} images, {label_count} labels")
