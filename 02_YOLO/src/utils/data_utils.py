import os
import yaml
from pathlib import Path
from omegaconf import DictConfig


def prepare_dataset_yaml(cfg: DictConfig) -> str:
    """
    Create dataset.yaml and prepare train/val/test directories
    by symlinking patient data.
    """
    data_root = Path(cfg.data.data_root)
    work_dir = Path(os.getcwd())  # Hydra changes cwd
    prepared_data_dir = work_dir / "data"
    
    # Create directories for train/val/test
    for split in ["train", "val", "test"]:
        (prepared_data_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (prepared_data_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Create symlinks for each patient's data
    for split in ["train", "val", "test"]:
        patient_ids = cfg.data.splits.get(split, [])
        for patient_id in patient_ids:
            patient_dir = data_root / patient_id
            
            # Symlink images
            src_images = patient_dir / "images"
            if src_images.exists():
                for img in src_images.glob("*.png"):
                    dst = prepared_data_dir / split / "images" / img.name
                    if not dst.exists():
                        dst.symlink_to(img)
            
            # Symlink labels
            src_labels = patient_dir / "labels"
            if src_labels.exists():
                for label in src_labels.glob("*.txt"):
                    dst = prepared_data_dir / split / "labels" / label.name
                    if not dst.exists():
                        dst.symlink_to(label)
    
    # Count images per split
    for split in ["train", "val", "test"]:
        img_count = len(list((prepared_data_dir / split / "images").glob("*.png")))
        label_count = len(list((prepared_data_dir / split / "labels").glob("*.txt")))
        print(f"{split}: {img_count} images, {label_count} labels")
    
    # Create dataset.yaml
    dataset_config = {
        "path": str(prepared_data_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(cfg.data.classes)},
    }
    
    # Add test only if patients are specified
    if cfg.data.splits.get("test"):
        dataset_config["test"] = "test/images"
    
    dataset_yaml_path = work_dir / "dataset.yaml"
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset YAML created at: {dataset_yaml_path}")
    return str(dataset_yaml_path)
