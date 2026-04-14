import csv
import shutil
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def train_yolo(cfg: DictConfig, dataset_yaml_path: str) -> None:
    """Train YOLO, then evaluate the best checkpoint and export CSV summaries."""

    if cfg.wandb.enabled:
        if wandb is None:
            raise ImportError("wandb is enabled in config, but the package is not installed.")
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name or cfg.experiment_name,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    model = YOLO(cfg.model.pretrained)
    model.train(
        data=dataset_yaml_path,
        epochs=cfg.train.epochs,
        imgsz=cfg.model.imgsz,
        batch=cfg.train.batch,
        workers=cfg.train.workers,
        device=cfg.train.device,
        optimizer=cfg.train.optimizer,
        lr0=cfg.train.lr0,
        lrf=cfg.train.lrf,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
        hsv_h=cfg.train.hsv_h,
        hsv_s=cfg.train.hsv_s,
        hsv_v=cfg.train.hsv_v,
        flipud=cfg.train.flipud,
        fliplr=cfg.train.fliplr,
        mosaic=cfg.train.mosaic,
        mixup=cfg.train.mixup,
        project=cfg.output_dir,
        name=cfg.experiment_name,
        save=cfg.train.save,
        plots=cfg.train.plots,
        seed=cfg.train.seed,
        deterministic=cfg.train.deterministic,
    )

    exp_dir = Path(cfg.output_dir) / cfg.experiment_name
    best_weights = exp_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_weights}")

    save_experiment_metadata(cfg, exp_dir, dataset_yaml_path)
    evaluate_checkpoint(best_weights, dataset_yaml_path, cfg, exp_dir)
    cleanup_experiment_outputs(exp_dir)

    if cfg.wandb.enabled and wandb is not None:
        wandb.finish()


def evaluate_checkpoint(
    best_weights: Path,
    dataset_yaml_path: str,
    cfg: DictConfig,
    exp_dir: Path,
) -> None:
    eval_model = YOLO(str(best_weights))

    for split in cfg.eval.splits:
        metrics = eval_model.val(
            data=dataset_yaml_path,
            split=split,
            imgsz=cfg.model.imgsz,
            batch=cfg.eval.batch or cfg.train.batch,
            workers=cfg.train.workers,
            device=cfg.train.device,
            project=cfg.output_dir,
            name=cfg.experiment_name,
            exist_ok=True,
            plots=cfg.eval.plots,
            save_json=cfg.eval.save_json,
        )
        write_metrics_csv(exp_dir / f"{split}_metrics.csv", split, metrics, best_weights)


def write_metrics_csv(output_path: Path, split: str, metrics, best_weights: Path) -> None:
    results = {"split": split, "weights": str(best_weights)}
    results.update(flatten_metrics(metrics.results_dict))

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)


def flatten_metrics(metrics_dict: dict) -> dict[str, float]:
    flattened = {}
    for key, value in metrics_dict.items():
        flattened[str(key).replace("/", "_")] = float(value)
    return flattened


def save_experiment_metadata(cfg: DictConfig, exp_dir: Path, dataset_yaml_path: str) -> None:
    metadata = {
        "experiment_name": cfg.experiment_name,
        "dataset_yaml": str(dataset_yaml_path),
        "data_root": cfg.data.data_root,
        "dataset_variant": cfg.data.dataset_variant,
        "split_setting": cfg.data.split_setting,
        "target_patient": cfg.data.target_patient,
        "patient_splits": OmegaConf.to_container(cfg.data.patient_splits, resolve=True),
    }

    output_path = exp_dir / "experiment_metadata.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata.keys()))
        writer.writeheader()
        writer.writerow(metadata)

    dataset_yaml = Path(dataset_yaml_path)
    split_manifest = dataset_yaml.parent / "split_manifest.csv"
    if dataset_yaml.exists():
        shutil.copy2(dataset_yaml, exp_dir / "dataset.yaml")
    if split_manifest.exists():
        shutil.copy2(split_manifest, exp_dir / "split_manifest.csv")


def cleanup_experiment_outputs(exp_dir: Path) -> None:
    """
    Clean up experiment outputs, keeping only essential files for later analysis.
    """
    if not exp_dir.exists():
        return

    weights_dir = exp_dir / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            if weight_file.name != "best.pt":
                weight_file.unlink()
                print(f"Removed: {weight_file}")

    pred_images = list(exp_dir.glob("*pred*.jpg")) + list(exp_dir.glob("*pred*.png"))
    if len(pred_images) > 6:
        for img in pred_images[6:]:
            img.unlink()
            print(f"Removed: {img}")

    items_to_remove = [
        "train_batch*.jpg",
        "val_batch*_labels.jpg",
        "labels*.jpg",
        "labels_correlogram.jpg",
        "confusion_matrix*.png",
        "F1_curve.png",
        "BoxF1_curve.png",
        "P_curve.png",
        "BoxP_curve.png",
        "R_curve.png",
        "BoxR_curve.png",
        "PR_curve.png",
        "BoxPR_curve.png",
    ]

    for pattern in items_to_remove:
        for item in exp_dir.glob(pattern):
            if item.is_file():
                item.unlink()
                print(f"Removed: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"Removed directory: {item}")

    print("Cleanup complete. Kept best.pt, evaluation CSVs, split manifest, and a few prediction images.")
