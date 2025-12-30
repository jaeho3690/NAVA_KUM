import os
import shutil
import glob
import wandb
from pathlib import Path
from ultralytics import YOLO
from omegaconf import DictConfig, OmegaConf


def train_yolo(cfg: DictConfig, dataset_yaml_path: str) -> None:
    """Train YOLO model with given configuration."""
    
    # Initialize wandb 
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name or cfg.experiment_name,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    # Load pretrained model
    model = YOLO(cfg.model.pretrained)
    
    # Train
    results = model.train(
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
    )
    
    # Clean up outputs - keep only essential files
    exp_dir = Path(cfg.output_dir) / cfg.experiment_name
    cleanup_experiment_outputs(exp_dir)
    
    # Finish wandb run
    if cfg.wandb.enabled:
        wandb.finish()
    
    return results


def cleanup_experiment_outputs(exp_dir: Path) -> None:
    """
    Clean up experiment outputs, keeping only:
    - weights/best.pt
    - 2 test result images
    """
    if not exp_dir.exists():
        return
    
    weights_dir = exp_dir / "weights"
    
    # Keep only best.pt, remove last.pt and other weight files
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            if weight_file.name != "best.pt":
                weight_file.unlink()
                print(f"Removed: {weight_file}")
    
    # Keep only 2 test result images, remove the rest
    # Look for val_batch*_pred.jpg or similar prediction images
    pred_images = list(exp_dir.glob("*pred*.jpg")) + list(exp_dir.glob("*pred*.png"))
    if len(pred_images) > 2:
        for img in pred_images[2:]:
            img.unlink()
            print(f"Removed: {img}")
    
    # Remove unnecessary files/directories
    items_to_remove = [
        "train_batch*.jpg",
        "val_batch*_labels.jpg", 
        "labels*.jpg",
        "labels_correlogram.jpg",
        "confusion_matrix*.png",
        "F1_curve.png",
        "P_curve.png", 
        "R_curve.png",
        "PR_curve.png",
        "results.csv",
    ]
    
    for pattern in items_to_remove:
        for item in exp_dir.glob(pattern):
            if item.is_file():
                item.unlink()
                print(f"Removed: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"Removed directory: {item}")
    
    print(f"Cleanup complete. Kept: best.pt and up to 2 prediction images")
