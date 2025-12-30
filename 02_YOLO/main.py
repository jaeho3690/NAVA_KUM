import hydra
from omegaconf import DictConfig, OmegaConf
from src.train import train_yolo
from src.utils.data_utils import prepare_dataset_yaml


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for YOLO training with Hydra configuration."""
    print(OmegaConf.to_yaml(cfg))
    
    # Prepare dataset.yaml from patient splits
    dataset_yaml_path = prepare_dataset_yaml(cfg)
    
    # Train model
    train_yolo(cfg, dataset_yaml_path)


if __name__ == "__main__":
    main()
