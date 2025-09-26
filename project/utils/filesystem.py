from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf


def create_experiment_dirs(config: DictConfig) -> Dict[str, Path]:
    """Create necessary directories for experiment."""
    log_dir = config.log_dir
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    base_dir = Path(log_dir)

    dirs = {
        "base": base_dir,
        "logs": base_dir / "logs",
        "checkpoints": base_dir / "checkpoints",
        "results": base_dir / "results",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def save_experiment_summary(
    config: DictConfig, results: Dict[str, Any], output_dir: str
):
    """Save experiment summary with config and results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert config to dict
    config_dict = OmegaConf.to_container(config, resolve=True)
    summary = {
        "config": config_dict,
        "results": results,
        "completed_at": datetime.now().isoformat(),
    }

    with open(output_path / "experiment_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
