#!/usr/bin/env python3
"""
Hyperparameter Optimization training script for GateL0RD models using Hydra.

This script handles HPO experiments using Optuna with Hydra-based configuration.

Usage:
    python train_hpo.py experiment=scaling_experiment
    python train_hpo.py hpo.n_trials=100
    python train_hpo.py model=v0 hpo.n_trials=50 hpo.timeout=7200
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Add project root to Python path  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl

from project.utils.config_hydra import (
    ConfigManager, ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, HPOConfig,
    save_experiment_summary
)

# Register configurations with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)
cs.store(group="model", name="v0", node=ModelConfig(version="v0"))
cs.store(group="model", name="v1", node=ModelConfig(version="v1"))
cs.store(group="model", name="v2", node=ModelConfig(version="v2"))
cs.store(group="model", name="v3", node=ModelConfig(version="v3"))
cs.store(group="data", name="billiard_ball", node=DataConfig(dataset_name="BilliardBall"))
cs.store(group="data", name="fetch_pick_place", node=DataConfig(dataset_name="FetchPickAndPlace"))
cs.store(group="training", name="hpo", node=TrainingConfig(max_epochs=50, deterministic=False))
cs.store(group="hpo", name="scaling", node=HPOConfig(study_name="scaling_experiment"))


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def setup_reproducibility(seed: int = 42):
    """Setup reproducibility for HPO."""
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = False  # Allow some non-determinism for performance
    torch.backends.cudnn.benchmark = True


def create_data_module(config: DictConfig):
    """Create and setup data module."""
    from project.datasets.timeseries import TimeSeriesDataModule
    
    # Determine data path based on dataset name
    if config.data.dataset_name == "BilliardBall":
        data_path = Path(config.data.data_dir) / "BilliardBall"
    elif config.data.dataset_name == "FetchPickAndPlace":
        data_path = Path(config.data.data_dir) / "FetchPickAndPlace"
    else:
        data_path = Path(config.data.data_dir) / config.data.dataset_name
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    
    return TimeSeriesDataModule(
        data_dir=str(data_path),
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        prediction_steps=config.data.prediction_steps,
        overlap_ratio=config.data.overlap_ratio,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        normalize=config.data.normalize,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )


def run_scaling_experiment(config: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Run parameter scaling HPO experiment."""
    logger.info(f"Starting parameter scaling HPO experiment: {config.name}")
    
    if not OmegaConf.select(config, "hpo"):
        raise ValueError("HPO configuration is required for HPO experiment")
    
    # Setup reproducibility
    setup_reproducibility(config.seed)
    
    # Create data module
    data_module = create_data_module(config)
    
    # Create HPO runner
    from project.hpo import GateL0RDHPORunner
    
    hpo_runner = GateL0RDHPORunner(
        data_module=data_module,
        study_name=config.hpo.study_name,
        storage_url=config.hpo.storage_url,
        log_dir=config.log_dir,
    )
    
    # Run scaling experiment
    study = hpo_runner.run_scaling_experiment(
        base_version=config.model.version,
        n_trials=config.hpo.n_trials,
        max_epochs=config.training.max_epochs,
        timeout=config.hpo.timeout,
    )
    
    # Analyze results
    analysis = hpo_runner.analyze_results(study)
    
    results = {
        'experiment_type': 'scaling',
        'base_version': config.model.version,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'analysis': analysis,
    }
    
    logger.info(f"Scaling experiment completed. Best value: {results['best_value']:.4f}")
    logger.info(f"Best version: {study.best_params.get('model_version', 'N/A')}")
    
    return results


def run_comparison_experiment(config: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Run architecture comparison HPO experiment."""
    logger.info(f"Starting architecture comparison HPO experiment: {config.name}")
    
    if not OmegaConf.select(config, "hpo"):
        raise ValueError("HPO configuration is required for HPO experiment")
    
    # Setup reproducibility
    setup_reproducibility(config.seed)
    
    # Create data module
    data_module = create_data_module(config)
    
    # Create HPO runner
    from project.hpo import GateL0RDHPORunner
    
    hpo_runner = GateL0RDHPORunner(
        data_module=data_module,
        study_name=config.hpo.study_name,
        storage_url=config.hpo.storage_url,
        log_dir=config.log_dir,
    )
    
    # Calculate trials per version
    n_trials_per_version = config.hpo.n_trials // 4  # Split across v0-v3
    if config.hpo.n_trials % 4 != 0:
        logger.warning(f"n_trials ({config.hpo.n_trials}) not divisible by 4. Using {n_trials_per_version} trials per version.")
    
    # Run architecture comparison
    studies = hpo_runner.run_architecture_comparison(
        n_trials_per_version=n_trials_per_version,
        max_epochs=config.training.max_epochs,
    )
    
    # Collect results for each version
    version_results = {}
    best_overall_value = float('inf')
    best_overall_version = None
    
    for version, study in studies.items():
        analysis = hpo_runner.analyze_results(study)
        version_results[version] = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'analysis': analysis,
        }
        
        # Track overall best
        if study.best_value < best_overall_value:
            best_overall_value = study.best_value
            best_overall_version = version
        
        logger.info(f"Version {version}: Best value {study.best_value:.4f}")
    
    results = {
        'experiment_type': 'comparison',
        'version_results': version_results,
        'best_overall_version': best_overall_version,
        'best_overall_value': best_overall_value,
        'total_trials': sum(len(study.trials) for study in studies.values()),
    }
    
    logger.info("Architecture comparison completed.")
    logger.info(f"Best overall version: {best_overall_version} (val_loss: {best_overall_value:.4f})")
    
    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """Main HPO function with Hydra."""
    # Setup logging
    logger = setup_logging("INFO")
    
    # Print configuration
    logger.info("HPO Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    
    # Validate configuration
    config_manager = ConfigManager()
    config_manager.validate_config(config)
    
    # Determine experiment type
    experiment_type = OmegaConf.select(config, "experiment_type", default="scaling")
    
    logger.info(f"Running {experiment_type} HPO experiment: {config.name}")
    logger.info(f"Dataset: {config.data.dataset_name}")
    logger.info(f"Trials: {config.hpo.n_trials if OmegaConf.select(config, 'hpo') else 'N/A'}")
    logger.info(f"Base version: {config.model.version}")
    
    try:
        # Run experiment based on type
        if experiment_type == 'scaling':
            results = run_scaling_experiment(config, logger)
        elif experiment_type == 'comparison':
            results = run_comparison_experiment(config, logger)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Save experiment summary
        experiment_dirs = config_manager.create_experiment_dirs(config)
        save_experiment_summary(config, results, str(experiment_dirs['results']))
        
        logger.info("HPO experiment completed successfully!")
        logger.info(f"Results saved to: {experiment_dirs['results']}")
        
        # Print summary
        if 'best_value' in results:
            logger.info(f"Best validation loss: {results['best_value']:.4f}")
        if 'best_params' in results:
            logger.info("Best parameters:")
            for param, value in results['best_params'].items():
                logger.info(f"  {param}: {value}")
        
    except Exception as e:
        logger.error(f"HPO experiment failed: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()