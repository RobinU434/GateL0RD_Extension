#!/usr/bin/env python3
"""
Training script for GateL0RD models using Hydra configuration.

This script handles single model training with Hydra-based configuration management and logging.

Usage:
    python train.py
    python train.py experiment=baseline_training
    python train.py model=v2 data=fetch_pick_place
    python train.py training.max_epochs=100 model.hidden_size=128
    python train.py --config-path=/path/to/configs --config-name=my_config
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger

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
cs.store(group="training", name="default", node=TrainingConfig())
cs.store(group="training", name="hpo", node=TrainingConfig(max_epochs=50, deterministic=False))


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


def setup_reproducibility(seed: int = 42, deterministic: bool = True):
    """Setup reproducibility for training."""
    pl.seed_everything(seed, workers=True)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_data_module(config: DictConfig):
    """Create and setup data module."""
    # Import here to avoid circular imports
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


def create_model(config: DictConfig, input_size: int, output_size: int):
    """Create and setup model."""
    from project.lightning_module import GateL0RDLightningModule
    
    return GateL0RDLightningModule(
        model_version=config.model.version,
        input_size=input_size,
        hidden_size=config.model.hidden_size,
        output_size=output_size,
        n_g_layers=config.model.n_g_layers,
        n_r_layers=config.model.n_r_layers,
        n_o_layers=config.model.n_o_layers,
        gate_noise_level=config.model.gate_noise_level,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer_name=config.training.optimizer_name,
        scheduler_name=config.training.scheduler_name,
        task_loss=config.training.task_loss,
        reg_lambda=config.training.reg_lambda,
        prediction_steps=config.data.prediction_steps,
        log_theta_stats=config.training.log_theta_stats,
    )


def create_callbacks(config: DictConfig, experiment_dirs: Dict[str, Path]) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Early stopping
    callbacks.append(
        EarlyStopping(
            monitor=config.training.monitor_metric,
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            mode='min',
            verbose=True,
        )
    )
    
    # Model checkpointing
    callbacks.append(
        ModelCheckpoint(
            dirpath=experiment_dirs['checkpoints'],
            filename='epoch-{epoch:02d}-val_loss-{val_loss:.3f}',
            monitor=config.training.monitor_metric,
            mode='min',
            save_top_k=config.training.save_top_k,
            save_last=True,
        )
    )
    
    # Progress bar
    callbacks.append(RichProgressBar())
    
    return callbacks


def create_loggers(config: DictConfig, experiment_dirs: Dict[str, Path]) -> list:
    """Create training loggers."""
    loggers = []

    # TensorBoard logger
    loggers.append(
        TensorBoardLogger(
            save_dir=experiment_dirs['logs'],
            name=config.name,
            version=None,
        )
    )

    # CSV logger
    loggers.append(
        CSVLogger(
            save_dir=experiment_dirs['logs'],
            name=config.name,
        )
    )

    # WandB logger (optional, only if wandb is installed and enabled in config)
    if OmegaConf.select(config, "logging.use_wandb", default=False):
        loggers.append(
            WandbLogger(
                name=config.name,
                save_dir=str(experiment_dirs['logs']),
                project=OmegaConf.select(config, "logging.wandb_project", default="GateL0RD"),
                entity=OmegaConf.select(config, "logging.wandb_entity", default=None),
                config=OmegaConf.to_container(config, resolve=True),
                log_model=True,
            )
        )
    return loggers


def run_training(config: DictConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Run standard training."""
    logger.info(f"Starting training: {config.name}")
    logger.info(f"Model version: {config.model.version}")
    
    # Setup reproducibility
    setup_reproducibility(config.seed, config.training.deterministic)
    
    # Create directories
    config_manager = ConfigManager()
    experiment_dirs = config_manager.create_experiment_dirs(config)
    
    # Save configuration
    
    # Create data module
    data_module = create_data_module(config)
    data_module.setup('fit')
    
    # Create model
    model = create_model(config, data_module.get_feature_dim(), data_module.get_feature_dim())
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        gpus=config.training.gpus if config.training.gpus > 0 else None,
        precision=config.training.precision,
        callbacks=create_callbacks(config, experiment_dirs),
        logger=create_loggers(config, experiment_dirs),
        log_every_n_steps=config.training.log_every_n_steps,
        deterministic=config.training.deterministic,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model if test data is available
    if data_module.test_dataloader() is not None:
        trainer.test(model, data_module)
    
    # Get results
    results = {
        'best_val_loss': float(trainer.callback_metrics.get('val_loss', float('inf'))),
        'total_epochs': trainer.current_epoch,
        'model_complexity': model.get_model_complexity(),
    }
    
    # Add test results if available
    if 'test_loss' in trainer.callback_metrics:
        results['test_loss'] = float(trainer.callback_metrics['test_loss'])
    
    logger.info(f"Training completed. Best val_loss: {results['best_val_loss']:.4f}")
    
    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function with Hydra."""
    # Setup logging
    logger = setup_logging("INFO")
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    
    # Validate configuration
    config_manager = ConfigManager()
    config_manager.validate_config(config)
    
    logger.info(f"Running training: {config.name}")
    logger.info(f"Model version: {config.model.version}")
    logger.info(f"Dataset: {config.data.dataset_name}")
    logger.info(f"Max epochs: {config.training.max_epochs}")
    
    try:
        # Run training
        results = run_training(config, logger)
        
        # Save experiment summary
        experiment_dirs = config_manager.create_experiment_dirs(config)
        save_experiment_summary(config, results, str(experiment_dirs['results']))
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {experiment_dirs['results']}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()