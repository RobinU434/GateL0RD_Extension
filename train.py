#!/usr/bin/env python3
"""
Training script for GateL0RD models.

This script handles single model training with configuration management and logging.

Usage:
    python train.py --config config/baseline_training.yaml
    python train.py --model.version v2 --data.dataset_name FetchPickAndPlace
    python train.py --training.max_epochs 100 --model.hidden_size 128
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
        
from project.utils.config import (
    ConfigManager, ExperimentConfig, 
    get_baseline_training_config, load_config_from_args, save_experiment_summary
)
from project.gatel0rd.lightning_module import GateL0RDLightningModule
from project.datasets.timeseries import TimeSeriesDataModule


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


def create_data_module(config: ExperimentConfig) -> TimeSeriesDataModule:
    """Create and setup data module."""
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


def create_model(config: ExperimentConfig, input_size: int, output_size: int) -> pl.LightningModule:
    """Create and setup model."""
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


def create_callbacks(config: ExperimentConfig, experiment_dirs: Dict[str, Path]) -> list:
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


def create_loggers(config: ExperimentConfig, experiment_dirs: Dict[str, Path]) -> list:
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
    if getattr(config, "logging", None) and getattr(config.logging, "use_wandb", False):
        loggers.append(
            WandbLogger(
                name=config.name,
                save_dir=str(experiment_dirs['logs']),
                project=getattr(config.logging, "wandb_project", "GateL0RD"),
                entity=getattr(config.logging, "wandb_entity", None),
                config=config,
                log_model=True,
            )
        )
    return loggers


def run_training(config: ExperimentConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Run standard training."""
    logger.info(f"Starting training: {config.name}")
    logger.info(f"Model version: {config.model.version}")
    
    # Setup reproducibility
    setup_reproducibility(config.seed, config.training.deterministic)
    
    # Create directories
    config_manager = ConfigManager()
    experiment_dirs = config_manager.create_experiment_dirs(config)
    
    # Save configuration
    config_manager.save_config(config, experiment_dirs['configs'] / 'experiment_config.yaml')
    
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


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GateL0RD models")
    
    # Configuration
    parser.add_argument('--config', type=str, help="Path to configuration file")
    
    # Quick overrides
    parser.add_argument('--model.version', type=str, choices=['v0', 'v1', 'v2', 'v3'])
    parser.add_argument('--data.dataset_name', type=str)
    parser.add_argument('--data.batch_size', type=int)
    parser.add_argument('--training.max_epochs', type=int)
    parser.add_argument('--training.learning_rate', type=float)
    parser.add_argument('--model.hidden_size', type=int)
    parser.add_argument('--model.n_g_layers', type=int)
    parser.add_argument('--model.n_r_layers', type=int)
    parser.add_argument('--model.gate_noise_level', type=float)
    parser.add_argument('--training.reg_lambda', type=float)
    
    # Logging
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Load configuration with overrides
    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ['config', 'log_level']}
    
    if args.config:
        config = load_config_from_args(args.config, **overrides)
    else:
        # Use default baseline training configuration
        config = get_baseline_training_config()
        
        # Apply overrides
        if overrides:
            from project.utils.config import _update_nested_dict
            import dataclasses
            
            config_dict = dataclasses.asdict(config)
            _update_nested_dict(config_dict, overrides)
            config = ConfigManager()._dict_to_config(config_dict)
    
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
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())