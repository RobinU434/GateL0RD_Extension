import logging
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from project.gatel0rd.lightning_module import create_model
from project.utils.devices import get_devices
from project.utils.filesystem import create_experiment_dirs, save_experiment_summary
from project.utils.logger import create_pl_loggers, setup_logging, setup_reproducibility
from project.datasets.build import create_data_module


def create_callbacks(config: DictConfig, experiment_dirs: Dict[str, Path]) -> list:
    """Create training callbacks."""
    callbacks = []
    print(experiment_dirs)
    # Early stopping
    callbacks.append(
        EarlyStopping(
            monitor=config.training.monitor_metric,
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            mode="min",
            verbose=True,
        )
    )

    # Model checkpointing
    callbacks.append(
        ModelCheckpoint(
            dirpath=experiment_dirs["checkpoints"],
            filename="epoch-{epoch:02d}-val_loss-{val_loss:.3f}",
            monitor=config.training.monitor_metric,
            mode="min",
            save_top_k=config.training.save_top_k,
            save_last=True,
        )
    )

    # Progress bar
    callbacks.append(RichProgressBar())

    return callbacks


def run_training(
    config: DictConfig, logger: logging.Logger, experiment_dirs: Dict[str, Path]
) -> Dict[str, Any]:
    """Run standard training."""
    logger.info("Starting training:")
    logger.info(f"Model version: {config.model.version}")

    # Setup reproducibility
    setup_reproducibility(config.seed, config.training.deterministic)

    # Create data module
    data_module = create_data_module(config)
    data_module.setup("fit")

    # Create model
    model = create_model(
        config, data_module.get_feature_dim(), data_module.get_feature_dim()
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        devices=get_devices(config.training.gpus, logger),
        precision=config.training.precision,
        callbacks=create_callbacks(config, experiment_dirs),
        logger=create_pl_loggers(config, experiment_dirs),
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
        "best_val_loss": float(trainer.callback_metrics.get("val_loss", float("inf"))),
        "total_epochs": trainer.current_epoch,
        "model_complexity": model.get_model_complexity(),
    }

    # Add test results if available
    if "test_loss" in trainer.callback_metrics:
        results["test_loss"] = float(trainer.callback_metrics["test_loss"])

    logger.info(f"Training completed. Best val_loss: {results['best_val_loss']:.4f}")

    return results


def train(config: DictConfig, log_level: str = "INFO"):
    logger = setup_logging(log_level)
    experiment_dirs = create_experiment_dirs(config)
    try:
        # Run training
        results = run_training(config, logger, experiment_dirs)

        # Save experiment summary
        save_experiment_summary(config, results, str(experiment_dirs["results"]))

        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {experiment_dirs['results']}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())
        raise
