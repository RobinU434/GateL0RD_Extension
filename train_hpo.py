#!/usr/bin/env python3
"""
Hyperparameter Optimization training script for GateL0RD models.

This script handles HPO experiments using Optuna with support for scaling
experiments and architecture comparisons.

Usage:
    python train_hpo.py --experiment_type scaling --n_trials 100
    python train_hpo.py --experiment_type comparison --n_trials_per_version 25
    python train_hpo.py --config config/scaling_experiment.yaml
    python train_hpo.py --base_version v0 --n_trials 50 --timeout 7200
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from project.utils.config import (ConfigManager, ExperimentConfig,
                            get_comparison_config,
                            get_scaling_experiment_config,
                            load_config_from_args, save_experiment_summary)
from project.datasets.timeseries import TimeSeriesDataModule
from project.scripts.hpo import GateL0RDHPORunner


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def setup_reproducibility(seed: int = 42):
    """Setup reproducibility for HPO."""
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = (
        False  # Allow some non-determinism for performance
    )
    torch.backends.cudnn.benchmark = True


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


def run_scaling_experiment(
    config: ExperimentConfig, logger: logging.Logger
) -> Dict[str, Any]:
    """Run parameter scaling HPO experiment."""
    logger.info(f"Starting parameter scaling HPO experiment: {config.name}")

    if config.hpo is None:
        raise ValueError("HPO configuration is required for HPO experiment")

    # Setup reproducibility
    setup_reproducibility(config.seed)

    # Create data module
    data_module = create_data_module(config)

    # Create HPO runner
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
        "experiment_type": "scaling",
        "base_version": config.model.version,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "analysis": analysis,
    }

    logger.info(
        f"Scaling experiment completed. Best value: {results['best_value']:.4f}"
    )
    logger.info(f"Best version: {study.best_params.get('model_version', 'N/A')}")

    return results


def run_comparison_experiment(
    config: ExperimentConfig, logger: logging.Logger
) -> Dict[str, Any]:
    """Run architecture comparison HPO experiment."""
    logger.info(f"Starting architecture comparison HPO experiment: {config.name}")

    if config.hpo is None:
        raise ValueError("HPO configuration is required for HPO experiment")

    # Setup reproducibility
    setup_reproducibility(config.seed)

    # Create data module
    data_module = create_data_module(config)

    # Create HPO runner
    hpo_runner = GateL0RDHPORunner(
        data_module=data_module,
        study_name=config.hpo.study_name,
        storage_url=config.hpo.storage_url,
        log_dir=config.log_dir,
    )

    # Calculate trials per version
    n_trials_per_version = config.hpo.n_trials // 4  # Split across v0-v3
    if config.hpo.n_trials % 4 != 0:
        logger.warning(
            f"n_trials ({config.hpo.n_trials}) not divisible by 4. Using {n_trials_per_version} trials per version."
        )

    # Run architecture comparison
    studies = hpo_runner.run_architecture_comparison(
        n_trials_per_version=n_trials_per_version,
        max_epochs=config.training.max_epochs,
    )

    # Collect results for each version
    version_results = {}
    best_overall_value = float("inf")
    best_overall_version = None

    for version, study in studies.items():
        analysis = hpo_runner.analyze_results(study)
        version_results[version] = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "analysis": analysis,
        }

        # Track overall best
        if study.best_value < best_overall_value:
            best_overall_value = study.best_value
            best_overall_version = version

        logger.info(f"Version {version}: Best value {study.best_value:.4f}")

    results = {
        "experiment_type": "comparison",
        "version_results": version_results,
        "best_overall_version": best_overall_version,
        "best_overall_value": best_overall_value,
        "total_trials": sum(len(study.trials) for study in studies.values()),
    }

    logger.info(f"Architecture comparison completed.")
    logger.info(
        f"Best overall version: {best_overall_version} (val_loss: {best_overall_value:.4f})"
    )

    return results


def run_custom_hpo(
    config: ExperimentConfig, hpo_config: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Any]:
    """Run custom HPO experiment with user-defined search space."""
    logger.info(f"Starting custom HPO experiment: {config.name}")

    # Setup reproducibility
    setup_reproducibility(config.seed)

    # Create data module
    data_module = create_data_module(config)

    # Create custom HPO objective
    from project.scripts.hpo import GateL0RDHPOObjective

    base_config = {
        "model_version": config.model.version,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "reg_lambda": config.training.reg_lambda,
        "optimizer_name": config.training.optimizer_name,
        "scheduler_name": config.training.scheduler_name,
    }

    objective = GateL0RDHPOObjective(
        data_module=data_module,
        base_config=base_config,
        hpo_config=hpo_config,
        max_epochs=config.training.max_epochs,
        log_dir=str(Path(config.log_dir) / "custom_hpo"),
    )

    # Create and run study
    import optuna

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=5
        ),
    )

    study.optimize(objective, n_trials=config.hpo.n_trials, timeout=config.hpo.timeout)

    results = {
        "experiment_type": "custom",
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "search_space": hpo_config,
    }

    logger.info(f"Custom HPO completed. Best value: {results['best_value']:.4f}")

    return results


def main():
    """Main HPO function."""
    parser = argparse.ArgumentParser(
        description="Run HPO experiments for GateL0RD models"
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--experiment_type",
        choices=["scaling", "comparison", "custom"],
        default="scaling",
        help="Type of HPO experiment to run",
    )

    # HPO specific arguments
    parser.add_argument(
        "--n_trials", type=int, default=100, help="Number of trials for HPO"
    )
    parser.add_argument(
        "--n_trials_per_version",
        type=int,
        default=25,
        help="Number of trials per version (for comparison experiment)",
    )
    parser.add_argument(
        "--base_version",
        type=str,
        choices=["v0", "v1", "v2", "v3"],
        default="v0",
        help="Base model version for scaling experiment",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for HPO experiment",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="gatel0rd_hpo",
        help="Name for the Optuna study",
    )
    parser.add_argument(
        "--storage_url",
        type=str,
        default=None,
        help="Database URL for study persistence",
    )

    # Quick overrides
    parser.add_argument("--data.dataset_name", type=str)
    parser.add_argument("--data.batch_size", type=int)
    parser.add_argument(
        "--training.max_epochs", type=int, default=50, help="Max epochs per trial"
    )
    parser.add_argument("--model.hidden_size", type=int)

    # Logging
    parser.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs/hpo", help="Directory for HPO logs"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Load configuration with overrides
    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ["config", "log_level", "experiment_type"]
    }

    # Handle HPO-specific overrides
    hpo_overrides = {}
    if args.n_trials:
        hpo_overrides["hpo.n_trials"] = args.n_trials
    if args.timeout:
        hpo_overrides["hpo.timeout"] = args.timeout
    if args.study_name:
        hpo_overrides["hpo.study_name"] = args.study_name
    if args.storage_url:
        hpo_overrides["hpo.storage_url"] = args.storage_url
    if args.base_version:
        hpo_overrides["model.version"] = args.base_version
    if args.log_dir:
        hpo_overrides["log_dir"] = args.log_dir

    overrides.update(hpo_overrides)

    if args.config:
        config = load_config_from_args(args.config, **overrides)
    else:
        # Use default based on experiment type
        if args.experiment_type == "scaling":
            config = get_scaling_experiment_config()
        elif args.experiment_type == "comparison":
            config = get_comparison_config()
        else:
            # For custom, use scaling as base
            config = get_scaling_experiment_config()
            config.experiment_type = "custom"

        # Apply overrides
        if overrides:
            import dataclasses

            from project.utils.config import _update_nested_dict

            config_dict = dataclasses.asdict(config)
            _update_nested_dict(config_dict, overrides)
            config = ConfigManager()._dict_to_config(config_dict)

    # Validate configuration
    config_manager = ConfigManager()
    config_manager.validate_config(config)

    logger.info(f"Running {args.experiment_type} HPO experiment: {config.name}")
    logger.info(f"Dataset: {config.data.dataset_name}")
    logger.info(f"Trials: {config.hpo.n_trials if config.hpo else 'N/A'}")
    logger.info(f"Base version: {config.model.version}")

    try:
        # Run experiment based on type
        if args.experiment_type == "scaling":
            results = run_scaling_experiment(config, logger)
        elif args.experiment_type == "comparison":
            results = run_comparison_experiment(config, logger)
        elif args.experiment_type == "custom":
            # For custom experiments, use the default scaling search space
            from project.scripts.hpo import get_default_scaling_config

            hpo_config = get_default_scaling_config()
            results = run_custom_hpo(config, hpo_config, logger)
        else:
            raise ValueError(f"Unknown experiment type: {args.experiment_type}")

        # Save experiment summary
        experiment_dirs = config_manager.create_experiment_dirs(config)
        save_experiment_summary(config, results, str(experiment_dirs["results"]))

        logger.info("HPO experiment completed successfully!")
        logger.info(f"Results saved to: {experiment_dirs['results']}")

        # Print summary
        if "best_value" in results:
            logger.info(f"Best validation loss: {results['best_value']:.4f}")
        if "best_params" in results:
            logger.info("Best parameters:")
            for param, value in results["best_params"].items():
                logger.info(f"  {param}: {value}")

        return 0

    except Exception as e:
        logger.error(f"HPO experiment failed: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
