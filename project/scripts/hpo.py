"""
Hyperparameter Optimization (HPO) framework for GateL0RD using Optuna.

This module provides comprehensive HPO capabilities for scaling experiments
focused on g and r network parameters across all GateL0RD versions.
"""

import json
from typing import Any, Dict, Optional
from pathlib import Path
import logging

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from project.gatel0rd.lightning_module import GateL0RDLightningModule
from project.datasets.timeseries import TimeSeriesDataModule


class GateL0RDHPOObjective:
    """
    Optuna objective function for GateL0RD hyperparameter optimization.

    Focuses on scaling experiments for g and r networks while maintaining
    or improving performance compared to baseline (v0).
    """

    def __init__(
        self,
        data_module: TimeSeriesDataModule,
        base_config: Dict[str, Any],
        hpo_config: Dict[str, Any],
        max_epochs: int = 50,
        gpus: int = 1,
        trial_timeout: Optional[int] = 3600,  # 1 hour per trial
        log_dir: str = "logs/hpo",
    ):
        """
        Initialize the HPO objective.

        Args:
            data_module: PyTorch Lightning data module
            base_config: Base configuration for model
            hpo_config: HPO search space configuration
            max_epochs: Maximum epochs per trial
            gpus: Number of GPUs to use
            trial_timeout: Maximum time per trial in seconds
            log_dir: Directory for logging
        """
        self.data_module = data_module
        self.base_config = base_config
        self.hpo_config = hpo_config
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.trial_timeout = trial_timeout
        self.log_dir = Path(log_dir)

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup feature dimensions
        self.input_size = data_module.get_feature_dim()
        self.output_size = data_module.get_feature_dim()

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for a single trial.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (validation loss - lower is better)
        """
        # Sample hyperparameters from search space
        params = self._sample_hyperparameters(trial)

        # Create model with sampled parameters
        model = GateL0RDLightningModule(
            input_size=self.input_size, output_size=self.output_size, **params
        )

        # Setup logging for this trial
        trial_logger = TensorBoardLogger(
            save_dir=self.log_dir, name=f"trial_{trial.number}", version=None
        )

        csv_logger = CSVLogger(
            save_dir=self.log_dir / f"trial_{trial.number}", name="metrics"
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True),
            ModelCheckpoint(
                dirpath=self.log_dir / f"trial_{trial.number}" / "checkpoints",
                filename="best-{epoch}-{val_loss:.3f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
            OptunaPruningCallback(trial, monitor="val_loss"),
        ]

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gpus=self.gpus if self.gpus > 0 else None,
            callbacks=callbacks,
            logger=[trial_logger, csv_logger],
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        try:
            # Train model
            trainer.fit(model, self.data_module)

            # Get best validation loss
            val_loss = trainer.callback_metrics["val_loss"].item()

            # Log trial results
            self._log_trial_results(trial, params, val_loss)

            return val_loss

        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {str(e)}")
            # Return high loss value for failed trials
            return float("inf")

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = self.base_config.copy()

        # Model version
        if "model_version" in self.hpo_config:
            params["model_version"] = trial.suggest_categorical(
                "model_version", self.hpo_config["model_version"]
            )

        # Architecture parameters (main HPO targets)
        if "hidden_size" in self.hpo_config:
            params["hidden_size"] = trial.suggest_categorical(
                "hidden_size", self.hpo_config["hidden_size"]
            )

        # Network scaling parameters - key focus areas
        if "n_g_layers" in self.hpo_config:
            params["n_g_layers"] = trial.suggest_int(
                "n_g_layers",
                min(self.hpo_config["n_g_layers"]),
                max(self.hpo_config["n_g_layers"]),
            )

        if "n_r_layers" in self.hpo_config:
            params["n_r_layers"] = trial.suggest_int(
                "n_r_layers",
                min(self.hpo_config["n_r_layers"]),
                max(self.hpo_config["n_r_layers"]),
            )

        if "n_o_layers" in self.hpo_config:
            params["n_o_layers"] = trial.suggest_int(
                "n_o_layers",
                min(self.hpo_config["n_o_layers"]),
                max(self.hpo_config["n_o_layers"]),
            )

        if "gate_noise_level" in self.hpo_config:
            params["gate_noise_level"] = trial.suggest_float(
                "gate_noise_level",
                min(self.hpo_config["gate_noise_level"]),
                max(self.hpo_config["gate_noise_level"]),
                log=True,
            )

        # Training parameters
        if "learning_rate" in self.hpo_config:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate",
                min(self.hpo_config["learning_rate"]),
                max(self.hpo_config["learning_rate"]),
                log=True,
            )

        if "weight_decay" in self.hpo_config:
            params["weight_decay"] = trial.suggest_float(
                "weight_decay",
                min(self.hpo_config["weight_decay"]),
                max(self.hpo_config["weight_decay"]),
                log=True,
            )

        if "reg_lambda" in self.hpo_config:
            params["reg_lambda"] = trial.suggest_float(
                "reg_lambda",
                min(self.hpo_config["reg_lambda"]),
                max(self.hpo_config["reg_lambda"]),
                log=True,
            )

        # Categorical parameters
        if "optimizer_name" in self.hpo_config:
            params["optimizer_name"] = trial.suggest_categorical(
                "optimizer_name", self.hpo_config["optimizer_name"]
            )

        if "scheduler_name" in self.hpo_config:
            params["scheduler_name"] = trial.suggest_categorical(
                "scheduler_name", self.hpo_config["scheduler_name"]
            )

        return params

    def _log_trial_results(
        self, trial: optuna.Trial, params: Dict[str, Any], val_loss: float
    ):
        """Log trial results to file."""
        results = {
            "trial_number": trial.number,
            "parameters": params,
            "val_loss": val_loss,
            "trial_state": trial.state.name if hasattr(trial, "state") else "COMPLETE",
        }

        results_file = self.log_dir / "trial_results.jsonl"
        with open(results_file, "a") as f:
            f.write(json.dumps(results) + "\n")


class OptunaPruningCallback(pl.Callback):
    """Callback for Optuna pruning integration."""

    def __init__(self, trial: optuna.Trial, monitor: str):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Check if trial should be pruned."""
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is not None:
            self.trial.report(current_score.item(), step=trainer.current_epoch)
            if self.trial.should_prune():
                message = f"Trial was pruned at epoch {trainer.current_epoch}."
                raise optuna.TrialPruned(message)


class GateL0RDHPORunner:
    """
    Main class for running GateL0RD hyperparameter optimization experiments.

    Provides methods for running different types of HPO experiments:
    - Scaling experiments (main focus)
    - Architecture comparison
    - Full hyperparameter search
    """

    def __init__(
        self,
        data_module: TimeSeriesDataModule,
        study_name: str = "gatel0rd_scaling",
        storage_url: Optional[str] = None,
        log_dir: str = "logs/hpo",
    ):
        """
        Initialize the HPO runner.

        Args:
            data_module: PyTorch Lightning data module
            study_name: Name for the Optuna study
            storage_url: Database URL for study storage (None for in-memory)
            log_dir: Directory for logging
        """
        self.data_module = data_module
        self.study_name = study_name
        self.storage_url = storage_url
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_study(
        self,
        direction: str = "minimize",
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """Create or load an Optuna study."""
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10, interval_steps=5
            )

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction=direction,
            pruner=pruner,
            load_if_exists=True,
        )

        return study

    def run_scaling_experiment(
        self,
        base_version: str = "v0",
        n_trials: int = 100,
        max_epochs: int = 50,
        timeout: Optional[int] = None,
    ) -> optuna.Study:
        """
        Run scaling experiment focusing on g and r network parameters.

        This is the main experiment type for parameter reduction while
        maintaining performance.
        """
        self.logger.info(
            f"Starting scaling experiment with base version {base_version}"
        )

        # Base configuration
        base_config = {
            "model_version": base_version,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "reg_lambda": 0.01,
            "optimizer_name": "adam",
            "scheduler_name": "plateau",
        }

        # HPO search space focused on scaling
        hpo_config = {
            "model_version": ["v0", "v1", "v2", "v3"],  # Compare all versions
            "hidden_size": [32, 64, 128],  # Test different scales
            "n_g_layers": [1, 2, 3, 4],  # Key parameter - gate network depth
            "n_r_layers": [1, 2, 3, 4],  # Key parameter - recurrent network depth
            "n_o_layers": [1, 2, 3],  # Output network depth
            "gate_noise_level": [0.1, 2.0],  # Gate noise range
        }

        study = self.create_study()
        objective = GateL0RDHPOObjective(
            data_module=self.data_module,
            base_config=base_config,
            hpo_config=hpo_config,
            max_epochs=max_epochs,
            log_dir=str(self.log_dir / "scaling_experiment"),
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Save results
        self._save_study_results(study, "scaling_experiment")

        return study

    def run_architecture_comparison(
        self,
        n_trials_per_version: int = 25,
        max_epochs: int = 50,
    ) -> Dict[str, optuna.Study]:
        """
        Run architecture comparison experiment.

        Compares all versions (v0-v3) with optimized hyperparameters for each.
        """
        self.logger.info("Starting architecture comparison experiment")

        results = {}

        for version in ["v0", "v1", "v2", "v3"]:
            self.logger.info(f"Optimizing version {version}")

            # Version-specific base config
            base_config = {
                "model_version": version,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "reg_lambda": 0.01,
            }

            # HPO search space per version
            hpo_config = {
                "hidden_size": [32, 64, 128, 256],
                "n_g_layers": [1, 2, 3, 4],
                "n_r_layers": [1, 2, 3, 4],
                "n_o_layers": [1, 2, 3],
                "gate_noise_level": [0.1, 2.0],
                "learning_rate": [1e-4, 1e-2],
                "weight_decay": [1e-5, 1e-3],
                "reg_lambda": [0.001, 0.1],
                "optimizer_name": ["adam", "adamw"],
            }

            study = self.create_study()
            study.study_name = f"{self.study_name}_{version}"

            objective = GateL0RDHPOObjective(
                data_module=self.data_module,
                base_config=base_config,
                hpo_config=hpo_config,
                max_epochs=max_epochs,
                log_dir=str(self.log_dir / f"architecture_comparison/{version}"),
            )

            study.optimize(objective, n_trials=n_trials_per_version)
            results[version] = study

            # Save intermediate results
            self._save_study_results(study, f"architecture_comparison_{version}")

        return results

    def _save_study_results(self, study: optuna.Study, experiment_name: str):
        """Save study results to files."""
        results_dir = self.log_dir / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        best_params = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }

        with open(results_dir / "best_results.json", "w") as f:
            json.dumps(best_params, f, indent=2)

        # Save all trials
        trials_data = []
        for trial in study.trials:
            trials_data.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
            )

        with open(results_dir / "all_trials.json", "w") as f:
            json.dump(trials_data, f, indent=2)

        self.logger.info(f"Results saved to {results_dir}")

    def analyze_results(self, study: optuna.Study) -> Dict[str, Any]:
        """Analyze and summarize HPO results."""
        analysis = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_complete_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "n_pruned_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
        }

        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            analysis["param_importance"] = importance
        except Exception as e:
            self.logger.warning(f"Could not compute parameter importance: {e}")

        return analysis


# Utility functions
def get_default_scaling_config() -> Dict[str, Any]:
    """Get default configuration for scaling experiments."""
    return {
        "model_version": ["v0", "v1", "v2", "v3"],
        "hidden_size": [32, 64, 128],
        "n_g_layers": [1, 2, 3, 4],
        "n_r_layers": [1, 2, 3, 4],
        "n_o_layers": [1, 2, 3],
        "gate_noise_level": [0.1, 2.0],
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "reg_lambda": [0.001, 0.01, 0.1],
    }


def estimate_parameter_reduction(
    baseline_params: Dict[str, Any],
    optimized_params: Dict[str, Any],
    input_size: int,
    output_size: int,
) -> Dict[str, float]:
    """Estimate parameter reduction between configurations."""
    # Create models with both configurations
    baseline_model = GateL0RDLightningModule(
        input_size=input_size, output_size=output_size, **baseline_params
    )

    optimized_model = GateL0RDLightningModule(
        input_size=input_size, output_size=output_size, **optimized_params
    )

    baseline_complexity = baseline_model.get_model_complexity()
    optimized_complexity = optimized_model.get_model_complexity()

    reductions = {}
    for key in baseline_complexity:
        baseline_val = baseline_complexity[key]
        optimized_val = optimized_complexity[key]

        if baseline_val > 0:
            reduction = (baseline_val - optimized_val) / baseline_val
            reductions[f"{key}_reduction"] = reduction
        else:
            reductions[f"{key}_reduction"] = 0.0

    return reductions
