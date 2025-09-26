"""
PyTorch Lightning module for training GateL0RD models.

This module provides a unified interface for training all GateL0RD versions (v0-v3)
with configurable hyperparameters and scaling experiments.
"""

from typing import Any, Dict, Optional, Tuple
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from project.gatel0rd.v0 import GateL0RDv0
from project.gatel0rd.v1 import GateL0RDv1
from project.gatel0rd.v2 import GateL0RDv2
from project.gatel0rd.v3 import GateL0RDv3
from project.gatel0rd.criterion import GateL0RDCriterion


class GateL0RDLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for GateL0RD training.

    Supports all versions (v0-v3) with configurable architecture scaling
    and hyperparameter optimization.
    """

    MODEL_VERSIONS = {
        "v0": GateL0RDv0,
        "v1": GateL0RDv1,
        "v2": GateL0RDv2,
        "v3": GateL0RDv3,
    }

    def __init__(
        self,
        # Model architecture
        model_version: str = "v0",
        input_size: int = 10,
        hidden_size: int = 64,
        output_size: int = 10,
        # Network scaling parameters (HPO targets)
        n_g_layers: int = 1,
        n_r_layers: int = 1,
        n_o_layers: int = 1,
        gate_noise_level: float = 1.0,
        # Training parameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adam",
        scheduler_name: str = "plateau",
        # Loss function parameters
        task_loss: str = "mse",
        reg_lambda: float = 0.01,
        # Prediction parameters
        prediction_steps: int = 1,
        # Logging
        log_theta_stats: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Save all hyperparameters
        self.save_hyperparameters()

        # Validate model version
        if model_version not in self.MODEL_VERSIONS:
            raise ValueError(
                f"Model version must be one of {list(self.MODEL_VERSIONS.keys())}"
            )

        self.model_version = model_version
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_g_layers = n_g_layers
        self.n_r_layers = n_r_layers
        self.n_o_layers = n_o_layers
        self.gate_noise_level = gate_noise_level
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.reg_lambda = reg_lambda
        self.prediction_steps = prediction_steps
        self.log_theta_stats = log_theta_stats

        # Build model
        self.model = self._build_model()

        # Setup loss function
        self.criterion = self._setup_criterion(task_loss)

        # Metrics storage
        self.train_losses = []
        self.val_losses = []

    def _build_model(self) -> nn.Module:
        """Build the GateL0RD model based on version and hyperparameters."""
        model_class = self.MODEL_VERSIONS[self.model_version]

        return model_class(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_g_layers=self.n_g_layers,
            n_r_layers=self.n_r_layers,
            n_o_layers=self.n_o_layers,
            gate_noise_level=self.gate_noise_level,
            batch_first=True,  # Use batch_first for easier handling
        )

    def _setup_criterion(self, task_loss: str) -> GateL0RDCriterion:
        """Setup the loss criterion."""
        if task_loss == "mse":
            task_criterion = nn.MSELoss()
        elif task_loss == "l1":
            task_criterion = nn.L1Loss()
        elif task_loss == "huber":
            task_criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown task loss: {task_loss}")

        return GateL0RDCriterion(
            task_criterion=task_criterion, reg_lambda=self.reg_lambda
        )

    def forward(
        self, x: torch.Tensor, h_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x, h_init=h_init)

    def _compute_loss_and_metrics(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> Dict[str, torch.Tensor]:
        """Compute loss and metrics for a batch."""
        # Extract data from batch
        input_seq = batch["input"]  # (batch_size, seq_len, feature_dim)
        target_seq = batch["target"]  # (batch_size, pred_steps, feature_dim)

        batch_size, seq_len, feature_dim = input_seq.shape

        # Forward pass
        outputs, final_hidden, theta = self.forward(input_seq)

        # outputs: (batch_size, seq_len - num_init_inputs, feature_dim)
        # We need the last prediction_steps outputs
        pred_outputs = outputs[:, -self.prediction_steps :, :]

        # Reshape for loss computation if needed
        if self.prediction_steps == 1:
            pred_outputs = pred_outputs.squeeze(1)  # (batch_size, feature_dim)
            target_seq = target_seq.squeeze(1)  # (batch_size, feature_dim)

        # Compute loss using GateL0RD criterion
        loss = self.criterion(pred_outputs, target_seq, theta=theta)

        # Compute additional metrics
        with torch.no_grad():
            # Task loss (without regularization)
            if self.prediction_steps == 1:
                task_loss = F.mse_loss(pred_outputs, target_seq)
            else:
                task_loss = F.mse_loss(pred_outputs, target_seq)

            # Regularization loss
            reg_loss = theta.mean()

            # Gate statistics
            theta_mean = theta.mean()
            theta_std = theta.std()
            theta_sparsity = (theta < 0.5).float().mean()  # Fraction of closed gates

            # Prediction error metrics
            pred_error = torch.abs(pred_outputs - target_seq).mean()

        # Log metrics
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_task_loss", task_loss, on_step=False, on_epoch=True)
        self.log(f"{prefix}_reg_loss", reg_loss, on_step=False, on_epoch=True)
        self.log(f"{prefix}_pred_error", pred_error, on_step=False, on_epoch=True)

        if self.log_theta_stats:
            self.log(f"{prefix}_theta_mean", theta_mean, on_step=False, on_epoch=True)
            self.log(f"{prefix}_theta_std", theta_std, on_step=False, on_epoch=True)
            self.log(
                f"{prefix}_theta_sparsity", theta_sparsity, on_step=False, on_epoch=True
            )

        return {
            "loss": loss,
            "task_loss": task_loss,
            "reg_loss": reg_loss,
            "pred_error": pred_error,
            "theta_mean": theta_mean,
            "theta_std": theta_std,
            "theta_sparsity": theta_sparsity,
        }

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        results = self._compute_loss_and_metrics(batch, "train")
        self.train_losses.append(results["loss"].item())
        return results["loss"]

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        results = self._compute_loss_and_metrics(batch, "val")
        self.val_losses.append(results["loss"].item())
        return results["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        results = self._compute_loss_and_metrics(batch, "test")
        return results["loss"]

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        input_seq = batch["input"]

        # Forward pass
        outputs, final_hidden, theta = self.forward(input_seq)

        # Get predictions (last prediction_steps outputs)
        predictions = outputs[:, -self.prediction_steps :, :]

        return {
            "predictions": predictions,
            "final_hidden": final_hidden,
            "theta": theta,
            "inputs": input_seq,
        }

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Setup optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.scheduler_name is None:
            return optimizer

        # Setup scheduler
        if self.scheduler_name.lower() == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        elif self.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=100,  # This should be set based on max_epochs
                eta_min=1e-6,
            )
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Unknown scheduler name: {self.scheduler_name}")

    def get_model_complexity(self) -> Dict[str, int]:
        """Get model complexity metrics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Get specific network parameters
        g_params = sum(p.numel() for p in self.model.cell.g.parameters())
        r_params = sum(p.numel() for p in self.model.cell.r.parameters())
        o_params = (
            sum(p.numel() for p in self.model.cell.out_enc.parameters())
            + sum(p.numel() for p in self.model.cell.fc_p.parameters())
            + sum(p.numel() for p in self.model.cell.fc_o.parameters())
        )

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "g_network_params": g_params,
            "r_network_params": r_params,
            "output_network_params": o_params,
        }

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log model complexity (only once)
        if self.current_epoch == 0:
            complexity = self.get_model_complexity()
            for key, value in complexity.items():
                self.log(f"model/{key}", value)

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        pass


# Utility functions for model analysis
def compare_model_versions(
    input_size: int,
    hidden_size: int,
    output_size: int,
    n_g_layers: int = 1,
    n_r_layers: int = 1,
) -> Dict[str, Dict[str, int]]:
    """Compare complexity of different GateL0RD versions."""
    comparison = {}

    for version in ["v0", "v1", "v2", "v3"]:
        model = GateL0RDLightningModule(
            model_version=version,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_g_layers=n_g_layers,
            n_r_layers=n_r_layers,
        )

        comparison[version] = model.get_model_complexity()

    return comparison


def suggest_hpo_search_space() -> Dict[str, Any]:
    """Suggest hyperparameter search space for optimization."""
    return {
        # Model architecture
        "model_version": ["v0", "v1", "v2", "v3"],
        "hidden_size": [32, 64, 128, 256],
        # Network scaling (main HPO targets)
        "n_g_layers": [1, 2, 3, 4],
        "n_r_layers": [1, 2, 3, 4],
        "n_o_layers": [1, 2, 3],
        "gate_noise_level": [0.1, 0.5, 1.0, 2.0],
        # Training parameters
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "reg_lambda": [0.001, 0.01, 0.1],
        # Optimization
        "optimizer_name": ["adam", "adamw"],
        "scheduler_name": ["plateau", "cosine"],
    }


def create_model(
    config: DictConfig, input_size: int, output_size: int
) -> pl.LightningModule:
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
