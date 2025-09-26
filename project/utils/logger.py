import logging
import sys
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger


def create_pl_loggers(config: DictConfig, experiment_dirs: Dict[str, Path]) -> list:
    """Create training loggers."""
    loggers = []

    # TensorBoard logger
    loggers.append(
        TensorBoardLogger(
            save_dir=experiment_dirs["logs"],
            name="tb_logs",
            version=None,
        )
    )

    # CSV logger
    loggers.append(
        CSVLogger(
            save_dir=experiment_dirs["logs"],
            name="csv_logs",
        )
    )

    # WandB logger (optional, only if wandb is installed and enabled in config)
    if getattr(config, "logging", None) and getattr(config.logging, "use_wandb", False):
        loggers.append(
            WandbLogger(
                name="wandb_logs",
                save_dir=str(experiment_dirs["logs"]),
                project=getattr(config.logging, "wandb_project", "GateL0RD"),
                entity=getattr(config.logging, "wandb_entity", None),
                config=config,
                log_model=True,
            )
        )
    return loggers


# =============================== LOGGING ===============================


class CustomFormatter(logging.Formatter):
    """class copied from:
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output"""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())

    logger = logging.getLogger(__name__)
    logger.propagate = False  # Prevent duplicate logs
    logger.addHandler(ch)

    return logger


def setup_reproducibility(seed: int = 42, deterministic: bool = True):
    """Setup reproducibility for training."""
    pl.seed_everything(seed, workers=True)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
