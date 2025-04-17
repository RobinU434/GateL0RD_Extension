
from typing import Any, Callable

import torch


class GateL0RDCriterion:
    def __init__(
        self, task_criterion: Callable[[Any], torch.Tensor], reg_lambda: float
    ):
        self.reg_lambda = reg_lambda
        self.task_criterion = task_criterion

    def forward(self, *args, theta: torch.Tensor) -> torch.Tensor:
        task_loss = self.task_criterion(*args)
        loss = task_loss + self.reg_lambda * theta.mean()
        return loss
    
    def __call__(self, *args, theta: torch.Tensor) -> torch.Tensor:
        return self.forward(*args, theta)