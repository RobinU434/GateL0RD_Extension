from typing import Tuple
from torch import nn
import torch

class ReTanh(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        return x.tanh().clamp(min=0, max=1)


class HeavisideST(torch.autograd.Function):
    """
    Heaviside activation function with straight through estimator
    """

    @staticmethod
    def forward(ctx, input):
        return torch.ceil(input).clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class GaussianNoise(nn.Module):
    def __init__(self, noise_level: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level
        self.noise_distr = torch.distributions.Normal(0, self.noise_level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = self.noise_distr.sample(x.shape)
            x = x + noise
        return x
    
    
class _GateL0RD(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = -1,
        n_g_layers: int = 1,
        n_r_layers: int = 1,
        n_o_layers: int = 1,
        gate_noise_level: float = 1,
        batch_first: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_g_layers = n_g_layers
        self.n_r_layers = n_r_layers
        self.n_o_layers = n_o_layers
        self.gate_noise_level = gate_noise_level
        self.batch_first = batch_first

        self._h_seq = []

    def _build_cell(self):
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, h_init: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): If batch_first (batch_size, seq_length, input_dim) else (seq_length, batch_size, input_dim)
            h_init (torch.Tensor, optional): Custom init hidden state. Expected dim (batch_size, hidden_dim). If None fall back to zeros. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - outputs: If batch_first (batch_size, seq_length, output_dim) else (seq_length, batch_size, output_dim)
                - hidden_out: (batch_size, hidden_dim)
                - theta_t: If batch_first (batch_size, seq_length, output_dim) else (seq_length, batch_size, output_dim)
        """
        self._h_seq = []

        if self.batch_first:
            # put seq dim in front
            x = x.permute(1, 0, 2)

        seq_len, batch_size, _ = x.shape

        if h_init is None:
            hx = torch.zeros((batch_size, self.hidden_size), device=x.device)
        else:
            hx = h_init

        self._h_seq.append(hx)
        y_s = []
        thetas = []
        for seq_idx in range(seq_len):
            y_t, hx, theta_t = self.cell.forward(x[seq_idx], hx)
            y_s.append(y_t)
            thetas.append(theta_t)
            self._h_seq.append(hx)

        self._h_seq = torch.stack(self._h_seq)
        y_s = torch.stack(y_s)
        thetas = torch.stack(thetas)

        if self.batch_first:
            y_s = y_s.permute(1, 0, 2)
            thetas = thetas.permute(1, 0, 2)

        return y_s, hx, thetas

    
def build_embedding_fc(
    input: int,
    output: int,
    n_layers: int = 1,
    hidden: int = -1,
    activation: str = "ReLU",
):
    if n_layers == 1:
        return nn.Linear(input, output)

    if hidden == -1:
        hidden = input

    layers = [nn.Linear(input, hidden)]
    for _ in range(n_layers - 1):
        layers.append(getattr(nn, activation)())
        layers.append(nn.Linear(hidden, hidden))
    layers = nn.Sequential(layers)
    return layers