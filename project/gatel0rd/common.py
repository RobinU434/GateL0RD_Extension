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

        self.cell_input_dim = 16
        self.cell_output_dim = 16
        # how many time steps of ts will you pass in to init the hidden state
        self.num_init_inputs = 2

        self._h_seq = []
        self.cell = self._build_cell()
        self.f_pre = create_fan_in(
            n_layers=3,
            input_dim=self.input_size,
            feature_dim=self.cell_input_dim,
            a_func="Tanh",
            final_activation=True,
        )
        self.f_init = create_fan_in(
            n_layers=3,
            input_dim=self.num_init_inputs * self.input_size,
            feature_dim=self.hidden_size,
            a_func="Tanh",
            final_activation=True,
        )
        self.f_out = create_fan_in(
            n_layers=2,
            input_dim=self.cell_output_dim,
            feature_dim=self.output_size,
            a_func="Tanh",
            final_activation=False,
        )

    def _build_cell(self):
        raise NotImplementedError
    
    def init_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (seq_length, batch_dim, feature_dim)

        Returns:
            torch.Tensor: initialized hidden state
        """
        if self.num_init_inputs > 0:
            return self.f_init.forward(x[:self.num_init_inputs])
        _, batch_dim, _ = x.shape
        return torch.zeros((batch_dim, self.hidden_size), device=x.device)

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

        # batch correction
        if self.batch_first:
            # put seq dim in front -> (seq_length, batch_dim, feature_dim)
            x = x.permute(1, 0, 2)

        seq_len, _  , _ = x.shape

        # init hidden state if needed
        if h_init is None:
            hx = self.init_hidden_state(x)
        else:
            hx = h_init

        # recurrent forward
        self._h_seq.append(hx)
        y_s = []
        thetas = []
        for seq_idx in range(seq_len):
            y_t, hx, theta_t = self.cell.forward(x[seq_idx], hx)
            y_s.append(y_t)
            thetas.append(theta_t)
            self._h_seq.append(hx)

        # organize outputs and logs
        self._h_seq = torch.stack(self._h_seq)
        y_s = torch.stack(y_s)
        thetas = torch.stack(thetas)

        # batch correction
        if self.batch_first:
            y_s = y_s.permute(1, 0, 2)
            thetas = thetas.permute(1, 0, 2)

        return y_s, hx, thetas

    def __repr__(self):
        return super().__repr__()


def create_fan_in(
    n_layers: int,
    input_dim: int,
    feature_dim: int,
    fan_offset: int = -1,
    a_func: str = "TanH",
    final_activation: bool = True,
) -> nn.Sequential:
    """Fan in type of network, decreasing features per layer by the power of 2

    Args:
        n_layers (int): number of layers
        input_dim (int): amount of input features
        feature_dim (int): amount of output neurons
        fan_offset (int): offset to the exponential coefficient of layer dimension
        a_func (str): activation function
        final_activation (bool): last module is an activation function

    Returns:
        nn.Sequential: preprocessing network
    """
    layers = []
    h_dim = input_dim
    for pre_l in range(n_layers):
        # Fan in type of network, decreasing features per layer
        pre_l_factor = pow(2, (n_layers - pre_l + fan_offset))
        layers.append(nn.Linear(h_dim, pre_l_factor * feature_dim))
        layers.append(getattr(nn, a_func)())
        h_dim = pre_l_factor * feature_dim

    if not final_activation:
        layers = layers[:-1]
    return nn.Sequential(*layers)
