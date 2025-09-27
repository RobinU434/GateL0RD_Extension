from typing import Tuple
from torch import nn
import torch

from project.gatel0rd.common import (
    _GateL0RD,
    GaussianNoise,
    HeavisideST,
    ReTanh,
    create_fan_in,
)


class GateL0RDCellv3(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = -1,
        n_g_layers: int = 1,
        n_r_layers: int = 1,
        n_o_layers: int = 1,
        gate_noise_level: float = 1,
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

        self.g = nn.Sequential(
            create_fan_in(
                n_layers=self.n_g_layers,
                input_dim=self.hidden_size,
                feature_dim=self.hidden_size,
                a_func="Tanh",
                fan_offset=-2,
                final_activation=False,
            ),
            GaussianNoise(self.gate_noise_level),
            ReTanh(),
        )

        self.r = create_fan_in(
            n_layers=self.n_r_layers,
            input_dim=self.input_size + self.hidden_size,
            feature_dim=self.hidden_size,
            a_func="Tanh",
            fan_offset=-2,
            final_activation=True,
        )

        assert self.n_o_layers > 0, "At least 1 layers for work load splitting required"
        self.out_enc = create_fan_in(
            n_layers=self.n_o_layers - 1,
            input_dim=self.input_size + self.hidden_size,
            feature_dim=self.input_size + self.hidden_size,
            a_func="Tanh",
            fan_offset=-2,
        )

        self.fc_p = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.output_size),
            nn.Tanh(),
        )
        self.fc_o = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.output_size),
            nn.Sigmoid(),
        )

    def forward(
        self, x_t: torch.Tensor, hx: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            x_t (torch.Tensor): (batch_size, input_features)
            hx (torch.Tensor, optional): (batch_size, hidden_size). Defaults to None.

        Returns: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - outputs: (batch_size, output_dim)
            - hidden_out: (batch_size, hidden_dim)
            - theta_t: (batch_size, hidden_dim)
        """
        assert len(x_t.shape) == 2, (
            f"Expected (batch_size, input_features) in x_t, but got: {x_t.shape}"
        )

        if hx is None:
            hx = torch.zeros((len(x_t), self.hidden_size), device=x_t.device)

        concat = torch.cat([x_t, hx], dim=1)
        candidate_hidden = self.r.forward(concat)

        lambda_t = self.g.forward(candidate_hidden)
        theta_t = HeavisideST.apply(lambda_t)

        new_hx = lambda_t * candidate_hidden + (1 - lambda_t) * hx

        concat = torch.cat([x_t, new_hx], dim=1)
        out_embed = self.out_enc.forward(concat)
        y_t = self.fc_p.forward(out_embed) * self.fc_o.forward(out_embed)

        return y_t, new_hx, theta_t


class GateL0RDv3(_GateL0RD):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=-1,
        n_pre_layers=3,
        n_init_layers=3,
        n_out_layers=2,
        n_g_layers=1,
        n_r_layers=1,
        n_o_layers=1,
        gate_noise_level=1,
        batch_first=False,
        factor_delta=0.1,
        num_warmup_steps=0,
        cell_input_dim=16,
        cell_output_dim=16,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_size,
            hidden_size,
            output_size,
            n_pre_layers,
            n_init_layers,
            n_out_layers,
            n_g_layers,
            n_r_layers,
            n_o_layers,
            gate_noise_level,
            batch_first,
            factor_delta,
            num_warmup_steps,
            cell_input_dim,
            cell_output_dim,
            *args,
            **kwargs,
        )

    def _build_cell(self):
        return GateL0RDCellv3(
            input_size=self.cell_input_dim,
            hidden_size=self.hidden_size,
            output_size=self.cell_output_dim,
            n_g_layers=self.n_g_layers,
            n_r_layers=self.n_r_layers,
            n_o_layers=self.n_o_layers,
            gate_noise_level=self.gate_noise_level,
        )
