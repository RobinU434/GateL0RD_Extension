from typing import Tuple
from torch import nn
import torch

from gatel0rd.common import (
    _GateL0RD,
    GaussianNoise,
    HeavisideST,
    ReTanh,
    build_embedding_fc,
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

        self.g = build_embedding_fc(
            self.hidden_size,
            self.hidden_size,
            self.n_g_layers,
            self.hidden_size,
            activation="ELU",
        )
        self.g = nn.Sequential(self.g, GaussianNoise(self.gate_noise_level), ReTanh())

        self.r = build_embedding_fc(
            self.input_size + self.hidden_size,
            self.hidden_size,
            self.n_g_layers,
            self.hidden_size,
            activation="ELU",
        )
        self.r = nn.Sequential(self.r, nn.Tanh())

        self.out_enc = build_embedding_fc(
            self.input_size + self.hidden_size,
            self.hidden_size,
            self.n_g_layers,
            self.hidden_size,
            activation="ELU",
        )
        self.fc_p = nn.Sequential(
            nn.ELU(), nn.Linear(self.hidden_size, self.output_size)
        )
        self.fc_o = nn.Sequential(
            nn.ELU(), nn.Linear(self.hidden_size, self.output_size), nn.Sigmoid()
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
        n_g_layers=1,
        n_r_layers=1,
        n_o_layers=1,
        gate_noise_level=1,
        batch_first=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_size,
            hidden_size,
            output_size,
            n_g_layers,
            n_r_layers,
            n_o_layers,
            gate_noise_level,
            batch_first,
            *args,
            **kwargs,
        )

    def _build_cell(self):
        return GateL0RDCellv3(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_g_layers=self.n_g_layers,
            n_r_layers=self.n_r_layers,
            n_o_layers=self.n_o_layers,
            gate_noise_level=self.gate_noise_level,
        )