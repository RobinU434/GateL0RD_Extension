from collections.abc import Callable
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
    """
    Base class for GateL0RD-style recurrent neural networks.

    This class implements a general framework for sequence modeling where the hidden state is updated recurrently using a custom cell, and the output can be either the absolute value or a delta (increment) over the input, resembling the solution of a discretized differential equation.

    Mathematical Formulation:
        Let x_t in R^{input_dim} be the input at time t, and h_t in R^{hidden_dim} the hidden state.
        The model processes sequences as follows:

        1. Preprocessing:
            x_t' = f_pre(x_t)
            where f_pre is a feedforward network (fan-in structure).

        2. Recurrent Cell:
            (y_t, h_{t+1}, θ_t) = cell(x_t', h_t)
            where cell is a user-defined module returning the cell output y_t, updated hidden state h_{t+1}, and optional gate parameters θ_t.

        3. Postprocessing:
            - If predict_deltas is False:
                output_t = f_out(y_t)
            - If predict_deltas is True:
                output_t = x_t + delta * f_out(y_t)
            where f_out is a feedforward network, and delta (factor_delta) is a scaling factor for the delta update.

        4. Sequence Generation:
            The model supports teacher forcing via a recurrent_mask, allowing the use of ground truth inputs or previous outputs at each time step.

    Initialization:
        The initial hidden state h_0 is computed from the first num_init_inputs inputs using a separate feedforward network f_init, or set to zeros if num_init_inputs is 0.

    Output:
        For a sequence of length T, the model returns:
            - outputs: sequence of outputs (T - num_init_inputs, batch_size, output_dim)
            - hidden_out: final hidden state
            - theta_t: sequence of gate parameters (if applicable)

    This base class is intended to be subclassed with a custom recurrent cell implementation via the _build_cell method.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = -1,
        n_pre_layers: int = 3,
        n_init_layers: int = 3,
        n_out_layers: int = 2,
        n_g_layers: int = 1,
        n_r_layers: int = 1,
        n_o_layers: int = 1,
        gate_noise_level: float = 1,
        batch_first: bool = False,
        factor_delta: float = 0.1,
        num_warmup_steps: int = 0,
        cell_input_dim: int = 16,
        cell_output_dim: int = 16,
        *args,
        **kwargs,
    ):
        """Initialize the _GateL0RD model.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state.
            output_size (int, optional): Size of the output features. Defaults to -1.
            n_g_layers (int, optional): Number of layers in the gating mechanism. Defaults to 1.
            n_r_layers (int, optional): Number of layers in the recurrent cell. Defaults to 1.
            n_o_layers (int, optional): Number of layers in the output projection. Defaults to 1.
            gate_noise_level (float, optional): Standard deviation of the Gaussian noise added to the gates. Defaults to 1.
            batch_first (bool, optional): If True, the input and output tensors are provided as (batch, seq, feature). Defaults to False.
            factor_delta (float, optional): Scaling factor for the delta update. Defaults to 0.1.
            num_init_inputs (int, optional): Number of initial inputs used to compute the initial hidden state. Defaults to 2.
            cell_input_dim (int, optional): Dimension of the cell input. Defaults to 16.
            cell_output_dim (int, optional): Dimension of the cell output. Defaults to 16.
        """
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_pre_layers = n_pre_layers
        self.n_init_layers = n_init_layers
        self.n_out_layers = n_out_layers
        self.n_g_layers = n_g_layers
        self.n_r_layers = n_r_layers
        self.n_o_layers = n_o_layers
        self.gate_noise_level = gate_noise_level
        self.batch_first = batch_first
        self.factor_delta = factor_delta
        self.num_warmup_steps = num_warmup_steps
        self.cell_input_dim = cell_input_dim
        self.cell_output_dim = cell_output_dim

        self._check_params()

        self._h_seq: torch.Tensor = None
        self.cell = self._build_cell()
        
        self.f_pre = self._build_pre_model()
        self.f_out = self._build_out_model()
        self.f_init = self._build_init_model()
        

    def _check_params(self):
        # check value ranges
        assert self.input_size > 0, "Input size has to be > 0"
        assert self.hidden_size > 0, "Hidden size has to be > 0"
        assert self.output_size > 0, "Output size has to be > 0"
        assert self.n_pre_layers >= 0, "Number of pre layers has to be >= 0"
        assert self.n_init_layers >= 0, "Number of init layers has to be >= 0"
        assert self.n_out_layers >= 0, "Number of out layers has to be >= 0"
        assert self.n_g_layers > 0, "Number of g layers has to be > 0"
        assert self.n_r_layers > 0, "Number of r layers has to be > 0"
        assert self.n_o_layers > 0, "Number of o layers has to be > 0"
        assert self.gate_noise_level >= 0, "Gate noise level has to be >= 0"
        assert self.factor_delta > 0, "Factor delta has to be > 0"
        assert self.num_warmup_steps >= 0, "Number of init inputs has to be >= 0"
        assert self.cell_input_dim > 0, "Cell input dim has to be > 0"
        assert self.cell_output_dim > 0, "Cell output dim has to be > 0"
        
    def _build_init_model(self) -> nn.Module:
        if self.num_warmup_steps > 0:
            return create_fan_in(
                n_layers=self.n_init_layers,
                input_dim=self.num_warmup_steps * self.input_size,
                feature_dim=self.hidden_size,
                a_func="Tanh",
                final_activation=True,
            )
        else:
            return None

    def _build_pre_model(self) -> nn.Module:
        if self.n_pre_layers > 0:
            return create_fan_in(
                n_layers=self.n_pre_layers,
                input_dim=self.input_size,
                feature_dim=self.cell_input_dim,
                a_func="Tanh",
                final_activation=True,
            )
        else:
            assert (
                self.cell_input_dim == self.input_size
            ), "If no pre layers are used, cell input dim has to match input size"
            return nn.Identity()

    
    def _build_out_model(self) -> nn.Module:
        if self.n_out_layers > 0:
            return create_fan_in(
                n_layers=self.n_out_layers,
                input_dim=self.cell_output_dim,
                feature_dim=self.output_size,
                a_func="Tanh",
                final_activation=False,
            )
        else:   
            assert (
                self.cell_output_dim == self.output_size
            ), "If no out layers are used, cell output dim has to match output size"
            return nn.Identity()

    def _build_cell(self) -> nn.Module:
        raise NotImplementedError

    def _delta_postprocess(
        self, cell_out: torch.Tensor, x_raw: torch.Tensor
    ) -> torch.Tensor:
        """solve differential equation to get new position
        x = x_raw + factor_delta * delta

        Args:
            cell_out (torch.Tensor): raw cell output (batch_dim, feature_dim)
            x_raw (torch.Tensor): raw input (batch_dim, feature_dim)

        Returns:
            torch.Tensor: postprocessed deltas
        """
        cell_out = self.f_out.forward(cell_out)
        res = x_raw + self.factor_delta * cell_out
        return res

    def _post_process(self, cell_out: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Postprocess the cell output.

        Args:
            cell_out (torch.Tensor): raw cell output (batch_dim, feature_dim)

        Returns:
            torch.Tensor: postprocessed output
        """
        cell_out = self.f_out.forward(cell_out)
        return cell_out

    def init_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state.

        Args:
            x (torch.Tensor): Input tensor (seq_length, batch_dim, feature_dim)

        Returns:
            torch.Tensor: Initialized hidden state (batch_dim, hidden_dim)
        """
        if self.num_warmup_steps > 0:
            return self.f_init.forward(x[: self.num_warmup_steps])

        _, batch_dim, _ = x.shape
        return torch.zeros((batch_dim, self.hidden_size), device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        h_init: torch.Tensor = None,
        recurrent_mask: torch.Tensor = None,
        predict_deltas: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the _GateL0RD model.

        Args:
            x (torch.Tensor): If batch_first (batch_size, seq_length, input_dim) else (seq_length, batch_size, input_dim)
            h_init (torch.Tensor, optional): Custom init hidden state. Expected dim (batch_size, hidden_dim). If None fall back to zeros. Defaults to None.
            recurrent_mask (torch.Tensor, optional): Mask for teacher forcing. Shape (seq_length, batch_size, 1). 1 means use input, 0 means use last output. Defaults to None.
            predict_deltas (bool, optional): Predict deltas instead of absolute values. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - outputs: If batch_first (batch_size, seq_length, output_dim) else (seq_length, batch_size, output_dim)
                - hidden_out: (batch_size, hidden_dim)
                - theta_t: If batch_first (batch_size, seq_length, output_dim) else (seq_length, batch_size, output_dim)
        """
        # batch correction
        if self.batch_first:
            # put seq dim in front -> (seq_length, batch_dim, feature_dim)
            x = x.permute(1, 0, 2)

        # init hidden state if needed
        if h_init is None:
            hx = self.init_hidden_state(x)
        else:
            hx = h_init
        seq_len = x.shape[0]

        if recurrent_mask is None:
            recurrent_mask = torch.ones((*x.shape[:2], 1))
        else:
            assert recurrent_mask.shape[:2] == x.shape, (
                "Recurrent mask shape has to be (seq_length, batch_size) to be compatible"
            )
            assert (recurrent_mask[self.num_warmup_steps] == 0).sum() == 0, (
                "Teacher forcing is required in the first recurrent input. Otherwise no information about start"
            )

        # selection of postprocess function
        if predict_deltas:
            post_process_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
                self._delta_postprocess
            )
        else:
            post_process_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
                self._post_process
            )

        # recurrent forward
        _h_seq = [hx]
        last_output = None
        y_s = []
        thetas = []
        for seq_idx in range(self.num_warmup_steps, seq_len):
            # teacher forcing
            if last_output is None or recurrent_mask[seq_idx]:
                x_tf = x[seq_idx]
            else:
                x_tf = last_output

            x_inp = self.f_pre.forward(x_tf)
            y_t, hx, theta_t = self.cell.forward(x_inp, hx)

            # postprocessing
            y_t = post_process_func(y_t, x_tf)
            last_output = y_t

            # collect outputs
            y_s.append(y_t)
            thetas.append(theta_t)
            _h_seq.append(hx)

        # organize outputs and logs
        self._h_seq = torch.stack(_h_seq)
        y_s = torch.stack(y_s)
        thetas = torch.stack(thetas)

        # batch correction
        if self.batch_first:
            y_s = y_s.permute(1, 0, 2)
            thetas = thetas.permute(1, 0, 2)

        return y_s, hx, thetas


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
        layers.append(nn.Linear(h_dim, int(pre_l_factor * feature_dim)))
        layers.append(getattr(nn, a_func)())
        h_dim = pre_l_factor * feature_dim

    if not final_activation:
        layers = layers[:-1]
    return nn.Sequential(*layers)
