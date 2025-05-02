import numpy as np
import torch
import torch.nn as nn
import math


class CombinedLeTE(nn.Module):
    """
    Combined LeTE that integrates both a Fourier-based and a Spline-based LeTE.

    This module takes scalar timestamps (shape: [batch_size, seq_len]) and produces
    time encodings of a specified dimension. A fraction `p` of the dimensions is allocated
    to the Fourier-based LeTE, and the rest is allocated
    to the Spline-based LeTE.

    Args:
        dim (int): Total dimension of the time encodings.
        p (float): Fraction that controls how many dimensions are allocated to
            the Fourier-based LeTE versus the Spline-based LeTE. Defaults to 0.5.
        layer_norm (bool): Whether to apply layer normalization to the outputs. Defaults to True.
        scale (bool): Whether to apply a learnable scale weight to each output dimension. Defaults to True.
        parameter_requires_grad (bool): Require gradient or not. Defaults to True.

    Attributes:
        dim_fourier (int): Number of dimensions allocated to the Fourier-based encoding.
        dim_spline (int): Number of dimensions allocated to the Spline-based encoding.
        w1_fourier (nn.Linear): A linear projection from 1D (timestamp) to `dim_fourier`.
        w1_spline (nn.Linear): A linear projection from 1D (timestamp) to `dim_spline`.
        w2_fourier (FourierSeries): The Fourier-based layer for time encoding.
        w2_spline (Spline): The Spline-based layer for time encoding.
        layernorm (nn.LayerNorm): If `layer_norm=True`, a layer normalization module of shape `dim`.
        scale_weight (nn.Parameter): If `scale=True`, a learnable 1D scale vector of shape (dim,).
    """

    def __init__(self, dim: int, p: float = 0.5, layer_norm: bool = True, scale: bool = True, parameter_requires_grad: bool = True):
        super().__init__()
        self.dim_fourier = math.floor(dim * p)
        self.dim_spline = dim - self.dim_fourier

        self.layer_norm = layer_norm
        self.scale = scale

        # Define the linear projections for Fourier and Spline
        # And initialize w1_fourier with a geometric progression in the weights
        if self.dim_fourier > 0:
            self.w1_fourier = nn.Linear(1, self.dim_fourier)
            fourier_vals = 1.0 / (10 ** np.linspace(0, 9, self.dim_fourier, dtype=np.float32))
            self.w1_fourier.weight = nn.Parameter(torch.from_numpy(fourier_vals).reshape(self.dim_fourier, -1))
            self.w1_fourier.bias = nn.Parameter(torch.zeros(self.dim_fourier))

        # Initialize w1_spline similarly
        if self.dim_spline > 0:
            self.w1_spline = nn.Linear(1, self.dim_spline)
            spline_vals = 1.0 / (10 ** np.linspace(0, 9, self.dim_spline, dtype=np.float32))
            self.w1_spline.weight = nn.Parameter(torch.from_numpy(spline_vals).reshape(self.dim_spline, -1))
            self.w1_spline.bias = nn.Parameter(torch.zeros(self.dim_spline))

        # Instantiate learnable non-linear transformation layers
        if self.dim_fourier > 0:
            self.w2_fourier = FourierSeries(dim_fourier=self.dim_fourier, grid_size_fourier=5)

        if self.dim_spline > 0:
            self.w2_spline = Spline(dim_spline=self.dim_spline, grid_size_spline=5)

        if self.dim_fourier == 0 or self.dim_spline == 0:
            self.scale = False
            self.layer_norm = False

        # Optional scaling vector
        if self.scale:
            self.scale_weight = nn.Parameter(torch.ones(dim))

        # Optional layer normalization
        if self.layer_norm:
            self.layernorm = nn.LayerNorm(dim)

        # Set requires_grad=False if needed
        if not parameter_requires_grad:
            if self.dim_fourier > 0:
                self.w1_fourier.weight.requires_grad = False
                self.w1_fourier.bias.requires_grad = False
                self.w2_fourier.requires_grad = False
            if self.dim_spline > 0:
                self.w1_spline.weight.requires_grad = False
                self.w1_spline.bias.requires_grad = False
                self.w2_spline.requires_grad = False
            if self.scale:
                self.scale_weight.requires_grad = False

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the time encodings for given timestamps.

        Args:
            timestamps (torch.Tensor): Shape (batch_size, seq_len). Each entry is a time value.

        Returns:
            torch.Tensor: Time encodings of shape (batch_size, seq_len, dim).
        """
        # Reshape timestamps to (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Project timestamps separately for Fourier and Spline
        if self.dim_fourier > 0:
            proj_fourier = self.w1_fourier(timestamps)  # (batch_size, seq_len, dim_fourier)
            output_fourier = self.w2_fourier(proj_fourier)  # (batch_size, seq_len, dim_fourier)
        else:
            # If p=1.0, dim_spline=0 => no Spline part, so set an empty tensor
            output_fourier = torch.zeros_like(timestamps[..., :0])

        if self.dim_spline > 0:
            proj_spline = self.w1_spline(timestamps)    # (batch_size, seq_len, dim_spline)
            output_spline = self.w2_spline(proj_spline)  # (batch_size, seq_len, dim_spline)
        else:
            # If p=0.0, dim_fourier=0 => no Fourier part, so set an empty tensor
            output_spline = torch.zeros_like(timestamps[..., :0])

        # Concatenate Fourier and Spline encodings along the last dimension
        output = torch.cat((output_fourier, output_spline), dim=-1)  # (batch_size, seq_len, dim)

        # Optionally apply LayerNorm
        if self.layer_norm:
            output = self.layernorm(output)

        # Optionally apply a learnable scaling
        if self.scale:
            # scale_weight: shape (dim, )
            # output: shape (batch_size, seq_len, dim)
            output = self.scale_weight * output

        return output


class FourierSeries(nn.Module):
    """
    Args:
        dim_fourier (int): Dimension of both the input and output features.
        grid_size_fourier (int): Number of frequency components (excluding constant term). Defaults to 5.

    Attributes:
        dim_fourier (int): Dimension of the input/output features.
        grid_size_fourier (int): Number of frequency components for the Fourier series.
        fourier_weight (torch.nn.Parameter): Learnable Fourier coefficients of shape (2, dim_fourier, dim_fourier, grid_size_fourier).
            The first index corresponds to cosine coefficients, the second to sine coefficients.
        bias (torch.nn.Parameter): Bias term of shape (dim_fourier,).
    """

    def __init__(self, dim_fourier: int, grid_size_fourier: int = 5):
        super().__init__()
        self.dim_fourier = dim_fourier
        self.grid_size_fourier = grid_size_fourier

        # fourier_weight shape: (2, dim_fourier, dim_fourier, grid_size_fourier)
        #   - 2 corresponds to cosine and sine parts, respectively
        #   - The layer outputs dim_fourier features given dim_fourier inputs
        self.fourier_weight = torch.nn.Parameter(torch.randn(2, self.dim_fourier, self.dim_fourier, grid_size_fourier) /
                                                (np.sqrt(self.dim_fourier) * np.sqrt(self.grid_size_fourier)))

        # Bias term, one per output dimension
        self.bias = nn.Parameter(torch.zeros(self.dim_fourier))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        out_shape = original_shape[0:-1] + (self.dim_fourier,)

        # Flatten everything except the last dim
        x = x.reshape(-1, self.dim_fourier)  # (N, dim_fourier)

        # Frequency indices k = 1..grid_size_fourier
        k = torch.arange(1, self.grid_size_fourier + 1, device=x.device)
        k = k.reshape(1, 1, 1, self.grid_size_fourier)  # shape: (1,1,1,K)

        # Reshape x to broadcast with k
        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1], 1)  # (N,1,dim_fourier,1)

        # Compute cos(k * x) and sin(k * x)
        c = torch.cos(k * x_reshaped)  # (N,1,dim_fourier,K)
        s = torch.sin(k * x_reshaped)  # (N,1,dim_fourier,K)

        # Sum up the contributions with the learned Fourier coefficients
        # fourier_weight[0:1] -> cosine part, fourier_weight[1:2] -> sine part
        y = torch.sum(c * self.fourier_weight[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fourier_weight[1:2], dim=(-2, -1))

        # Add bias
        y += self.bias

        # Reshape back to original shape
        y = y.reshape(out_shape)
        return y


class Spline(nn.Module):
    """
    Args:
        dim_spline (int): Dimension of the spline input (and typically output).
        grid_size_spline (int): Number of spline knots (excluding boundary). Defaults to 5.
        order_spline (int): Spline order (degree). Defaults to 3.
        grid_range (list): Value range of the grid. Defaults to [-1, 1].

    Attributes:
        dim_spline (int): Dimension of the input/output.
        grid_size_spline (int): Number of spline knots in each dimension.
        order_spline (int): Spline order (degree).
        grid (torch.Tensor): The registered buffer containing grid points of shape (dim_spline, grid_size_spline + 2 * order_spline + 1).
        base_weight (torch.nn.Parameter): Linear weight for the base branch (dim_spline x dim_spline).
        spline_weight (torch.nn.Parameter): Learnable B-spline coefficients (dim_spline x dim_spline x (grid_size_spline + order_spline)).
    """
    def __init__(self, dim_spline: int, grid_size_spline: int = 5, order_spline: int = 3, grid_range: list = [-1, 1]):
        super().__init__()
        self.dim_spline = dim_spline
        self.grid_size_spline = grid_size_spline
        self.order_spline = order_spline

        # Compute grid spacing
        h = (grid_range[1] - grid_range[0]) / float(self.grid_size_spline)

        # Build the grid for each dimension
        grid = torch.arange(-self.order_spline, self.grid_size_spline + self.order_spline + 1)
        grid = grid * h + grid_range[0]
        grid = grid.expand(self.dim_spline, -1).contiguous()
        self.register_buffer("grid", grid)

        # Base weight for the linear+activation branch
        self.base_weight = nn.Parameter(torch.Tensor(self.dim_spline, self.dim_spline))

        # Spline coefficients
        self.spline_weight = nn.Parameter(torch.Tensor(self.dim_spline, self.dim_spline, self.grid_size_spline + self.order_spline))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.dim_spline)  # flatten all but the last dimension

        # Base branch: tanh + linear
        base_output = nn.functional.linear(torch.tanh(x), self.base_weight)  # shape: (N, dim_spline)

        # If the input batch is empty, return a zero-like tensor
        if x.size(0) == 0:
            spline_output = torch.zeros_like(base_output)
        else:
            # Evaluate B-spline basis
            # b_splines(x).shape -> (N, dim_spline, grid_size_spline + order_spline)
            # Flatten for linear
            b_splines_val = self.b_splines(x).view(x.size(0), -1)

            # Reshape spline_weight for linear operation
            w = self.spline_weight.view(self.dim_spline, -1)
            spline_output = nn.functional.linear(b_splines_val, w)

        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.dim_spline)
        return output

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # A: (dim_spline, batch_size, grid_size_spline + order_spline)
        A = self.b_splines(x).transpose(0, 1)
        # B: (dim_spline, batch_size, dim_spline)
        B = y.transpose(0, 1)

        # Compute pseudo-inverse of A
        A_pinv = torch.pinverse(A)  # shape: (dim_spline, grid_size_spline+order_spline, batch_size)

        # Solve the least squares problem using the pseudo-inverse
        # shape: (dim_spline, grid_size_spline+order_spline, batch_size)
        solution = torch.bmm(A_pinv, B)

        # Permute to (batch_size, dim_spline, grid_size_spline+order_spline) -> (dim_spline, dim_spline, grid_size_spline+order_spline)
        result = solution.permute(2, 0, 1).contiguous()
        return result

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        # grid: (dim_spline, grid_size_spline + 2*order_spline + 1)
        grid = self.grid
        x = x.unsqueeze(-1)  # shape: (N, dim_spline, 1)

        # bases will mark where x lies between [grid_i, grid_{i+1})
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # shape: (N, dim_spline, ?)

        # Recursively elevate the basis to higher spline orders
        for k in range(1, self.order_spline + 1):
            # Segment lengths in the left and right directions
            left_num = (x - grid[:, :-(k + 1)])
            left_den = (grid[:, k:-1] - grid[:, :-(k + 1)])

            right_num = (grid[:, k + 1:] - x)
            right_den = (grid[:, k + 1:] - grid[:, 1:-k])

            bases = (left_num / left_den) * bases[:, :, :-1] + (right_num / right_den) * bases[:, :, 1:]

        return bases.contiguous()
