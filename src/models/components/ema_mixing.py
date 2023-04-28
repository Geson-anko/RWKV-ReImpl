import torch
import torch.nn as nn
from torch import Tensor


class EMAMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W = nn.Linear(dim, dim, bias=False)
        self.factor = nn.Parameter(torch.rand(dim))
        self.sigmoid = nn.Sigmoid()
        self.x_mix_last = None

    # (len, batch, dim) -> (len, batch, dim)
    def forward(self, x: Tensor) -> Tensor:
        len = x.shape[0]
        if self.x_mix_last is None:
            self.x_mix_last = torch.zeros(
                x.shape[1:], dtype=x.dtype, device=x.device
            )  # (batch, dim)
        factor = self.sigmoid(self.factor)  # (dim)
        factor_progression = torch.pow(factor, torch.arange(len).unsqueeze(-1))  # (len, dim)
        fft_factor_progression = torch.fft.rfft(factor_progression, n=len * 2, dim=0)  # (?, dim)
        fft_x = torch.fft.rfft(x, n=len * 2, dim=0)  # (?, batch, dim)
        x_conv_factor_progression = torch.fft.irfft(
            fft_x * fft_factor_progression.unsqueeze(1), dim=0
        ).narrow(
            0, 0, len
        )  # (len, batch, dim)
        x_mix = (
            self.x_mix_last.unsqueeze(0) * factor_progression.unsqueeze(1) * factor
            + (1 - factor) * x_conv_factor_progression
        )
        self.x_mix_last = x_mix[-1]
        return self.W(x_mix)

    def clear_hidden(self):
        self.x_mix_last = None
