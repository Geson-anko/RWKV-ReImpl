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

    # (len, dim) -> (len, dim)
    def forward(self, x: Tensor):
        len = x.shape[0]
        if self.x_mix_last is None:
            self.x_mix_last = torch.randn(self.dim)
        factor = self.sigmoid(self.factor)
        factor_progression = torch.pow(factor, torch.arange(len).unsqueeze(-1))
        fft_factor_progression = torch.fft.rfft(factor_progression, n=len * 2, dim=0)
        fft_x = torch.fft.rfft(x, n=len * 2, dim=0)
        x_conv_factor_progression = torch.fft.irfft(fft_x * fft_factor_progression, dim=0).narrow(
            0, 0, len
        )
        x_mix = (
            self.x_mix_last * factor_progression * factor
            + (1 - factor) * x_conv_factor_progression
        )
        self.x_mix_last = x_mix
        return self.W(x_mix)
