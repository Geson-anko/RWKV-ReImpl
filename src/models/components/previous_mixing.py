import torch
import torch.nn as nn
from torch import Tensor


class PreviousMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mix_factor = nn.Parameter(torch.rand(dim))
        self.x_last = None

    # (len, *, dim) -> (len, *, dim)
    def forward(self, x: Tensor) -> Tensor:
        if self.x_last is None:
            self.x_last = torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)
        x_shift = x.roll(shifts=1, dims=0)
        x_shift[0] = self.x_last
        self.x_last = x[-1]
        return x * self.mix_factor + x_shift * (1 - self.mix_factor)

    def clear_hidden(self):
        self.x_last = None
