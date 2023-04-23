import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.ema_mixing import EMAMixing


class ChannelMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ema_mixing_k = EMAMixing(dim)
        self.ema_mixing_r = EMAMixing(dim)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    # (len, dim) -> (len, dim)
    def forward(self, x: Tensor) -> Tensor:
        vk = self.Wv(torch.pow(self.relu(self.ema_mixing_k(x)), 2))
        return self.sigmoid(self.ema_mixing_r(x)) * vk
