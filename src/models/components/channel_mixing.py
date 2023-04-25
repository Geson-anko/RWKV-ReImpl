import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.previous_mixing import PreviousMixing


class ChannelMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.previous_mixing_k = PreviousMixing(dim)
        self.previous_mixing_r = PreviousMixing(dim)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    # (len, dim) -> (len, dim)
    def forward(self, x: Tensor) -> Tensor:
        vk = self.Wv(torch.pow(self.relu(self.previous_mixing_k(x)), 2))
        return self.sigmoid(self.previous_mixing_r(x)) * vk

    def clear_hidden(self):
        self.previous_mixing_k.clear_hidden()
        self.previous_mixing_r.clear_hidden()
