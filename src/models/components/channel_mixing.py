import torch
import torch.nn as nn
from torch import Tensor

from .previous_mixing import PreviousMixing


class ChannelMixing(nn.Module):
    def __init__(self, dim: int, hidden_dim_factor: int = 4):
        super().__init__()
        self.dim = dim
        self.hidden_dim_factor = hidden_dim_factor
        hidden_dim = dim * hidden_dim_factor
        self.previous_mixing_k = PreviousMixing(dim)
        self.previous_mixing_r = PreviousMixing(dim)
        self.Wk = nn.Linear(dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, dim, bias=False)
        self.Wr = nn.Linear(dim, dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    # (len, *, dim) -> (len, *, dim)
    def forward(self, x: Tensor) -> Tensor:
        vk = self.Wv(torch.pow(self.relu(self.Wk(self.previous_mixing_k(x))), 2))
        return self.sigmoid(self.Wr(self.previous_mixing_r(x))) * vk

    def clear_hidden(self):
        self.previous_mixing_k.clear_hidden()
        self.previous_mixing_r.clear_hidden()
