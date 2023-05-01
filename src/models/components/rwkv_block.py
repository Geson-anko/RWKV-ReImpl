import torch
import torch.nn as nn
from torch import Tensor

from .channel_mixing import ChannelMixing
from .time_mixing import TimeMixing


class RWKVBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.time_mixing = TimeMixing(dim)
        self.channel_mixing = ChannelMixing(dim)
        self.layer_norm = nn.LayerNorm(dim)

    # (len, batch, dim) -> (len, batch, dim)
    def forward(self, x: Tensor) -> Tensor:
        x_ = self.layer_norm(x)
        x_ = self.time_mixing(x_)
        x = x + x_
        x_ = self.layer_norm(x)
        x_ = self.channel_mixing(x_)
        x = x + x_
        return x

    def clear_hidden(self):
        self.time_mixing.clear_hidden()
        self.channel_mixing.clear_hidden()
