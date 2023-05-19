import copy

import torch
import torch.nn as nn
from torch import Tensor

from .rwkv_block import RWKVBlock


class RWKV(nn.Module):
    def __init__(self, dim: int, depth: int, hidden_dim_factor: int = 4):
        super().__init__()
        self.block_list = nn.ModuleList([RWKVBlock(dim, hidden_dim_factor) for _ in range(depth)])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x

    def clear_hidden(self):
        for block in self.block_list:
            block.clear_hidden()
