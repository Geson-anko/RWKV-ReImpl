import torch
import torch.nn as nn
from torch import Tensor

from .rwkv import RWKV


class RWKVLang(nn.Module):
    def __init__(self, model: nn.Module, dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layernorm_out = nn.LayerNorm(dim)
        self.linear_out = nn.Linear(dim, vocab_size)
        self.rwkv = model

    # (len, batch) -> (len, batch, vocab_size)
    def forward(self, x: Tensor):
        x = self.embedding(x)
        x = self.rwkv(x)
        x = self.layernorm_out(x)
        x = self.linear_out(x)
        return x.softmax(dim=-1)

    def clear_hidden(self):
        self.rwkv.clear_hidden()
