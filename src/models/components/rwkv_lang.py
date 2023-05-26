import math

import torch
import torch.nn as nn
from torch import Tensor

from .rwkv import RWKV


class RWKVLang(nn.Module):
    def __init__(self, model: nn.Module, dim: int, vocab_size: int):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

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
        return x

    def clear_hidden(self):
        self.rwkv.clear_hidden()

    def init_weights(self):
        nn.init.orthogonal_(
            self.embedding.weight, gain=1e-4 * math.sqrt(max(self.dim, self.vocab_size))
        )
        nn.init.orthogonal_(
            self.linear_out.weight, gain=0.5 * math.sqrt(self.vocab_size / self.dim)
        )
        self.rwkv.init_weights()
