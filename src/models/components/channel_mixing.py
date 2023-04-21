import torch
import torch.nn as nn
from torch import Tensor

class ChannelMixing(nn.Module):
    def __init__(self, dim: int, len: int):
        super().__init__()
        self.dim = dim
        self.len = len
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wr = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.mix_k = nn.Parameter(torch.rand(dim))
        self.mix_r = nn.Parameter(torch.rand(dim))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.last_x_mix_k = None
        self.last_x_mix_r = None

    # (len, dim) -> (len, dim)
    def forward(self, x: Tensor) -> Tensor:
        if self.last_x_mix_k is None:
            self.last_x_mix_k = torch.randn(self.dim)
        mix_k = self.sigmoid(self.mix_k)
        mix_k_progression = torch.pow(mix_k, torch.arange(self.len).unsqueeze(-1)) # (len, dim)
        fft_mix_k_progression = torch.fft.rfft(mix_k_progression, n=self.len*2, dim=0)
        fft_x = torch.fft.rfft(x, n=self.len*2, dim=0)
        x_conv_mix_k_progression = torch.fft.irfft(fft_x*fft_mix_k_progression, dim=0).narrow(0,0,self.len)
        x_mix_k = self.last_x_mix_k * mix_k_progression * mix_k + (1-mix_k) * x_conv_mix_k_progression
        k = self.Wk(x_mix_k)
        self.last_x_mix_k = x_mix_k

        if self.last_x_mix_r is None:
            self.last_x_mix_r = torch.randn(self.dim)
        mix_r = self.sigmoid(self.mix_r)
        mix_r_progression = torch.pow(mix_r, torch.arange(self.len).unsqueeze(-1)) # (len, dim)
        fft_mix_r_progression = torch.fft.rfft(mix_r_progression, n=self.len*2, dim=0)
        fft_x = torch.fft.rfft(x, n=self.len*2, dim=0)
        x_conv_mix_r_progression = torch.fft.irfft(fft_x*fft_mix_r_progression, dim=0).narrow(0,0,self.len)
        x_mix_r = self.last_x_mix_r * mix_r_progression * mix_r + (1-mix_r) * x_conv_mix_r_progression
        r = self.Wr(x_mix_r)
        self.last_x_mix_r = x_mix_r

        vk = self.Wv(torch.pow(self.relu(k),2))
        return self.sigmoid(r) * vk

