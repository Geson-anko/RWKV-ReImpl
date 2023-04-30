import torch
import torch.nn as nn
from torch import Tensor

from .previous_mixing import PreviousMixing


class TimeMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.p_mix_k = PreviousMixing(dim)
        self.p_mix_v = PreviousMixing(dim)
        self.p_mix_r = PreviousMixing(dim)
        self.numerator_last = None
        self.denominator_last = None
        self.W_out = nn.Linear(dim, dim, bias=False)
        self.w = nn.Parameter(torch.rand(dim))
        self.u = nn.Parameter(torch.rand(dim))
        self.sigmoid = nn.Sigmoid()
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wr = nn.Linear(dim, dim, bias=False)

    # (len, batch, dim) -> (len, batch, dim)
    def forward(self, x: Tensor) -> Tensor:
        if self.numerator_last is None:
            self.numerator_last = torch.zeros(
                x.shape[1:], dtype=x.dtype, device=x.device
            )  # (batch, dim)
        if self.denominator_last is None:
            self.denominator_last = torch.zeros(
                x.shape[1:], dtype=x.dtype, device=x.device
            )  # (batch, dim)
        len = x.shape[0]
        k = self.Wk(self.p_mix_k(x))
        v = self.Wv(self.p_mix_v(x))
        r = self.Wr(self.p_mix_r(x))
        w = -torch.exp(self.w)
        exp_w = torch.exp(w)
        exp_w_progression = torch.exp(
            w.unsqueeze(0) * torch.arange(len).unsqueeze(1)
        )  # (len, dim)
        exp_k = torch.exp(k)
        exp_k_v = exp_k * v
        fft_exp_w_progression = torch.fft.rfft(exp_w_progression, n=len * 2, dim=0)  # (?, dim)
        fft_exp_k = torch.fft.rfft(exp_k, n=len * 2, dim=0)  # (?, batch, dim)
        fft_exp_k_v = torch.fft.rfft(exp_k_v, n=len * 2, dim=0)  # (?, batch, dim)
        conv_ewp_ek = torch.fft.irfft(
            fft_exp_w_progression.unsqueeze(1) * fft_exp_k, dim=0
        ).narrow(
            0, 0, len
        )  # (len, batch, dim)
        conv_ewp_ekv = torch.fft.irfft(
            fft_exp_w_progression.unsqueeze(1) * fft_exp_k_v, dim=0
        ).narrow(
            0, 0, len
        )  # (len, batch, dim)
        denominator = (
            self.denominator_last.unsqueeze(0) * exp_w_progression.unsqueeze(1) * exp_w
            + conv_ewp_ek
        )
        self.denominator_last = denominator[-1]
        numerator = (
            self.numerator_last.unsqueeze(0) * exp_w_progression.unsqueeze(1) * exp_w
            + conv_ewp_ekv
        )
        self.numerator_last = numerator[-1]
        exp_u = torch.exp(self.u)
        denominator_out = (denominator - exp_k) / exp_w + exp_k * exp_u
        numerator_out = (numerator - exp_k_v) / exp_w + exp_k_v * exp_u
        return self.W_out(self.sigmoid(r) * numerator_out / denominator_out)

    def clear_hidden(self):
        self.numerator_last = None
        self.denominator_last = None
        self.p_mix_k.clear_hidden()
        self.p_mix_v.clear_hidden()
        self.p_mix_r.clear_hidden()
