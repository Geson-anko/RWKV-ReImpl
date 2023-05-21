import pytest
import torch

from src.models.components.ema_mixing import EMAMixing


@pytest.mark.parametrize(
    """
    len,
    batch,
    dim,
    """,
    [
        (1024, 1, 512),
        (1024, 4, 1024),
        (2048, 16, 1024),
    ],
)
def test_ema_mixing(len, batch, dim):
    ema_mixing = EMAMixing(dim)
    x = torch.rand(len, batch, dim)
    assert ema_mixing.x_mix_last is None
    x = ema_mixing(x)
    assert ema_mixing.x_mix_last is not None
    assert "x_mix_last" in ema_mixing.state_dict()
    x = ema_mixing(x)
    ema_mixing.clear_hidden()
    assert ema_mixing.x_mix_last is None
    x = ema_mixing(x)
    assert ema_mixing.x_mix_last is not None
    x = ema_mixing(x)
    assert x.size() == torch.Size([len, batch, dim])
