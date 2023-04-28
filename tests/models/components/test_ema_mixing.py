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
    x = ema_mixing(x)
    x = ema_mixing(x)
    ema_mixing.clear_hidden()
    x = ema_mixing(x)
    x = ema_mixing(x)
    assert x.size() == torch.Size([len, batch, dim])
