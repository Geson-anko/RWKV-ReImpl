import pytest
import torch

from src.models.components.ema_mixing import EMAMixing

@pytest.mark.parametrize(
    """
    len,
    dim,
    """,
    [
        (1024, 512),
        (1024, 1024),
        (2048, 1024),
    ],
)
def test_ema_mixing(len, dim):
    ema_mixing = EMAMixing(dim)
    x = torch.rand(len, dim)
    x = ema_mixing(x)
    x = ema_mixing(x)
    assert x.size() == torch.Size([len, dim])
