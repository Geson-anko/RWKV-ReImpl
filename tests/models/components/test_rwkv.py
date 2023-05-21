import pytest
import torch

from src.models.components.rwkv import RWKV


@pytest.mark.parametrize(
    """
    len,
    batch,
    dim,
    depth,
    """,
    [
        (1024, 1, 32, 2),
        (1024, 4, 64, 1),
        (512, 16, 16, 3),
    ],
)
def test_rwkv(len, batch, dim, depth):
    rwkv = RWKV(dim, depth)
    rwkv.init_weights()
    x = torch.rand(len, batch, dim)
    x = rwkv(x)
    x = rwkv(x)
    rwkv.clear_hidden()
    x = rwkv(x)
    x = rwkv(x)
    assert x.size() == torch.Size([len, batch, dim])
