import pytest
import torch

from src.models.components.rwkv_block import RWKVBlock


@pytest.mark.parametrize(
    """
    len,
    batch,
    dim,
    """,
    [
        (1024, 1, 32),
        (1024, 4, 64),
        (512, 16, 16),
    ],
)
def test_rwkv_block(len, batch, dim):
    rwkv_block = RWKVBlock(dim)
    rwkv_block.init_weights()
    x = torch.rand(len, batch, dim)
    x = rwkv_block(x)
    x = rwkv_block(x)
    rwkv_block.clear_hidden()
    x = rwkv_block(x)
    x = rwkv_block(x)
    assert x.size() == torch.Size([len, batch, dim])
