import pytest
import torch

from src.models.components.rwkv import RWKV
from src.models.components.rwkv_lang import RWKVLang


@pytest.mark.parametrize(
    """
    len,
    batch,
    dim,
    depth,
    vocab_size,
    """,
    [
        (1024, 1, 32, 2, 123),
        (1024, 4, 64, 1, 432),
        (512, 16, 16, 3, 915),
    ],
)
def test_rwkv_lang(len, batch, dim, depth, vocab_size):
    rwkv_lang = RWKVLang(RWKV(dim, depth), dim, vocab_size)
    assert rwkv_lang.vocab_size == vocab_size
    assert rwkv_lang.dim == dim

    x = torch.randint(vocab_size, (len, batch))
    y = rwkv_lang(x)
    y = rwkv_lang(x)
    rwkv_lang.clear_hidden()
    y = rwkv_lang(x)
    y = rwkv_lang(x)
    assert y.size() == torch.Size([len, batch, vocab_size])
