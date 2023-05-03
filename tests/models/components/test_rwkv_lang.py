import pytest
import torch

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
        (1024, 1, 512, 2, 123),
        (1024, 4, 1024, 1, 432),
        (512, 16, 256, 3, 915),
    ],
)
def test_rwkv_lang(len, batch, dim, depth, vocab_size):
    rwkv_lang = RWKVLang(dim, depth, vocab_size)
    x = torch.randint(vocab_size, (len, batch))
    y = rwkv_lang(x)
    y = rwkv_lang(x)
    rwkv_lang.clear_hidden()
    y = rwkv_lang(x)
    y = rwkv_lang(x)
    assert y.size() == torch.Size([len, batch, vocab_size])
