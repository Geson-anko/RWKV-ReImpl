from src.utils.generations import rwkv_generate_fixed_length
from src.models.components.rwkv_lang import RWKVLang, RWKV
import pytest
import torch

@pytest.mark.parametrize(
    """
    dim,
    num_generating_tokens,
    prepare_hidden_state_recursively,
    """,
    [
        (32, 64, True),
        (32, 128, False),
    ]
)
def test_rwkv_generate_fixed_length(
    dim: int,
    num_generating_tokens: int,
    prepare_hidden_state_recursively: bool,
):
    vocab_size = 32
    net = RWKVLang(
        RWKV(dim, depth=1),
        dim=dim,
        vocab_size=vocab_size,
    )

    prompt_tokens = torch.randint(1, vocab_size, (num_generating_tokens // 2,),dtype=torch.long)

    generated_tokens = rwkv_generate_fixed_length(
        net,
        prompt_tokens,
        num_generating_tokens,
        prepare_hidden_state_recursively
    )
    assert generated_tokens.shape == (num_generating_tokens,)
    assert generated_tokens.dtype == torch.long




