import pytest
import torch

from src.models.components.channel_mixing import ChannelMixing


@pytest.mark.parametrize(
    """
    len,
    dim,
    hidden_dim_factor,
    """,
    [
        (1024, 16, 4),
        (1024, 32, 2),
        (2048, 16, 1),
    ],
)
def test_channel_mixing(len, dim, hidden_dim_factor):
    channel_mixing = ChannelMixing(dim, hidden_dim_factor)

    assert channel_mixing.dim == dim
    assert channel_mixing.hidden_dim_factor == hidden_dim_factor

    other_axes = torch.randint(1, 4, (torch.randint(1, 3, (1,)).item(),))
    shape = (len, *other_axes, dim)
    x = torch.rand(shape)
    x = channel_mixing(x)
    x = channel_mixing(x)
    channel_mixing.clear_hidden()
    x = channel_mixing(x)
    x = channel_mixing(x)
    assert x.size() == shape
