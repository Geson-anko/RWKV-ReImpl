import pytest
import torch

from src.models.components.channel_mixing import ChannelMixing


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
def test_channel_mixing(len, dim):
    channel_mixing = ChannelMixing(dim)
    other_axes = torch.randint(1, 4, (torch.randint(1, 3, (1,)).item(),))
    shape = (len, *other_axes, dim)
    x = torch.rand(shape)
    x = channel_mixing(x)
    x = channel_mixing(x)
    channel_mixing.clear_hidden()
    x = channel_mixing(x)
    x = channel_mixing(x)
    assert x.size() == shape
