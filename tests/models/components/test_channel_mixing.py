import torch
import pytest
from src.models.components.channel_mixing import ChannelMixing

@pytest.mark.parametrize(
    """
    len,
    dim,
    """,
    [
        (1024,1024),
        (2048,1024),
        (1024,512),
    ]
)
def test_channel_mixing(len, dim):
    channel_mixing = ChannelMixing(len, dim)
    x = torch.rand(len, dim)
    y = channel_mixing(x)
    assert y.size() == torch.Size([len, dim])