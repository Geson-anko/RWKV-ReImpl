import pytest
import torch

from src.models.components.previous_mixing import PreviousMixing


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
def test_previous_mixing(len, dim):
    previous_mixing = PreviousMixing(dim)
    x = torch.rand(len, dim)
    x = previous_mixing(x)
    x = previous_mixing(x)
    previous_mixing.clear_hidden()
    x = previous_mixing(x)
    x = previous_mixing(x)
    assert x.size() == torch.Size([len, dim])
