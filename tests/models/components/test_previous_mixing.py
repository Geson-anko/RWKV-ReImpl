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
    other_axes = torch.randint(1, 4, (torch.randint(1, 3, (1,)).item(),))
    shape = (len, *other_axes, dim)
    x = torch.rand(shape)
    x = previous_mixing(x)
    x = previous_mixing(x)
    previous_mixing.clear_hidden()
    x = previous_mixing(x)
    x = previous_mixing(x)
    assert x.size() == shape
