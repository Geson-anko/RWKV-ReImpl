import pytest
import torch

from src.models.components.time_mixing import TimeMixing


@pytest.mark.parametrize(
    """
    len,
    batch,
    dim,
    """,
    [
        (1024, 1, 512),
        (1024, 4, 1024),
        (512, 16, 256),
    ],
)
def test_time_mixing(len, batch, dim):
    time_mixing = TimeMixing(dim)
    x = torch.rand(len, batch, dim)
    x = time_mixing(x)
    x = time_mixing(x)
    time_mixing.clear_hidden()
    assert time_mixing.numerator_last is None
    assert time_mixing.denominator_last is None
    x = time_mixing(x)
    x = time_mixing(x)
    assert x.size() == torch.Size([len, batch, dim])
