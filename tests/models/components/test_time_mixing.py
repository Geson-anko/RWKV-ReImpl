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
    assert time_mixing.numerator_last is None
    assert time_mixing.denominator_last is None
    x = time_mixing(x)
    assert time_mixing.numerator_last is not None
    assert time_mixing.denominator_last is not None
    assert "numerator_last" in time_mixing.state_dict()
    assert "denominator_last" in time_mixing.state_dict()

    x = time_mixing(x)
    time_mixing.clear_hidden()
    assert time_mixing.numerator_last is None
    assert time_mixing.denominator_last is None
    x = time_mixing(x)
    assert time_mixing.numerator_last is not None
    assert time_mixing.denominator_last is not None
    x = time_mixing(x)
    assert x.size() == torch.Size([len, batch, dim])


def test_init_weights():
    time_mixing = TimeMixing(512)
    time_mixing.init_weights()
    torch.testing.assert_close(time_mixing.Wk.weight, torch.zeros_like(time_mixing.Wk.weight))
    torch.testing.assert_close(time_mixing.Wr.weight, torch.zeros_like(time_mixing.Wr.weight))
    torch.testing.assert_close(
        time_mixing.W_out.weight, torch.zeros_like(time_mixing.W_out.weight)
    )
    torch.testing.assert_close(
        time_mixing.Wv.weight @ time_mixing.Wv.weight.T,
        torch.eye(512),
    )
