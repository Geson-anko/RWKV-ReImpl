import pytest

from src.models.components.warmup_exp_decay_lr_lambda import WarmupExpDecayLRLambda


@pytest.fixture
def warmup_exp_decay_lr_lambda():
    return WarmupExpDecayLRLambda(
        base_lr=0.1,
        init_lr=0.01,
        max_lr=0.2,
        final_lr=0.001,
        warmup_steps=10,
        max_steps=100,
    )


def test__init__(warmup_exp_decay_lr_lambda: WarmupExpDecayLRLambda):
    assert warmup_exp_decay_lr_lambda.base_lr == 0.1
    assert warmup_exp_decay_lr_lambda.init_lr == 0.01
    assert warmup_exp_decay_lr_lambda.max_lr == 0.2
    assert warmup_exp_decay_lr_lambda.final_lr == 0.001
    assert warmup_exp_decay_lr_lambda.warmup_steps == 10
    assert warmup_exp_decay_lr_lambda.gamma == pytest.approx(0.005 ** (1 / 90), rel=1e-6)


def test_warmup(warmup_exp_decay_lr_lambda: WarmupExpDecayLRLambda):
    assert warmup_exp_decay_lr_lambda(0) == 0.01 / 0.1
    assert warmup_exp_decay_lr_lambda(5) == pytest.approx(
        (0.01 + 0.5 * (0.2 - 0.01)) / 0.1, rel=1e-6
    )
    assert warmup_exp_decay_lr_lambda(10) == pytest.approx(0.2 / 0.1, rel=1e-6)


def test_decay(warmup_exp_decay_lr_lambda: WarmupExpDecayLRLambda):
    assert warmup_exp_decay_lr_lambda(11) == pytest.approx(
        (0.2 * warmup_exp_decay_lr_lambda.gamma) / 0.1, rel=1e-6
    )
    assert warmup_exp_decay_lr_lambda(20) == pytest.approx(
        (0.2 * warmup_exp_decay_lr_lambda.gamma**10) / 0.1, rel=1e-6
    )


def test_final_lr(warmup_exp_decay_lr_lambda: WarmupExpDecayLRLambda):
    warmup_exp_decay_lr_lambda.final_lr = 0.001
    assert warmup_exp_decay_lr_lambda(5000) == 0.001 / 0.1
