from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig):
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig):
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_dummy_text_data_dir(dummy_text_data_dir: Path):
    assert dummy_text_data_dir.exists()
    assert dummy_text_data_dir.is_dir()
    text_files = list(dummy_text_data_dir.glob("*.txt"))
    assert len(text_files) == 10
    assert all([f.is_file() for f in text_files])

    for i in range(len(text_files)):
        fp = dummy_text_data_dir / f"dummy_text_{i}.txt"
        with open(fp) as f:
            assert f.read() == f"dummy text\n<tag {i}>"
