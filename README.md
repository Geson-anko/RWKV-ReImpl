# Your Project Name

The Reimplementation of RWKV.

## Description

RWKVを再現実装し、その性能を確かめます。
<https://github.com/BlinkDL/RWKV-LM>

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/Geson-anko/RWKV-ReImpl.git
cd RWKV-ReImpl

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
