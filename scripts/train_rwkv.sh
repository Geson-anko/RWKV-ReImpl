#!/bin/bash
# Run from root folder with: bash scripts/train_rwkv.sh

python src/train.py experiment=rwkv_wiki trainer=gpu
