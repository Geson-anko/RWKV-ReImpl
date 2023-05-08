#!/bin/bash

screen python src/train.py \
    experiment=rwkv_wiki \
    trainer=gpu \
    data.num_workers=1 \
    model.monitoring_interval=1000 \
    callbacks.model_checkpoint.every_n_train_steps=1000 \
    data.batch_size=16 \
    trainer.max_steps=1_000_000 \
