#!/bin/bash

# Run training with Hydra configuration
torchrun --standalone --nproc_per_node=1 train.py \
    experiment=pomngpt_full \
    training.batch_size=8 \
    training.accumulation=12 \
    model.gpt.log_stats=1 \
    evaluation.val_loss_every=10