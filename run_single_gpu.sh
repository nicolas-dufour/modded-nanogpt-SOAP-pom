#!/bin/bash

# Run training with Hydra configuration
torchrun --standalone --nproc_per_node=1 train.py \
    experiment=pomngpt_full_no_matrix_norm_no_suv \
    training.batch_size=8 \
    training.accumulation=12 \
    model.gpt.log_stats=1 \
    evaluation.val_loss_every=10