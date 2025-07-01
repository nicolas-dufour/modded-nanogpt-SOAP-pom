#!/bin/bash

# Run training with Hydra configuration
torchrun --standalone --nproc_per_node=1 train.py \
    experiment=pomngpt_justnorm_matrix \
    training.batch_size=24 \
    training.accumulation=16