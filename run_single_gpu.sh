#!/bin/bash

# Run training with Hydra configuration
torchrun --standalone --nproc_per_node=1 train.py \
    experiment=pomngpt_no_norm_layers \
    training.batch_size=24 \
    training.accumulation=16