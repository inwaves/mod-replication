#!/bin/bash

python train.py \
  --batch_size 16 \
  --capacity_fraction 0.125 \
  --epochs 10 \
  --model gpt2 \
  --log_every_n_steps 100 \
  # --use_mod

