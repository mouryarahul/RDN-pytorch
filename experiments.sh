#!/bin/bash

python train.py --train-file "datasets_PIL/DIV2K_bicubic_x2.h5" \
                --eval-file "datasets_PIL/Set5_bicubic_x2.h5" \
                --outputs-dir "learned_Models/PIL" \
                --scale 2 \
                --num-features 64 \
                --growth-rate 64 \
                --num-blocks 16 \
                --num-layers 8 \
                --lr 1e-4 \
                --batch-size 16 \
                --patch-size 32 \
                --num-epochs 800 \
                --num-workers 4 \
                --seed 123                