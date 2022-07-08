#!/bin/bash

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/DIV2K/DIV2K_train_HR" \
#                   --output-path "datasets_PIL/DIV2K_bicubic_x2.h5" \
#                   --interp-method "PIL" \
#                   --scale 2

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/benchmark/Set5/HR" \
#                   --output-path "datasets_PIL/Set5_bicubic_x2.h5" \
#                   --interp-method "PIL" \
#                   --scale 2 \
#                   --eval              

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/DIV2K/DIV2K_train_HR" \
#                   --output-path "datasets_PIL/DIV2K_bicubic_x3.h5" \
#                   --interp-method "PIL" \
#                   --scale 3

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/benchmark/Set5/HR" \
#                   --output-path "datasets_PIL/Set5_bicubic_x3.h5" \
#                   --interp-method "PIL" \
#                   --scale 3 \
#                   --eval                  

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/DIV2K/DIV2K_train_HR" \
#                   --output-path "datasets_PIL/DIV2K_bicubic_x4.h5" \
#                   --interp-method "PIL" \
#                   --scale 4  

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/benchmark/Set5/HR" \
#                   --output-path "datasets_PIL/Set5_bicubic_x4.h5" \
#                   --interp-method "PIL" \
#                   --scale 4 \
#                   --eval                                                      
                                    

#python train.py --train-file "datasets_PIL/DIV2K_bicubic_x2.h5" \
#                --eval-file "datasets_PIL/Set5_bicubic_x2.h5" \
#                --outputs-dir "learned_Models/PIL" \
#                --scale 2 \
#                --num-features 64 \
#                --growth-rate 64 \
#                --num-blocks 16 \
#                --num-layers 8 \
#                --lr 1e-4 \
#                --batch-size 16 \
#                --patch-size 32 \
#                --num-epochs 800 \
#                --num-workers 4 \
#                --seed 123                
#
#python train.py --train-file "datasets_PIL/DIV2K_bicubic_x3.h5" \
#                --eval-file "datasets_PIL/Set5_bicubic_x3.h5" \
#                --outputs-dir "learned_Models/PIL" \
#                --scale 3 \
#                --num-features 64 \
#                --growth-rate 64 \
#                --num-blocks 16 \
#                --num-layers 8 \
#                --lr 1e-4 \
#                --batch-size 16 \
#                --patch-size 32 \
#                --num-epochs 800 \
#                --num-workers 4 \
#                --seed 123                 
#
#python train.py --train-file "datasets_PIL/DIV2K_bicubic_x4.h5" \
#                --eval-file "datasets_PIL/Set5_bicubic_x4.h5" \
#                --outputs-dir "learned_Models/PIL" \
#                --scale 4 \
#                --num-features 64 \
#                --growth-rate 64 \
#                --num-blocks 16 \
#                --num-layers 8 \
#                --lr 1e-4 \
#                --batch-size 16 \
#                --patch-size 32 \
#                --num-epochs 800 \
#                --num-workers 4 \
#                --seed 123                 
#




# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/DIV2K/DIV2K_train_HR" \
#                   --output-path "datasets_TORCH/DIV2K_bicubic_x2.h5" \
#                   --interp-method "TORCH" \
#                   --scale 2

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/benchmark/Set5/HR" \
#                   --output-path "datasets_TORCH/Set5_bicubic_x2.h5" \
#                   --interp-method "TORCH" \
#                   --scale 2 \
#                   --eval              

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/DIV2K/DIV2K_train_HR" \
#                   --output-path "datasets_TORCH/DIV2K_bicubic_x3.h5" \
#                   --interp-method "TORCH" \
#                   --scale 3

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/benchmark/Set5/HR" \
#                   --output-path "datasets_TORCH/Set5_bicubic_x3.h5" \
#                   --interp-method "TORCH" \
#                   --scale 3 \
#                   --eval                  

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/DIV2K/DIV2K_train_HR" \
#                   --output-path "datasets_TORCH/DIV2K_bicubic_x4.h5" \
#                   --interp-method "TORCH" \
#                   --scale 4  

# python prepare.py --images-dir "/home/rm181/Workspace/GitHub/DIV2K/benchmark/Set5/HR" \
#                   --output-path "datasets_TORCH/Set5_bicubic_x4.h5" \
#                   --interp-method "TORCH" \
#                   --scale 4 \
#                   --eval  


python train.py --train-file "datasets_TORCH/DIV2K_bicubic_x2.h5" \
                --eval-file "datasets_TORCH/Set5_bicubic_x2.h5" \
                --outputs-dir "learned_Models/TORCH" \
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

python train.py --train-file "datasets_TORCH/DIV2K_bicubic_x3.h5" \
                --eval-file "datasets_TORCH/Set5_bicubic_x3.h5" \
                --outputs-dir "learned_Models/TORCH" \
                --scale 3 \
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

python train.py --train-file "datasets_TORCH/DIV2K_bicubic_x4.h5" \
                --eval-file "datasets_TORCH/Set5_bicubic_x4.h5" \
                --outputs-dir "learned_Models/TORCH" \
                --scale 4 \
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

