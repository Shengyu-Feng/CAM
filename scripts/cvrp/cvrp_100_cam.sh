#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
export PRECISION=no # safe to use fp16
export PROJECT=cvrp_100_cam
export EXP=cam

accelerate launch --mixed_precision=$PRECISION \
    --num_processes=$PROCESSES main.py \
    --train_path $DATADIR/CVRP/cvrp100_uniform_c50_128k_1.txt \
    --valid_path $DATADIR/CVRP/cvrp100_uniform_c50_128k_2.txt \
    --test_path $DATADIR/CVRP/cvrp100_hgs-20s_15.563.txt \
    --problem cvrp \
    --samples_per_epoch 1024 \
    --valid_sample 16 \
    --batch_size 1 \
    --mini_batch_size 8 \
    --num_t 10 \
    --num_k 4 \
    --num_tp 10 \
    --num_kp 16 \
    --num_h 128 \
    --num_l 6 \
    --method CAM \
    --model AnGNN \
    --epochs 100 \
    --N_anneal 100 \
    --patience 20 \
    --lr 0.0002 \
    --save_dir $MODELDIR/$PROJECT/$EXP \
    --wandb_name $PROJECT \
    --wandb_run_name $EXP \
    --do_train \