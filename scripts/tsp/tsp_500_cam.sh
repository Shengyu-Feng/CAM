#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
export PRECISION=no # safe to use fp16
export PROJECT=tsp_500_cam
export EXP=cam

accelerate launch --mixed_precision=$PRECISION \
    --num_processes=$PROCESSES main.py \
    --train_path $DATADIR/TSP/tsp500_train_concorde.txt \
    --valid_path $DATADIR/TSP/tsp500_valid_concorde.txt \
    --test_path $DATADIR/TSP/tsp500_test_concorde.txt \
    --problem tsp \
    --samples_per_epoch 1024 \
    --valid_sample 16 \
    --batch_size 4 \
    --sparse_factor 50 \
    --num_t 10 \
    --num_k 16 \
    --num_tp 10 \
    --num_kp 8 \
    --num_h 256 \
    --num_l 12 \
    --method CAM \
    --model AnGNN \
    --epochs 50 \
    --N_anneal 50 \
    --patience 20 \
    --lr 0.0002 \
    --save_dir $MODELDIR/$PROJECT/$EXP \
    --wandb_name $PROJECT \
    --wandb_run_name $EXP \
    --do_train 