#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
export PRECISION=no
export PROJECT=mis_rb_large_cam
export EXP=cam

accelerate launch --mixed_precision=$PRECISION \
    --num_processes=$PROCESSES main.py \
    --train_path $DATADIR/MIS/RB_large_train \
    --valid_path $DATADIR/MIS/RB_large_valid \
    --test_path $DATADIR/MIS/RB_large_test \
    --problem mis \
    --samples_per_epoch 1000 \
    --valid_sample 64 \
    --batch_size 1 \
    --mini_batch_size 128 \
    --num_t 50 \
    --num_k 20 \
    --num_tp 50 \
    --num_kp 20 \
    --num_h 64 \
    --num_l 6 \
    --method CAM \
    --model EncodeProcessDecode \
    --tau0 0.1 \
    --epochs 1500 \
    --N_anneal 1500 \
    --patience 400 \
    --lr 0.0005 \
    --save_dir $MODELDIR/$PROJECT/$EXP \
    --wandb_name $PROJECT \
    --wandb_run_name $EXP \
    --do_train
