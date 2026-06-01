#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
export PRECISION=no
export PROJECT=mis_rb_small_cam
export EXP=cam

accelerate launch --mixed_precision=$PRECISION \
    --num_processes=$PROCESSES main.py \
    --train_path $DATADIR/MIS/RB_small_train \
    --valid_path $DATADIR/MIS/RB_small_valid \
    --test_path $DATADIR/MIS/RB_small_test \
    --problem mis \
    --samples_per_epoch 4000 \
    --valid_sample 64 \
    --batch_size 4 \
    --mini_batch_size 256 \
    --num_t 50 \
    --num_k 20 \
    --num_tp 50 \
    --num_kp 10 \
    --num_h 64 \
    --num_l 8 \
    --method CAM \
    --model EncodeProcessDecode \
    --epochs 1000 \
    --N_anneal 1000 \
    --patience 200 \
    --lr 0.0005 \
    --save_dir $MODELDIR/$PROJECT/$EXP \
    --wandb_name $PROJECT \
    --wandb_run_name $EXP \
    --do_train
