#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
export PRECISION=no
export PROJECT=mis_er_cam
export EXP=cam

accelerate launch --mixed_precision=$PRECISION \
    --num_processes=$PROCESSES main.py \
    --train_path $DATADIR/MIS/ER_train \
    --valid_path $DATADIR/MIS/ER_valid \
    --test_path $DATADIR/MIS/ER_test \
    --problem mis \
    --samples_per_epoch 1000 \
    --valid_sample 64 \
    --batch_size 1 \
    --mini_batch_size 512 \
    --num_t 50 \
    --num_k 20 \
    --num_tp 50 \
    --num_kp 20 \
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