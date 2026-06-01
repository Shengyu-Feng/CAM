#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
export PRECISION=no
export PROJECT=mcut_ba_large_cam
export EXP=cam

accelerate launch --mixed_precision=$PRECISION \
    --num_processes=$PROCESSES main.py \
    --train_path $DATADIR/MCut/BA_large_train \
    --valid_path $DATADIR/MCut/BA_large_valid \
    --test_path $DATADIR/MCut/BA_large_test \
    --problem mcut \
    --samples_per_epoch 1000 \
    --valid_sample 64 \
    --batch_size 1 \
    --mini_batch_size 256 \
    --num_t 50 \
    --num_k 20 \
    --num_tp 50 \
    --num_kp 20 \
    --num_h 64 \
    --num_l 6 \
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
