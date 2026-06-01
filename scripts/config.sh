#!/bin/bash
# Machine-specific configuration.
# Edit DATADIR to point to the data root on this machine.

# For trouble shooting, you can set the following environment variable to disable NCCL P2P communication.
# export NCCL_P2P_DISABLE=1

# For wandb logging, you can set the following environment variable to your wandb API key.
# export WANDB_API_KEY=your_wandb_api_key

export DATADIR=./CAM_data
export MODELDIR=./CAM_models

# set the number of processes to use for distributed training. This should be set to the number of GPUs available on the machine.
export PROCESSES=8
