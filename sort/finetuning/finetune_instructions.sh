#!/bin/bash

export PATH=$PATH:/sbin
export GPUS_PER_NODE=8

bash -c 'CUDA_LAUNCH_BLOCKING=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO \
python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes 1 \
finetuning.py --config-name=config_instruction.yaml'