#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m torch.distributed.launch --nproc_per_node=7 --master_port 29502 --use_env experiments_quan_bn_SNN.py \
 --model resnet18 --WeightBit 4 --ActBit 4 --batch_size 64  --output_per_timestep --dataPath "/data/ImageNet" --quanmodel /home/user/model_pool/resnet18_quan_w4_a4_best.pth --maxTimeStep 16 \
