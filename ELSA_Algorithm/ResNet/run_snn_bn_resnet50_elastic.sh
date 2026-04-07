#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 --master_port 29502 --use_env experiments_quan_bn_SNN.py \
 --model resnet50 --WeightBit 4 --ActBit 3 --batch_size 1 --elastic --output_per_timestep --dataPath "/data/ImageNet" --quanmodel ../model_pool/resnet50_quan_w4_a3_best.pth --maxTimeStep 24 \
