#!/bin/bash

# 设置变量
DATASET="CVUSA"
L1_WEIGHT_GRD="0"
PERCEPTUAL_WEIGHT_GRD="1"
SKIP="0"
HEIGHT_PLANE_NUM="1"
INPUT_TYPE="pol"
MODEL_TYPE="primary"
BATCH_SIZE="14"
NAME="primary_2080_1"
GPU_IDS="0"

# 运行 Python 脚本
kernprof -l -v main.py \
    --dataset $DATASET \
    --l1_weight_grd $L1_WEIGHT_GRD \
    --perceptual_weight_grd $PERCEPTUAL_WEIGHT_GRD \
    --skip $SKIP \
    --heightPlaneNum $HEIGHT_PLANE_NUM \
    --input_type $INPUT_TYPE \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE \
    --name $NAME \
    --gpu_ids $GPU_IDS \
    --lambda_L1 0.1 \
    > output_v1.log