#!/bin/bash

# 设置变量
DATASET="CVUSA"
L1_WEIGHT_GRD="0"
PERCEPTUAL_WEIGHT_GRD="1"
SKIP="0"
HEIGHT_PLANE_NUM="64"
INPUT_TYPE="estimated_height"
MODEL_TYPE="primary"
BATCH_SIZE="12"
NAME="primary_2080_4"
GPU_IDS="3"

# 运行 Python 脚本
python main.py \
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
    --lambda_L1 1 \
    --lr 0.0002
