#!/bin/bash

# 设置变量
DATASET="CVUSA"
L1_WEIGHT_GRD="0"
PERCEPTUAL_WEIGHT_GRD="1"
SKIP="0"
HEIGHT_PLANE_NUM="64"
INPUT_TYPE="pol"
MODEL_TYPE="pix2pix"
BATCH_SIZE="8"
NAME="pix2pix_2080"
GPU_IDS="1"

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
    --gpu_ids $GPU_IDS
