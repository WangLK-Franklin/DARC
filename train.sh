#!/bin/bash

# 设置要使用的 GPU ID
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 运行训练脚本
python example_train/dsact/dsact_mlp_mujoco_world.py \
    --env_id="gym_ant" \
    --algorithm="DSACT" \
    --trainer_mode="encode_test" \
    --enable_cuda=True \
    --seed=12345 \
    --max_iteration=1500000 \
    --buffer_max_size=1000000 \
    --eval_interval=2500

echo "Training finished!" 