#!/bin/bash

# Base directory for data
DATA_DIR="data/"

# Base experiment name
BASE_EXP="melee_bc"

# Learning rates to test
LRS=(1e-3)

# Number of layers to test

LAYERS=(15)

# Hidden layer sizes to test
SIZES=(1024)

# Policy types to test
POLICIES=(gpt_ar)

# Run ablation studies
for lr in "${LRS[@]}"; do
    for n_layers in "${LAYERS[@]}"; do
        for size in "${SIZES[@]}"; do
            for policy in "${POLICIES[@]}"; do
                # Calculate actual buffer size (convert from thousands)
                # actual_buf_size=$((buf_size * 1000))
                
                # Create experiment name
                exp_name="${BASE_EXP}_${policy}_lr=${lr}_layers=${n_layers}_size=${size}"
                
                echo "Running experiment: ${exp_name}"
                
                # Run training
                python -m src.scripts.train_melee \
                    --data_dir "${DATA_DIR}" \
                    --exp_name "${exp_name}" \
                    --n_iter 5000 \
                    --batch_size 512 \
                    --learning_rate ${lr} \
                    --n_layers ${n_layers} \
                    --size ${size} \
                    --max_replay_buffer_size 100000000 \
                    --save_freq 1000 \
                    --val_freq 10 \
                    --policy_type ${policy} \
                    --seed 1
                
                echo "Completed experiment: ${exp_name}"
                echo "----------------------------------------"
            done
        done
    done
done
