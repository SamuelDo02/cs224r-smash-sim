#!/bin/bash

# Base directory for data
DATA_DIR="data/"

# Base experiment name
BASE_EXP="melee_bc"

# Learning rates to test
LRS=(1e-3)

# Number of layers to test
LAYERS=(2 4 8)

# Hidden layer sizes to test
SIZES=(32)

# Replay buffer sizes to test (in thousands)
BUFFER_SIZES=(1000000)

# Run ablation studies
for lr in "${LRS[@]}"; do
    for n_layers in "${LAYERS[@]}"; do
        for size in "${SIZES[@]}"; do
            for buf_size in "${BUFFER_SIZES[@]}"; do
                # Calculate actual buffer size (convert from thousands)
                actual_buf_size=$((buf_size * 1000))
                
                # Create experiment name
                exp_name="${BASE_EXP}_lr=${lr}_layers=${n_layers}_size=${size}"
                
                echo "Running experiment: ${exp_name}"
                
                # Run training
                python -m src.scripts.train_melee \
                    --data_dir "${DATA_DIR}" \
                    --exp_name "${exp_name}" \
                    --n_iter 5000 \
                    --batch_size 64 \
                    --learning_rate ${lr} \
                    --n_layers ${n_layers} \
                    --size ${size} \
                    --max_replay_buffer_size ${actual_buf_size} \
                    --save_freq 10 \
                    --val_freq 5 \
                    --seed 1
                
                echo "Completed experiment: ${exp_name}"
                echo "----------------------------------------"
            done
        done
    done
done
