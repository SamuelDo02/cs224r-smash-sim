#!/bin/bash

# Default parameters
LOGDIR="../../data/ppo_training"
EXP_NAME="ppo_evaluation"
POLICY_TYPE="mlp"
N_LAYERS=2
SIZE=64
LEARNING_RATE=3e-4
BATCH_SIZE=64
N_EPISODES=10
MAX_STEPS=1000
FRAME_WINDOW=5
MAX_REPLAY_BUFFER_SIZE=1000000

# PPO-specific parameters
GAMMA=0.99
GAE_LAMBDA=0.95
PPO_EPOCHS=10
CLIP_RATIO=0.2
VALUE_COEF=0.5
ENTROPY_COEF=0.01

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --policy_type)
            POLICY_TYPE="$2"
            shift 2
            ;;
        --n_layers)
            N_LAYERS="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --n_episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --frame_window)
            FRAME_WINDOW="$2"
            shift 2
            ;;
        --max_replay_buffer_size)
            MAX_REPLAY_BUFFER_SIZE="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --gae_lambda)
            GAE_LAMBDA="$2"
            shift 2
            ;;
        --ppo_epochs)
            PPO_EPOCHS="$2"
            shift 2
            ;;
        --clip_ratio)
            CLIP_RATIO="$2"
            shift 2
            ;;
        --value_coef)
            VALUE_COEF="$2"
            shift 2
            ;;
        --entropy_coef)
            ENTROPY_COEF="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Run evaluation script
python eval_melee.py \
    --logdir "$LOGDIR" \
    --exp_name "$EXP_NAME" \
    --method ppo \
    --policy_type "$POLICY_TYPE" \
    --n_layers "$N_LAYERS" \
    --size "$SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --n_episodes "$N_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --frame_window "$FRAME_WINDOW" \
    --max_replay_buffer_size "$MAX_REPLAY_BUFFER_SIZE" \
    --gamma "$GAMMA" \
    --gae_lambda "$GAE_LAMBDA" \
    --ppo_epochs "$PPO_EPOCHS" \
    --clip_ratio "$CLIP_RATIO" \
    --value_coef "$VALUE_COEF" \
    --entropy_coef "$ENTROPY_COEF" 