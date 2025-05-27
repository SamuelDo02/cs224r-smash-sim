"""
Train a behavior cloning agent for Super Smash Bros. Melee using pre-processed pickle files
"""

import os
import time
import argparse
from pathlib import Path

import torch
import numpy as np
import wandb
from tqdm import tqdm

from agents.bc_agent import BCAgent
from infrastructure.melee_env import MeleeEnv
from data.pkl_replay_buffer import PKLReplayBuffer

def train_bc(params):
    """
    Runs behavior cloning with the specified parameters
    """
    #######################
    ## INIT
    #######################

    # Initialize wandb
    wandb.init(
        project="melee-bc",
        name=params['exp_name'],
        config=params
    )

    # Create the Melee environment
    env = MeleeEnv()

    if 'seed' not in params:
        params['seed'] = np.random.randint(0, 1000000)
    print(f'Using seed: {params["seed"]}')
    # Set random seeds
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    #######################
    ## LOAD EXPERT DATA
    #######################
    
    print('Loading expert data from...', params['data_dir'])
    replay_buffer = PKLReplayBuffer(max_size=params['max_replay_buffer_size'])
    
    # Load training data
    train_dir = os.path.join(params['data_dir'], 'train')
    print("\nLoading training data...")
    replay_buffer.add_directory(train_dir)
    print(f"Loaded {replay_buffer.current_size} frames")
    
    # Optionally load validation data
    val_dir = os.path.join(params['data_dir'], 'val')
    if os.path.exists(val_dir):
        print("\nLoading validation data...")
        val_buffer = PKLReplayBuffer(max_size=params['max_replay_buffer_size'])
        val_buffer.add_directory(val_dir)
        print(f"Loaded {val_buffer.current_size} frames")
    else:
        val_buffer = None

    #######################
    ## SETUP AGENT
    #######################

    agent_params = {
        'ac_dim': env.action_space.shape[0],
        'ob_dim': env.observation_space.shape[0] * params['frame_window'],
        'n_layers': params['n_layers'],
        'size': params['size'],
        'learning_rate': params['learning_rate'],
        'max_replay_buffer_size': params['max_replay_buffer_size'],
        'policy_type': params['policy_type']
    }

    agent = BCAgent(env, agent_params)
    
    #######################
    ## TRAIN AGENT
    #######################
    
    print('\nStarting training...')
    total_steps = params['n_iter'] * params['batch_size']
    steps_so_far = 0
    best_val_loss = float('inf')
    
    # Calculate and log total number of parameters
    total_params = sum(p.numel() for p in agent.actor.parameters())
    print(f'\nTotal parameters: {total_params:,}')
    wandb.run.summary['total_parameters'] = total_params

    pbar = tqdm(range(params['n_iter']), desc='Training')
    for itr in pbar:
        # Sample training data
        observations, actions = replay_buffer.sample(params['batch_size'], 
                                                     frame_window=params['frame_window'])
        # Train the agent
        train_log = agent.train(observations, actions)
        # print(train_log)
        # Print log probs every 500 iterations
        # if (itr + 1) % 500 == 0:
        #     print(agent.actor.get_action(observations[0]), actions[0], observations[0])
        steps_so_far += params['batch_size']
        
        # Update progress bar
        pbar.set_postfix({
            'Steps': f'{steps_so_far}/{total_steps}',
            'Loss': f'{train_log["Training Loss"]:.4f}'
        })
        
        # Log to wandb
        wandb.log({
            "train/loss": train_log["Training Loss"],
            "train/success_rate": train_log["Success Rate"],
            "train/steps": steps_so_far
        })
        
        # Log all accuracy metrics
        for metric_name, metric_value in train_log.items():
            if metric_name not in ["Training Loss", "Success Rate"]:
                wandb.log({
                    f"train/{metric_name}": metric_value
                })
        
        # Validate if we have validation data
        if val_buffer is not None and (itr + 1) % params['val_freq'] == 0:
            val_observations, val_actions = val_buffer.sample(params['batch_size'],
                                                              frame_window=params['frame_window'])
            val_log = agent.train(val_observations, val_actions, train=False)
            val_loss = val_log["Training Loss"]
            # print(f'Validation Loss: {val_loss:.4f}')
            # print(f'Validation Loss: {val_loss:.4f}')
            # print(f'Validation Success Rate: {val_log["Success Rate"]:.4f}')
            
            # Log validation metrics
            wandb.log({
                "val/loss": val_loss,
                "val/success_rate": val_log["Success Rate"],
                "val/best_loss": best_val_loss
            })
            
            # Log all validation accuracy metrics
            for metric_name, metric_value in val_log.items():
                if metric_name not in ["Training Loss", "Success Rate"]:
                    wandb.log({
                        f"val/{metric_name}": metric_value
                    })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(params['logdir'], 'best_policy.pt')
                agent.save(save_path)
                # print(f'New best model saved to {save_path}')
                
                # Log best model to wandb
                wandb.save(save_path)
        
        # Save periodic checkpoint
        if (itr + 1) % params['save_freq'] == 0:
            save_path = os.path.join(params['logdir'], f'policy_itr_{itr+1}.pt')
            agent.save(save_path)
            # print(f'Checkpoint saved to {save_path}')
            wandb.save(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing train/val/test subdirectories with .pkl files')
    parser.add_argument('--exp_name', type=str, required=True,
                      help='Name of the experiment')
    parser.add_argument('--n_iter', type=int, default=1000,
                      help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate for the policy')
    parser.add_argument('--n_layers', type=int, default=2,
                      help='Number of layers in policy network')
    parser.add_argument('--size', type=int, default=64,
                      help='Size of hidden layers in policy network')
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000,
                      help='Maximum size of replay buffer')
    parser.add_argument('--save_freq', type=int, default=10,
                      help='How often to save policy checkpoints')
    parser.add_argument('--val_freq', type=int, default=5,
                      help='How often to run validation')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 1000000),
                      help='Random seed')
    parser.add_argument('--frame_window', type=int, default=5,
                      help='How many frames to use per observation')
    parser.add_argument('--policy_type', type=str, default='mlp',
                      help='Type of policy network to use')
    args = parser.parse_args()

    # Create experiment directory
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    logdir = f"melee_bc_{args.exp_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    logdir = os.path.join(data_path, logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    # Convert arguments to dictionary
    params = vars(args)
    params['logdir'] = logdir
    
    # Run training
    train_bc(params)

if __name__ == "__main__":
    main() 