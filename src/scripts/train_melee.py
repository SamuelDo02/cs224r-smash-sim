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
from agents.iql_agent import IQLAgent
from agents.ppo_agent import PPOAgent
from infrastructure.melee_env import MeleeEnv
from data.pkl_replay_buffer import PKLReplayBuffer

def train_agent(params):
    """
    Runs training with the specified parameters
    """
    #######################
    ## INIT
    #######################

    name = "train-" + params['exp_name']

    # Initialize wandb
    wandb.init(
        project=f"melee-{params['method']}",
        name=name,
        config=params
    )

    # Create the Melee environment
    env = MeleeEnv(replay_dir_subfolder=name)

    if 'seed' not in params:
        params['seed'] = np.random.randint(0, 1000000)
    print(f'Using seed: {params["seed"]}')
    # Set random seeds
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    #######################
    ## LOAD EXPERT DATA
    #######################
    
    if params['method'] in ['bc', 'iql']:
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
        'policy_type': params['policy_type'],
        'method': params['method']
    }

    # Add PPO-specific parameters
    if params['method'] == 'ppo':
        agent_params.update({
            'batch_size': params['batch_size'],
            'gamma': params['gamma'],
            'gae_lambda': params['gae_lambda'],
            'ppo_epochs': params['ppo_epochs'],
            'clip_ratio': params['clip_ratio'],
            'value_coef': params['value_coef'],
            'entropy_coef': params['entropy_coef']
        })

    if params['method'] == 'iql':
        print('Initializing IQL agent...')
        agent = IQLAgent(env, agent_params)
    elif params['method'] == 'ppo':
        print('Initializing PPO agent...')
        agent = PPOAgent(env, agent_params)
    else:
        print('Initializing BC agent...')
        agent = BCAgent(env, agent_params)

    if params['loaddir'] is not None:
        model_path = os.path.join(params['loaddir'], 'best_policy.pt')
        print(f'Loading model from {model_path}')
        if params['policy_type'] == 'mlp':
            agent.actor.load_state_dict(torch.load(model_path))
        elif params['policy_type'] == 'gpt_ar':
            state_dict = torch.load(model_path)
            agent.value_net.load_state_dict(state_dict['value_state_dict'])
            agent.q_net.load_state_dict(state_dict['q_state_dict'])
            agent.actor.load_state_dict(state_dict['policy_state_dict'])
    
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
        if params['method'] == 'ppo':
            # PPO training loop
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            while episode_steps < params['max_steps']:
                # Select action
                action, log_prob, value = agent.select_action(obs)
                
                # Take step in environment
                next_obs, reward, done, info = env.step(action)
                
                # Store transition
                agent.memory.store_memory(
                    obs, action, log_prob, value, reward, done
                )
                
                # Update observation and counters
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                
                # End episode if done
                if done:
                    break
            
            # Train agent if we have enough samples
            if len(agent.memory.states) >= params['batch_size']:
                train_log = agent.train(
                    agent.memory.states,
                    agent.memory.actions,
                    agent.memory.rewards,
                    agent.memory.vals,
                    agent.memory.dones
                )
                steps_so_far += params['batch_size']
        else:
            # BC/IQL training loop
            observations, actions = replay_buffer.sample(params['batch_size'], 
                                                         frame_window=params['frame_window'])
            train_log = agent.train(observations, actions)
            steps_so_far += params['batch_size']
        
        # Update progress bar
        pbar.set_postfix({
            'Steps': f'{steps_so_far}/{total_steps}',
            'Loss': f'{train_log["Training Loss"]:.4f}'
        })
        
        # Log to wandb
        wandb.log({
            "train/loss": train_log["Training Loss"],
            # "train/success_rate": train_log["Success Rate"],
            "train/steps": steps_so_far
        })
        
        # Log all accuracy metrics
        for metric_name, metric_value in train_log.items():
            if metric_name not in ["Training Loss", "Success Rate"]:
                wandb.log({
                    f"train/{metric_name}": metric_value
                })
        
        # Validate if we have validation data (only for BC/IQL)
        if params['method'] in ['bc', 'iql'] and val_buffer is not None and (itr + 1) % params['val_freq'] == 0:
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
                # "val/success_rate": val_log["Success Rate"],
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

        # eval the best model on the Env
        
    print('Stopping environment...')
    env.stop()
    print('Training complete!')

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
    parser.add_argument('--method', type=str, default='bc',
                      choices=['bc', 'iql', 'ppo'],
                      help='Method to use for training')
    parser.add_argument('--loaddir', type=str, default=None,
                      help='Directory to load model from')
    
    # PPO-specific arguments
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor for PPO')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                      help='GAE lambda for PPO')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                      help='Number of PPO epochs')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                      help='PPO clip ratio')
    parser.add_argument('--value_coef', type=float, default=0.5,
                      help='Value loss coefficient for PPO')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                      help='Entropy coefficient for PPO')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Maximum steps per episode for PPO')
    
    args = parser.parse_args()

    # Create experiment directory
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    logdir = f"{args.exp_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    logdir = os.path.join(data_path, logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    # Convert arguments to dictionary
    params = vars(args)
    params['logdir'] = logdir
    
    # Run training
    train_agent(params)

if __name__ == "__main__":
    main() 