"""
Evaluate a trained agent on the Melee environment
"""

import os
import argparse
from agents.iql_agent import IQLAgent
import torch
import numpy as np
from pathlib import Path
from collections import deque

from agents.bc_agent import BCAgent
from agents.ppo_agent import PPOAgent
from infrastructure.melee_env import MeleeEnv
import wandb


def evaluate_model(params):
    """
    Evaluate a trained model on the Melee environment
    """

    name = "eval-" + params['exp_name']

    wandb.init(
        project=f"melee-{params['method']}",
        name=name,
        config=params
    )

    # Create the Melee environment
    env = MeleeEnv(replay_dir_subfolder=name)

    # Set random seeds
    if 'seed' not in params:
        params['seed'] = np.random.randint(0, 1000000)
    print(f'Using seed: {params["seed"]}')
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    # Setup agent
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

    if params['method'] == 'bc':
        agent = BCAgent(env, agent_params)
    elif params['method'] == 'iql':
        agent = IQLAgent(env, agent_params)
    elif params['method'] == 'ppo':
        agent = PPOAgent(env, agent_params)

    # Load the best model
    model_path = os.path.join(params['logdir'], 'best_policy.pt')
    print(f'Loading model from {model_path}')
    agent.load(model_path)
    agent.actor.eval()  # Set to evaluation mode

    # Evaluation loop
    print('Starting evaluation...')
    total_reward = 0
    wins = 0
    
    for episode in range(params['n_episodes']):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Episode loop
        while episode_steps < params['max_steps']:
            # Get action from policy
            with torch.no_grad():
                if params['policy_type'] == 'gpt_ar':
                    stacked_obs = torch.from_numpy(np.stack(past_observations)).float().unsqueeze(0)
                    action = agent.actor.get_action(stacked_obs)
                else:
                    action = agent.actor.get_action(obs)
                
                wandb.log({"action": action.tolist()}, step=episode_steps)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            # Log step metrics
            wandb.log({
                "step_reward": reward,
                "step": episode_steps
            })
            
            # End episode if done
            if done:
                if reward > 0:  # Positive reward indicates a win
                    wins += 1
                break
        
        total_reward += episode_reward
        logger.info(f"Episode {episode} - Reward: {episode_reward:.2f}")
    
    # Print final statistics
    avg_reward = total_reward / params['n_episodes']
    win_rate = wins / params['n_episodes']
    print(f'\nEvaluation complete!')
    print(f'Average reward over {params["n_episodes"]} episodes: {avg_reward:.2f}')
    print(f'Win rate: {win_rate:.2%}')

    # Log final metrics
    wandb.log({
        "avg_reward": avg_reward,
        "win_rate": win_rate,
        "total_episodes": params['n_episodes']
    })
    wandb.finish()

    # Clean up
    env.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True,
                      help='Directory containing the trained model')
    parser.add_argument('--n_episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    parser.add_argument('--n_layers', type=int, default=15,
                      help='Number of layers in policy network')
    parser.add_argument('--size', type=int, default=1024,
                      help='Size of hidden layers in policy network')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate for the policy')
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000,
                      help='Maximum size of replay buffer')
    parser.add_argument('--frame_window', type=int, default=5,
                      help='How many frames to use per observation')
    parser.add_argument('--policy_type', type=str, default='mlp',
                      help='Type of policy network to use')
    parser.add_argument('--exp_name', type=str, required=True,
                      help='Name of the experiment')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Maximum steps per episode')
    parser.add_argument('--method', type=str, default='bc',
                      choices=['bc', 'iql', 'ppo'],
                      help='Method to use for training')
    parser.add_argument('--seed', type=int, default=1,
                      help='Random seed')
    
    # PPO-specific arguments
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for PPO')
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
    
    args = parser.parse_args()

    # Convert arguments to dictionary
    params = vars(args)
    
    # Run evaluation
    evaluate_model(params)

if __name__ == "__main__":
    main() 

