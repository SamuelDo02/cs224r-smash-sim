"""
Evaluate a trained behavior cloning agent on the Melee environment
"""

import os
import argparse
from agents.iql_agent import IQLAgent
import torch
import numpy as np
from pathlib import Path
from collections import deque

from agents.bc_agent import BCAgent
from infrastructure.melee_env import MeleeEnv
import wandb


def evaluate_model(params):
    """
    Evaluate a trained model on the Melee environment
    """

    name = "eval-" + params['exp_name']

    wandb.init(
        project="melee-bc",
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
    if params['method'] == 'bc':
        agent = BCAgent(env, agent_params)
    elif params['method'] == 'iql':
        agent = IQLAgent(env, agent_params)

    # Load the best model
    model_path = os.path.join(params['logdir'], 'best_policy.pt')
    print(f'Loading model from {model_path}')
    if params['method'] == 'bc':
        agent.actor.load_state_dict(torch.load(model_path))
    elif params['method'] == 'iql':
        state_dict = torch.load(model_path)
        agent.value_net.load_state_dict(state_dict['value_state_dict'])
        agent.q_net.load_state_dict(state_dict['q_state_dict'])
        agent.actor.load_state_dict(state_dict['policy_state_dict'])
        agent.value_net.eval()
        agent.q_net.eval()
    agent.actor.eval()  # Set to evaluation mode

    # Evaluation loop
    print('Starting evaluation...')
    total_reward = 0
    step_count = 0
    max_steps = params['n_steps']
    obs = env.get_frame_history()
    done = False
    past_observations = deque(maxlen=params['frame_window'])
    for i in range(params['frame_window'] - 1):
        past_observations.append(np.zeros_like(obs))
    past_observations.append(obs)

    while step_count < max_steps:
        # Get action from policy
        with torch.no_grad():
            if params['policy_type'] == 'gpt_ar':
                stacked_obs = torch.from_numpy(np.stack(past_observations)).float().unsqueeze(0)
                # print(stacked_obs.shape)
                action = agent.actor.get_action(stacked_obs)
            else:
                action = agent.actor.get_action(obs)
            
            # print(f'Action: {action} for step {step_count + 1}')
            wandb.log({"action": action.tolist()}, step=step_count)
        
        # Take step in environment
        obs, reward, done, info = env.step(action)
        past_observations.append(obs)
        total_reward += reward

        # print(f'Step {step_count + 1} finished with reward: {reward:.2f}')
        # print(f'Info: {info}')
        wandb.log({
            "step_reward": reward,
            "step": step_count
        })

        total_reward += reward
        step_count += 1

        if done:
            print(f'Eval finished with reward: {total_reward:.2f}')
            break
            

    # Print final results
    avg_reward = total_reward / max_steps
    print(f'\nEvaluation complete!')
    print(f'Average reward over {max_steps} episodes: {avg_reward:.2f}')

    # Log final metrics
    wandb.log({
        "avg_reward": avg_reward,
        "total_steps": max_steps
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
    parser.add_argument('--n_steps', type=int, default=10000,
                      help='Number of steps to evaluate')
    parser.add_argument('--method', type=str, default='bc',
                      help='Method to use for training')
    parser.add_argument('--seed', type=int, default=1,
                      help='Random seed')
    args = parser.parse_args()

    # Convert arguments to dictionary
    params = vars(args)
    
    # Run evaluation
    evaluate_model(params)

if __name__ == "__main__":
    main() 

