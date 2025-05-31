"""
Evaluate a trained behavior cloning agent on the Melee environment
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path

from agents.bc_agent import BCAgent
from infrastructure.melee_env import MeleeEnv
import wandb


def evaluate_model(params):
    """
    Evaluate a trained model on the Melee environment
    """

    wandb.init(
        project="melee-bc",
        name="eval-" + params['exp_name'],
        config=params
    )

    # Create the Melee environment
    env = MeleeEnv()

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
        'policy_type': params['policy_type']
    }

    agent = BCAgent(env, agent_params)

    # Load the best model
    model_path = os.path.join(params['logdir'], 'best_policy.pt')
    print(f'Loading model from {model_path}')
    agent.actor.load_state_dict(torch.load(model_path))
    agent.actor.eval()  # Set to evaluation mode

    # Evaluation loop
    print('Starting evaluation...')
    total_reward = 0
    step_count = 0
    max_steps = params['n_steps']
    obs = env.get_frame_history()
    done = False

    while step_count < max_steps:
        # Get action from policy
        with torch.no_grad():
            action = agent.actor.get_action(obs)
            # action = np.random.uniform(-1, 1, env.action_space.shape[0])
            # action = np.clip(action, -1.0, 1.0)
            # action = np.sin(2 * np.pi * step_count / 1000) * np.ones(env.action_space.shape[0])
            # Keep first 4 values, set rest to 0
  
            # action = np.clip(action, -1.0, 1.0)
            # action = np.zeros(env.action_space.shape[0])
            print(f'Action: {action} for step {step_count + 1}')
            # print(f'Action: {action} for episode {episode_count + 1}')
            wandb.log({"action": action.tolist()}, step=step_count)
        
        # Take step in environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(f'Step {step_count + 1} finished with reward: {reward:.2f}')
        print(f'Info: {info}')
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
    args = parser.parse_args()

    # Convert arguments to dictionary
    params = vars(args)
    
    # Run evaluation
    evaluate_model(params)

if __name__ == "__main__":
    main() 