import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from policies.MLP_policy import MLPPolicySL
from policies.Transformer_policy import TransformerPolicySL
from policies.GPT_policy import GPTPolicy
from policies.GPT_AR_policy import GPTARPolicy
from policies.preprocessor import Preprocessor

class ReplayBuffer:
    """
    Experience replay buffer for DQN
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network agent for Smash with experience replay and target network
    """
    def __init__(self, env, params):
        # Initialize variables
        self.env = env
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create policy class as our Q-network
        if self.params['policy_type'] == 'transformer':
            self.q_network = TransformerPolicySL(
                self.params['ac_dim'],
                self.params['ob_dim'],
                self.params['n_layers'],
                self.params['size'],
                learning_rate=self.params['learning_rate'],
            )
            self.target_network = TransformerPolicySL(
                self.params['ac_dim'],
                self.params['ob_dim'],
                self.params['n_layers'],
                self.params['size'],
                learning_rate=self.params['learning_rate'],
            )
        elif self.params['policy_type'] == 'gpt':
            self.q_network = GPTPolicy(
                preprocessor=Preprocessor()
            )
            self.target_network = GPTPolicy(
                preprocessor=Preprocessor()
            )
        elif self.params['policy_type'] == 'gpt_ar':
            self.q_network = GPTARPolicy(
                preprocessor=Preprocessor()
            )
            self.target_network = GPTARPolicy(
                preprocessor=Preprocessor()
            )
        else:
            self.q_network = MLPPolicySL(
                self.params['ac_dim'],
                self.params['ob_dim'],
                self.params['n_layers'],
                self.params['size'],
                learning_rate=self.params['learning_rate'],
            )
            self.target_network = MLPPolicySL(
                self.params['ac_dim'],
                self.params['ob_dim'],
                self.params['n_layers'],
                self.params['size'],
                learning_rate=self.params['learning_rate'],
            )
        
        # Initialize target network with same weights as Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=params.get('buffer_size', 100000))
        
        # DQN hyperparameters
        self.gamma = params.get('gamma', 0.99)  # discount factor
        self.epsilon = params.get('epsilon_start', 1.0)  # exploration rate
        self.epsilon_min = params.get('epsilon_min', 0.01)
        self.epsilon_decay = params.get('epsilon_decay', 0.995)
        self.target_update_freq = params.get('target_update_freq', 1000)
        self.batch_size = params.get('batch_size', 64)
        self.update_counter = 0
        
    def select_action(self, state, eval_mode=False):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: current state
            eval_mode: if True, use greedy policy (no exploration)
            
        Returns:
            int: selected action
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.params['ac_dim'])
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Normalize observations like in IQL
            state = (state - state.mean(dim=0)) / (state.std(dim=0) + 1e-8)
            dist = self.q_network.forward(state)
            return dist.mean.argmax().item()
    
    def update_epsilon(self):
        """Decay epsilon value"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, train=True):
        """
        Train the DQN agent
        
        Args:
            ob_no: batch of observations
            ac_na: batch of actions
            re_n: batch of rewards
            next_ob_no: batch of next observations
            terminal_n: batch of terminal flags
            train: whether to update the networks
            
        Returns:
            dict: training statistics
        """
        # Store transitions in replay buffer
        for i in range(len(ob_no)):
            self.replay_buffer.push(
                ob_no[i], ac_na[i], re_n[i], next_ob_no[i], terminal_n[i]
            )
        
        if not train or len(self.replay_buffer) < self.batch_size:
            return {'Training Loss': 0.0}
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors and normalize observations
        states = torch.FloatTensor(states).to(self.device)
        states = (states - states.mean(dim=0)) / (states.std(dim=0) + 1e-8)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_states = (next_states - next_states.mean(dim=0)) / (next_states.std(dim=0) + 1e-8)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values using policy distribution
        current_dist = self.q_network.forward(states)
        current_q_values = current_dist.mean.gather(1, actions.long().unsqueeze(1))
        
        # Get next Q values using target network
        with torch.no_grad():
            next_dist = self.target_network.forward(next_states)
            next_q_values = next_dist.mean.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        if train:
            self.q_network.update(states, actions, train=True)
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update exploration rate
        self.update_epsilon()
        
        return {
            'Training Loss': loss.item(),
            'Epsilon': self.epsilon
        }
    
    def save(self, path):
        """
        Save the Q-network and target network
        """
        # Save just the policy state dicts since we're using MLP policies
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """
        Load the Q-network and target network
        """
        checkpoint = torch.load(path)
        # Handle both old and new format
        if 'q_network_state_dict' in checkpoint:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        else:
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon'] 