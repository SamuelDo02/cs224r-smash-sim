"""
PPO (Proximal Policy Optimization) agent for Melee
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from policies.MLP_policy import MLPPolicySL
from policies.Transformer_policy import TransformerPolicySL
from policies.GPT_policy import GPTPolicy
from policies.GPT_AR_policy import GPTARPolicy
from policies.preprocessor import Preprocessor

class PPOMemory:
    """
    Memory buffer for PPO that stores transitions
    """
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), \
               np.array(self.probs), np.array(self.vals), \
               np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class PPOAgent:
    """
    PPO agent for Melee that uses a policy network and value network
    """
    def __init__(self, env, params):
        # Initialize variables
        self.env = env
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create policy class as our actor
        print(f'Initializing {self.params["policy_type"]} policy...')   
        if self.params['policy_type'] == 'transformer':
            self.actor = TransformerPolicySL(
                self.params['ac_dim'],
                self.params['ob_dim'],
                self.params['n_layers'],
                self.params['size'],
                learning_rate=self.params['learning_rate'],
            )
        elif self.params['policy_type'] == 'gpt':
            self.actor = GPTPolicy(
                preprocessor=Preprocessor()
            )
        elif self.params['policy_type'] == 'gpt_ar':
            self.actor = GPTARPolicy(
                preprocessor=Preprocessor()
            )
        else:
            self.actor = MLPPolicySL(
                self.params['ac_dim'],
                self.params['ob_dim'],
                self.params['n_layers'],
                self.params['size'],
                learning_rate=self.params['learning_rate'],
            )
        
        # Initialize value network
        self.critic = nn.Sequential(
            nn.Linear(params['ob_dim'], params['size']),
            nn.ReLU(),
            nn.Linear(params['size'], params['size']),
            nn.ReLU(),
            nn.Linear(params['size'], 1)
        ).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params['learning_rate'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params['learning_rate'])
        
        self.memory = PPOMemory(batch_size=params.get('batch_size', 64))
        
        self.gamma = params.get('gamma', 0.99)  
        self.gae_lambda = params.get('gae_lambda', 0.95)  
        self.ppo_epochs = params.get('ppo_epochs', 10)  
        self.clip_ratio = params.get('clip_ratio', 0.2)  
        self.value_coef = params.get('value_coef', 0.5)  
        self.entropy_coef = params.get('entropy_coef', 0.01)  

    def select_action(self, state, eval_mode=False):
        """
        Select action using the policy network
        
        Args:
            state: current state
            eval_mode: if True, use mean of distribution (no exploration)
            
        Returns:
            tuple: (action, log_prob, value)
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state = (state - state.mean(dim=0)) / (state.std(dim=0) + 1e-8)
            
            dist = self.actor.forward(state)
            value = self.critic(state)
            
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation
        
        Args:
            rewards: list of rewards
            values: list of value predictions
            dones: list of done flags
            
        Returns:
            tuple: (advantages, returns)
        """
        advantages = []
        returns = []
        running_return = 0
        previous_value = 0
        running_advantage = 0

        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            running_return = r + self.gamma * running_return * (1-d)
            returns.insert(0, running_return)

            td_error = r + self.gamma * previous_value * (1-d) - v
            running_advantage = td_error + self.gamma * self.gae_lambda * running_advantage * (1-d)
            advantages.insert(0, running_advantage)

            previous_value = v

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, train=True):
        """
        Train the PPO agent
        
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
        if not train:
            return {'Training Loss': 0.0}
            
        # Store transitions in memory
        for i in range(len(ob_no)):
            self.memory.store_memory(
                ob_no[i], ac_na[i], re_n[i], next_ob_no[i], terminal_n[i]
            )
        
        # Generate batches
        states, actions, old_probs, old_vals, rewards, dones, batches = self.memory.generate_batches()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        old_vals = torch.FloatTensor(old_vals).to(self.device)
        
        # Normalize observations
        states = (states - states.mean(dim=0)) / (states.std(dim=0) + 1e-8)
        
        # Compute GAE
        advantages, returns = self.compute_gae(rewards, old_vals, dones)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            for batch in batches:
                # Get batch data
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]
                
                # Get current policy distribution and value
                dist = self.actor.forward(batch_states)
                values = self.critic(batch_states)
                
                # Compute new log probs and entropy
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute policy loss
                ratio = torch.exp(new_log_probs - batch_old_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        # Clear memory
        self.memory.clear_memory()
        
        return {
            'Training Loss': loss.item(),
            'Policy Loss': policy_loss.item(),
            'Value Loss': value_loss.item(),
            'Entropy': entropy.item()
        }

    def save(self, path):
        """
        Save the policy and value networks
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        Load the policy and value networks
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict']) 