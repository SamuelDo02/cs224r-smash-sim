import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policies.MLP_policy import MLPPolicySL
from policies.Transformer_policy import TransformerPolicySL
from policies.GPT_policy import GPTPolicy
from policies.GPT_AR_policy import GPTARPolicy
from policies.preprocessor import Preprocessor
from .iql_reward import IQLReward

class IQLAgent:
    """
    Implicit Q-Learning agent that learns a policy using value function learning
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
        
        # Initialize value function with smaller network
        self.value_net = nn.Sequential(
            nn.Linear(params['ob_dim'], params['size']//2),
            nn.ReLU(),
            nn.Linear(params['size']//2, params['size']//4),
            nn.ReLU(), 
            nn.Linear(params['size']//4, 1)
        ).to(self.device)
        
        # Initialize Q-function with smaller network
        q_input_dim = params['ob_dim'] + params['ac_dim']
        self.q_net = nn.Sequential(
            nn.Linear(q_input_dim, params['size']//2),
            nn.ReLU(),
            nn.Linear(params['size']//2, params['size']//4),
            nn.ReLU(),
            nn.Linear(params['size']//4, 1)
        ).to(self.device)
        # Initialize optimizers with gradient clipping
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=params['learning_rate'])
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=params['learning_rate']/10)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=params['learning_rate'])
        
        # Initialize reward function
        self.reward_fn = IQLReward(
            temperature=10.0,
            expectile=0.9995,
            max_grad_norm=1.0
        )
        
        
    def train(self, ob_no, ac_na, train=True):
        """
        Train the policy using IQL
        
        Args:
            ob_no: batch of observations
            ac_na: batch of actions to imitate
            train: whether to update the networks or just compute loss
        Returns:
            dict: training statistics
        """
        # Convert to tensors and normalize observations
        obs = torch.FloatTensor(ob_no).to(self.device)
        actions = torch.FloatTensor(ac_na).to(self.device)
        
        # Normalize observations
        obs = (obs - obs.mean(dim=0)) / (obs.std(dim=0) + 1e-8)
        
        # Update value function
        with torch.no_grad():
            q_values = self.q_net(torch.cat([obs, actions], dim=-1))
        value_pred = self.value_net(obs)
        value_loss = self.reward_fn.compute_value_loss(q_values, value_pred)
        
        if train:
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.reward_fn.max_grad_norm)
            self.value_optimizer.step()
        
        # Update Q-function
        with torch.no_grad():
            value_pred = self.value_net(obs)
        q_pred = self.q_net(torch.cat([obs, actions], dim=-1))
        q_loss = self.reward_fn.compute_q_loss(q_pred, value_pred)
        
        if train:
            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.reward_fn.max_grad_norm)
            self.q_optimizer.step()
        
        # Update policy with weighted BC loss
        with torch.no_grad():
            value_pred = self.value_net(obs)
            q_values = self.q_net(torch.cat([obs, actions], dim=-1))
            _, weights = self.reward_fn.compute_advantages(q_values, value_pred)
        
        # Get policy predictions
        dist = self.actor.forward(obs)
        log_probs = dist.log_prob(actions)
        bc_loss = -log_probs.sum(dim=-1).mean()
        
        # Apply importance weights to BC loss
        weighted_bc_loss = bc_loss * weights.mean()
        
        if train:
            self.optimizer.zero_grad()
            weighted_bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.reward_fn.max_grad_norm)
            self.optimizer.step()
        
        return {
            'Training Loss': weighted_bc_loss.item(),
            'Value Loss': value_loss.item(),
            'Q Loss': q_loss.item(),
            'BC Loss': bc_loss.item(),
        }
    
    def save(self, path):
        """
        Save the policy and value/Q networks
        """
        torch.save({
            'policy_state_dict': self.actor.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'q_state_dict': self.q_net.state_dict()
        }, path) 