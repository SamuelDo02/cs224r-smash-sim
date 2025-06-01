"""
Behavior cloning agent for Melee
"""
from policies.GPT_AR_policy import GPTARPolicy
from policies.GPT_policy import GPTPolicy
from policies.MLP_policy import MLPPolicySL
from policies.Transformer_policy import TransformerPolicySL
from policies.preprocessor import Preprocessor
import torch

class BCAgent:
    """
    A behavior cloning agent for Melee

    Attributes
    ----------
    actor : MLPPolicySL
        An MLP that outputs an agent's actions given its observations
    """
    def __init__(self, env, agent_params):
        # Initialize variables
        self.env = env
        self.agent_params = agent_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.agent_params)
        # Create policy class as our actor
        if self.agent_params['policy_type'] == 'transformer':
            self.actor = TransformerPolicySL(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                learning_rate=self.agent_params['learning_rate'],
            )
        elif self.agent_params['policy_type'] == 'gpt':
            self.actor = GPTPolicy(
                preprocessor=Preprocessor()
            )
        elif self.agent_params['policy_type'] == 'gpt_ar':
            self.actor = GPTARPolicy(
                preprocessor=Preprocessor()
            )
        else:
            self.actor = MLPPolicySL(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                learning_rate=self.agent_params['learning_rate'],
            )

    def train(self, ob_no, ac_na, train=True):
        """
        Train the policy using behavior cloning

        Args:
            ob_no: batch of observations
            ac_na: batch of actions to imitate
            train: whether to update the network or just compute loss
        Returns:
            dict: training statistics
        """
        return self.actor.update(ob_no, ac_na, train=train)

    def save(self, path):
        """
        Save the policy to a file
        """
        return self.actor.save(path)
