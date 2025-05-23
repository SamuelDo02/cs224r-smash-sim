"""
Behavior cloning agent for Melee
"""
from src.policies.MLP_policy import MLPPolicySL

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

        # Create policy class as our actor
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
