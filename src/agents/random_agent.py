"""
Random Agent for UNO.
"""

import numpy as np
import random

class RandomAgent:
    """
    A random agent that selects actions uniformly at random from legal actions.
    """
    
    def __init__(self, num_actions, np_random=None):
        """
        Initialize the random agent.
        
        Args:
            num_actions (int): Number of possible actions
            np_random: Random number generator (optional)
        """
        self.num_actions = num_actions
        self.np_random = np_random if np_random is not None else np.random.RandomState()
        
    def use_raw(self, state):
        """
        Use the agent to choose an action when in raw format.
        
        Args:
            state: Current game state (raw format from RLCard)
            
        Returns:
            int: Selected action
        """
        # Extract legal actions from state
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = state['legal_actions']
            # RLCard returns OrderedDict for legal_actions, convert to list
            if hasattr(legal_actions, 'keys'):
                legal_actions = list(legal_actions.keys())
        else:
            # Fallback to all actions if legal actions not available
            legal_actions = list(range(self.num_actions))
        
        # Randomly select from legal actions
        if legal_actions:
            action = self.np_random.choice(legal_actions)
            return int(action)  # Ensure Python int, not numpy int
        else:
            # If no legal actions (shouldn't happen), return first action
            return 0
    
    def eval_step(self, state):
        """
        Use the agent for evaluation (same as use_raw for random agent).
        
        Args:
            state: Current game state
            
        Returns:
            tuple: (action, probs) where probs is uniform over legal actions
        """
        action = self.use_raw(state)
        
        # Create uniform probability distribution over legal actions
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = state['legal_actions']
            # RLCard returns OrderedDict for legal_actions, convert to list
            if hasattr(legal_actions, 'keys'):
                legal_actions = list(legal_actions.keys())
            
            probs = np.zeros(self.num_actions)
            if legal_actions:
                prob_value = 1.0 / len(legal_actions)
                for a in legal_actions:
                    probs[a] = prob_value
        else:
            probs = np.ones(self.num_actions) / self.num_actions
            
        return action, probs
    
    def set_device(self, device):
        """Set device (compatibility with other agents)."""
        pass  # Random agent doesn't use GPU
    
    def train(self):
        """Set agent to training mode (compatibility with other agents)."""
        pass  # Random agent doesn't train
    
    def eval(self):
        """Set agent to evaluation mode (compatibility with other agents)."""
        pass  # Random agent is always in eval mode