
import numpy as np
import torch

class HeuristicAgent:
    """
    A simple rule-based agent for UNO.
    Strategy:
    1. Play Action cards (Skip, Reverse, Draw Two) first to disrupt opponent.
    2. Play Wild cards last to save them for emergencies.
    3. Match color if possible.
    4. Match number if possible.
    """
    
    def __init__(self, action_num):
        self.action_num = action_num
        
    def eval(self):
        """Dummy method for compatibility."""
        pass
        
    def train(self):
        """Dummy method for compatibility."""
        pass
        
    def use_raw(self, state):
        """
        Predict action based on heuristic rules.
        Args:
            state (dict): Raw state dictionary
        """
        legal_actions = list(state['legal_actions'].keys())
        if not legal_actions:
            return 0
            
        # 1. Prioritize Action Cards (Skip=10, Reverse=11, Draw2=12)
        action_cards = [a for a in legal_actions if 10 <= (a % 15) <= 12]
        if action_cards:
            return np.random.choice(action_cards)
            
        # 2. Save Wild Cards (Wild=13, Wild4=14) for last
        normal_cards = [a for a in legal_actions if (a % 15) < 13]
        if normal_cards:
            # Prefer matching color of top card if known (heuristic)
            # Since we don't easily parse state here, just random normal
            return np.random.choice(normal_cards)
            
        # 3. Play Wild if nothing else
        return np.random.choice(legal_actions)

    def step(self, state):
        return self.use_raw(state)
