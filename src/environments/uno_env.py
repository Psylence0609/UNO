"""
UNO Environment Wrapper for RLCard.
"""

import rlcard
from rlcard.envs.registration import register, make
import numpy as np

# Patch RLCard UNO environment's buggy _decode_action method
def _patch_rlcard_uno():
    """Patch the RLCard UNO environment to fix the OrderedDict bug."""
    try:
        from rlcard.envs.uno import UnoEnv, ACTION_LIST
        
        if not hasattr(UnoEnv, '_patched'):
            # Store original method
            original_decode = UnoEnv._decode_action
            
            def patched_decode_action(self, action_id):
                """Fixed _decode_action that handles OrderedDict properly."""
                legal_ids = self._get_legal_actions()
                # FIX: Convert OrderedDict to list if needed
                if hasattr(legal_ids, 'keys'):
                    legal_ids = list(legal_ids.keys())
                
                if action_id in legal_ids:
                    return ACTION_LIST[action_id]
                else:
                    # If action not legal, randomly choose from legal actions
                    return ACTION_LIST[np.random.choice(legal_ids)]
            
            UnoEnv._decode_action = patched_decode_action
            UnoEnv._patched = True
            print("✅ RLCard UNO environment patched successfully")
    except Exception as e:
        print(f"⚠️  Could not patch RLCard UNO: {e}")

_patch_rlcard_uno()

class UnoEnvironment:
    """
    Wrapper for RLCard UNO environment with additional functionality.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the UNO environment.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.env = rlcard.make('uno', config={'seed': seed})
        self.seed = seed
        
        # Environment properties
        self.num_actions = self.env.num_actions
        self.num_players = self.env.num_players
        self.state_shape = self.env.state_shape
        
    def reset(self):
        """
        Reset the environment and return initial state.
        
        Returns:
            tuple: (state, player_id)
        """
        state, player_id = self.env.reset()
        return state, player_id
    
    def step(self, action, raw_action=False):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            raw_action (bool): Whether the action is in raw format
            
        Returns:
            tuple: (next_state, next_player_id)
        """
        # Ensure action is Python int to avoid RLCard issues
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)
        
        # Workaround for RLCard UNO bug with action 60 (draw card)
        # RLCard has a bug where it passes OrderedDict to np.random.choice
        # We'll use raw_action=True to bypass the buggy _decode_action method
        if raw_action:
            return self.env.step(action, raw_action=True)
        else:
            try:
                return self.env.step(action)
            except (ValueError, TypeError) as e:
                if "must be 1-dimensional or an integer" in str(e) or "OrderedDict" in str(e):
                    # Retry with raw_action=True to bypass buggy _decode_action
                    return self.env.step(action, raw_action=True)
                else:
                    raise
    
    def is_over(self):
        """Check if the game is over."""
        return self.env.is_over()
    
    def get_payoffs(self):
        """Get the payoffs for all players."""
        return self.env.get_payoffs()
    
    def get_legal_actions(self):
        """Get legal actions for the current player."""
        # Get the current state to extract legal actions
        state = self.env.get_state(self.env.get_player_id())
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = state['legal_actions']
            # RLCard returns OrderedDict for legal_actions, convert to list
            if hasattr(legal_actions, 'keys'):
                return list(legal_actions.keys())
            return legal_actions
        return []
    
    def get_perfect_information(self):
        """Get perfect information state (for debugging/analysis)."""
        return self.env.get_perfect_information()
    
    def render(self, mode='human'):
        """Render the current game state."""
        if mode == 'human':
            print("=== UNO Game State ===")
            perfect_info = self.get_perfect_information()
            print(f"Current player: {perfect_info.get('current_player', 'Unknown')}")
            print(f"Direction: {perfect_info.get('direction', 'Unknown')}")
            print(f"Top card: {perfect_info.get('top_card', 'Unknown')}")
            
            # Show hand sizes for all players
            for i, hand_size in enumerate(perfect_info.get('hand_sizes', [])):
                print(f"Player {i}: {hand_size} cards")
                
        return self.env.render(mode=mode)
    
    def decode_action(self, action_id):
        """
        Decode action ID to human-readable format.
        
        Args:
            action_id (int): Action ID to decode
            
        Returns:
            str: Human-readable action description
        """
        try:
            return self.env.decode_action(action_id)
        except:
            return f"Action_{action_id}"
    
    def get_state_representation(self, state):
        """
        Convert state to a more usable format for neural networks.
        
        Args:
            state: Raw state from environment
            
        Returns:
            np.array: Processed state representation
        """
        if isinstance(state, dict):
            # Extract relevant features from state dictionary
            features = []
            
            # Add observation vector
            if 'obs' in state:
                obs = state['obs']
                if isinstance(obs, (list, np.ndarray)):
                    features.extend(obs)
                else:
                    features.append(obs)
            
            # Add legal actions mask
            if 'legal_actions' in state:
                legal_actions_dict = state['legal_actions']
                legal_mask = np.zeros(self.num_actions)
                # Handle OrderedDict from RLCard
                if hasattr(legal_actions_dict, 'keys'):
                    for action in legal_actions_dict.keys():
                        legal_mask[action] = 1
                else:
                    for action in legal_actions_dict:
                        legal_mask[action] = 1
                features.extend(legal_mask)
            
            return np.array(features, dtype=np.float32)
        
        elif isinstance(state, (list, np.ndarray)):
            return np.array(state, dtype=np.float32)
        
        else:
            # If state is in an unexpected format, try to convert it
            return np.array([state] if np.isscalar(state) else state, dtype=np.float32)