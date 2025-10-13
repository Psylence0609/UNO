"""
Debug script to understand RLCard UNO state structure.
"""

import rlcard
import numpy as np

def debug_uno_environment():
    """Debug the UNO environment to understand state structure."""
    print("🔍 Debugging UNO Environment State Structure")
    print("=" * 60)
    
    # Create environment
    env = rlcard.make('uno')
    
    # Reset and get initial state
    state, player_id = env.reset()
    
    print(f"Initial player: {player_id}")
    print(f"State type: {type(state)}")
    print(f"State keys (if dict): {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
    print()
    
    # Examine state structure
    if isinstance(state, dict):
        for key, value in state.items():
            print(f"Key '{key}':")
            print(f"  Type: {type(value)}")
            print(f"  Shape: {getattr(value, 'shape', 'No shape attribute')}")
            if hasattr(value, '__len__') and len(value) < 20:
                print(f"  Value: {value}")
            elif isinstance(value, (list, np.ndarray)) and len(value) < 100:
                print(f"  Length: {len(value)}")
                print(f"  First few elements: {value[:10] if len(value) > 10 else value}")
            print()
    else:
        print(f"State value: {state}")
    
    # Get legal actions
    try:
        # Check different ways to get legal actions
        state_dict = env.get_state(player_id)
        print(f"State dict keys: {state_dict.keys()}")
        print(f"Raw legal actions: {state_dict.get('raw_legal_actions', 'Not found')}")
        print(f"Legal actions: {state_dict.get('legal_actions', 'Not found')}")
        
        # Test action space mapping
        print(f"\nAction space exploration:")
        raw_actions = state_dict.get('raw_legal_actions', [])
        legal_action_ids = list(state_dict.get('legal_actions', {}).keys())
        
        for i, (action_id, raw_action) in enumerate(zip(legal_action_ids, raw_actions)):
            print(f"  Action ID {action_id} -> Raw action: {raw_action}")
            if i >= 5:  # Limit output
                break
                
    except Exception as e:
        print(f"Error getting legal actions: {e}")
        
    print()
    
    # Environment info
    print(f"Number of actions: {env.num_actions}")
    print(f"Number of players: {env.num_players}")
    print(f"State shape: {env.state_shape}")

if __name__ == "__main__":
    debug_uno_environment()