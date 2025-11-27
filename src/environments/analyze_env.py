"""
Comprehensive RLCard UNO Environment Analysis.
"""

import rlcard
import numpy as np
from collections import Counter

def analyze_uno_environment():
    """Comprehensive analysis of the UNO environment."""
    print(" COMPREHENSIVE UNO ENVIRONMENT ANALYSIS")
    print("=" * 70)
    
    env = rlcard.make('uno')
    
    # Basic environment info
    print(" BASIC ENVIRONMENT INFORMATION")
    print("-" * 40)
    print(f"Number of actions: {env.num_actions}")
    print(f"Number of players: {env.num_players}")  
    print(f"State shape: {env.state_shape}")
    print()
    
    # Action space analysis
    print(" ACTION SPACE ANALYSIS")
    print("-" * 40)
    
    # Run multiple games to see action space
    action_mapping = {}
    all_raw_actions = set()
    
    for game in range(5):
        state, player_id = env.reset()
        
        while not env.is_over():
            legal_actions = state['legal_actions']
            raw_legal_actions = state['raw_legal_actions']
            
            # Map action IDs to raw actions
            for action_id, raw_action in zip(legal_actions.keys(), raw_legal_actions):
                action_mapping[action_id] = raw_action
                all_raw_actions.add(raw_action)
            
            # Take random action
            action_id = np.random.choice(list(legal_actions.keys()))
            state, player_id = env.step(action_id)
            
            if len(action_mapping) >= env.num_actions * 0.8:  # Stop when we have most actions
                break
        
        if len(action_mapping) >= env.num_actions * 0.8:
            break
    
    print(f"Total unique raw actions discovered: {len(all_raw_actions)}")
    print(f"Action mappings discovered: {len(action_mapping)}")
    
    # Categorize actions
    colors = ['r', 'g', 'b', 'y']
    numbers = [str(i) for i in range(10)]
    special_actions = ['skip', 'reverse', 'draw_2', 'wild', 'wild_draw_4']
    
    action_categories = {
        'number_cards': [],
        'skip_cards': [],
        'reverse_cards': [],
        'draw_2_cards': [],
        'wild_cards': [],
        'wild_draw_4_cards': [],
        'other': []
    }
    
    for raw_action in all_raw_actions:
        if 'wild_draw_4' in raw_action:
            action_categories['wild_draw_4_cards'].append(raw_action)
        elif 'wild' in raw_action:
            action_categories['wild_cards'].append(raw_action)
        elif 'draw_2' in raw_action:
            action_categories['draw_2_cards'].append(raw_action)
        elif 'skip' in raw_action:
            action_categories['skip_cards'].append(raw_action)
        elif 'reverse' in raw_action:
            action_categories['reverse_cards'].append(raw_action)
        elif any(num in raw_action for num in numbers):
            action_categories['number_cards'].append(raw_action)
        else:
            action_categories['other'].append(raw_action)
    
    for category, actions in action_categories.items():
        print(f"{category}: {len(actions)} actions")
        if actions:
            print(f"  Examples: {actions[:5]}")
    print()
    
    # State representation analysis
    print(" STATE REPRESENTATION ANALYSIS")
    print("-" * 40)
    
    state, player_id = env.reset()
    
    print("State dictionary keys:")
    for key in state.keys():
        print(f"  {key}: {type(state[key])}")
    
    # Analyze observation tensor
    obs = state['obs']
    print(f"\nObservation tensor shape: {obs.shape}")
    print(f"Observation tensor type: {obs.dtype}")
    print(f"Observation value range: [{obs.min()}, {obs.max()}]")
    
    # Analyze each channel of the observation
    for i in range(obs.shape[0]):
        channel = obs[i]
        non_zero = np.count_nonzero(channel)
        print(f"  Channel {i}: {non_zero} non-zero elements out of {channel.size}")
    
    # Raw observation analysis
    raw_obs = state['raw_obs']
    print(f"\nRaw observation keys: {list(raw_obs.keys())}")
    print(f"Player hand: {raw_obs['hand']}")
    print(f"Target card: {raw_obs['target']}")
    print(f"Number of cards per player: {raw_obs['num_cards']}")
    print()
    
    # Game dynamics analysis
    print(" GAME DYNAMICS ANALYSIS")
    print("-" * 40)
    
    game_lengths = []
    win_distribution = [0, 0]  # For 2 players
    
    for game in range(100):
        state, player_id = env.reset()
        steps = 0
        
        while not env.is_over():
            legal_actions = list(state['legal_actions'].keys())
            action = np.random.choice(legal_actions)
            state, player_id = env.step(action)
            steps += 1
            
            if steps > 500:  # Prevent infinite loops
                break
        
        game_lengths.append(steps)
        payoffs = env.get_payoffs()
        winner = np.argmax(payoffs)
        win_distribution[winner] += 1
    
    print(f"Average game length: {np.mean(game_lengths):.1f} ± {np.std(game_lengths):.1f}")
    print(f"Min/Max game length: {min(game_lengths)}/{max(game_lengths)}")
    print(f"Win distribution: Player 0: {win_distribution[0]}%, Player 1: {win_distribution[1]}%")
    print()
    
    # Action frequency analysis
    print(" ACTION FREQUENCY ANALYSIS")
    print("-" * 40)
    
    action_frequency = Counter()
    
    for game in range(50):
        state, player_id = env.reset()
        
        while not env.is_over():
            legal_actions = list(state['legal_actions'].keys())
            action = np.random.choice(legal_actions)
            
            if action in action_mapping:
                action_frequency[action_mapping[action]] += 1
            
            state, player_id = env.step(action)
    
    print("Most common actions:")
    for action, count in action_frequency.most_common(10):
        print(f"  {action}: {count} times")
    
    print("\n Environment analysis completed!")
    print("\n KEY INSIGHTS:")
    print("  • Action space: 61 discrete actions (cards + special moves)")
    print("  • State representation: 4x4x15 tensor + additional info")
    print("  • Game length: Highly variable (short games possible)")
    print("  • Action categories: Number cards, special cards, wild cards")
    print("  • Legal actions: Dynamically filtered based on game rules")

if __name__ == "__main__":
    analyze_uno_environment()