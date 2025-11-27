
import sys
import os
import torch
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.environments.uno_env import UnoEnvironment
from src.agents.dmc_agent_advanced_opponent import AdvancedDMCAgentWithOpponent

def test_dron_fix():
    print("🚀 Testing DRON Fix...")
    
    # 1. Initialize Environment
    env = UnoEnvironment()
    state, player_id = env.reset()
    print("✅ Environment initialized")
    
    # 2. Initialize Agent
    config = {
        'environment': {'num_players': 2},
        'opponent_modeling': {
            'history_size': 10,
            'hidden_layers': [64, 32],
            'strategy_dim': 16,
            'use_attention': True,
            'use_temporal': True,
            'sequence_feature_size': None # Should be auto-detected
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 4
        },
        'network': {
            'hidden_layers': [128, 64]
        },
        'dmc': {}
    }
    
    # Calculate state size dynamically
    sample_state, _ = env.reset()
    features = []
    if 'obs' in sample_state:
        features.extend(sample_state['obs'].flatten())
    features.extend(np.zeros(env.num_actions))
    state_size = len(features)
    
    agent = AdvancedDMCAgentWithOpponent(
        state_size=state_size,
        action_size=env.num_actions,
        config=config
    )
    print(f"✅ Agent initialized. Sequence feature size: {agent.sequence_feature_size}")
    
    # 3. Test Feature Extraction (Static + Sequence)
    # Simulate some opponent moves
    agent.update_opponent_history(action=1, opponent_id=1, hand_size=6)
    agent.update_opponent_history(action=2, opponent_id=1, hand_size=5)
    
    static, sequence = agent.extract_opponent_features(state)
    print(f"✅ Features extracted.")
    print(f"   Static shape: {static.shape}")
    print(f"   Sequence shape: {sequence.shape}")
    
    assert sequence.shape == (10, agent.sequence_feature_size), f"Expected (10, {agent.sequence_feature_size}), got {sequence.shape}"
    
    # 4. Test Act (Forward Pass)
    action = agent.act(state, legal_actions=[0, 1, 2])
    print(f"✅ Action selected: {action}")
    
    # 5. Test Learn (Backward Pass)
    # Store some dummy transitions
    for _ in range(5):
        agent.store_transition(
            state=state,
            action=0,
            reward=1.0,
            next_state=state,
            done=False,
            opponent_features=(static, sequence)
        )
        
    stats = agent.learn_episode()
    print(f"✅ Learning step completed.")
    print(f"   Stats: {stats}")
    
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    test_dron_fix()
