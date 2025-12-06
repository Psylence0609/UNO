"""
Main entry point for the UNO RL project.
"""

import sys
import os
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.environments.uno_env import UnoEnvironment
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    # Adjust path if running from scripts
    if not os.path.exists(config_path):
        config_path = os.path.join(project_root, config_path)
        
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """Main function to run experiments."""
    print("UNO Reinforcement Learning Project")
    print("=" * 50)
    
    config = load_config()
    
    print("Setting up UNO environment...")
    env = UnoEnvironment()
    
    print(f"Environment created successfully!")
    print(f"State shape: {env.state_shape}")
    print(f"Action space: {env.num_actions}")
    print(f"Number of players: {env.num_players}")
    
    print("\nTesting random agents...")
    agents = [RandomAgent(env.num_actions) for _ in range(env.num_players)]
    
    state, player_id = env.reset()
    print(f"Starting test game with player {player_id}")
    
    for step in range(10):
        if env.is_over():
            break
            
        action = agents[player_id].use_raw(state)
        state, next_player_id = env.step(action, raw_action=False)
        
        print(f"  Step {step + 1}: Player {player_id} took action {action}")
        player_id = next_player_id
        
        if env.is_over():
            payoffs = env.get_payoffs()
            print(f"Game ended! Payoffs: {payoffs}")
            break
    
    print("\nBasic environment test completed successfully!")
    print("\nNext steps:")
    print("  1. Implement DQN agent")
    print("  2. Create training infrastructure")
    print("  3. Set up evaluation framework")

if __name__ == "__main__":
    main()