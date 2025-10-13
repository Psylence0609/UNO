#!/usr/bin/env python3
"""Quick evaluation script for comparing DQN and DMC+MCTS models."""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append('.')

from src.environments.uno_env import UnoEnvironment
from src.agents.dqn_agent_new import DQNAgent
from src.agents.dmc_agent import DMCAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator

def main():
    """Quick evaluation of available models."""
    print("=== Quick Model Evaluation ===")
    
    # Initialize environment
    env = UnoEnvironment()
    evaluator = Evaluator(env)
    
    # Check what models are available
    models_available = {
        'DQN Original': 'models/uno_dqn_final.pth',
        'DQN New': 'models/uno_dqn_new_final.pth', 
        'DMC+MCTS': 'models/uno_dmc_final.pth'
    }
    
    agents = {}
    results = {}
    
    print("\nChecking available models...")
    for name, path in models_available.items():
        if os.path.exists(path):
            print(f"✅ Found {name}: {path}")
            try:
                if 'DQN Original' in name:
                    # Original DQN with [512, 256] architecture
                    agent = DQNAgent(env.state_dim, env.action_dim, hidden_dims=[512, 256])
                else:
                    # New models with unified [256, 128] architecture
                    if 'DQN' in name:
                        agent = DQNAgent(env.state_dim, env.action_dim, hidden_dims=[256, 128])
                    else:
                        agent = DMCAgent(env.state_dim, env.action_dim, hidden_dims=[256, 128])
                
                agent.load_model(path)
                agents[name] = agent
                print(f"   Loaded successfully")
                
            except Exception as e:
                print(f"   ❌ Failed to load: {e}")
        else:
            print(f"❌ Missing {name}: {path}")
    
    # Add random agent
    agents['Random'] = RandomAgent(env.action_dim)
    
    print(f"\nLoaded agents: {list(agents.keys())}")
    
    # Evaluate each agent vs random
    print("\n=== Evaluation vs Random Agent ===")
    for name, agent in agents.items():
        if name != 'Random':
            try:
                win_rate, avg_turns = evaluator.evaluate_agent(agent, num_games=500)
                results[name] = {'win_rate': win_rate, 'avg_turns': avg_turns}
                print(f"{name:15} - Win Rate: {win_rate:.3f} ({win_rate*100:.1f}%), Avg Turns: {avg_turns:.1f}")
            except Exception as e:
                print(f"{name:15} - Error: {e}")
    
    # Head-to-head comparisons between trained agents
    trained_agents = {k: v for k, v in agents.items() if k != 'Random'}
    
    if len(trained_agents) >= 2:
        print("\n=== Head-to-Head Comparisons ===")
        agent_names = list(trained_agents.keys())
        
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                name1, name2 = agent_names[i], agent_names[j]
                agent1, agent2 = trained_agents[name1], trained_agents[name2]
                
                try:
                    win_rate = evaluator.head_to_head_evaluation(agent1, agent2, num_games=300)
                    print(f"{name1:15} vs {name2:15}: {win_rate:.3f} ({win_rate*100:.1f}%)")
                except Exception as e:
                    print(f"{name1:15} vs {name2:15}: Error - {e}")
    
    # Summary
    print("\n=== SUMMARY ===")
    if results:
        best_agent = max(results.items(), key=lambda x: x[1]['win_rate'])
        print(f"Best performing agent: {best_agent[0]} with {best_agent[1]['win_rate']:.3f} ({best_agent[1]['win_rate']*100:.1f}%) win rate")
        
        print("\nAll results:")
        for name, result in sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True):
            print(f"  {name:15}: {result['win_rate']:.3f} ({result['win_rate']*100:.1f}%) win rate, {result['avg_turns']:.1f} avg turns")
    else:
        print("No models available for evaluation")

if __name__ == "__main__":
    main()