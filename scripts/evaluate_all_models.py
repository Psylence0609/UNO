#!/usr/bin/env python3
"""Comprehensive evaluation of all UNO models."""

import os
import sys
sys.path.append('.')

from src.environments.uno_env import UnoEnvironment
from src.agents.dqn_agent_new import DQNAgent
from src.agents.dmc_agent import DMCAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator
import yaml
import numpy as np

def load_config():
    """Load configuration."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_vs_random(evaluator, agent, num_games=500):
    """Evaluate agent against random baseline."""
    random_agent = RandomAgent(evaluator.env.num_actions)
    
    # Agent as player 0, random as player 1
    results = evaluator.evaluate_agents([agent, random_agent], num_games=num_games, verbose=False)
    
    agent_wins = results['wins'][0]
    win_rate = agent_wins / num_games
    avg_turns = np.mean(results['game_lengths'])
    
    return win_rate, avg_turns

def head_to_head(evaluator, agent1, agent2, num_games=300):
    """Evaluate two agents head-to-head."""
    results = evaluator.evaluate_agents([agent1, agent2], num_games=num_games, verbose=False)
    
    agent1_wins = results['wins'][0]
    win_rate = agent1_wins / num_games
    
    return win_rate

def main():
    """Evaluate all available models."""
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    
    # Initialize environment
    env = UnoEnvironment()
    evaluator = Evaluator(env)
    config = load_config()
    
    # Calculate state dimension
    sample_state, _ = env.reset()
    state_dim = sample_state['obs'].flatten().shape[0] + env.num_actions
    
    # Model configurations
    models = {
        'DQN Original': {
            'file': 'models/custom/dqn_final.pth',
            'type': 'dqn',
            'architecture': [256, 128]
        },
        'DQN+MCTS': {
            'file': 'models/custom/dqn_mcts_final.pth',
            'type': 'dqn',
            'architecture': [256, 128]
        },
        'DMC Old': {
            'file': 'models/custom/dmc_mcts_final.pth',
            'type': 'dmc',
            'architecture': [256, 128]
        },
        'DMC Ep10k': {
            'file': 'models/custom/dmc_episode_10000.pth',
            'type': 'dmc',
            'architecture': [256, 128]
        }
    }
    
    # Load available models
    agents = {}
    print("\nLoading Models...")
    print("-" * 80)
    
    for name, info in models.items():
        if os.path.exists(info['file']):
            try:
                if info['type'] == 'dqn':
                    agent = DQNAgent(state_dim, env.num_actions, config)
                    agent.load(info['file'])
                else:  # dmc
                    agent = DMCAgent(state_dim, env.num_actions, config)
                    agent.load(info['file'])
                
                # Set to evaluation mode
                agent.epsilon = 0.0
                agents[name] = agent
                print(f"{name:20s} loaded from {info['file']}")
            except Exception as e:
                print(f"{name:20s} failed to load: {e}")
        else:
            print(f"{name:20s} file not found: {info['file']}")
    
    if len(agents) == 0:
        print("\nNo models loaded!")
        return
    
    # Evaluate vs Random
    print("\n" + "=" * 80)
    print("PERFORMANCE VS RANDOM BASELINE")
    print("=" * 80)
    
    results = {}
    for name, agent in agents.items():
        print(f"\nEvaluating {name}... (500 games)")
        try:
            win_rate, avg_turns = evaluate_vs_random(evaluator, agent, num_games=500)
            results[name] = {
                'vs_random_wr': win_rate,
                'avg_turns': avg_turns
            }
            print(f"   Win Rate: {win_rate:.1%}")
            print(f"   Avg Turns: {avg_turns:.1f}")
        except Exception as e:
            print(f"   Error: {e}")
            results[name] = {'vs_random_wr': 0, 'avg_turns': 0}
    
    # Head-to-head comparisons
    if len(agents) > 1:
        print("\n" + "=" * 80)
        print("HEAD-TO-HEAD COMPARISONS")
        print("=" * 80)
        
        agent_names = list(agents.keys())
        h2h_results = {}
        
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                name1, name2 = agent_names[i], agent_names[j]
                agent1, agent2 = agents[name1], agents[name2]
                
                print(f"\n{name1} vs {name2} (300 games)")
                try:
                    win_rate = head_to_head(evaluator, agent1, agent2, num_games=300)
                    h2h_results[f"{name1} vs {name2}"] = win_rate
                    print(f"   {name1} wins: {win_rate:.1%}")
                    print(f"   {name2} wins: {1-win_rate:.1%}")
                except Exception as e:
                    print(f"   Error: {e}")
    
    # Generate comparison table
    print("\n" + "=" * 80)
    print("COMPLETE COMPARISON TABLE")
    print("=" * 80)
    
    print("\n")
    print(" Model               vs Random    Avg Turns   ")
    print("")
    
    for name in sorted(results.keys(), key=lambda x: results[x]['vs_random_wr'], reverse=True):
        wr = results[name]['vs_random_wr']
        turns = results[name]['avg_turns']
        print(f" {name:18s}  {wr:10.1%}  {turns:10.1f} ")
    
    print("")
    
    # Head-to-head matrix
    if 'h2h_results' in locals() and h2h_results:
        print("\n")
        print("           HEAD-TO-HEAD RESULTS                  ")
        print("")
        
        for matchup, wr in sorted(h2h_results.items()):
            parts = matchup.split(' vs ')
            print(f"  {parts[0]:15s} {wr:>5.1%} vs {1-wr:>5.1%}  {parts[1]:15s}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        best = max(results.items(), key=lambda x: x[1]['vs_random_wr'])
        print(f"\nBest Overall: {best[0]} ({best[1]['vs_random_wr']:.1%} vs Random)")
        
        print("\nKey Insights:")
        print(f"   • Total models evaluated: {len(results)}")
        if 'h2h_results' in locals():
            print(f"   • Head-to-head matchups: {len(h2h_results)}")
        print(f"   • All models using unified [256, 128] architecture")
        print(f"   • Random baseline ~50% due to UNO's high variance")

if __name__ == "__main__":
    main()
