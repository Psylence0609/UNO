#!/usr/bin/env python3
"""
Evaluate DMC agent with opponent modeling against diverse opponents.
Tests performance against random, aggressive, conservative, and strategic opponents.
"""

import os
import sys
import torch
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append('.')

from src.environments.uno_env import UnoEnvironment
from src.agents.dmc_agent_with_opponent import DMCAgentWithOpponentModeling
from src.agents.dmc_agent import DMCAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator
from src.evaluation.statistical_analysis import (
    compute_confidence_interval,
    generate_statistical_summary,
    format_statistical_results
)


class AggressiveAgent:
    """
    Aggressive agent that plays cards quickly.
    Prefers playing cards over drawing.
    """
    
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
    
    def use_raw(self, state):
        if 'legal_actions' in state:
            legal_actions = list(state['legal_actions'].keys())
            # Prefer non-draw actions (action 60 is draw)
            non_draw_actions = [a for a in legal_actions if a != 60]
            if non_draw_actions:
                return np.random.choice(non_draw_actions)
            else:
                return legal_actions[0] if legal_actions else 0
        return 0


class ConservativeAgent:
    """
    Conservative agent that holds cards longer.
    Prefers drawing over playing when uncertain.
    """
    
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
    
    def use_raw(self, state):
        if 'legal_actions' in state:
            legal_actions = list(state['legal_actions'].keys())
            # Prefer draw action if available (action 60 is draw)
            if 60 in legal_actions:
                return 60
            else:
                return np.random.choice(legal_actions)
        return 0


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def evaluate_against_opponent(
    agent,
    opponent,
    opponent_name: str,
    num_games: int = 500,
    num_runs: int = 5,
    base_seed: int = 42
) -> dict:
    win_rates = []
    avg_turns_list = []
    
    for run_idx in range(num_runs):
        seed = base_seed + run_idx * 1000
        env = UnoEnvironment(seed=seed)
        evaluator = Evaluator(env)
        
        if hasattr(agent, 'reset_opponent_tracking'):
            agent.reset_opponent_tracking(num_players=env.num_players)
        
        results = evaluator.evaluate_agents([agent, opponent], num_games=num_games, verbose=False)
        win_rate = results['win_rates'][0]
        avg_turns = results['avg_game_length']
        
        win_rates.append(win_rate)
        avg_turns_list.append(avg_turns)
    
    win_rates_array = np.array(win_rates)
    mean_wr, ci_lower, ci_upper = compute_confidence_interval(win_rates_array)
    std_wr = np.std(win_rates_array)
    mean_turns = np.mean(avg_turns_list)
    
    return {
        'opponent_name': opponent_name,
        'mean_win_rate': mean_wr,
        'std_win_rate': std_wr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_avg_turns': mean_turns,
        'num_runs': num_runs,
        'num_games_per_run': num_games
    }


def main():
    print("=" * 80)
    print("OPPONENT MODELING EVALUATION")
    print("=" * 80)
    
    config = load_config()
    
    env = UnoEnvironment(seed=42)
    evaluator = Evaluator(env)
    
    sample_state, _ = env.reset()
    state_dim = sample_state['obs'].flatten().shape[0] + env.num_actions
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"\nLoading Models...")
    print("-" * 80)
    
    baseline_model_path = 'models/custom/dmc_episode_10000.pth'
    if not os.path.exists(baseline_model_path):
        print(f"  Baseline model not found: {baseline_model_path}")
        print("   Skipping baseline comparison")
        baseline_agent = None
    else:
        try:
            baseline_agent = DMCAgent(state_dim, env.num_actions, config)
            baseline_agent.load(baseline_model_path)
            baseline_agent.epsilon = 0.0
            print(f"Baseline DMC agent loaded")
        except Exception as e:
            print(f"  Failed to load baseline: {e}")
            baseline_agent = None
    
    opponent_model_path = 'models/custom/dmc_opponent_modeling_final.pth'
    if not os.path.exists(opponent_model_path):
        print(f"  Opponent modeling model not found: {opponent_model_path}")
        print("   Please train the model first using train_dmc_with_opponent.py")
        opponent_agent = None
    else:
        try:
            opponent_agent = DMCAgentWithOpponentModeling(
                state_size=state_dim,
                action_size=env.num_actions,
                config=config
            )
            opponent_agent.load(opponent_model_path)
            opponent_agent.epsilon = 0.0
            print(f"DMC with opponent modeling loaded")
        except Exception as e:
            print(f"  Failed to load opponent modeling agent: {e}")
            import traceback
            traceback.print_exc()
            opponent_agent = None
    
    if opponent_agent is None:
        print("\nCannot proceed without opponent modeling agent")
        return
    
    opponents = {
        'Random': RandomAgent(env.num_actions),
        'Aggressive': AggressiveAgent(env.num_actions),
        'Conservative': ConservativeAgent(env.num_actions)
    }
    
    if baseline_agent is not None:
        opponents['Strategic (DMC)'] = baseline_agent
    
    num_games_per_run = 300
    num_runs = 5
    
    print(f"\nEvaluation Configuration:")
    print(f"   Games per run: {num_games_per_run}")
    print(f"   Number of runs: {num_runs}")
    print(f"   Total games per opponent: {num_runs * num_games_per_run}")
    
    print(f"\nEvaluating DMC with Opponent Modeling...")
    print("-" * 80)
    
    results_opponent_modeling = {}
    
    for opponent_name, opponent in opponents.items():
        print(f"\nEvaluating against {opponent_name}...")
        try:
            result = evaluate_against_opponent(
                opponent_agent, opponent, opponent_name,
                num_games=num_games_per_run, num_runs=num_runs, base_seed=42
            )
            results_opponent_modeling[opponent_name] = result
            
            print(f"   Mean Win Rate: {result['mean_win_rate']:.1%} ± {result['std_win_rate']:.1%}")
            print(f"   95% CI: [{result['ci_lower']:.1%}, {result['ci_upper']:.1%}]")
            print(f"   Avg Turns: {result['mean_avg_turns']:.1f}")
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    results_baseline = {}
    
    if baseline_agent is not None:
        print(f"\nEvaluating Baseline DMC (for comparison)...")
        print("-" * 80)
        
        for opponent_name, opponent in opponents.items():
            if opponent_name == 'Strategic (DMC)':
                continue
            
            print(f"\nEvaluating baseline against {opponent_name}...")
            try:
                result = evaluate_against_opponent(
                    baseline_agent, opponent, opponent_name,
                    num_games=num_games_per_run, num_runs=num_runs, base_seed=42
                )
                results_baseline[opponent_name] = result
                
                print(f"   Mean Win Rate: {result['mean_win_rate']:.1%} ± {result['std_win_rate']:.1%}")
                print(f"   95% CI: [{result['ci_lower']:.1%}, {result['ci_upper']:.1%}]")
            except Exception as e:
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print("\n")
    print(" Opponent Type             With Opponent Model   Baseline DMC          Improvement  ")
    print("")
    
    for opponent_name in results_opponent_modeling.keys():
        if opponent_name == 'Strategic (DMC)':
            continue
        
        om_result = results_opponent_modeling[opponent_name]
        om_wr = om_result['mean_win_rate']
        
        if opponent_name in results_baseline:
            baseline_result = results_baseline[opponent_name]
            baseline_wr = baseline_result['mean_win_rate']
            improvement = om_wr - baseline_wr
            improvement_str = f"{improvement:+.1%}"
        else:
            baseline_wr = None
            improvement_str = "N/A"
        
        print(f" {opponent_name:24s}  {om_wr:18.1%}  ", end="")
        if baseline_wr is not None:
            print(f"{baseline_wr:18.1%}  {improvement_str:12s} ")
        else:
            print(f"{'N/A':18s}  {improvement_str:12s} ")
    
    print("")
    
    if results_baseline:
        print("\n" + "=" * 80)
        print("STATISTICAL COMPARISON")
        print("=" * 80)
        
        for opponent_name in results_opponent_modeling.keys():
            if opponent_name not in results_baseline:
                continue
            
            om_result = results_opponent_modeling[opponent_name]
            baseline_result = results_baseline[opponent_name]
            
            improvement = om_result['mean_win_rate'] - baseline_result['mean_win_rate']
            improvement_pct = improvement * 100
            
            print(f"\n{opponent_name}:")
            print(f"   Opponent Modeling: {om_result['mean_win_rate']:.1%} (CI: [{om_result['ci_lower']:.1%}, {om_result['ci_upper']:.1%}])")
            print(f"   Baseline DMC: {baseline_result['mean_win_rate']:.1%} (CI: [{baseline_result['ci_lower']:.1%}, {baseline_result['ci_upper']:.1%}])")
            print(f"   Improvement: {improvement:+.1%} ({improvement_pct:+.2f} percentage points)")
    
    output_file = 'results/opponent_modeling_evaluation.csv'
    os.makedirs('results', exist_ok=True)
    
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Agent', 'Opponent', 'Mean Win Rate', 'Std Win Rate', 'CI Lower', 'CI Upper',
            'Mean Avg Turns', 'N Runs', 'N Games Per Run'
        ])
        
        for opponent_name, result in results_opponent_modeling.items():
            writer.writerow([
                'DMC with Opponent Modeling',
                opponent_name,
                f"{result['mean_win_rate']:.4f}",
                f"{result['std_win_rate']:.4f}",
                f"{result['ci_lower']:.4f}",
                f"{result['ci_upper']:.4f}",
                f"{result['mean_avg_turns']:.2f}",
                result['num_runs'],
                result['num_games_per_run']
            ])
        
        if results_baseline:
            for opponent_name, result in results_baseline.items():
                writer.writerow([
                    'Baseline DMC',
                    opponent_name,
                    f"{result['mean_win_rate']:.4f}",
                    f"{result['std_win_rate']:.4f}",
                    f"{result['ci_lower']:.4f}",
                    f"{result['ci_upper']:.4f}",
                    f"{result['mean_avg_turns']:.2f}",
                    result['num_runs'],
                    result['num_games_per_run']
                ])
    
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"   Opponent types tested: {len(results_opponent_modeling)}")
    print(f"   Total games played: {len(results_opponent_modeling) * num_runs * num_games_per_run}")
    
    if results_opponent_modeling:
        best_opponent = max(results_opponent_modeling.keys(), 
                          key=lambda x: results_opponent_modeling[x]['mean_win_rate'])
        worst_opponent = min(results_opponent_modeling.keys(), 
                           key=lambda x: results_opponent_modeling[x]['mean_win_rate'])
        
        print(f"   Best performance: vs {best_opponent} ({results_opponent_modeling[best_opponent]['mean_win_rate']:.1%})")
        print(f"   Worst performance: vs {worst_opponent} ({results_opponent_modeling[worst_opponent]['mean_win_rate']:.1%})")
        
        if results_baseline:
            avg_improvement = np.mean([
                results_opponent_modeling[opp]['mean_win_rate'] - results_baseline[opp]['mean_win_rate']
                for opp in results_baseline.keys()
            ])
            print(f"   Average improvement over baseline: {avg_improvement:+.1%}")


if __name__ == "__main__":
    main()
