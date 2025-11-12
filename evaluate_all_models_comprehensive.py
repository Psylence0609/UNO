#!/usr/bin/env python3
"""
Comprehensive evaluation of all UNO models including custom and RLCard models.
Evaluates all models with multiple runs and statistical analysis.
"""

import os
import sys
import torch
import numpy as np
import yaml
import csv
from tqdm import tqdm

sys.path.append('.')

from src.environments.uno_env import UnoEnvironment
from src.agents.dqn_agent_new import DQNAgent
from src.agents.dmc_agent import DMCAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator
from src.evaluation.statistical_analysis import (
    compute_confidence_interval,
    compare_models_multiple_runs,
    pairwise_comparisons,
    generate_statistical_summary,
    format_statistical_results
)
from rlcard.agents.dmc_agent.model import DMCAgent as RLCardDMCAgent


class RLCardDMCAgentWrapper:
    """Wrapper to make RLCard's DMCAgent compatible with our evaluation framework."""
    
    def __init__(self, rlcard_agent):
        self.agent = rlcard_agent
        self.agent.eval()
        self.agent.exp_epsilon = 0.0
    
    def use_raw(self, state):
        if 'raw_legal_actions' not in state and 'legal_actions' in state:
            legal_actions = state['legal_actions']
            if hasattr(legal_actions, 'keys'):
                state['raw_legal_actions'] = list(legal_actions.keys())
            else:
                state['raw_legal_actions'] = list(legal_actions)
        
        try:
            action, _ = self.agent.eval_step(state)
            return action
        except Exception as e:
            return self.agent.step(state)


def load_config():
    """Load configuration."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_rlcard_dmc_model(model_path, state_shape, action_shape, device="cpu"):
    """Load RLCard DMC model from checkpoint."""
    if device == "cpu":
        map_location = "cpu"
    else:
        map_location = f"cuda:{device}"
    
    checkpoint = torch.load(model_path, map_location=map_location)
    
    agent = RLCardDMCAgent(
        state_shape=state_shape,
        action_shape=action_shape,
        mlp_layers=[512, 512, 512, 512, 512],
        exp_epsilon=0.0,
        device=device
    )
    
    if "model_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["model_state_dict"][0])
    else:
        agent.load_state_dict(checkpoint)
    
    return RLCardDMCAgentWrapper(agent)


def evaluate_vs_random(agent, num_games=500, seed=42):
    """Evaluate agent against random baseline."""
    # Create environment with specified seed
    env = UnoEnvironment(seed=seed)
    evaluator = Evaluator(env)
    random_agent = RandomAgent(evaluator.env.num_actions)
    results = evaluator.evaluate_agents([agent, random_agent], num_games=num_games, verbose=False)
    agent_wins = results['wins'][0]
    win_rate = agent_wins / num_games
    avg_turns = np.mean(results['game_lengths'])
    return win_rate, avg_turns


def evaluate_model_multiple_runs(agent, num_runs=10, num_games_per_run=500, base_seed=42):
    """
    Evaluate a model multiple times with different seeds for statistical robustness.
    
    Args:
        agent: Agent to evaluate
        num_runs: Number of evaluation runs
        num_games_per_run: Number of games per run
        base_seed: Base seed for randomization
        
    Returns:
        Tuple of (win_rates_list, avg_turns_list)
    """
    win_rates = []
    avg_turns_list = []
    
    for run_idx in range(num_runs):
        seed = base_seed + run_idx * 1000
        win_rate, avg_turns = evaluate_vs_random(agent, num_games=num_games_per_run, seed=seed)
        win_rates.append(win_rate)
        avg_turns_list.append(avg_turns)
    
    return np.array(win_rates), np.array(avg_turns_list)


def main():
    """Evaluate all available models with statistical analysis."""
    print("=" * 80)
    print("🏆 COMPREHENSIVE MODEL EVALUATION WITH STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Configuration
    num_runs = 10
    num_games_per_run = 500
    base_seed = 42
    
    print(f"\n📊 Evaluation Configuration:")
    print(f"   Number of runs per model: {num_runs}")
    print(f"   Games per run: {num_games_per_run}")
    print(f"   Total games per model: {num_runs * num_games_per_run}")
    
    # Initialize environment
    env = UnoEnvironment(seed=base_seed)
    evaluator = Evaluator(env)
    config = load_config()
    
    # Calculate state dimension
    sample_state, _ = env.reset()
    state_dim = sample_state['obs'].flatten().shape[0] + env.num_actions
    state_shape = sample_state['obs'].shape
    action_shape = (env.num_actions,)
    
    # Check device
    if torch.cuda.is_available():
        device = "0"
    elif torch.backends.mps.is_available():
        device = "cpu"
    else:
        device = "cpu"
    
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
        },
        'RLCard DMC (100M)': {
            'file': 'models/rlcard/rlcard_dmc_100M.tar',
            'type': 'rlcard_dmc',
            'architecture': [512, 512, 512, 512, 512]
        },
        'RLCard DMC+MCTS (100M)': {
            'file': 'models/rlcard/rlcard_dmc_mcts_100M.tar',
            'type': 'rlcard_dmc',
            'architecture': [512, 512, 512, 512, 512]
        }
    }
    
    # Load agents and evaluate with multiple runs
    agents = {}
    win_rates_dict = {}
    avg_turns_dict = {}
    results = {}
    
    print("\n📦 Loading Models...")
    print("-" * 80)
    
    # First, load all agents
    for name, info in models.items():
        if not os.path.exists(info['file']):
            print(f"⚠️  {name:30s} - File not found: {info['file']}")
            continue
        
        print(f"\n📥 Loading {name}...")
        try:
            if info['type'] == 'dqn':
                agent = DQNAgent(state_dim, env.num_actions, config)
                agent.load(info['file'])
                agent.epsilon = 0.0
            elif info['type'] == 'dmc':
                agent = DMCAgent(state_dim, env.num_actions, config)
                agent.load(info['file'])
                agent.epsilon = 0.0
            elif info['type'] == 'rlcard_dmc':
                agent = load_rlcard_dmc_model(
                    info['file'], state_shape, action_shape, device
                )
            
            agents[name] = agent
            print(f"   ✅ Loaded successfully")
            
        except Exception as e:
            print(f"   ❌ Error loading: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Evaluate each agent with multiple runs
    print("\n🎮 Evaluating Models (Multiple Runs)...")
    print("-" * 80)
    
    for name, agent in tqdm(agents.items(), desc="Evaluating models"):
        print(f"\n🔄 Evaluating {name} ({num_runs} runs)...")
        try:
            win_rates, avg_turns_array = evaluate_model_multiple_runs(
                agent, num_runs=num_runs, 
                num_games_per_run=num_games_per_run, base_seed=base_seed
            )
            
            win_rates_dict[name] = win_rates.tolist()
            avg_turns_dict[name] = avg_turns_array.tolist()
            
            # Compute statistics
            mean_wr, ci_lower_wr, ci_upper_wr = compute_confidence_interval(win_rates)
            mean_turns = np.mean(avg_turns_array)
            std_wr = np.std(win_rates)
            meets_threshold = ci_lower_wr >= 0.55  # Conservative: CI lower bound >= 55%
            
            results[name] = {
                'mean_win_rate': mean_wr,
                'std_win_rate': std_wr,
                'ci_lower': ci_lower_wr,
                'ci_upper': ci_upper_wr,
                'mean_avg_turns': mean_turns,
                'meets_55_threshold': meets_threshold,
                'type': models[name]['type'],
                'architecture': models[name]['architecture'],
                'file': models[name]['file'],
                'num_runs': num_runs
            }
            
            status = "✅ MEETS 55%" if meets_threshold else "❌ Below 55%"
            print(f"   Mean Win Rate: {mean_wr:.1%} ± {std_wr:.1%} {status}")
            print(f"   95% CI: [{ci_lower_wr:.1%}, {ci_upper_wr:.1%}]")
            print(f"   Avg Turns: {mean_turns:.1f}")
            
        except Exception as e:
            print(f"   ❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Statistical Analysis
    print("\n" + "=" * 80)
    print("📊 STATISTICAL ANALYSIS")
    print("=" * 80)
    
    if len(win_rates_dict) > 0:
        # Generate statistical summary
        statistical_summary = generate_statistical_summary(win_rates_dict)
        
        # Print formatted results
        print(format_statistical_results(statistical_summary))
        
        # Summary table with confidence intervals
        print("\n" + "=" * 80)
        print("📊 EVALUATION RESULTS SUMMARY (with 95% Confidence Intervals)")
        print("=" * 80)
        
        print("\n┌──────────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
        print("│ Model                            │ Mean Win Rate│ 95% CI       │ Avg Turns    │ Meets 55%?   │")
        print("├──────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
        
        for name in sorted(results.keys(), key=lambda x: results[x]['mean_win_rate'], reverse=True):
            r = results[name]
            mean_wr = r['mean_win_rate']
            ci_lower = r['ci_lower']
            ci_upper = r['ci_upper']
            turns = r['mean_avg_turns']
            meets = "✅ YES" if r['meets_55_threshold'] else "❌ NO"
            print(f"│ {name:32s} │ {mean_wr:11.1%} │ [{ci_lower:5.1%},{ci_upper:5.1%}] │ {turns:11.1f} │ {meets:12s} │")
        
        print("└──────────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    
    # Models meeting threshold
    print("\n" + "=" * 80)
    print("🎯 MODELS MEETING 55% THRESHOLD")
    print("=" * 80)
    
    meeting_threshold = [name for name, r in results.items() if r['meets_55_threshold']]
    if meeting_threshold:
        for name in meeting_threshold:
            r = results[name]
            print(f"\n✅ {name}")
            print(f"   Mean Win Rate: {r['mean_win_rate']:.1%} (95% CI: [{r['ci_lower']:.1%}, {r['ci_upper']:.1%}])")
            print(f"   Architecture: {r['architecture']}")
            print(f"   File: {r['file']}")
    else:
        print("\n⚠️  No models currently meet the 55% threshold (using conservative CI lower bound >= 55%).")
        print("   Consider retraining or using best available models.")
    
    # Save results to CSV
    output_file = 'results/model_evaluation_results.csv'
    os.makedirs('results', exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Mean Win Rate', 'Std Win Rate', 'CI Lower', 'CI Upper',
            'Mean Avg Turns', 'Meets 55% Threshold', 'Type', 'Architecture', 'File', 'N Runs'
        ])
        for name, r in sorted(results.items(), key=lambda x: x[1]['mean_win_rate'], reverse=True):
            writer.writerow([
                name,
                f"{r['mean_win_rate']:.4f}",
                f"{r['std_win_rate']:.4f}",
                f"{r['ci_lower']:.4f}",
                f"{r['ci_upper']:.4f}",
                f"{r['mean_avg_turns']:.2f}",
                r['meets_55_threshold'],
                r['type'],
                str(r['architecture']),
                r['file'],
                r['num_runs']
            ])
    
    # Save statistical summary
    if len(win_rates_dict) > 0:
        summary_file = 'results/statistical_summary.csv'
        statistical_summary['model_summary'].to_csv(summary_file, index=False)
        print(f"\n💾 Statistical summary saved to: {summary_file}")
        
        pairwise_file = 'results/pairwise_comparisons.csv'
        statistical_summary['pairwise_comparisons'].to_csv(pairwise_file, index=False)
        print(f"💾 Pairwise comparisons saved to: {pairwise_file}")
    
    print(f"💾 Results saved to: {output_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("📈 FINAL SUMMARY")
    print("=" * 80)
    print(f"   Total models evaluated: {len(results)}")
    print(f"   Models meeting 55% threshold: {len(meeting_threshold)}")
    print(f"   Models below 55% threshold: {len(results) - len(meeting_threshold)}")
    print(f"   Total games played: {len(results) * num_runs * num_games_per_run}")
    
    if meeting_threshold:
        best = max(meeting_threshold, key=lambda x: results[x]['mean_win_rate'])
        print(f"   Best performing model: {best} ({results[best]['mean_win_rate']:.1%}, CI: [{results[best]['ci_lower']:.1%}, {results[best]['ci_upper']:.1%}])")
    elif results:
        best = max(results.keys(), key=lambda x: results[x]['mean_win_rate'])
        print(f"   Best available model: {best} ({results[best]['mean_win_rate']:.1%}, CI: [{results[best]['ci_lower']:.1%}, {results[best]['ci_upper']:.1%}])")
    
    return results, statistical_summary if len(win_rates_dict) > 0 else None


if __name__ == "__main__":
    main()

