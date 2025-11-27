"""
Multi-Player Tournament Evaluation for UNO RL Agents.
Supports 2, 3, and 4 player tournaments via command line arguments.
"""

import os
import sys
import numpy as np
from itertools import combinations
from tqdm import tqdm
import json
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.environments.uno_env import UnoEnvironment
from src.agents.dmc_agent_advanced_opponent import AdvancedDMCAgentWithOpponent
from src.agents.dmc_agent import DMCAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent


class TournamentMultiPlayer:
    """Multi-Player Round-Robin Tournament for UNO agents."""

    def __init__(self, agents_dict, num_players=3, num_games=1000):
        """
        Initialize tournament.

        Args:
            agents_dict: Dictionary of {name: agent} pairs
            num_players: Number of players per game (2, 3, or 4)
            num_games: Number of games per matchup
        """
        self.agents_dict = agents_dict
        self.num_players = num_players
        self.num_games = num_games
        self.env = UnoEnvironment(seed=42)

        # Results storage
        self.results = {name: {'wins': 0, 'games': 0, 'placements': []}
                       for name in agents_dict.keys()}
        self.matchup_results = {}

    def _normalize_matchup_key(self, matchup):
        """Normalize a matchup tuple by sorting the agent names."""
        return tuple(sorted(matchup))

    def play_game(self, agents):
        """Play a single multi-player game."""
        state, player_id = self.env.reset()

        while not self.env.is_over():
            current_agent = agents[player_id]

            # Get action
            try:
                action = current_agent.use_raw(state)
            except:
                # Fallback for agents without use_raw
                legal_actions = list(state['legal_actions'].keys())
                action = np.random.choice(legal_actions)

            # Take step
            state, player_id = self.env.step(action)

        # Get payoffs
        payoffs = self.env.get_payoffs()
        return payoffs

    def run_matchup(self, agent_names):
        """Run all games for a specific multi-player matchup."""
        agents = [self.agents_dict[name] for name in agent_names]
        
        # Set agents to eval mode
        for agent in agents:
            if hasattr(agent, 'eval'):
                agent.eval()
        
        wins = {name: 0 for name in agent_names}
        placements = {name: [] for name in agent_names}
        
        for _ in tqdm(range(self.num_games), desc=f"{agent_names[0][:10]} vs {agent_names[1][:10]} vs {agent_names[2][:10]}", leave=False):
            payoffs = self.play_game(agents)
            
            # Determine winner and placements
            winner_idx = np.argmax(payoffs)
            wins[agent_names[winner_idx]] += 1
            
            # Record placements (1st, 2nd, 3rd)
            sorted_indices = np.argsort(payoffs)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                placements[agent_names[idx]].append(rank)
        
        # Calculate win rates
        matchup_key = tuple(sorted(agent_names))
        self.matchup_results[matchup_key] = {
            name: {
                'wins': wins[name],
                'win_rate': wins[name] / self.num_games,
                'avg_placement': np.mean(placements[name])
            }
            for name in agent_names
        }
        
        # Update overall results
        for name in agent_names:
            self.results[name]['wins'] += wins[name]
            self.results[name]['games'] += self.num_games
            self.results[name]['placements'].extend(placements[name])
        
        return self.matchup_results[matchup_key]
    
    def run_tournament(self, skip_completed=None):
        """Run full round-robin tournament."""
        agent_names = list(self.agents_dict.keys())

        # Generate all combinations based on number of players
        matchups = list(combinations(agent_names, self.num_players))

        # Filter out completed matchups if provided
        if skip_completed:
            original_count = len(matchups)
            # Normalize both generated matchups and completed keys for comparison
            normalized_completed = {self._normalize_matchup_key(eval(k)) for k in skip_completed}
            matchups = [m for m in matchups if self._normalize_matchup_key(m) not in normalized_completed]
            skipped_count = original_count - len(matchups)
            print(f"⏭️  Skipping {skipped_count} completed matchups")

        print(f"\n🏆 Starting {self.num_players}-Player Tournament")
        print(f"{'='*80}")
        print(f"Agents: {len(agent_names)}")
        print(f"Matchups: {len(matchups)}")
        print(f"Games per Matchup: {self.num_games}")
        print(f"Total Games: {len(matchups) * self.num_games}")
        print(f"{'='*80}\n")

        for matchup in tqdm(matchups, desc="Tournament Progress"):
            self.run_matchup(matchup)
        
        # Calculate final statistics
        for name in agent_names:
            self.results[name]['overall_win_rate'] = (
                self.results[name]['wins'] / self.results[name]['games']
            )
            self.results[name]['avg_placement'] = np.mean(self.results[name]['placements'])
        
        return self.results
    
    def print_leaderboard(self):
        """Print tournament leaderboard."""
        sorted_agents = sorted(
            self.results.items(),
            key=lambda x: x[1]['overall_win_rate'],
            reverse=True
        )
        
        print(f"\n{'='*80}")
        print(f"🏆 TOURNAMENT LEADERBOARD")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Agent':<25} {'Win Rate':<12} {'Avg Place':<12} {'Wins':<10}")
        print(f"{'-'*80}")
        
        for rank, (name, stats) in enumerate(sorted_agents, 1):
            print(f"{rank:<6} {name:<25} {stats['overall_win_rate']:.1%}      "
                  f"{stats['avg_placement']:.2f}          "
                  f"{stats['wins']}/{stats['games']}")
        
        print(f"{'='*80}\n")

    def print_leaderboard_from_data(self, results_data):
        """Print tournament leaderboard from data dictionary."""
        sorted_agents = sorted(
            results_data['overall'].items(),
            key=lambda x: x[1]['overall_win_rate'],
            reverse=True
        )

        print(f"\n{'='*80}")
        print(f"🏆 TOURNAMENT LEADERBOARD")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Agent':<25} {'Win Rate':<12} {'Avg Place':<12} {'Wins':<10}")
        print(f"{'-'*80}")

        for rank, (name, stats) in enumerate(sorted_agents, 1):
            print(f"{rank:<6} {name:<25} {stats['overall_win_rate']:<12.1%} "
                  f"{stats['avg_placement']:<12.1f} {stats['wins']:<10}")
        print(f"{'='*80}\n")

    def save_results(self, filepath='tournament_results.json'):
        """Save tournament results to JSON."""
        with open(filepath, 'w') as f:
            json.dump({
                'overall': self.results,
                'matchups': {str(self._normalize_matchup_key(k)): v for k, v in self.matchup_results.items()}
            }, f, indent=2)
        print(f"✅ Results saved to {filepath}")


def load_agent(agent_type, model_path=None, config=None):
    """Load an agent from checkpoint."""
    env = UnoEnvironment()

    if agent_type == 'dron':
        agent = AdvancedDMCAgentWithOpponent(
            state_size=301,  # From your setup
            action_size=env.num_actions,
            config=config or {}
        )
        if model_path:
            agent.load(model_path)
        agent.eval()
        return agent

    elif agent_type == 'dmc':
        # Old DMC models used different architecture
        old_dmc_config = {
            'network': {
                'hidden_layers': [256, 128]  # Old architecture
            }
        }
        agent = DMCAgent(
            state_size=301,
            action_size=env.num_actions,
            config=old_dmc_config
        )
        if model_path:
            agent.load(model_path)
        agent.eval()
        return agent

    elif agent_type == 'dqn':
        agent = DQNAgent(
            state_size=301,
            action_size=env.num_actions,
            config=config or {}
        )
        if model_path:
            agent.load(model_path)
        agent.eval()
        return agent

    elif agent_type == 'rlcard_dmc':
        # Load RLCard DMC model
        try:
            from evaluate_rlcard_dmc import load_rlcard_dmc_model
            # Use RLCard's actual state shape for UNO
            state_shape = [4, 4, 15]  # RLCard UNO state shape
            action_shape = [61]  # RLCard UNO action size
            device = "cpu"  # Use CPU for evaluation
            agent = load_rlcard_dmc_model(model_path, state_shape, action_shape, device)
            return agent
        except ImportError:
            print(f"⚠️  RLCard not available, cannot load {model_path}")
            raise ValueError("RLCard not installed")

    elif agent_type == 'rlcard_dqn':
        # For now, we'll use the custom DQN as RLCard DQN placeholder
        # since we don't have a separate RLCard DQN wrapper
        print("⚠️  RLCard DQN not implemented yet, using custom DQN")
        agent = DQNAgent(
            state_size=301,
            action_size=env.num_actions,
            config=config or {}
        )
        if model_path:
            agent.load(model_path)
        agent.eval()
        return agent

    elif agent_type == 'heuristic':
        return HeuristicAgent(env.num_actions)

    elif agent_type == 'random':
        return RandomAgent(env.num_actions)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    """Run the tournament."""
    import yaml

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-player UNO tournament evaluation')
    parser.add_argument('--players', type=int, choices=[2, 3, 4], default=3,
                       help='Number of players per game (2, 3, or 4)')
    parser.add_argument('--games', type=int, default=1000,
                       help='Number of games per matchup')
    args = parser.parse_args()

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Define agents to evaluate - all your trained models
    agents = {
        # Custom models
        'DRON (57.5%)': load_agent('dron', 'models/custom/dmc_advanced_opponent_best.pth', config),
        'DMC + MCTS': load_agent('dmc', 'models/custom/dmc_mcts_final.pth'),
        'DMC': load_agent('dmc', 'models/custom/dmc_episode_10000.pth'),
        'DQN + MCTS': load_agent('dqn', 'models/custom/dqn_mcts_final.pth', config),
        'DQN': load_agent('dqn', 'models/custom/dqn_final.pth', config),

        # RLCard built-in models
        'RLCard DMC': load_agent('rlcard_dmc', 'models/rlcard/rlcard_dmc_100M.tar'),

        # Baselines
        'Heuristic': load_agent('heuristic'),
        'Random': load_agent('random')
    }

    # Results file based on number of players
    existing_results_file = f'tournament_{args.players}player_results.json'
    existing_data = {'overall': {}, 'matchups': {}}
    completed_matchups = set()

    if os.path.exists(existing_results_file):
        print(f"📂 Found existing results file: {existing_results_file}")
        try:
            with open(existing_results_file, 'r') as f:
                existing_data = json.load(f)
                if 'matchups' in existing_data:
                    # Normalize existing matchup keys for consistency
                    normalized_matchups = {}
                    for key_str, value in existing_data['matchups'].items():
                        try:
                            key_tuple = eval(key_str)
                            normalized_key = str(tuple(sorted(key_tuple)))
                            normalized_matchups[normalized_key] = value
                        except:
                            # If parsing fails, keep original
                            normalized_matchups[key_str] = value
                    existing_data['matchups'] = normalized_matchups
                    completed_matchups = set(existing_data['matchups'].keys())
                    print(f"✅ Found {len(completed_matchups)} completed matchups to skip")
        except Exception as e:
            print(f"⚠️  Could not read existing results: {e}")
            existing_data = {'overall': {}, 'matchups': {}}

    # Run tournament with skip logic
    tournament = TournamentMultiPlayer(agents, num_players=args.players, num_games=args.games)
    new_overall_results = tournament.run_tournament(skip_completed=completed_matchups)

    # Merge results with existing data
    print("🔄 Merging new results with existing data...")

    # Merge overall statistics
    for agent_name, agent_stats in new_overall_results.items():
        if agent_name in existing_data['overall']:
            # Merge stats for existing agents
            existing_stats = existing_data['overall'][agent_name]
            existing_stats['wins'] += agent_stats['wins']
            existing_stats['games'] += agent_stats['games']
            existing_stats['overall_win_rate'] = existing_stats['wins'] / existing_stats['games']
            if not np.isnan(agent_stats['avg_placement']) and not np.isnan(existing_stats['avg_placement']):
                existing_stats['avg_placement'] = (
                    (existing_stats['avg_placement'] * (existing_stats['games'] - agent_stats['games']) +
                     agent_stats['avg_placement'] * agent_stats['games']) / existing_stats['games']
                )
            existing_stats['placements'].extend(agent_stats['placements'])
        else:
            # Add new agent
            existing_data['overall'][agent_name] = agent_stats.copy()

    # Add new matchups
    existing_data['matchups'].update({str(k): v for k, v in tournament.matchup_results.items()})

    print(f"📊 Final results: {len(existing_data['overall'])} agents, {len(existing_data['matchups'])} matchups")

    # Print leaderboard with merged results
    tournament.print_leaderboard_from_data(existing_data)

    # Save merged results back to the same file
    with open(existing_results_file, 'w') as f:
        json.dump(existing_data, f, indent=2)
    print(f"💾 Results saved to {existing_results_file}")


if __name__ == "__main__":
    main()
