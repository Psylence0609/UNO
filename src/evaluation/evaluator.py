"""
Evaluation framework for UNO RL agents.
"""

import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    """
    Evaluation framework for testing UNO agents.
    """
    
    def __init__(self, env):
        """
        Initialize the evaluator.
        
        Args:
            env: UNO environment instance
        """
        self.env = env
        self.results_history = []
        
    def evaluate_agents(self, agents, num_games=1000, verbose=True):
        """
        Evaluate agents by playing multiple games.
        
        Args:
            agents (list): List of agents to evaluate
            num_games (int): Number of games to play
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Evaluation results
        """
        if len(agents) != self.env.num_players:
            raise ValueError(f"Expected {self.env.num_players} agents, got {len(agents)}")
        
        results = {
            'wins': [0] * self.env.num_players,
            'game_lengths': [],
            'payoffs_history': [],
            'total_games': num_games
        }
        
        start_time = time.time()
        
        for game_idx in range(num_games):
            if verbose and (game_idx + 1) % (num_games // 10) == 0:
                print(f"Progress: {game_idx + 1}/{num_games} games completed")
            
            # Play one game
            game_length, payoffs = self._play_single_game(agents)
            
            # Record results
            results['game_lengths'].append(game_length)
            results['payoffs_history'].append(payoffs)
            
            # Determine winner (player with highest payoff)
            winner = np.argmax(payoffs)
            results['wins'][winner] += 1
        
        # Calculate statistics
        results['win_rates'] = [wins / num_games for wins in results['wins']]
        results['avg_game_length'] = np.mean(results['game_lengths'])
        results['std_game_length'] = np.std(results['game_lengths'])
        results['evaluation_time'] = time.time() - start_time
        
        # Store results
        self.results_history.append(results)
        
        if verbose:
            self._print_results(results, agents)
            
        return results
    
    def _play_single_game(self, agents):
        """
        Play a single game between agents.
        
        Args:
            agents (list): List of agents
            
        Returns:
            tuple: (game_length, payoffs)
        """
        # Reset environment
        state, player_id = self.env.reset()
        game_length = 0
        
        # Play until game ends
        while not self.env.is_over():
            # Get action from current player
            action = agents[player_id].use_raw(state)
            
            # Take step
            state, player_id = self.env.step(action)
            game_length += 1
            
            # Safety check for infinite games
            if game_length > 1000:
                print(f"Warning: Game exceeded 1000 steps, ending early")
                break
        
        # Get final payoffs
        payoffs = self.env.get_payoffs()
        
        return game_length, payoffs
    
    def _print_results(self, results, agents):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for i, agent in enumerate(agents):
            agent_name = agent.__class__.__name__
            print(f"Player {i} ({agent_name}):")
            print(f"  Wins: {results['wins'][i]}/{results['total_games']}")
            print(f"  Win Rate: {results['win_rates'][i]:.1%}")
        
        print(f"\nGame Statistics:")
        print(f"  Average Game Length: {results['avg_game_length']:.1f} ± {results['std_game_length']:.1f}")
        print(f"  Evaluation Time: {results['evaluation_time']:.2f} seconds")
        print("="*60)
    
    def plot_results(self, results=None, save_path=None):
        """
        Plot evaluation results.
        
        Args:
            results (dict): Results to plot (if None, use last results)
            save_path (str): Path to save the plot
        """
        if results is None:
            if not self.results_history:
                print("No results to plot")
                return
            results = self.results_history[-1]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Win rates
        player_names = [f"Player {i}" for i in range(len(results['win_rates']))]
        bars = ax1.bar(player_names, results['win_rates'])
        ax1.set_title('Win Rates')
        ax1.set_ylabel('Win Rate')
        ax1.set_ylim(0, 1)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, results['win_rates']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # Game length distribution
        ax2.hist(results['game_lengths'], bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title('Game Length Distribution')
        ax2.set_xlabel('Game Length (steps)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(results['avg_game_length'], color='red', linestyle='--', 
                   label=f'Mean: {results["avg_game_length"]:.1f}')
        ax2.legend()
        
        # Payoffs over time (last 100 games)
        recent_payoffs = results['payoffs_history'][-100:]
        if recent_payoffs:
            payoffs_array = np.array(recent_payoffs)
            for i in range(payoffs_array.shape[1]):
                ax3.plot(payoffs_array[:, i], label=f'Player {i}', alpha=0.7)
            ax3.set_title('Payoffs (Last 100 Games)')
            ax3.set_xlabel('Game')
            ax3.set_ylabel('Payoff')
            ax3.legend()
        
        # Cumulative wins
        cumulative_wins = np.zeros((len(results['payoffs_history']), len(results['wins'])))
        for i, payoffs in enumerate(results['payoffs_history']):
            winner = np.argmax(payoffs)
            if i > 0:
                cumulative_wins[i] = cumulative_wins[i-1].copy()
            cumulative_wins[i, winner] += 1
        
        for i in range(cumulative_wins.shape[1]):
            ax4.plot(cumulative_wins[:, i], label=f'Player {i}')
        ax4.set_title('Cumulative Wins')
        ax4.set_xlabel('Game')
        ax4.set_ylabel('Cumulative Wins')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def compare_multiple_evaluations(self, save_path=None):
        """
        Compare results from multiple evaluations.
        
        Args:
            save_path (str): Path to save the comparison plot
        """
        if len(self.results_history) < 2:
            print("Need at least 2 evaluations to compare")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Win rates comparison
        eval_names = [f"Evaluation {i+1}" for i in range(len(self.results_history))]
        num_players = len(self.results_history[0]['win_rates'])
        
        x = np.arange(len(eval_names))
        width = 0.35 / num_players
        
        for player in range(num_players):
            win_rates = [results['win_rates'][player] for results in self.results_history]
            offset = (player - num_players/2 + 0.5) * width
            ax1.bar(x + offset, win_rates, width, label=f'Player {player}')
        
        ax1.set_title('Win Rates Comparison')
        ax1.set_ylabel('Win Rate')
        ax1.set_xlabel('Evaluation')
        ax1.set_xticks(x)
        ax1.set_xticklabels(eval_names)
        ax1.legend()
        
        # Average game length comparison
        avg_lengths = [results['avg_game_length'] for results in self.results_history]
        std_lengths = [results['std_game_length'] for results in self.results_history]
        
        ax2.bar(eval_names, avg_lengths, yerr=std_lengths, capsize=5)
        ax2.set_title('Average Game Length Comparison')
        ax2.set_ylabel('Average Game Length')
        ax2.set_xlabel('Evaluation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()