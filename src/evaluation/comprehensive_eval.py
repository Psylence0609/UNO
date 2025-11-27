"""
Comprehensive evaluation comparing Random, DQN, and DMC+MCTS agents.
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.uno_env import UnoEnvironment
from src.agents.dmc_agent import DMCAgent
from src.agents.dqn_agent_new import DQNAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluator import Evaluator

class ComprehensiveEvaluation:
    """Comprehensive evaluation of all UNO agents."""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the evaluator.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup environment
        self.env = UnoEnvironment(seed=self.config['environment']['seed'])
        
        # Calculate state size
        sample_state, _ = self.env.reset()
        features = []
        if 'obs' in sample_state:
            features.extend(sample_state['obs'].flatten())
        features.extend(np.zeros(self.env.num_actions))
        self.state_size = len(features)
        
        # Setup agents
        self.agents = {}
        self.load_agents()
        
        # Setup evaluator
        self.evaluator = Evaluator(self.env)
        
        # Setup results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_agents(self):
        """Load all available agents."""
        # Random agent (always available)
        self.agents['Random'] = RandomAgent(self.env.num_actions)
        
        # DQN agent (if model exists)
        try:
            dqn_agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.env.num_actions,
                config=self.config
            )
            
            # Try to load best model
            model_paths = [
                "models/dqn_final.pth",
                "models/dqn_best.pth"
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    dqn_agent.load(model_path)
                    dqn_agent.epsilon = 0.0  # No exploration during evaluation
                    self.agents['DQN'] = dqn_agent
                    print(f" Loaded DQN from {model_path}")
                    break
            
            if 'DQN' not in self.agents:
                print("  No trained DQN model found")
                
        except Exception as e:
            print(f" Failed to load DQN: {e}")
        
        # DMC agent (if model exists)
        try:
            dmc_agent = DMCAgent(
                state_size=self.state_size,
                action_size=self.env.num_actions,
                config=self.config
            )
            
            # Try to load best model
            model_paths = [
                "models/dmc_mcts_final.pth",
                "models/dmc_best.pth"
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    dmc_agent.load(model_path)
                    dmc_agent.epsilon = 0.0  # No exploration during evaluation
                    dmc_agent.eval()  # Set to evaluation mode
                    self.agents['DMC+MCTS'] = dmc_agent
                    print(f" Loaded DMC+MCTS from {model_path}")
                    break
            
            if 'DMC+MCTS' not in self.agents:
                print("  No trained DMC+MCTS model found")
                
        except Exception as e:
            print(f" Failed to load DMC+MCTS: {e}")
        
        print(f"Available agents: {list(self.agents.keys())}")
    
    def pairwise_evaluation(self, agent1_name, agent2_name, num_games=1000):
        """
        Evaluate two agents against each other.
        
        Args:
            agent1_name (str): Name of first agent
            agent2_name (str): Name of second agent
            num_games (int): Number of games to play
            
        Returns:
            dict: Detailed evaluation results
        """
        agent1 = self.agents[agent1_name]
        agent2 = self.agents[agent2_name]
        
        print(f" Evaluating {agent1_name} vs {agent2_name} ({num_games} games)")
        
        # Play games in both positions
        results_p1 = self.evaluator.evaluate_agents(
            [agent1, agent2], num_games // 2, verbose=False
        )
        
        results_p2 = self.evaluator.evaluate_agents(
            [agent2, agent1], num_games // 2, verbose=False
        )
        
        # Combine results
        total_wins_agent1 = results_p1['wins'][0] + results_p2['wins'][1]
        total_wins_agent2 = results_p1['wins'][1] + results_p2['wins'][0]
        
        win_rate_agent1 = total_wins_agent1 / num_games
        win_rate_agent2 = total_wins_agent2 / num_games
        
        avg_game_length = (results_p1['avg_game_length'] + results_p2['avg_game_length']) / 2
        
        return {
            'agent1': agent1_name,
            'agent2': agent2_name,
            'agent1_wins': total_wins_agent1,
            'agent2_wins': total_wins_agent2,
            'agent1_win_rate': win_rate_agent1,
            'agent2_win_rate': win_rate_agent2,
            'avg_game_length': avg_game_length,
            'total_games': num_games
        }
    
    def round_robin_tournament(self, num_games=1000):
        """
        Run a round-robin tournament between all agents.
        
        Args:
            num_games (int): Number of games per matchup
            
        Returns:
            dict: Tournament results
        """
        print(" Running Round-Robin Tournament")
        print("=" * 50)
        
        agent_names = list(self.agents.keys())
        results = []
        
        # Play all pairwise matchups
        for i, agent1 in enumerate(agent_names):
            for j, agent2 in enumerate(agent_names):
                if i != j:  # Don't play against self
                    matchup_result = self.pairwise_evaluation(agent1, agent2, num_games)
                    results.append(matchup_result)
        
        # Compile tournament standings
        standings = {}
        for agent in agent_names:
            standings[agent] = {
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_games': 0
            }
        
        for result in results:
            agent1, agent2 = result['agent1'], result['agent2']
            
            standings[agent1]['wins'] += result['agent1_wins']
            standings[agent1]['losses'] += result['agent2_wins']
            standings[agent1]['total_games'] += result['total_games']
            
            standings[agent2]['wins'] += result['agent2_wins']
            standings[agent2]['losses'] += result['agent1_wins']
            standings[agent2]['total_games'] += result['total_games']
        
        # Calculate win rates
        for agent in standings:
            total = standings[agent]['total_games']
            if total > 0:
                standings[agent]['win_rate'] = standings[agent]['wins'] / total
        
        return {
            'pairwise_results': results,
            'standings': standings
        }
    
    def detailed_analysis(self, num_games=2000):
        """
        Run detailed analysis of all agents.
        
        Args:
            num_games (int): Number of games for detailed analysis
            
        Returns:
            dict: Comprehensive analysis results
        """
        print(" Running Detailed Analysis")
        print("=" * 50)
        
        analysis_results = {}
        
        # Tournament results
        tournament_results = self.round_robin_tournament(num_games)
        analysis_results['tournament'] = tournament_results
        
        # Individual agent analysis
        for agent_name in self.agents.keys():
            if agent_name != 'Random':  # Skip random for individual analysis
                print(f"\n Analyzing {agent_name}")
                
                # Performance vs Random
                vs_random = self.pairwise_evaluation(agent_name, 'Random', num_games)
                
                agent_analysis = {
                    'vs_random': vs_random,
                    'performance_metrics': self._calculate_performance_metrics(agent_name, vs_random)
                }
                
                analysis_results[agent_name] = agent_analysis
        
        return analysis_results
    
    def _calculate_performance_metrics(self, agent_name, vs_random_result):
        """Calculate detailed performance metrics for an agent."""
        win_rate = vs_random_result['agent1_win_rate']
        avg_length = vs_random_result['avg_game_length']
        
        # Performance categories
        if win_rate >= 0.70:
            performance_level = "Excellent"
        elif win_rate >= 0.60:
            performance_level = "Good"
        elif win_rate >= 0.55:
            performance_level = "Above Average"
        else:
            performance_level = "Poor"
        
        # Efficiency (shorter games might indicate better play)
        if avg_length <= 15:
            efficiency = "High"
        elif avg_length <= 20:
            efficiency = "Medium"
        else:
            efficiency = "Low"
        
        return {
            'win_rate': win_rate,
            'performance_level': performance_level,
            'efficiency': efficiency,
            'avg_game_length': avg_length
        }
    
    def generate_report(self, analysis_results):
        """Generate a comprehensive report."""
        print("\n COMPREHENSIVE EVALUATION REPORT")
        print("=" * 60)
        
        # Tournament standings
        print("\n TOURNAMENT STANDINGS")
        print("-" * 30)
        standings = analysis_results['tournament']['standings']
        sorted_agents = sorted(standings.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        
        for rank, (agent, stats) in enumerate(sorted_agents, 1):
            print(f"{rank}. {agent:12} | Win Rate: {stats['win_rate']:.1%} | "
                  f"Record: {stats['wins']}-{stats['losses']}")
        
        # Detailed performance analysis
        print("\n PERFORMANCE ANALYSIS")
        print("-" * 30)
        
        for agent_name in ['DQN', 'DMC+MCTS']:
            if agent_name in analysis_results:
                metrics = analysis_results[agent_name]['performance_metrics']
                print(f"\n{agent_name}:")
                print(f"  Win Rate vs Random: {metrics['win_rate']:.1%}")
                print(f"  Performance Level: {metrics['performance_level']}")
                print(f"  Efficiency: {metrics['efficiency']}")
                print(f"  Avg Game Length: {metrics['avg_game_length']:.1f}")
        
        # Head-to-head comparison
        print("\n HEAD-TO-HEAD COMPARISONS")
        print("-" * 30)
        
        pairwise_results = analysis_results['tournament']['pairwise_results']
        for result in pairwise_results:
            if result['agent1'] != 'Random' and result['agent2'] != 'Random':
                print(f"{result['agent1']} vs {result['agent2']}: "
                      f"{result['agent1_win_rate']:.1%} - {result['agent2_win_rate']:.1%}")
        
        # Research insights
        print("\n RESEARCH INSIGHTS")
        print("-" * 30)
        
        if 'DMC+MCTS' in analysis_results and 'DQN' in analysis_results:
            dmc_wr = analysis_results['DMC+MCTS']['performance_metrics']['win_rate']
            dqn_wr = analysis_results['DQN']['performance_metrics']['win_rate']
            
            if dmc_wr > dqn_wr:
                improvement = ((dmc_wr - dqn_wr) / dqn_wr) * 100
                print(f" DMC+MCTS outperforms DQN by {improvement:.1f}% relative improvement")
                print(" MCTS reward shaping successfully enhances learning")
            else:
                print("  DMC+MCTS did not outperform DQN baseline")
        
        # Save results
        self._save_results(analysis_results)
        
        return analysis_results
    
    def _save_results(self, analysis_results):
        """Save results to files."""
        # Create CSV with tournament results
        standings_data = []
        for agent, stats in analysis_results['tournament']['standings'].items():
            standings_data.append({
                'Agent': agent,
                'Wins': stats['wins'],
                'Losses': stats['losses'],
                'Win_Rate': stats['win_rate'],
                'Total_Games': stats['total_games']
            })
        
        df_standings = pd.DataFrame(standings_data)
        df_standings.to_csv(os.path.join(self.results_dir, 'tournament_standings.csv'), index=False)
        
        # Create CSV with pairwise results
        pairwise_data = []
        for result in analysis_results['tournament']['pairwise_results']:
            pairwise_data.append({
                'Agent_1': result['agent1'],
                'Agent_2': result['agent2'],
                'Agent_1_Wins': result['agent1_wins'],
                'Agent_2_Wins': result['agent2_wins'],
                'Agent_1_Win_Rate': result['agent1_win_rate'],
                'Agent_2_Win_Rate': result['agent2_win_rate'],
                'Avg_Game_Length': result['avg_game_length']
            })
        
        df_pairwise = pd.DataFrame(pairwise_data)
        df_pairwise.to_csv(os.path.join(self.results_dir, 'pairwise_results.csv'), index=False)
        
        print(f" Results saved to {self.results_dir}/")
    
    def create_visualizations(self, analysis_results):
        """Create visualizations of the results."""
        print(" Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Win rates bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Tournament standings
        standings = analysis_results['tournament']['standings']
        agents = list(standings.keys())
        win_rates = [standings[agent]['win_rate'] for agent in agents]
        
        bars = ax1.bar(agents, win_rates, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        ax1.set_title('Tournament Win Rates', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Win Rate')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # Performance vs Random
        if len([a for a in agents if a != 'Random']) >= 2:
            learned_agents = [a for a in agents if a != 'Random']
            vs_random_rates = []
            
            for agent in learned_agents:
                if agent in analysis_results:
                    vs_random_rates.append(
                        analysis_results[agent]['performance_metrics']['win_rate']
                    )
            
            if vs_random_rates:
                bars2 = ax2.bar(learned_agents, vs_random_rates, 
                               color=['#2ca02c', '#1f77b4'])
                ax2.set_title('Win Rate vs Random Agent', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Win Rate vs Random')
                ax2.set_ylim(0, 1)
                ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
                ax2.legend()
                
                # Add value labels
                for bar, rate in zip(bars2, vs_random_rates):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Visualizations saved to {self.results_dir}/")


def main():
    """Main function to run comprehensive evaluation."""
    print(" UNO Agents Comprehensive Evaluation")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create evaluator
    evaluator = ComprehensiveEvaluation()
    
    if len(evaluator.agents) < 2:
        print(" Need at least 2 agents for evaluation")
        return
    
    print(f"Agents loaded: {list(evaluator.agents.keys())}")
    print()
    
    # Run comprehensive analysis
    analysis_results = evaluator.detailed_analysis(num_games=2000)
    
    # Generate report
    evaluator.generate_report(analysis_results)
    
    # Create visualizations
    evaluator.create_visualizations(analysis_results)
    
    print("\n EVALUATION COMPLETED!")
    print("=" * 50)
    print("Check the 'results/' directory for detailed outputs")


if __name__ == "__main__":
    main()