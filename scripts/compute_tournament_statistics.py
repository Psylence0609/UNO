"""
Compute statistical analysis for tournament results.
Calculates 95% confidence intervals, performs significance tests, and generates detailed reports.
"""

import json
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path


class TournamentStatisticalAnalyzer:
    """Performs comprehensive statistical analysis on tournament results."""
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer with tournament results.
        
        Args:
            results_file: Path to tournament results JSON file
        """
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.overall_results = self.data.get('overall', {})
        self.matchup_results = self.data.get('matchups', {})
        self.num_players = self._infer_num_players()
    
    def _infer_num_players(self) -> int:
        """Infer number of players from matchup data."""
        for matchup_key in self.matchup_results.keys():
            try:
                matchup = eval(matchup_key)
                return len(matchup)
            except:
                pass
        return 0
    
    def compute_win_rate_confidence_interval(
        self, 
        wins: int, 
        games: int, 
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute confidence interval for win rate (proportion).
        Uses Wilson score interval for better coverage with proportions.
        
        Args:
            wins: Number of wins
            games: Total number of games
            confidence: Confidence level (default 0.95)
        
        Returns:
            (win_rate, lower_bound, upper_bound)
        """
        if games == 0:
            return 0.0, 0.0, 0.0
        
        p = wins / games
        n = games
        
        # For large samples, we can also use normal approximation with t-distribution
        # Treating each game as a Bernoulli trial
        z = stats.t.ppf((1 + confidence) / 2, df=n-1) if n < 30 else stats.norm.ppf((1 + confidence) / 2)
        
        # Standard error for proportion
        se = np.sqrt(p * (1 - p) / n)
        
        # Confidence interval
        margin = z * se
        lower = max(0.0, p - margin)
        upper = min(1.0, p + margin)
        
        return p, lower, upper
    
    def compute_agent_statistics(self) -> pd.DataFrame:
        """
        Compute comprehensive statistics for each agent.
        
        Returns:
            DataFrame with agent statistics including confidence intervals
        """
        stats_data = []
        
        for agent_name, agent_data in self.overall_results.items():
            wins = agent_data.get('wins', 0)
            games = agent_data.get('games', 0)
            
            if games == 0:
                continue
            
            # Compute confidence interval for win rate
            win_rate, ci_lower, ci_upper = self.compute_win_rate_confidence_interval(wins, games)
            
            # Standard error
            se = np.sqrt(win_rate * (1 - win_rate) / games)
            
            stats_data.append({
                'Agent': agent_name,
                'Wins': wins,
                'Games': games,
                'Win_Rate': win_rate,
                'CI_Lower_95': ci_lower,
                'CI_Upper_95': ci_upper,
                'SE': se,
                'Avg_Placement': agent_data.get('avg_placement', np.nan)
            })
        
        df = pd.DataFrame(stats_data)
        df = df.sort_values('Win_Rate', ascending=False)
        
        return df
    
    def pairwise_significance_test(
        self, 
        agent1_stats: Dict,
        agent2_stats: Dict
    ) -> Dict:
        """
        Perform pairwise significance test between two agents.
        Uses z-test for proportions.
        
        Args:
            agent1_stats: Statistics for agent 1 (wins, games, win_rate)
            agent2_stats: Statistics for agent 2 (wins, games, win_rate)
        
        Returns:
            Dictionary with test results
        """
        w1, n1, p1 = agent1_stats['wins'], agent1_stats['games'], agent1_stats['win_rate']
        w2, n2, p2 = agent2_stats['wins'], agent2_stats['games'], agent2_stats['win_rate']
        
        if n1 == 0 or n2 == 0:
            return {
                'z_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'mean_diff': 0.0
            }
        
        # Pooled proportion
        p_pool = (w1 + w2) / (n1 + n2)
        
        # Standard error for difference in proportions
        se_diff = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        if se_diff == 0:
            return {
                'z_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'mean_diff': p1 - p2
            }
        
        # Z-statistic for difference in proportions
        z_stat = (p1 - p2) / se_diff
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'highly_significant': p_value < 0.01,
            'mean_diff': p1 - p2
        }
    
    def compute_pairwise_comparisons(self, agent_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all pairwise comparisons between agents.
        
        Args:
            agent_stats_df: DataFrame with agent statistics
        
        Returns:
            DataFrame with pairwise comparison results
        """
        comparisons = []
        
        agents = agent_stats_df['Agent'].tolist()
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i >= j:
                    continue
                
                agent1_data = agent_stats_df[agent_stats_df['Agent'] == agent1].iloc[0]
                agent2_data = agent_stats_df[agent_stats_df['Agent'] == agent2].iloc[0]
                
                stats1 = {
                    'wins': agent1_data['Wins'],
                    'games': agent1_data['Games'],
                    'win_rate': agent1_data['Win_Rate']
                }
                
                stats2 = {
                    'wins': agent2_data['Wins'],
                    'games': agent2_data['Games'],
                    'win_rate': agent2_data['Win_Rate']
                }
                
                test_result = self.pairwise_significance_test(stats1, stats2)
                
                comparisons.append({
                    'Agent_1': agent1,
                    'Agent_2': agent2,
                    'Win_Rate_1': stats1['win_rate'],
                    'Win_Rate_2': stats2['win_rate'],
                    'Difference': test_result['mean_diff'],
                    'Z_Statistic': test_result['z_statistic'],
                    'P_Value': test_result['p_value'],
                    'Significant': test_result['significant'],
                    'Highly_Significant': test_result.get('highly_significant', False)
                })
        
        return pd.DataFrame(comparisons)
    
    def compute_head_to_head_statistics(self, agent1_name: str, agent2_name: str = None) -> pd.DataFrame:
        """
        Compute head-to-head statistics for an agent against all others.
        
        Args:
            agent1_name: Name of the primary agent
            agent2_name: Optional name of specific opponent (None for all opponents)
        
        Returns:
            DataFrame with head-to-head statistics
        """
        h2h_stats = []
        
        for matchup_key, matchup_data in self.matchup_results.items():
            try:
                matchup = eval(matchup_key)
            except:
                continue
            
            if agent1_name not in matchup:
                continue
            
            if agent2_name and agent2_name not in matchup:
                continue
            
            # Get agent1's stats in this matchup
            if agent1_name in matchup_data:
                agent1_stats = matchup_data[agent1_name]
                wins = agent1_stats['wins']
                
                # Infer number of games from win rate
                if agent1_stats['win_rate'] > 0:
                    games = int(wins / agent1_stats['win_rate'])
                else:
                    games = 0
                
                # Compute CI for this matchup
                win_rate, ci_lower, ci_upper = self.compute_win_rate_confidence_interval(wins, games)
                
                # Get opponent names
                opponents = [name for name in matchup if name != agent1_name]
                
                h2h_stats.append({
                    'Opponent': ', '.join(opponents) if len(opponents) > 1 else opponents[0] if opponents else 'N/A',
                    'Wins': wins,
                    'Games': games,
                    'Win_Rate': win_rate,
                    'CI_Lower_95': ci_lower,
                    'CI_Upper_95': ci_upper
                })
        
        df = pd.DataFrame(h2h_stats)
        if not df.empty:
            df = df.sort_values('Win_Rate', ascending=False)
        
        return df
    
    def generate_report(self) -> str:
        """
        Generate comprehensive statistical report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 100)
        report.append(f"STATISTICAL ANALYSIS REPORT: {self.num_players}-PLAYER TOURNAMENT")
        report.append("=" * 100)
        report.append("")
        
        # Compute agent statistics
        agent_stats = self.compute_agent_statistics()
        
        report.append("OVERALL AGENT PERFORMANCE (with 95% Confidence Intervals)")
        report.append("-" * 100)
        report.append(f"{'Rank':<6} {'Agent':<30} {'Win Rate':<12} {'95% CI':<25} {'Wins':<15} {'Avg Place':<12}")
        report.append("-" * 100)
        
        for idx, row in agent_stats.iterrows():
            rank = idx + 1 if idx == 0 else idx
            ci_str = f"[{row['CI_Lower_95']:.1%}, {row['CI_Upper_95']:.1%}]"
            wins_str = f"{row['Wins']}/{row['Games']}"
            report.append(
                f"{rank:<6} {row['Agent']:<30} {row['Win_Rate']:<12.1%} {ci_str:<25} "
                f"{wins_str:<15} {row['Avg_Placement']:<12.2f}"
            )
        
        report.append("")
        report.append("=" * 100)
        report.append("PAIRWISE STATISTICAL COMPARISONS")
        report.append("=" * 100)
        report.append("")
        
        # Compute pairwise comparisons
        comparisons = self.compute_pairwise_comparisons(agent_stats)
        
        # Show significant comparisons
        significant_comparisons = comparisons[comparisons['Significant'] == True]
        
        report.append(f"Total comparisons: {len(comparisons)}")
        report.append(f"Statistically significant differences (p < 0.05): {len(significant_comparisons)}")
        report.append("")
        report.append("TOP 20 MOST SIGNIFICANT DIFFERENCES:")
        report.append("-" * 100)
        report.append(f"{'Agent 1':<25} {'Agent 2':<25} {'Diff':<12} {'Z-stat':<10} {'P-value':<12} {'Sig':<6}")
        report.append("-" * 100)
        
        top_comparisons = comparisons.sort_values('P_Value').head(20)
        for _, row in top_comparisons.iterrows():
            sig_marker = "***" if row['Highly_Significant'] else ("**" if row['Significant'] else "")
            report.append(
                f"{row['Agent_1']:<25.24} {row['Agent_2']:<25.24} "
                f"{row['Difference']:>11.1%} {row['Z_Statistic']:>9.2f} "
                f"{row['P_Value']:>11.4f} {sig_marker:<6}"
            )
        
        report.append("")
        report.append("Legend: *** p < 0.01 (highly significant), ** p < 0.05 (significant)")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_statistics_to_csv(self, output_dir: str = 'results/statistics'):
        """
        Save statistical analysis to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Agent statistics
        agent_stats = self.compute_agent_statistics()
        agent_stats_file = f"{output_dir}/agent_statistics_{self.num_players}player.csv"
        agent_stats.to_csv(agent_stats_file, index=False)
        print(f"✓ Saved agent statistics to {agent_stats_file}")
        
        # Pairwise comparisons
        comparisons = self.compute_pairwise_comparisons(agent_stats)
        comparisons_file = f"{output_dir}/pairwise_comparisons_{self.num_players}player.csv"
        comparisons.to_csv(comparisons_file, index=False)
        print(f"✓ Saved pairwise comparisons to {comparisons_file}")


def analyze_all_tournaments():
    """Analyze all tournament results (2, 3, and 4 players)."""
    
    print("\n" + "=" * 100)
    print("TOURNAMENT STATISTICAL ANALYSIS")
    print("=" * 100)
    print("\nThis script computes:")
    print("  • 95% confidence intervals for win rates (using t-distribution)")
    print("  • Z-tests for pairwise significance testing")
    print("  • Comprehensive statistical reports")
    print("=" * 100)
    print()
    
    tournament_files = [
        'results/tournament_2player_results.json',
        'results/tournament_3player_results.json',
        'results/tournament_4player_results.json'
    ]
    
    all_reports = []
    
    for results_file in tournament_files:
        if not Path(results_file).exists():
            print(f"⚠ Skipping {results_file} (not found)")
            continue
        
        print(f"\n{'='*100}")
        print(f"Analyzing: {results_file}")
        print(f"{'='*100}")
        
        analyzer = TournamentStatisticalAnalyzer(results_file)
        
        # Generate report
        report = analyzer.generate_report()
        all_reports.append(report)
        print(report)
        
        # Save CSV files
        analyzer.save_statistics_to_csv()
        
        # Save text report
        report_file = f"results/statistics/statistical_report_{analyzer.num_players}player.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Saved report to {report_file}")
    
    # Save combined report
    combined_report_file = "results/statistics/combined_statistical_analysis.txt"
    with open(combined_report_file, 'w') as f:
        f.write("\n\n".join(all_reports))
    print(f"\n{'='*100}")
    print(f"✓ Saved combined report to {combined_report_file}")
    print(f"{'='*100}")
    
    # Generate summary table for all complexity levels
    print("\n" + "="*100)
    print("CROSS-COMPLEXITY SUMMARY")
    print("="*100)
    print()
    
    summary_data = []
    for results_file in tournament_files:
        if not Path(results_file).exists():
            continue
        
        analyzer = TournamentStatisticalAnalyzer(results_file)
        agent_stats = analyzer.compute_agent_statistics()
        
        for _, row in agent_stats.iterrows():
            summary_data.append({
                'Players': analyzer.num_players,
                'Agent': row['Agent'],
                'Win_Rate': row['Win_Rate'],
                'CI_Lower': row['CI_Lower_95'],
                'CI_Upper': row['CI_Upper_95'],
                'Games': row['Games']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Pivot to show agents across different player counts
    pivot_data = []
    for agent in summary_df['Agent'].unique():
        agent_data = {'Agent': agent}
        for players in [2, 3, 4]:
            player_data = summary_df[(summary_df['Agent'] == agent) & (summary_df['Players'] == players)]
            if not player_data.empty:
                row = player_data.iloc[0]
                agent_data[f'{players}P_WinRate'] = row['Win_Rate']
                agent_data[f'{players}P_CI'] = f"[{row['CI_Lower']:.1%}, {row['CI_Upper']:.1%}]"
            else:
                agent_data[f'{players}P_WinRate'] = np.nan
                agent_data[f'{players}P_CI'] = 'N/A'
        
        # Calculate scaling (2-player to 4-player change)
        if not np.isnan(agent_data.get('2P_WinRate', np.nan)) and not np.isnan(agent_data.get('4P_WinRate', np.nan)):
            agent_data['Scaling_2to4'] = agent_data['4P_WinRate'] - agent_data['2P_WinRate']
        else:
            agent_data['Scaling_2to4'] = np.nan
        
        pivot_data.append(agent_data)
    
    pivot_df = pd.DataFrame(pivot_data)
    pivot_df = pivot_df.sort_values('4P_WinRate', ascending=False)
    
    print(f"{'Agent':<30} {'2-Player':<20} {'3-Player':<20} {'4-Player':<20} {'Scaling':<12}")
    print("-" * 105)
    for _, row in pivot_df.iterrows():
        wr_2p = f"{row['2P_WinRate']:.1%}" if not np.isnan(row['2P_WinRate']) else "N/A"
        wr_3p = f"{row['3P_WinRate']:.1%}" if not np.isnan(row['3P_WinRate']) else "N/A"
        wr_4p = f"{row['4P_WinRate']:.1%}" if not np.isnan(row['4P_WinRate']) else "N/A"
        scaling = f"{row['Scaling_2to4']:+.1%}" if not np.isnan(row['Scaling_2to4']) else "N/A"
        
        print(f"{row['Agent']:<30} {wr_2p:<20} {wr_3p:<20} {wr_4p:<20} {scaling:<12}")
    
    # Save cross-complexity summary
    summary_file = "results/statistics/cross_complexity_summary.csv"
    pivot_df.to_csv(summary_file, index=False)
    print(f"\n✓ Saved cross-complexity summary to {summary_file}")
    
    print("\n" + "="*100)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*100)
    print("\nGenerated files:")
    print("  • results/statistics/agent_statistics_{2,3,4}player.csv")
    print("  • results/statistics/pairwise_comparisons_{2,3,4}player.csv")
    print("  • results/statistics/statistical_report_{2,3,4}player.txt")
    print("  • results/statistics/combined_statistical_analysis.txt")
    print("  • results/statistics/cross_complexity_summary.csv")
    print("="*100)


if __name__ == "__main__":
    analyze_all_tournaments()

