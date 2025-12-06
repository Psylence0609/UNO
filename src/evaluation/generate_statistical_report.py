#!/usr/bin/env python3
"""
Generate statistical analysis report from evaluation results.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append('.')

from src.evaluation.statistical_analysis import (
    generate_statistical_summary,
    format_statistical_results
)


def load_evaluation_results(results_file='results/model_evaluation_results.csv'):
    """
    Load evaluation results from CSV file.
    
    Args:
        results_file: Path to results CSV file
        
    Returns:
        Dictionary mapping model names to lists of win rates
    """
    if not os.path.exists(results_file):
        print(f"  Results file not found: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    
    # Check if we have multiple runs or single evaluation
    if 'N Runs' in df.columns:
        # We have statistical summary, need to reconstruct win rates
        # For now, return None and use summary data directly
        return None
    
    # If we have individual run data, group by model
    # For now, assume we're using the comprehensive evaluation script output
    return None


def generate_report_from_summary(
    summary_file='results/statistical_summary.csv',
    pairwise_file='results/pairwise_comparisons.csv',
    output_file='results/statistical_analysis_report.md'
):
    """
    Generate markdown report from statistical summary files.
    
    Args:
        summary_file: Path to statistical summary CSV
        pairwise_file: Path to pairwise comparisons CSV
        output_file: Path to output markdown file
    """
    os.makedirs('results', exist_ok=True)
    
    report = []
    report.append("# Statistical Analysis Report")
    report.append("")
    report.append("This report provides comprehensive statistical analysis of all model evaluations.")
    report.append("")
    report.append("---")
    report.append("")
    
    # Load summary data
    if os.path.exists(summary_file):
        df_summary = pd.read_csv(summary_file)
        
        report.append("## Model Performance Summary")
        report.append("")
        report.append("### Performance Metrics with 95% Confidence Intervals")
        report.append("")
        report.append("| Model | Mean Win Rate | Std Dev | 95% CI Lower | 95% CI Upper | Min | Max | N Runs |")
        report.append("|-------|---------------|---------|--------------|--------------|-----|-----|--------|")
        
        for _, row in df_summary.iterrows():
            report.append(
                f"| {row['Model']} | {row['Mean']:.4f} | {row['Std']:.4f} | "
                f"{row['CI_Lower']:.4f} | {row['CI_Upper']:.4f} | "
                f"{row['Min']:.4f} | {row['Max']:.4f} | {int(row['N_Runs'])} |"
            )
        
        report.append("")
        report.append("### Key Findings")
        report.append("")
        
        # Find best and worst models
        best_model = df_summary.loc[df_summary['Mean'].idxmax()]
        worst_model = df_summary.loc[df_summary['Mean'].idxmin()]
        
        report.append(f"1. **Best Performing Model**: {best_model['Model']} with {best_model['Mean']:.1%} win rate (95% CI: [{best_model['CI_Lower']:.1%}, {best_model['CI_Upper']:.1%}])")
        report.append(f"2. **Worst Performing Model**: {worst_model['Model']} with {worst_model['Mean']:.1%} win rate (95% CI: [{worst_model['CI_Lower']:.1%}, {worst_model['CI_Upper']:.1%}])")
        
        # Models meeting 55% threshold
        meeting_55 = df_summary[df_summary['CI_Lower'] >= 0.55]
        if len(meeting_55) > 0:
            report.append(f"3. **Models Meeting 55% Threshold**: {len(meeting_55)} model(s)")
            for _, row in meeting_55.iterrows():
                report.append(f"   - {row['Model']}: {row['Mean']:.1%} (CI: [{row['CI_Lower']:.1%}, {row['CI_Upper']:.1%}])")
        else:
            report.append("3. **Models Meeting 55% Threshold**: None (using conservative CI lower bound >= 55%)")
        
        report.append("")
    
    # Load pairwise comparisons
    if os.path.exists(pairwise_file):
        df_pairwise = pd.read_csv(pairwise_file)
        
        report.append("## Pairwise Statistical Comparisons")
        report.append("")
        report.append("### Statistical Significance Tests")
        report.append("")
        report.append("Comparisons include:")
        report.append("- Independent t-test (parametric)")
        report.append("- Mann-Whitney U test (non-parametric)")
        report.append("- Cohen's d effect size")
        report.append("")
        report.append("| Model 1 | Model 2 | Mean Diff | T-test p-value | U-test p-value | Cohen's d | Effect Size | Significant? |")
        report.append("|---------|---------|-----------|----------------|----------------|-----------|-------------|--------------|")
        
        for _, row in df_pairwise.iterrows():
            t_sig = "***" if row['T_Significant'] else ""
            u_sig = "***" if row['U_Significant'] else ""
            sig_mark = "Yes" if (row['T_Significant'] or row['U_Significant']) else "No"
            
            report.append(
                f"| {row['Model_1']} | {row['Model_2']} | {row['Mean_Diff']:.4f} | "
                f"{row['T_P_Value']:.4f}{t_sig} | {row['U_P_Value']:.4f}{u_sig} | "
                f"{row['Cohens_D']:.4f} | {row['Effect_Size']} | {sig_mark} |"
            )
        
        report.append("")
        report.append("*** indicates p < 0.05 (statistically significant)")
        report.append("")
        
        # Summary of significant differences
        significant_comparisons = df_pairwise[
            (df_pairwise['T_Significant']) | (df_pairwise['U_Significant'])
        ]
        
        if len(significant_comparisons) > 0:
            report.append("### Significant Differences")
            report.append("")
            for _, row in significant_comparisons.iterrows():
                direction = ">" if row['Mean_Diff'] > 0 else "<"
                report.append(
                    f"- {row['Model_1']} {direction} {row['Model_2']}: "
                    f"Mean difference = {row['Mean_Diff']:.4f}, "
                    f"Effect size = {row['Effect_Size']} (Cohen's d = {row['Cohens_D']:.4f})"
                )
            report.append("")
    
    # Performance gap analysis
    if os.path.exists(summary_file):
        df_summary = pd.read_csv(summary_file)
        
        report.append("## Performance Gap Analysis")
        report.append("")
        report.append("### RLCard vs Custom Models")
        report.append("")
        
        rlcard_models = df_summary[df_summary['Model'].str.contains('RLCard', na=False)]
        custom_models = df_summary[~df_summary['Model'].str.contains('RLCard', na=False)]
        
        if len(rlcard_models) > 0 and len(custom_models) > 0:
            best_rlcard = rlcard_models.loc[rlcard_models['Mean'].idxmax()]
            best_custom = custom_models.loc[custom_models['Mean'].idxmax()]
            
            gap = best_rlcard['Mean'] - best_custom['Mean']
            gap_pct = gap * 100
            
            report.append(f"**Performance Gap**:")
            report.append(f"- Best RLCard Model: {best_rlcard['Model']} ({best_rlcard['Mean']:.1%})")
            report.append(f"- Best Custom Model: {best_custom['Model']} ({best_custom['Mean']:.1%})")
            report.append(f"- Gap: {gap:.1%} ({gap_pct:.2f} percentage points)")
            report.append("")
            report.append("**Analysis**:")
            if gap > 0.05:
                report.append(f"- RLCard models significantly outperform custom models ({gap:.1%} gap)")
                report.append("- This suggests that parallel training architecture is more important than network complexity")
            elif gap > 0.02:
                report.append(f"- RLCard models moderately outperform custom models ({gap:.1%} gap)")
            else:
                report.append(f"- Performance is similar between RLCard and custom models ({gap:.1%} gap)")
            report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("### For Best Performance")
    report.append("")
    if os.path.exists(summary_file):
        df_summary = pd.read_csv(summary_file)
        best_model = df_summary.loc[df_summary['Mean'].idxmax()]
        report.append(f"1. **Use {best_model['Model']}**: Achieves {best_model['Mean']:.1%} win rate with 95% CI [{best_model['CI_Lower']:.1%}, {best_model['CI_Upper']:.1%}]")
    report.append("2. **Evaluate RLCard Models**: Confirm performance with comprehensive evaluation")
    report.append("3. **Consider Retraining**: Custom models below 55% threshold may benefit from:")
    report.append("   - More training episodes")
    report.append("   - Better hyperparameter tuning")
    report.append("   - Parallel training architecture")
    report.append("")
    
    report.append("### For Research")
    report.append("")
    report.append("1. **Custom DMC**: Best custom model with three-headed architecture")
    report.append("2. **MCTS Integration**: Investigate why MCTS reward shaping provides only modest gains")
    report.append("3. **Checkpoint Selection**: Study why some checkpoints outperform final checkpoints")
    report.append("4. **Opponent Modeling**: Train and evaluate DMC with opponent modeling")
    report.append("")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f" Statistical analysis report generated: {output_file}")
    return output_file


def main():
    """Main function to generate statistical report."""
    print("=" * 80)
    print(" GENERATING STATISTICAL ANALYSIS REPORT")
    print("=" * 80)
    
    # Check if summary files exist
    summary_file = 'results/statistical_summary.csv'
    pairwise_file = 'results/pairwise_comparisons.csv'
    
    if not os.path.exists(summary_file):
        print(f"  Statistical summary file not found: {summary_file}")
        print("   Please run evaluate_all_models_comprehensive.py first")
        return
    
    # Generate report
    output_file = generate_report_from_summary(summary_file, pairwise_file)
    
    print(f"\n Report generated successfully: {output_file}")
    print("\n Next steps:")
    print("   1. Review the statistical analysis report")
    print("   2. Run RLCard model evaluations if not already done")
    print("   3. Train and evaluate opponent modeling model")
    print("   4. Update documentation with final results")


if __name__ == "__main__":
    main()

