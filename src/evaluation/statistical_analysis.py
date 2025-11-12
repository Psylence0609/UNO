"""
Statistical analysis utilities for model evaluation.
Provides functions for computing confidence intervals, statistical tests, and effect sizes.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional
import pandas as pd


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a sample of data.
    
    Args:
        data: Array of sample values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    
    # Use t-distribution for small samples, normal for large samples
    if n < 30:
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_critical * std_err
    else:
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin = z_critical * std_err
    
    lower = mean - margin
    upper = mean + margin
    
    return mean, lower, upper


def independent_t_test(sample1: np.ndarray, sample2: np.ndarray) -> Dict[str, float]:
    """
    Perform independent t-test between two samples.
    
    Args:
        sample1: First sample
        sample2: Second sample
        
    Returns:
        Dictionary with test statistics:
        - t_statistic: t-statistic value
        - p_value: p-value
        - df: degrees of freedom
        - mean_diff: difference in means
        - significant: boolean indicating if p < 0.05
    """
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    
    # Calculate degrees of freedom (Welch's t-test)
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    n1 = len(sample1)
    n2 = len(sample2)
    
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    mean_diff = np.mean(sample1) - np.mean(sample2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'df': df,
        'mean_diff': mean_diff,
        'significant': p_value < 0.05,
        'highly_significant': p_value < 0.01
    }


def mann_whitney_u_test(sample1: np.ndarray, sample2: np.ndarray) -> Dict[str, float]:
    """
    Perform Mann-Whitney U test (non-parametric test) between two samples.
    
    Args:
        sample1: First sample
        sample2: Second sample
        
    Returns:
        Dictionary with test statistics:
        - u_statistic: U-statistic value
        - p_value: p-value
        - significant: boolean indicating if p < 0.05
    """
    u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
    
    return {
        'u_statistic': u_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'highly_significant': p_value < 0.01
    }


def cohens_d(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size between two samples.
    
    Cohen's d interpretation:
    - |d| < 0.2: Negligible effect
    - 0.2 <= |d| < 0.5: Small effect
    - 0.5 <= |d| < 0.8: Medium effect
    - |d| >= 0.8: Large effect
    
    Args:
        sample1: First sample
        sample2: Second sample
        
    Returns:
        Cohen's d value
    """
    n1 = len(sample1)
    n2 = len(sample2)
    
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean1 - mean2) / pooled_std
    return d


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        String interpretation
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def compare_models_multiple_runs(
    results_dict: Dict[str, List[float]],
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Compare multiple models with multiple evaluation runs.
    
    Args:
        results_dict: Dictionary mapping model names to lists of win rates from multiple runs
        confidence: Confidence level for intervals
        
    Returns:
        DataFrame with statistical summary for each model
    """
    summary_data = []
    
    for model_name, win_rates in results_dict.items():
        win_rates_array = np.array(win_rates)
        
        mean, lower, upper = compute_confidence_interval(win_rates_array, confidence)
        std = np.std(win_rates_array)
        n = len(win_rates_array)
        
        summary_data.append({
            'Model': model_name,
            'Mean': mean,
            'Std': std,
            'CI_Lower': lower,
            'CI_Upper': upper,
            'N_Runs': n,
            'Min': np.min(win_rates_array),
            'Max': np.max(win_rates_array)
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Mean', ascending=False)
    
    return df


def pairwise_comparisons(
    results_dict: Dict[str, List[float]],
    test_type: str = 'both'
) -> pd.DataFrame:
    """
    Perform pairwise statistical comparisons between all models.
    
    Args:
        results_dict: Dictionary mapping model names to lists of win rates
        test_type: 't-test', 'mann-whitney', or 'both'
        
    Returns:
        DataFrame with pairwise comparison results
    """
    model_names = list(results_dict.keys())
    comparisons = []
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:
                continue
            
            sample1 = np.array(results_dict[model1])
            sample2 = np.array(results_dict[model2])
            
            comparison = {
                'Model_1': model1,
                'Model_2': model2,
                'Mean_1': np.mean(sample1),
                'Mean_2': np.mean(sample2),
                'Mean_Diff': np.mean(sample1) - np.mean(sample2)
            }
            
            # Perform statistical tests
            if test_type in ['t-test', 'both']:
                t_test_result = independent_t_test(sample1, sample2)
                comparison['T_Statistic'] = t_test_result['t_statistic']
                comparison['T_P_Value'] = t_test_result['p_value']
                comparison['T_Significant'] = t_test_result['significant']
            
            if test_type in ['mann-whitney', 'both']:
                u_test_result = mann_whitney_u_test(sample1, sample2)
                comparison['U_Statistic'] = u_test_result['u_statistic']
                comparison['U_P_Value'] = u_test_result['p_value']
                comparison['U_Significant'] = u_test_result['significant']
            
            # Calculate effect size
            d = cohens_d(sample1, sample2)
            comparison['Cohens_D'] = d
            comparison['Effect_Size'] = interpret_effect_size(d)
            
            comparisons.append(comparison)
    
    df = pd.DataFrame(comparisons)
    return df


def generate_statistical_summary(
    results_dict: Dict[str, List[float]],
    confidence: float = 0.95
) -> Dict:
    """
    Generate comprehensive statistical summary for model comparison.
    
    Args:
        results_dict: Dictionary mapping model names to lists of win rates
        confidence: Confidence level for intervals
        
    Returns:
        Dictionary with comprehensive statistical summary
    """
    summary = {
        'model_summary': compare_models_multiple_runs(results_dict, confidence),
        'pairwise_comparisons': pairwise_comparisons(results_dict, test_type='both')
    }
    
    return summary


def format_statistical_results(summary: Dict) -> str:
    """
    Format statistical summary as a readable string.
    
    Args:
        summary: Statistical summary dictionary
        
    Returns:
        Formatted string
    """
    output = []
    output.append("=" * 80)
    output.append("STATISTICAL ANALYSIS SUMMARY")
    output.append("=" * 80)
    
    # Model summary
    output.append("\nMODEL PERFORMANCE SUMMARY (with 95% Confidence Intervals)")
    output.append("-" * 80)
    df_summary = summary['model_summary']
    for _, row in df_summary.iterrows():
        output.append(
            f"{row['Model']:30s} | "
            f"Mean: {row['Mean']:.4f} | "
            f"CI: [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}] | "
            f"Std: {row['Std']:.4f} | "
            f"N: {int(row['N_Runs'])}"
        )
    
    # Pairwise comparisons
    output.append("\nPAIRWISE COMPARISONS")
    output.append("-" * 80)
    df_comparisons = summary['pairwise_comparisons']
    
    for _, row in df_comparisons.iterrows():
        output.append(f"\n{row['Model_1']} vs {row['Model_2']}")
        output.append(f"  Mean Difference: {row['Mean_Diff']:.4f}")
        output.append(f"  T-test: p = {row['T_P_Value']:.4f} {'***' if row['T_Significant'] else ''}")
        output.append(f"  Mann-Whitney U: p = {row['U_P_Value']:.4f} {'***' if row['U_Significant'] else ''}")
        output.append(f"  Cohen's d: {row['Cohens_D']:.4f} ({row['Effect_Size']} effect)")
    
    output.append("\n" + "=" * 80)
    output.append("*** indicates p < 0.05 (statistically significant)")
    output.append("=" * 80)
    
    return "\n".join(output)

