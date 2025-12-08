# Statistical Analysis - Summary

## What Was Done

This document summarizes the statistical analysis performed on tournament evaluation results to validate the claims made in the research paper.

## Methodology

### Confidence Intervals
- **Method**: 95% confidence intervals using t-distribution (for n < 30) or normal distribution (for n ≥ 30)
- **Standard Error**: Computed as SE = √(p(1-p)/n) for win rate proportions
- **Formula**: CI = p ± t_{α/2, df} × SE

### Significance Testing
- **Test**: Z-test for difference in proportions (two independent samples)
- **Null Hypothesis**: H₀: p₁ = p₂ (no difference in win rates)
- **Significance Level**: α = 0.05 (p < 0.05 indicates significance)
- **Two-tailed**: Tests whether agents differ in either direction

### Sample Sizes
- **2-Player**: 28,000 games (7,000 per agent)
- **3-Player**: 56,000+ games (21,000 per agent)
- **4-Player**: 70,000 games (35,000 per agent)
- **Total**: 154,000 games

## Key Findings

### 1. DRON Scalability 
- **2-Player**: 53.4% [52.2%, 54.6%]
- **3-Player**: 53.2% [52.5%, 53.9%]
- **4-Player**: 54.2% [53.6%, 54.7%]
- **Scaling**: +0.8% (slight improvement, confidence intervals overlap)
- **Conclusion**: Performance remains stable/improves with complexity

### 2. DMC+MCTS Resilience
- **2-Player**: 45.9% [44.7%, 47.1%]
- **3-Player**: 47.3% [46.6%, 48.0%]
- **4-Player**: 47.4% [46.9%, 48.0%]
- **Scaling**: +1.5% (positive trend)
- **Conclusion**: MCTS provides algorithmic resilience

### 3. RLCard DMC Catastrophic Failure
- **2-Player**: 57.7% [56.5%, 58.8%]
- **3-Player**: 30.5% [29.8%, 31.1%]
- **4-Player**: 8.3% [8.0%, 8.6%]
- **Scaling**: -49.4% (dramatic collapse)
- **Conclusion**: Highly optimized implementation fails in complex multi-agent settings

### 4. Statistical Significance
- **2-Player**: 20/28 comparisons significant (p < 0.05)
- **3-Player**: 28/28 comparisons significant (p < 0.05)
- **4-Player**: 27/28 comparisons significant (p < 0.05)
- **Conclusion**: Virtually all performance differences are statistically significant

## Generated Files

All statistical analysis results are saved in `results/statistics/`:

### CSV Files (for further analysis)
- `agent_statistics_2player.csv` - Per-agent stats with CIs for 2-player
- `agent_statistics_3player.csv` - Per-agent stats with CIs for 3-player
- `agent_statistics_4player.csv` - Per-agent stats with CIs for 4-player
- `pairwise_comparisons_2player.csv` - All pairwise significance tests for 2-player
- `pairwise_comparisons_3player.csv` - All pairwise significance tests for 3-player
- `pairwise_comparisons_4player.csv` - All pairwise significance tests for 4-player
- `cross_complexity_summary.csv` - Scaling analysis across all complexity levels

### Text Reports
- `statistical_report_2player.txt` - Complete 2-player statistical report
- `statistical_report_3player.txt` - Complete 3-player statistical report
- `statistical_report_4player.txt` - Complete 4-player statistical report
- `combined_statistical_analysis.txt` - All reports combined

### Documentation
- `docs/STATISTICAL_ANALYSIS_REPORT.md` - Comprehensive markdown report with interpretation

## How to Reproduce

Run the statistical analysis script:

```bash
python scripts/compute_tournament_statistics.py
```

This will:
1. Load tournament results from `results/tournament_{2,3,4}player_results.json`
2. Compute 95% confidence intervals for all agents
3. Perform pairwise significance tests
4. Generate all CSV and text reports
5. Create cross-complexity summary