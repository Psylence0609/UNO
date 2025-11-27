# Statistical Analysis Report

This report provides comprehensive statistical analysis of all model evaluations.

---

## Model Performance Summary

### Performance Metrics with 95% Confidence Intervals

| Model | Mean Win Rate | Std Dev | 95% CI Lower | 95% CI Upper | Min | Max | N Runs |
|-------|---------------|---------|--------------|--------------|-----|-----|--------|
| RLCard DMC+MCTS (100M) | 0.6134 | 0.0227 | 0.5963 | 0.6305 | 0.5760 | 0.6420 | 10 |
| RLCard DMC (100M) | 0.6038 | 0.0149 | 0.5925 | 0.6151 | 0.5660 | 0.6240 | 10 |
| DQN+MCTS | 0.5116 | 0.0168 | 0.4989 | 0.5243 | 0.4780 | 0.5360 | 10 |
| DMC Ep10k | 0.5004 | 0.0179 | 0.4869 | 0.5139 | 0.4720 | 0.5360 | 10 |
| DQN Original | 0.4978 | 0.0209 | 0.4820 | 0.5136 | 0.4700 | 0.5360 | 10 |
| DMC Old | 0.4742 | 0.0151 | 0.4628 | 0.4856 | 0.4520 | 0.4960 | 10 |

### Key Findings

1. **Best Performing Model**: RLCard DMC+MCTS (100M) with 61.3% win rate (95% CI: [59.6%, 63.0%])
2. **Worst Performing Model**: DMC Old with 47.4% win rate (95% CI: [46.3%, 48.6%])
3. **Models Meeting 55% Threshold**: 2 model(s)
   - RLCard DMC+MCTS (100M): 61.3% (CI: [59.6%, 63.0%])
   - RLCard DMC (100M): 60.4% (CI: [59.3%, 61.5%])

## Pairwise Statistical Comparisons

### Statistical Significance Tests

Comparisons include:
- Independent t-test (parametric)
- Mann-Whitney U test (non-parametric)
- Cohen's d effect size

| Model 1 | Model 2 | Mean Diff | T-test p-value | U-test p-value | Cohen's d | Effect Size | Significant? |
|---------|---------|-----------|----------------|----------------|-----------|-------------|--------------|
| DQN Original | DQN+MCTS | -0.0138 | 0.1403 | 0.1298 | -0.6899 | Medium | No |
| DQN Original | DMC Old | 0.0236 | 0.0133*** | 0.0374*** | 1.2273 | Large | Yes |
| DQN Original | DMC Ep10k | -0.0026 | 0.7800 | 0.7910 | -0.1268 | Negligible | No |
| DQN Original | RLCard DMC (100M) | -0.1060 | 0.0000*** | 0.0002*** | -5.5360 | Large | Yes |
| DQN Original | RLCard DMC+MCTS (100M) | -0.1156 | 0.0000*** | 0.0002*** | -5.0298 | Large | Yes |
| DQN+MCTS | DMC Old | 0.0374 | 0.0001*** | 0.0010*** | 2.2188 | Large | Yes |
| DQN+MCTS | DMC Ep10k | 0.0112 | 0.1880 | 0.1613 | 0.6120 | Medium | No |
| DQN+MCTS | RLCard DMC (100M) | -0.0922 | 0.0000*** | 0.0002*** | -5.5002 | Large | Yes |
| DQN+MCTS | RLCard DMC+MCTS (100M) | -0.1018 | 0.0000*** | 0.0002*** | -4.8388 | Large | Yes |
| DMC Old | DMC Ep10k | -0.0262 | 0.0035*** | 0.0081*** | -1.5013 | Large | Yes |
| DMC Old | RLCard DMC (100M) | -0.1296 | 0.0000*** | 0.0002*** | -8.1864 | Large | Yes |
| DMC Old | RLCard DMC+MCTS (100M) | -0.1392 | 0.0000*** | 0.0002*** | -6.8560 | Large | Yes |
| DMC Ep10k | RLCard DMC (100M) | -0.1034 | 0.0000*** | 0.0002*** | -5.9555 | Large | Yes |
| DMC Ep10k | RLCard DMC+MCTS (100M) | -0.1130 | 0.0000*** | 0.0002*** | -5.2513 | Large | Yes |
| RLCard DMC (100M) | RLCard DMC+MCTS (100M) | -0.0096 | 0.3026 | 0.3069 | -0.4746 | Small | No |

*** indicates p < 0.05 (statistically significant)

### Significant Differences

- DQN Original > DMC Old: Mean difference = 0.0236, Effect size = Large (Cohen's d = 1.2273)
- DQN Original < RLCard DMC (100M): Mean difference = -0.1060, Effect size = Large (Cohen's d = -5.5360)
- DQN Original < RLCard DMC+MCTS (100M): Mean difference = -0.1156, Effect size = Large (Cohen's d = -5.0298)
- DQN+MCTS > DMC Old: Mean difference = 0.0374, Effect size = Large (Cohen's d = 2.2188)
- DQN+MCTS < RLCard DMC (100M): Mean difference = -0.0922, Effect size = Large (Cohen's d = -5.5002)
- DQN+MCTS < RLCard DMC+MCTS (100M): Mean difference = -0.1018, Effect size = Large (Cohen's d = -4.8388)
- DMC Old < DMC Ep10k: Mean difference = -0.0262, Effect size = Large (Cohen's d = -1.5013)
- DMC Old < RLCard DMC (100M): Mean difference = -0.1296, Effect size = Large (Cohen's d = -8.1864)
- DMC Old < RLCard DMC+MCTS (100M): Mean difference = -0.1392, Effect size = Large (Cohen's d = -6.8560)
- DMC Ep10k < RLCard DMC (100M): Mean difference = -0.1034, Effect size = Large (Cohen's d = -5.9555)
- DMC Ep10k < RLCard DMC+MCTS (100M): Mean difference = -0.1130, Effect size = Large (Cohen's d = -5.2513)

## Performance Gap Analysis

### RLCard vs Custom Models

**Performance Gap**:
- Best RLCard Model: RLCard DMC+MCTS (100M) (61.3%)
- Best Custom Model: DQN+MCTS (51.2%)
- Gap: 10.2% (10.18 percentage points)

**Analysis**:
- RLCard models significantly outperform custom models (10.2% gap)
- This suggests that parallel training architecture is more important than network complexity

## Recommendations

### For Best Performance

1. **Use RLCard DMC+MCTS (100M)**: Achieves 61.3% win rate with 95% CI [59.6%, 63.0%]
2. **Evaluate RLCard Models**: Confirm performance with comprehensive evaluation
3. **Consider Retraining**: Custom models below 55% threshold may benefit from:
   - More training episodes
   - Better hyperparameter tuning
   - Parallel training architecture

### For Research

1. **Custom DMC**: Best custom model with three-headed architecture
2. **MCTS Integration**: Investigate why MCTS reward shaping provides only modest gains
3. **Checkpoint Selection**: Study why some checkpoints outperform final checkpoints
4. **Opponent Modeling**: Train and evaluate DMC with opponent modeling
