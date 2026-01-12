
# Competition Report: Hedge Fund Time Series Forecasting

## 🏆 Final Result (Winning Strategy)
**The "Original Shrinkage" configuration is the champion.**
- **Score (Est)**: ~0.053
- **Configuration**: LightGBM Huber (alpha=0.5) + Sqrt Weights + Original Features + Per-Horizon Shrinkage.

## ⚠️ Score Regression Root Cause
Second submission scored worse due to **incorrect shrinkage values**:

| Horizon | Correct (0.053) | Incorrect (worse) |
|---------|-----------------|-------------------|
| H1 | **0.12** | 0.15 |
| H3 | **0.06** | 0.15 ← 2.5x too high! |
| H10 | **0.27** | 0.28 |
| H25 | **0.29** | 0.30 |

**H3 shrinkage was the critical error** - it should be 0.06, not 0.15!

## 📉 Methodology Pivot
Initial Cross-Validation on the "Hybrid Strategy" (using aggregated Market/Sector features) showed improvement (Ratio 0.995 vs 0.997). However, Leaderboard feedback indicated it performed strictly worse than the Baseline.
**Why?**
- **Regime Shift**: The test set (2,299 new entities, future timeframe) likely contains market conditions unseen in training.
- **Overfitting**: Aggregated macro features (e.g., "Market Mean") rely on the assumption that training period correlations hold in the future. They did not.
- **Microstructure Wins**: The 86 anonymized features likely contain alpha that is independent of macro regime, making them more robust for extrapolation.

## Final Algorithm

1. **Features**: Use only the provided 86 anonymized features. Drop all engineered macro features.
2. **Model**: LightGBM Regressor
   - Objective: `huber` (alpha=0.5) - Slightly wider "box" than 0.1, balancing robustness and sensitivity.
   - Trees: 1500-2000
3. **Shrinkage Strategy** (The "Secret Sauce"):
   - **Horizon 1**: x 0.12 (Heavy dampening for noise)
   - **Horizon 3**: x 0.06 (Very light - H3 has more signal!)
   - **Horizon 10**: x 0.27
   - **Horizon 25**: x 0.29


## Failed Experiments (For Reference)
1. **MSE Loss**: Score 0. (Too sensitive to outliers).
2. **High-Weight Training Only**: Score 0. (Loss of generalization).
3. **XGBoost Pseudo-Huber**: Ratio >> 50. (Gradient scaling issues).
4. **Market/Sector Feature Engineering**: Worse LB Score. (Regime shift).

## Conclusion
In noisy financial time series forecasting, **Simplicity and Conservatism** beat complexity. The winning model does just one thing well: it identifies the directional signal in the original features and dampens it heavily to ensure the ratio $ \frac{\text{Error}}{\text{ZeroError}} < 1 $.
