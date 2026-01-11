
# Competition Report: Hedge Fund Time Series Forecasting

## 🏆 Final Result (Winning Strategy)
**The "Robust Baseline" is the champion.**
- **Score (Est)**: ~0.053
- **Configuration**: LightGBM Huber (alpha=0.5) + Sqrt Weights + Original Features + Aggressive Shrinkage.

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
   - Trees: 2000
3. **Shrinkage Strategy** (The "Secret Sauce"):
   - Test data is noisier than Validation data. We must shrink more aggressively than CV suggests.
   - **Horizon 1**: x 0.15 (Signal is almost non-existent)
   - **Horizon 3**: x 0.15
   - **Horizon 10**: x 0.28
   - **Horizon 25**: x 0.30

## Failed Experiments (For Reference)
1. **MSE Loss**: Score 0. (Too sensitive to outliers).
2. **High-Weight Training Only**: Score 0. (Loss of generalization).
3. **XGBoost Pseudo-Huber**: Ratio >> 50. (Gradient scaling issues).
4. **Market/Sector Feature Engineering**: Worse LB Score. (Regime shift).

## Conclusion
In noisy financial time series forecasting, **Simplicity and Conservatism** beat complexity. The winning model does just one thing well: it identifies the directional signal in the original features and dampens it heavily to ensure the ratio $ \frac{\text{Error}}{\text{ZeroError}} < 1 $.
