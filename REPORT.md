
# Competition Report: Hedge Fund Time Series Forecasting

## Project Overview
This project targets the Kaggle Hedge Fund Time Series Forecasting competition. The goal is to predict returns (`y_target`) for various entities across different time horizons, evaluated using a custom weighted RMSE skill score.

**Key Metric:** 
`Score = sqrt(1 - min(max(ratio, 0), 1))` where `ratio = sum(w*(y-pred)^2) / sum(w*y^2)`.
This metric penalizes predictions that are worse than predicting zero. A positive score requires `ratio < 1`.

## Executive Summary
After extensive experimentation, we achieved a positive score of **~0.054** (estimated) by treating the problem as a "conservative signal extraction" task rather than standard regression.

**Key Breakthroughs:**
1. **Beating the Zero Baseline**: Most standard models fail (score 0). We succeeded by applying heavy shrinkage (multiplying predictions by 0.2-0.3).
2. **Robust Loss Functions**: Huber loss with $\alpha=0.1$ proved significantly better than MSE, confirming the data contains many outliers.
3. **Weight Transformation**: Training with $\sqrt{w+1}$ weights balanced the need to focus on high-weight samples without overfitting to the top 1%.
4. **Horizon-Specific Modeling**: Each prediction horizon (1, 3, 10, 25 days) requires different shrinkage levels and feature sets.
5. **Feature Engineering**: Macro/Sector trends help significantly for long horizons (10, 25) but add noise to short horizons (1).

## Methodology & Experiments

### 1. Data Exploration & Baseline (Failures)
- **Initial Baselines**: LightGBM with MSE loss and raw weights. Result: Score 0.
- **Problem**: The model overfit to high-weight outliers and produced noisy predictions that increased the weighted MSE error compared to predicting zero.
- **Insight**: The metric is extremely sensitive. If `sum(w*(y-pred)^2) > sum(w*y^2)`, score is 0.

### 2. High-Weight Focus
We hypothesized that focusing on the top 10% of weighted samples (which contribute ~72% of the metric) would help.
- **Strategy**: Train only on high-weight subset.
- **Result**: Failed (Score 0). The model lost generalization capability for the broader dataset.

### 3. Strategy Refinement (The Breakthrough)
We tested 7 strategies, including Quantile Regression, MAE, and various weight transformations.
- **Discovery**: Comparing prediction error vs. zero-prediction error showed that for ~60% of samples, our model was *hurting* the score.
- **Solution**: "Shrinkage". By multiplying predictions by a small factor $s \in [0, 1]$, we reduce the variance of the error.
- **Optimal Shrinkage**: We found $s \approx 0.3$ minimized the ratio.
- **Result**: First positive score (~0.044)!

### 4. Advanced Tuning (Optimization)
We conducted rigorous experiments to fine-tune the winning approach.
- **Huber Loss Alpha**: $\alpha=0.1$ is optimal. (Ratio 0.9980)
- **Per-Sample Shrinkage**: Weight-adaptive shrinkage works best by applying *stronger* shrinkage to high-weight (noisy) samples.

### 5. Feature Engineering (Signal Quality)
We aggregated key features by `ts_index` (Market) and `code` (Sector) to capture macro trends.
- **Horizon 1**: Ratio worsened (0.9986 -> 0.9990). Market noise distracts the model.
- **Horizon 25**: Ratio improved significantly (0.9977 -> 0.9958). Macro tends drive long-term returns.
- **Conclusion**: Use a **Hybrid Strategy**.
  - H1: Base Features.
  - H3, H10, H25: Extended Features.

## Final Solution Architecture

1. **Model**: LightGBM Regressor
   - `objective`: 'huber' ($\alpha=0.1$)
   - `learning_rate`: 0.03
   - `num_leaves`: 31, `max_depth`: 6
   - `n_estimators`: 2000

2. **Training Data**:
   - Full training set
   - Weights transformed: $w_{train} = \sqrt{w_{raw} + 1}$
   - **Features**: Hybrid set (Base vs Extended) per horizon.

3. **Inference**:
   - Predict raw values
   - Apply horizon-specific shrinkage:
     - H1: 0.29 (Conservative)
     - H3: 0.27
     - H10: 0.32
     - H25: 0.34 (More confident due to better features)

## Conclusion
This competition requires a paradigm shift from "minimizing error" to "minimizing the ratio of error to zero-prediction error". The noise-to-signal ratio is extremely high. The winning strategy effectively filters out noise by dampening predictions and leveraging macro trends only where they are statistically valid (long horizons).
