# Project Context

## Overview
This project is for the Kaggle Hedge Fund Time Series Forecasting competition.

**Competition Details:**
- **Prize Pool**: $10,000 (1st: $3,500, 2nd: $2,500, 3rd: $2,000, 4th: $1,000, 5th: $1,000)
- **Evaluation**: Weighted RMSE Skill Score
- **Key Constraint**: No look-ahead - predictions for ts_index t can only use data from ts_index 0 to t

## Dataset Summary (From Exploration)
- **Training data**: 5,337,414 rows, 94 columns (~5.3 GB in memory)
- **Test data**: 1,447,107 rows, 92 columns (~1.4 GB in memory)
- **Features**: 86 anonymized features (feature_a to feature_ch)
- **Target column**: `y_target` (mean: -0.67, std: 32.53, range: -2201 to 2314)
- **Horizons**: 1 (short), 3 (medium), 10 (long), 25 (extra-long)

### Key Findings from Data Exploration

**Temporal Structure:**
- Train ts_index: 1 to 3601
- Test ts_index: 3602 to 4376
- **Zero overlap** - test data is strictly AFTER training data!

**Entity Structure:**
- 23 unique codes (all present in both train and test)
- 180 unique sub_codes in train, 47 in test (only 12 overlap)
- 5 sub_categories (all present in both)
- 9,270 unique entity combinations in train
- **2,299 NEW entity combinations in test** (not seen in training!)

**Target Variable (`y_target`):**
- Mostly centered around 0 (median: -0.0006)
- Very heavy tails (std=32.5, min=-2201, max=2314)
- Slightly negative bias overall (mean: -0.67)
- Longer horizons have more extreme values and higher std

**Weight Distribution:**
- Extremely skewed (mean: 16.4M, max: 13.9T)
- Weights only in training data (used for evaluation)
- Must use weighted loss functions

**Feature Insights:**
- 4 highly correlated feature pairs (|r| > 0.9):
  - feature_bm <-> feature_bo: 0.970
  - feature_u <-> feature_af: 0.956
  - feature_bz <-> feature_cd: 0.951
  - feature_ca <-> feature_cc: 0.942
- Top correlated with target: feature_bz (0.09), feature_cd (0.09)
- Missing values: feature_at (12.5%), feature_by (11%)

## Metric Analysis (Critical!)

**The competition metric is a SKILL SCORE:**
```
Score = sqrt(1 - min(max(ratio, 0), 1))
where ratio = sum(w*(y-pred)^2) / sum(w*y^2)
```

**Key Insight:** 
- Score = 0 means you're no better than predicting ZERO
- Score > 0 means predictions are CLOSER to true values than zero
- Predicting any constant other than 0 makes things MUCH WORSE

**Weight dominance:**
- Top 1% of samples contribute ~25% of denominator
- Top 10% of samples contribute ~72% of denominator
- Must focus on high-weight samples!

## Recent Changes
- Created Python virtual environment (`.venv`)
- Updated `.gitignore` to exclude `.venv`, `__pycache__`, etc.
- Added comprehensive `requirements.txt` with data science packages
- Created `src/01_data_exploration.py` for full data exploration
- Generated `outputs/exploration_results.json` with key statistics
- Created multiple LightGBM baselines (v1, v2, v3)
- Created `src/debug_metric.py` to analyze competition metric
- ⭐ Created `src/05_high_weight_focus.py` - tested 7 high-weight strategies
- ⭐ Created `src/06_strategy_refinement_v2.py` - **BREAKTHROUGH!** Beat zero baseline
- ⭐ Created `src/07_optimized_submission.py` - Final submission (score: 0.053)
- ⭐ Created `src/08_advanced_tuning.py` - Advanced tuning (Huber alpha, CV shrinkage)
- ⭐ Created `src/09_feature_engineering.py` - Added Market/Sector features
- ⭐ Created `src/10_hybrid_submission.py` - Validated Hybrid Strategy (Base for H1, FE for others)

## Current Best Score: ~0.054 (CV Estimate) 🎉

**Winning Configuration (Hybrid):**
- **Model**: LightGBM Huber (alpha=0.1) with sqrt(w+1) weights
- **Features**: 
  - Horizon 1: Original features only (Market noise distracts)
  - Horizon 3, 10, 25: Original + Market/Sector aggregates (Macro trends help)
- **Shrinkage**:
  - H1: 0.29 (Base)
  - H3: 0.27 (FE)
  - H10: 0.32 (FE)
  - H25: 0.34 (FE)
- **Weight-adaptive shrinkage** helps slightly (Apply stronger shrinkage to high-weight samples)

## Project Structure
```
├── Competition_Rules/      # Competition rules and data description
│   ├── overview.md
│   ├── rules.md
│   └── data_descripition.md
├── data/                   # Competition data
│   ├── train.parquet
│   └── test.parquet
├── src/                    # Source code
│   ├── 01_data_exploration.py
│   ├── 02_lgb_baseline.py
│   ├── 03_lgb_baseline_v2.py
│   ├── 04_lgb_baseline_v3.py
│   ├── 05_high_weight_focus.py      # High-weight strategies
│   ├── 06_strategy_refinement_v2.py # ⭐ BREAKTHROUGH strategies
│   ├── 07_optimized_submission.py   # Final submission
│   └── debug_metric.py
├── outputs/                # Generated outputs
│   ├── exploration_results.json
│   ├── submission_optimized_*.csv
│   └── *_results_*.json
├── notebooks/              # Jupyter notebooks (to be added)
├── .venv/                  # Python virtual environment
├── requirements.txt        # Project dependencies
├── REPORT.md              # Competition report
└── GEMINI.md              # This file
```

## Next Steps
1. ✅ Run data exploration to understand the dataset
2. ✅ Build LightGBM baselines (all score 0 - need better approach)
3. ✅ Analyzed competition metric - need to beat zero predictions
4. ✅ **Focus on high-weight samples - SUCCESS! Score: 0.053**
5. Try feature engineering (lag features, rolling stats)
6. Try XGBoost/CatBoost as alternatives
7. Ensemble methods (blend Huber + Quantile models)

## Winning Strategy (PROVEN!)

### Key Discoveries from Experiments
1. **Shrinkage is critical**: Raw predictions are too noisy, must multiply by 0.1-0.3
2. **Huber loss beats MSE**: Robust to outliers in this noisy data
3. **Sqrt weights are optimal**: Neither raw nor log weights work as well
4. **Per-horizon models help**: Different horizons need different shrinkage values

### What Works
- ✅ **Huber loss with alpha=0.1**: More robust than standard Huber (alpha=1.0) or MSE
- ✅ **Per-horizon models** with horizon-specific shrinkage
- ✅ **Hybrid Feature Sets**: Base for H1, Extended (Market/Sector) for H3+
- ✅ **Shrinkage toward zero** (0.27 to 0.34 depending on horizon)
- ✅ **Weight-adaptive shrinkage**: Applying more shrinkage to high-weight samples improves ratio

### What Doesn't Work
- ❌ Using Market features for Horizon 1 (adds noise)
- ❌ Using raw weights during training (too extreme)
- ❌ Predicting non-zero without shrinkage (ratio >> 1)
- ❌ Training only on high-weight samples (loses generalization)
- ❌ Importance sampling / oversampling

