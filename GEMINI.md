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

## Recent Changes
- Created Python virtual environment (`.venv`)
- Updated `.gitignore` to exclude `.venv`, `__pycache__`, etc.
- Added comprehensive `requirements.txt` with data science packages
- Created `src/01_data_exploration.py` for full data exploration
- Generated `outputs/exploration_results.json` with key statistics

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
│   └── 01_data_exploration.py
├── outputs/                # Generated outputs
│   └── exploration_results.json
├── notebooks/              # Jupyter notebooks (to be added)
├── .venv/                  # Python virtual environment
├── requirements.txt        # Project dependencies
└── GEMINI.md              # This file
```

## Next Steps
1. ✅ Run data exploration to understand the dataset
2. Build baseline model with LightGBM
3. Implement proper time-based cross-validation
4. Feature engineering based on exploration findings
5. Advanced modeling and ensembling

## Winning Strategy

### Critical Competition Insights
1. **No look-ahead allowed**: Must predict ts_index t using only data from 0 to t
2. **Test is AFTER train**: Focus on recent training data, consider recency weighting
3. **New entities in test**: 2,299 entity combinations not in training - need robust global model
4. **Weighted RMSE**: Use competition metric in cross-validation and as loss function
5. **Heavy-tailed target**: Consider robust loss functions or clipping

### Modeling Approaches to Try
1. **Gradient Boosting**: LightGBM, XGBoost, CatBoost
2. **Horizon-specific models**: Different models for each horizon (1, 3, 10, 25)
3. **Entity embeddings**: For handling categorical variables
4. **Feature engineering**: Lag features, rolling statistics (respecting no look-ahead)
5. **Ensemble methods**: Stacking multiple approaches
