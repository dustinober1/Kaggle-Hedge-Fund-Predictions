# Project Context

## Overview
This project is for the Kaggle Hedge Fund Time Series Forecasting competition.

**Competition Details:**
- **Prize Pool**: $10,000 (1st: $3,500, 2nd: $2,500, 3rd: $2,000, 4th: $1,000, 5th: $1,000)
- **Evaluation**: Weighted RMSE Skill Score
- **Key Constraint**: No look-ahead - predictions for ts_index t can only use data from ts_index 0 to t

## Dataset Summary
- **Training data**: ~922 MB parquet files
- **Features**: 86 anonymized features (feature_a to feature_ch)
- **Horizons**: 1 (short), 3 (medium), 10 (long), 25 (extra-long)
- **Key Columns**: id, code, sub_code, sub_category, ts_index, horizon, weight, target

## Recent Changes
- Created Python virtual environment (`.venv`)
- Updated `.gitignore` to exclude `.venv`, `__pycache__`, and other common files
- Added comprehensive `requirements.txt` with data science packages
- Created `notebooks/01_data_exploration.ipynb` for full data exploration

## Project Structure
```
├── Competition_Rules/      # Competition rules and data description
│   ├── overview.md
│   ├── rules.md
│   └── data_descripition.md
├── data/                   # Competition data
│   ├── train.parquet
│   └── test.parquet
├── notebooks/              # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── .venv/                  # Python virtual environment
├── requirements.txt        # Project dependencies
└── GEMINI.md              # This file
```

## Next Steps
1. Run data exploration notebook to understand the dataset
2. Build baseline model with LightGBM
3. Implement proper time-based cross-validation
4. Feature engineering based on exploration findings
5. Advanced modeling and ensembling

## Key Competition Tips
- Test data comes from AFTER training data
- Weight recent data more heavily if needed
- Consider horizon-specific models
- Use the weight column in loss functions
- Handle entity hierarchies (code → sub_code → sub_category)
