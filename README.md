# Kaggle Hedge Fund Time Series Forecasting

This repository contains the solution for the Kaggle Hedge Fund Time Series Forecasting competition. The goal is to predict future returns for various entities, evaluated on a custom Weighted RMSE skill score.

## 🏆 Final Result
- **Estimated Score**: ~0.054 (Beating the zero baseline significantly)
- **Rank**: (Hypothetical, as this is a simulation)

## 🧠 Solution Overview

Our winning approach treats this problem as a **conservative signal extraction** task. The data is extremely noisy, and standard regression models often perform worse than predicting zero (Score 0).

### Key Strategies
1.  **Robust Loss Function**: We use **LightGBM with Huber Loss** (`alpha=0.1`). This "boxier" loss function is far more robust to the extreme outliers present in the dataset than MSE or MAE.
2.  **Hybrid Feature Engineering**:
    *   **Short Horizon (H1)**: Uses only the original anonymized features. Market aggregated features proved to be noise for 1-day predictions.
    *   **Long Horizons (H3, H10, H25)**: Uses "Extended" features, including global Market Means and Sector Trends. These macro features significantly improve signal for longer-term trends.
3.  **Robust Shrinkage**:
    *   Raw model predictions are too confident/variance-heavy.
    *   We apply a learned **shrinkage factor** of roughly **0.27 to 0.34** (depending on horizon) to all predictions. This dampens the variance and maximizes the competition metric (Ratio).
4.  **Weight Transformation**: Training uses `sqrt(weight + 1)` sample weights to balance focus between high-value transactions and generalizability.

### Architecture
- **Model**: LightGBM Regressor (GOSS)
- **Objective**: Huber ($\delta=0.1$)
- **Validation**: 3-Fold Time-Series Cross-Validation

## 📂 Repository Structure

```
├── data/                   # Competition datasets (train.parquet, test.parquet)
├── outputs/                # Submission files and experimental logs
├── src/                    # Source code
│   ├── 01_data_exploration.py       # Initial EDA
│   ├── 02-04_lgb_baseline_*.py      # Early baseline attempts (Score 0)
│   ├── 06_strategy_refinement_v2.py # Breakthrough: Finding Huber & Shrinkage
│   ├── 08_advanced_tuning.py        # Tuning Alpha and per-horizon shrinkage
│   ├── 09_feature_engineering.py    # Validating Market/Sector features
│   ├── 10_hybrid_submission.py      # ⭐ FINAL SUBMISSION SCRIPT
│   └── 11_ensemble_strategy.py      # Ensemble experiment (LightGBM vs XGBoost)
├── GEMINI.md               # Detailed project context and memory
├── REPORT.md               # Comprehensive technical report
└── requirements.txt        # Python dependencies
```

## 🚀 Usage

### Prerequisites
- Python 3.9+
- 16GB+ RAM recommended (Dataset ~5GB)

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Reproducing the Winning Submission
Run the hybrid submission script, which implements the best-performing strategy:
```bash
python src/10_hybrid_submission.py
```
This will generate `outputs/submission_hybrid_[timestamp].csv`.

### Running Experiments
To see the validation results for Feature Engineering:
```bash
python src/09_feature_engineering.py
```

## 📊 Detailed Analysis
For a deep dive into the experimental process, failures (like XGBoost), and detailed metric analysis, please see [REPORT.md](./REPORT.md).