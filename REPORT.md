# Hedge Fund Time Series Forecasting - Competition Report

**Competition**: [Kaggle - Hedge Fund Time Series Forecasting](https://www.kaggle.com/competitions/ts-forecasting)  
**Prize Pool**: $10,000  
**Last Updated**: 2026-01-10

---

## 📋 Executive Summary

This report documents our approach to the Kaggle Hedge Fund Time Series Forecasting competition. The goal is to predict a continuous numerical value (`y_target`) using 86 anonymized features across multiple forecast horizons (1, 3, 10, 25).

**Current Status**: Initial exploration and baseline models complete. All baselines score 0 on validation, indicating the need for more sophisticated approaches that can beat the trivial "predict zero" baseline.

---

## 📊 Dataset Overview

### Data Size
| Dataset | Rows | Columns | Memory |
|---------|------|---------|--------|
| Train | 5,337,414 | 94 | ~5.3 GB |
| Test | 1,447,107 | 92 | ~1.4 GB |

### Key Columns
- **id**: Unique identifier (compound key)
- **code**: 23 unique entity codes
- **sub_code**: 180 in train, 47 in test (only 12 overlap!)
- **sub_category**: 5 categories
- **ts_index**: Temporal index (train: 1-3601, test: 3602-4376)
- **horizon**: Forecast horizon (1, 3, 10, 25)
- **weight**: Sample weight for evaluation (ONLY in train)
- **y_target**: Target variable to predict
- **feature_a to feature_ch**: 86 anonymized features

### Critical Observations

1. **No Temporal Overlap**: Test ts_index (3602-4376) is strictly AFTER train (1-3601)
2. **New Entities in Test**: 2,299 entity combinations in test that don't appear in training
3. **Target Distribution**: Centered at 0 (median: -0.0006, mean: -0.67, std: 32.5)
4. **Extreme Weight Skew**: Weights range from 0 to 13.9 trillion

---

## 🎯 Competition Metric Analysis

### The Skill Score Formula
```
Score = sqrt(1 - min(max(ratio, 0), 1))
where ratio = sum(w * (y - pred)^2) / sum(w * y^2)
```

### Key Insights

| Finding | Value | Implication |
|---------|-------|-------------|
| Score when predicting 0 | 0.000 | This is the baseline to beat |
| Score when predicting mean (-0.67) | 0.000 | Worse than zero! (ratio = 21,521) |
| Top 1% samples contribution | 24.67% | Weight concentration is extreme |
| Top 10% samples contribution | 72.22% | Must focus on high-weight samples |

### What This Means
- **The metric is a "skill score"** - it measures how much better you are than predicting zero
- **Score = 0** means your predictions are no better than zero
- **Score > 0** requires predictions that are CLOSER to true values than zero is
- **Any constant prediction makes things worse** - the model must learn actual patterns

---

## 🔬 Experiments Conducted

### Experiment 1: Data Exploration
**File**: `src/01_data_exploration.py`  
**Status**: ✅ Complete

**Key Findings**:
- 86 features with varying missing rates (0-12.5%)
- 4 highly correlated feature pairs (|r| > 0.9):
  - feature_bm ↔ feature_bo: 0.970
  - feature_u ↔ feature_af: 0.956
  - feature_bz ↔ feature_cd: 0.951
  - feature_ca ↔ feature_cc: 0.942
- Top features correlated with target: feature_bz (0.09), feature_cd (0.09)
- Target has heavy tails with extreme outliers (min: -2201, max: 2314)

---

### Experiment 2: LightGBM Baseline v1
**File**: `src/02_lgb_baseline.py`  
**Status**: ✅ Complete

**Configuration**:
- Time-based 80/20 split (train ts_index 1-2881, val 2882-3601)
- Used sample weights during training
- Early stopping with 50 rounds patience

**Results**:
| Metric | Value |
|--------|-------|
| Best Iteration | 29 |
| Validation Score | 0.054 |
| Horizon 1 Score | 0.000 |
| Horizon 3 Score | 0.000 |
| Horizon 10 Score | 0.059 |
| Horizon 25 Score | 0.074 |

**Observations**:
- Model stopped very early (29 iterations)
- Short horizons (1, 3) scored 0
- Using extreme weights during training may cause issues

---

### Experiment 3: LightGBM Baseline v2 (Per-Horizon Models)
**File**: `src/03_lgb_baseline_v2.py`  
**Status**: ✅ Complete

**Configuration**:
- Separate models for each horizon
- Normalized weights using log1p for training
- Evaluated with original weights

**Results**:
| Horizon | Best Iteration | Score |
|---------|----------------|-------|
| 1 | 31 | 0.000 |
| 3 | 54 | 0.000 |
| 10 | 33 | 0.000 |
| 25 | 22 | 0.000 |
| **Overall** | - | **0.000** |

**Observations**:
- Weight normalization didn't help
- All horizons scored 0 when evaluated with original weights

---

### Experiment 4: Baseline Comparison v3
**File**: `src/04_lgb_baseline_v3.py`  
**Status**: ✅ Complete

**Approaches Tested**:

| Approach | Validation Score |
|----------|-----------------|
| Zero Baseline | 0.000 |
| Mean Baseline | 0.000 |
| Horizon Mean Baseline | 0.000 |
| LightGBM Single (no weights) | 0.000 |
| LightGBM Per-Horizon (no weights) | 0.000 |

**Observations**:
- ALL approaches score 0
- This confirms the difficulty of beating the zero baseline
- The signal-to-noise ratio in the data is very low

---

### Experiment 5: Metric Deep Dive
**File**: `src/debug_metric.py`  
**Status**: ✅ Complete

**Key Findings**:
```
Predicting ZERO:
  Ratio: 1.000000
  Score: 0.000000

Predicting MEAN (-0.3878):
  Ratio: 21521.372026  (MUCH WORSE!)
  Score: 0.000000

Predicting PERFECTLY:
  Ratio: 0.000000
  Score: 1.000000
```

**Conclusion**: The metric design means we need predictions that are genuinely closer to true values than zero. Any systematic bias makes things worse.

---

### Experiment 6: High-Weight Sample Focus (7 Strategies)
**File**: `src/05_high_weight_focus.py`  
**Status**: ✅ Complete

**Strategies Tested**:
| Strategy | Description | Ratio | Score |
|----------|-------------|-------|-------|
| S1 | Train on top 10% weights only | 364,680 | 0.000 |
| S2 | Importance sampling (5x oversample) | 139,495 | 0.000 |
| S3 | Log-transformed weights | 9.93 | 0.000 |
| S4 | Selective prediction (90% threshold) | 425,647 | 0.000 |
| S5 | Per-horizon high-weight models | 412,188 | 0.000 |
| S6 | **Sqrt-transformed weights** | **1.02** | 0.000 |
| S7 | Confidence blend | 5,479 | 0.000 |

**Key Finding**: Strategy 6 (sqrt weights) achieved ratio=1.019, very close to beating zero (need ratio < 1.0)!

---

### Experiment 7: Strategy Refinement v2 ⭐ BREAKTHROUGH!
**File**: `src/06_strategy_refinement_v2.py`  
**Status**: ✅ Complete - **FIRST SUCCESS!**

**Key Insight**: Apply **shrinkage** to predictions (multiply by small factor) to reduce variance.

**Results**:
| Strategy | Best Shrinkage | Ratio | Score | Beats Zero? |
|----------|----------------|-------|-------|-------------|
| **Huber Loss + Sqrt Weights** | 0.30 | **0.998** | **0.044** | ✅ YES! |
| Quantile (0.5) | 0.80 | 0.9996 | 0.020 | ✅ YES! |
| Weight Power Sweep (p=0.6) | 0.20 | 0.9997 | 0.017 | ✅ YES! |
| Per-Horizon + Shrinkage | varies | 0.9998 | 0.013 | ✅ YES! |
| Zero Baseline | N/A | 1.000 | 0.000 | - |

**Winning Configuration**:
- Loss function: **Huber** (robust to outliers)
- Weight transformation: **sqrt(weight + 1)**
- Shrinkage factor: **0.30** (predictions × 0.30)
- This achieved **ratio = 0.998**, meaning predictions are 0.2% better than zero!

---

### Experiment 8: Optimized Submission
**File**: `src/07_optimized_submission.py`  
**Status**: ✅ Complete

**Final Approach**: Per-Horizon Huber Models with Horizon-Specific Shrinkage

| Horizon | Shrinkage | Ratio |
|---------|-----------|-------|
| 1 | 0.12 | 0.9996 |
| 3 | 0.06 | 0.9998 |
| 10 | 0.27 | 0.9976 |
| 25 | 0.29 | 0.9961 |
| **Overall** | - | **0.9972** |

**Final Validation Score**: **0.0529** 🎉

**Submission Statistics**:
- Mean prediction: -0.076
- Std: 0.225
- Range: [-2.19, 1.82]

---

## 📁 Project Structure

```
Kaggle-hedgefundTimeSeriesForecasting/
├── Competition_Rules/
│   ├── overview.md
│   ├── rules.md
│   └── data_descripition.md
├── data/
│   ├── train.parquet (922 MB)
│   └── test.parquet (146 MB)
├── src/
│   ├── 01_data_exploration.py       # Full EDA
│   ├── 02_lgb_baseline.py           # LightGBM v1
│   ├── 03_lgb_baseline_v2.py        # Per-horizon models
│   ├── 04_lgb_baseline_v3.py        # Baseline comparison
│   ├── 05_high_weight_focus.py      # High-weight strategies
│   ├── 06_strategy_refinement_v2.py # ⭐ Breakthrough strategies
│   ├── 07_optimized_submission.py   # Final submission generator
│   └── debug_metric.py              # Metric analysis
├── outputs/
│   ├── exploration_results.json
│   ├── submission_*.csv             # Various submissions
│   └── *_results_*.json             # Experiment results
├── .venv/                           # Python environment
├── requirements.txt
├── GEMINI.md                        # AI assistant context
└── REPORT.md                        # This file
```

---

## 🎯 Next Steps (Priority Order)

### 1. ✅ High-Weight Sample Focus - DONE!
- ✅ Tried 7 different strategies for high-weight samples
- ✅ Found sqrt weight transformation works best
- ✅ Discovered shrinkage is key to beating zero baseline

### 2. Improve Score Further
- Fine-tune Huber loss parameters (alpha/delta)
- Try different shrinkage values per-sample
- Cross-validate shrinkage on multiple time folds

### 3. Feature Engineering (Next Priority)
- Lag features (respecting no look-ahead constraint)
- Rolling statistics (mean, std, min, max over past windows)
- Entity-level aggregations
- Time-based features (recent vs old data)

### 4. Alternative Modeling Approaches
- **XGBoost/CatBoost**: May handle noise differently
- **Neural Networks**: For entity embeddings
- **Ensemble**: Blend Huber + Quantile models

### 5. Validation Strategy Refinement
- Multiple time-based folds
- More robust shrinkage estimation
- Entity-aware validation

---

## 📝 Lessons Learned

1. **Understand the metric first**: The skill score design heavily influences strategy
2. **Weights matter enormously**: Top 10% of samples dominate the evaluation
3. **Simple baselines are powerful**: Predicting zero is the benchmark to beat
4. **Signal-to-noise is low**: 86 features but top correlation with target is only 0.09
5. **New entities in test**: 2,299 unseen entity combinations require robust models
6. ⭐ **Shrinkage is critical**: Raw model predictions are too noisy; shrinking them towards zero dramatically improves performance
7. ⭐ **Huber loss is robust**: Handles outliers better than MSE for this noisy data
8. ⭐ **Sqrt weights balance importance**: Neither raw weights (too extreme) nor uniform weights (ignores importance) work well

---

## 🏆 Current Best Results

| Metric | Value |
|--------|-------|
| **Validation Score** | **0.0529** |
| Validation Ratio | 0.9972 |
| Approach | Per-Horizon Huber + Shrinkage |
| Weight Transform | sqrt(weight + 1) |

---

## 📚 References

- [Competition Overview](https://www.kaggle.com/competitions/ts-forecasting/overview)
- [Competition Data Description](https://www.kaggle.com/competitions/ts-forecasting/data)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Huber Loss](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)

---

*This report will be updated as we make progress on the competition.*

