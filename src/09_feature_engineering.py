"""
Hedge Fund Time Series Forecasting - Feature Engineering
======================================================

Building on the robust Huber model, this script adds:
1. Market-level features (aggregated mean/std per ts_index)
2. Sector-level features (aggregated per ts_index + code)
3. Rolling statistics of market trends

Goal: Provide the model with "Global Context" since individual 
entity behaviors are noisy.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Define paths
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Competition Metric
# =============================================================================
def weighted_rmse_score(y_true, y_pred, weights):
    """Competition metric"""
    denom = np.sum(weights * y_true ** 2)
    if denom == 0:
        return 0.0
    ratio = np.sum(weights * (y_true - y_pred) ** 2) / denom
    clipped = np.clip(ratio, 0.0, 1.0)
    return float(np.sqrt(1.0 - clipped))


def get_ratio(y_true, y_pred, weights):
    """Get ratio component"""
    denom = np.sum(weights * y_true ** 2)
    if denom == 0:
        return float('inf')
    return np.sum(weights * (y_true - y_pred) ** 2) / denom


# =============================================================================
# Feature Engineering
# =============================================================================
def engineer_features(train, test):
    """
    Create new features for both train and test.
    Optimized for memory usage.
    """
    print("\n=== Feature Engineering ===")
    
    # Identify high-impact features (correlated with target or weights)
    # Based on data exploration: feature_bz, feature_cd, feature_u, feature_af
    key_features = ['feature_bz', 'feature_cd', 'feature_u', 'feature_af']
    
    # Combine for global calculations
    print("  Preparing data for aggregation...")
    # We only need specific columns for aggregation to save memory
    agg_cols = ['ts_index', 'code'] + key_features
    combined = pd.concat([train[agg_cols], test[agg_cols]], axis=0).reset_index(drop=True)
    
    # 1. Market Features (Global Mean/Std per timestamp)
    print("  Creating Market Features (by ts_index)...")
    market_stats = combined.groupby('ts_index')[key_features].agg(['mean', 'std'])
    market_stats.columns = [f'market_{col}_{stat}' for col, stat in market_stats.columns]
    
    # 2. Sector Features (Mean per code per timestamp)
    print("  Creating Sector Features (by ts_index + code)...")
    sector_stats = combined.groupby(['ts_index', 'code'])[key_features].mean()
    sector_stats.columns = [f'sector_{col}_mean' for col in sector_stats.columns]
    
    # 3. Rolling Market Features (Market Trend)
    # We apply rolling window on the UNIQUE market_stats index (ts_index)
    print("  Creating Rolling Market Trends...")
    rolling_windows = [5, 20]
    rolling_feats = []
    
    for window in rolling_windows:
        # Calculate rolling mean on the market averages
        rolled = market_stats[[f'market_{f}_mean' for f in key_features]].rolling(window=window).mean()
        rolled.columns = [f'market_rolling{window}_{f}' for f in key_features]
        rolling_feats.append(rolled)
        
    all_market_features = pd.concat([market_stats] + rolling_feats, axis=1)
    
    # Merge back to Train
    print("  Merging aggregates to Train...")
    # Merge market stats
    train = train.merge(all_market_features, on='ts_index', how='left')
    # Merge sector stats
    train = train.merge(sector_stats, on=['ts_index', 'code'], how='left')
    
    # Merge back to Test
    print("  Merging aggregates to Test...")
    test = test.merge(all_market_features, on='ts_index', how='left')
    test = test.merge(sector_stats, on=['ts_index', 'code'], how='left')
    
    # Fill NAs (rolling windows create NAs at start)
    new_cols = [c for c in train.columns if c not in Combined_original_cols]
    train[new_cols] = train[new_cols].fillna(0)
    test[new_cols] = test[new_cols].fillna(0)
    
    print(f"  Added {len(new_cols)} new features")
    return train, test, new_cols


# =============================================================================
# Workflow
# =============================================================================
def load_data():
    """Load data"""
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    return train, test

def main():
    print("=" * 80)
    print("HEDGE FUND FORECASTING: FEATURE ENGINEERING & VALIDATION")
    print("=" * 80)
    
    # 1. Load Data
    train, test = load_data()
    original_cols = train.columns.tolist()
    global Combined_original_cols 
    Combined_original_cols = set(original_cols)
    
    # 2. Feature Engineering
    train_eng, test_eng, new_features = engineer_features(train, test)
    
    # Define features to use (original + new)
    original_features = [c for c in train.columns if c.startswith('feature_')]
    feature_cols = original_features + new_features
    
    print(f"Total features: {len(feature_cols)} (Original: {len(original_features)}, New: {len(new_features)})")
    
    # 3. Validation Strategy (Time Series CV)
    # We will use the robust CV shrinkage logic
    print("\n" + "=" * 60)
    print("VALIDATING WITH NEW FEATURES (3-Fold Time-Series CV)")
    print("=" * 60)
    
    # Use Huber alpha=0.1 (Winner)
    params = {
        'objective': 'huber',
        'alpha': 0.1,
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'n_estimators': 1500,
        'random_state': 42,
        'verbose': -1,
    }
    
    # CV Setup
    n_folds = 3
    val_ratio = 0.15
    ts_max = train_eng['ts_index'].max()
    ts_min = train_eng['ts_index'].min()
    ts_range = ts_max - ts_min
    val_size = int(ts_range * val_ratio)
    total_val_space = val_size * n_folds
    train_end = ts_max - total_val_space
    
    splits = []
    for fold in range(n_folds):
        val_start = train_end + fold * val_size + 1
        val_end = val_start + val_size - 1
        train_mask = train_eng['ts_index'] <= (val_start - 1)
        val_mask = (train_eng['ts_index'] >= val_start) & (train_eng['ts_index'] <= val_end)
        splits.append((train_mask, val_mask))

    # Run CV
    horizons = sorted(train_eng['horizon'].unique())
    horizon_results = {h: {'shrinkages': [], 'ratios': []} for h in horizons}
    overall_ratios = []

    for fold, (train_mask, val_mask) in enumerate(splits):
        print(f"\n--- Fold {fold + 1} ---")
        train_fold = train_eng[train_mask]
        val_fold = train_eng[val_mask]
        
        all_preds = np.zeros(len(val_fold))
        
        for horizon in horizons:
            train_h = train_fold[train_fold['horizon'] == horizon]
            val_h = val_fold[val_fold['horizon'] == horizon]
            
            if len(val_h) == 0: continue
            
            X_train = train_h[feature_cols]
            y_train = train_h['y_target']
            # Sqrt weights
            train_weights = np.sqrt(train_h['weight'].values + 1)
            
            X_val = val_h[feature_cols]
            y_val = val_h['y_target'].values
            w_val = val_h['weight'].values
            
            train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
            model = lgb.train(
                params, train_data, num_boost_round=1500,
                callbacks=[lgb.log_evaluation(period=99999)]
            )
            
            raw_preds = model.predict(X_val)
            
            # Find optimal shrinkage
            best_s, best_r = 0, float('inf')
            for s in np.arange(0.0, 0.6, 0.01):
                r = get_ratio(y_val, raw_preds * s, w_val)
                if r < best_r:
                    best_r = r
                    best_s = s
            
            horizon_results[horizon]['shrinkages'].append(best_s)
            horizon_results[horizon]['ratios'].append(best_r)
            
            mask = val_fold['horizon'] == horizon
            all_preds[mask.values] = raw_preds * best_s
            
            print(f"  Horizon {horizon}: ratio={best_r:.6f}, shrinkage={best_s:.2f}")
            
        fold_ratio = get_ratio(val_fold['y_target'].values, all_preds, val_fold['weight'].values)
        overall_ratios.append(fold_ratio)
        print(f"  --> Fold Ratio: {fold_ratio:.6f}")

    # Results
    avg_ratio = np.mean(overall_ratios)
    std_ratio = np.std(overall_ratios)
    
    print("\n" + "=" * 60)
    print("RESULTS WITH FEATURE ENGINEERING")
    print("=" * 60)
    print(f"Average CV Ratio: {avg_ratio:.6f} ± {std_ratio:.6f}")
    
    if avg_ratio < 1.0:
        score = np.sqrt(1 - avg_ratio)
        print(f"Expected Score: {score:.6f}")
    else:
        print("Score: 0.0 (Ratio >= 1.0)")
        
    # Baseline comparison (from previous run)
    print(f"\nPrevious Baseline (No FE): Ratio ~0.9977, Score ~0.048")
    if avg_ratio < 0.9977:
        print("IMPROVEMENT: YES!")
    else:
        print("IMPROVEMENT: NO")
        
    return

if __name__ == "__main__":
    main()
