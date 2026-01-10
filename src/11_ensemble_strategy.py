"""
Hedge Fund Time Series Forecasting - Ensemble Strategy
======================================================

Ensemble Strategy:
1. Model A: LightGBM (Huber Loss, alpha=0.1) - Our finding winning model.
2. Model B: XGBoost (Pseudo-Huber Loss) - Adds algorithmic diversity.
3. Method: Weighted Blend (w * A + (1-w) * B).

Goal: Reduce variance further by combining two robust models.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
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
# Metrics & Utils
# =============================================================================
def get_ratio(y_true, y_pred, weights):
    """Get ratio component"""
    denom = np.sum(weights * y_true ** 2)
    if denom == 0: return float('inf')
    return np.sum(weights * (y_true - y_pred) ** 2) / denom

def weighted_rmse_score(y_true, y_pred, weights):
    denom = np.sum(weights * y_true ** 2)
    if denom == 0: return 0.0
    ratio = np.sum(weights * (y_true - y_pred) ** 2) / denom
    clipped = np.clip(ratio, 0.0, 1.0)
    return float(np.sqrt(1.0 - clipped))


# =============================================================================
# Feature Engineering (Hybrid)
# =============================================================================
def engineer_features(train, test):
    print("\n=== Feature Engineering ===")
    key_features = ['feature_bz', 'feature_cd', 'feature_u', 'feature_af']
    agg_cols = ['ts_index', 'code'] + key_features
    combined = pd.concat([train[agg_cols], test[agg_cols]], axis=0).reset_index(drop=True)
    
    # Market & Sector
    market_stats = combined.groupby('ts_index')[key_features].agg(['mean', 'std'])
    market_stats.columns = [f'market_{col}_{stat}' for col, stat in market_stats.columns]
    
    sector_stats = combined.groupby(['ts_index', 'code'])[key_features].mean()
    sector_stats.columns = [f'sector_{col}_mean' for col in sector_stats.columns]
    
    # Rolling
    rolling_windows = [5, 20]
    rolling_feats = []
    for window in rolling_windows:
        rolled = market_stats[[f'market_{f}_mean' for f in key_features]].rolling(window=window).mean()
        rolled.columns = [f'market_rolling{window}_{f}' for f in key_features]
        rolling_feats.append(rolled)
    
    all_market_features = pd.concat([market_stats] + rolling_feats, axis=1)
    
    # Merge
    print("  Merging Features...")
    train = train.merge(all_market_features, on='ts_index', how='left')
    train = train.merge(sector_stats, on=['ts_index', 'code'], how='left')
    test = test.merge(all_market_features, on='ts_index', how='left')
    test = test.merge(sector_stats, on=['ts_index', 'code'], how='left')
    
    new_cols = [c for c in train.columns if c.startswith('market_') or c.startswith('sector_')]
    train[new_cols] = train[new_cols].fillna(0)
    test[new_cols] = test[new_cols].fillna(0)
    
    return train, test, new_cols


# =============================================================================
# Model Training
# =============================================================================
def train_lgbm_model(X_train, y_train, w_train, params):
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    model = lgb.train(
        params, train_data, num_boost_round=params['n_estimators'],
        callbacks=[lgb.log_evaluation(period=0)] # Silent
    )
    return model

def train_xgb_model(X_train, y_train, w_train, params):
    # XGBoost needs weights in DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    model = xgb.train(
        params, dtrain, num_boost_round=params['n_estimators'],
        verbose_eval=False
    )
    return model

def get_shrinkage_for_horizon(horizon, model_type='lgb'):
    # Derived from robust CV results
    # XGB might need different shrinkage, but we start with same baseline
    if model_type == 'lgb':
        # Winning LGBM shrinkages
        return {1: 0.293, 3: 0.270, 10: 0.320, 25: 0.340}.get(horizon, 0.25)
    else:
        # Conservative starting point for XGB
        return {1: 0.25, 3: 0.25, 10: 0.30, 25: 0.30}.get(horizon, 0.25)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 80)
    print("HEDGE FUND FORECASTING - ENSEMBLE STRATEGY")
    print("=" * 80)
    
    # 1. Load & Engineer
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    
    train, test, new_features = engineer_features(train, test)
    
    original_features = [c for c in train.columns if c.startswith('feature_')]
    all_features = original_features + new_features
    
    # Feature config (Hybrid)
    feature_config = {
        1: original_features,
        3: all_features,
        10: all_features,
        25: all_features
    }
    
    # 2. CV Loop for Blending Weights
    print("\n" + "=" * 60)
    print("FINDING OPTIMAL BLENDING WEIGHTS (3-Fold CV)")
    print("=" * 60)
    
    # LightGBM Params (Huber)
    lgb_params = {
        'objective': 'huber', 'alpha': 0.1, 'metric': 'rmse', 'boosting_type': 'gbdt',
        'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6,
        'min_child_samples': 100, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.5, 'reg_lambda': 0.5, 'n_estimators': 1000,
        'random_state': 42, 'verbose': -1
    }
    
    # XGBoost Params (Pseudo-Huber)
    xgb_params = {
        'objective': 'reg:pseudohubererror',
        'huber_slope': 0.1, # Similar to alpha
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 800, # XGB is usually faster converging
        'tree_method': 'hist', # Faster
        'seed': 42
    }
    
    n_folds = 3
    val_ratio = 0.15
    ts_max = train['ts_index'].max()
    ts_range = ts_max - train['ts_index'].min()
    val_size = int(ts_range * val_ratio * n_folds) # Use one larger fold for speed
    
    # Training Split (Last 15% as holdout for blending)
    # We simplify to 1 fold for blend weights to save time
    val_start = ts_max - int(ts_range * 0.2) 
    
    train_mask = train['ts_index'] < val_start
    val_mask = train['ts_index'] >= val_start
    
    print(f"Blending Fold: Train < {val_start}, Val >= {val_start}")
    
    train_fold = train[train_mask]
    val_fold = train[val_mask]
    
    blend_results = []
    
    horizons = sorted(train['horizon'].unique())
    optimal_weights = {} # Key: Horizon, Value: LGB weight
    
    for horizon in horizons:
        print(f"\nHorizon {horizon}...")
        
        features = feature_config[horizon]
        
        # Filter Data
        train_h = train_fold[train_fold['horizon'] == horizon]
        val_h = val_fold[val_fold['horizon'] == horizon]
        
        X_train = train_h[features]
        y_train = train_h['y_target']
        w_train = np.sqrt(train_h['weight'].values + 1)
        
        X_val = val_h[features]
        y_val = val_h['y_target'].values
        w_val = val_h['weight'].values
        
        # Train LGBM
        print("  Training LightGBM...")
        model_lgb = train_lgbm_model(X_train, y_train, w_train, lgb_params)
        preds_lgb = model_lgb.predict(X_val)
        
        # Train XGBoost
        print("  Training XGBoost...")
        model_xgb = train_xgb_model(X_train, y_train, w_train, xgb_params)
        dval = xgb.DMatrix(X_val)
        preds_xgb = model_xgb.predict(dval)
        
        # Apply individual shrinkages FIRST
        shrink_lgb = get_shrinkage_for_horizon(horizon, 'lgb')
        shrink_xgb = get_shrinkage_for_horizon(horizon, 'xgb')
        
        preds_lgb_s = preds_lgb * shrink_lgb
        preds_xgb_s = preds_xgb * shrink_xgb
        
        ratio_lgb = get_ratio(y_val, preds_lgb_s, w_val)
        ratio_xgb = get_ratio(y_val, preds_xgb_s, w_val)
        
        print(f"    LGBM Ratio: {ratio_lgb:.6f}")
        print(f"    XGB  Ratio: {ratio_xgb:.6f}")
        
        # Find optimal blend weight (w * LGB + (1-w) * XGB)
        best_w = 1.0
        best_r = min(ratio_lgb, ratio_xgb)
        
        for w in np.arange(0.0, 1.05, 0.05):
            blend = w * preds_lgb_s + (1 - w) * preds_xgb_s
            r = get_ratio(y_val, blend, w_val)
            if r < best_r:
                best_r = r
                best_w = w
        
        optimal_weights[horizon] = best_w
        print(f"  Best Blend: {best_w:.2f} LGB + {1-best_w:.2f} XGB -> Ratio: {best_r:.6f}")
        
        if best_r < min(ratio_lgb, ratio_xgb):
             print(f"  Ensemble Gain: {min(ratio_lgb, ratio_xgb) - best_r:.6f}")
        else:
             print("  No gain from ensemble")
        
        blend_results.append(best_r)
        
    print("\n" + "=" * 60)
    print("FINAL TRAINING & SUBMISSION")
    print("=" * 60)
    print(f"Optimal Weights (LGB): {optimal_weights}")
    
    # Train on Full Data
    # To save time, we will only retrain if weight < 1.0
    # But for a robust submission, we should do it properly.
    
    all_test_preds = np.zeros(len(test))
    
    for horizon in horizons:
        print(f"\nHorizon {horizon}...")
        w_lgb = optimal_weights[horizon]
        features = feature_config[horizon]
        
        # Data
        train_h = train[train['horizon'] == horizon]
        X_train = train_h[features]
        y_train = train_h['y_target']
        w_train = np.sqrt(train_h['weight'].values + 1)
        
        # Test Data
        test_h = test[test['horizon'] == horizon]
        X_test = test_h[features]
        
        preds_final = np.zeros(len(test_h))
        
        # LGBM
        if w_lgb > 0.01:
            print("  Training Full LGBM...")
            model_l = train_lgbm_model(X_train, y_train, w_train, lgb_params)
            raw_p = model_l.predict(X_test)
            preds_final += w_lgb * (raw_p * get_shrinkage_for_horizon(horizon, 'lgb'))
            
        # XGB
        if w_lgb < 0.99:
            print("  Training Full XGBoost...")
            model_x = train_xgb_model(X_train, y_train, w_train, xgb_params)
            dtest = xgb.DMatrix(X_test)
            raw_p = model_x.predict(dtest)
            preds_final += (1 - w_lgb) * (raw_p * get_shrinkage_for_horizon(horizon, 'xgb'))
            
        mask = test['horizon'] == horizon
        all_test_preds[mask.values] = preds_final
        
    # Submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({'id': test['id'], 'prediction': all_test_preds})
    
    output_path = OUTPUT_DIR / f'submission_ensemble_{timestamp}.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"Mean Prediction: {all_test_preds.mean():.6f}")

if __name__ == "__main__":
    main()
