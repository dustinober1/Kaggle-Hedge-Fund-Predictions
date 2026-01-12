"""
Hedge Fund Time Series Forecasting - IMPROVED SUBMISSION
==========================================================

Root Cause Analysis:
- Second submission used WRONG shrinkage for H3 (0.15 vs 0.06)
- This 2.5x higher shrinkage hurt performance

Improvements:
1. Restore exact original shrinkage values (H1: 0.12, H3: 0.06, H10: 0.27, H25: 0.29)
2. Test alpha=0.1 (better CV ratio in advanced tuning)
3. Test weight-adaptive shrinkage (best overall ratio)
4. Finer grid search for shrinkage optimization (0.01 steps)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent.parent / 'ts-forecasting'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


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


def time_based_split(train, val_ratio=0.2):
    """Split data based on ts_index"""
    ts_max = train['ts_index'].max()
    ts_min = train['ts_index'].min()
    ts_range = ts_max - ts_min
    val_threshold = ts_max - int(ts_range * val_ratio)
    
    train_df = train[train['ts_index'] <= val_threshold].copy()
    val_df = train[train['ts_index'] > val_threshold].copy()
    return train_df, val_df


def train_per_horizon_models(train_df, feature_cols, alpha=0.5, weight_power=0.5):
    """Train separate Huber models per horizon."""
    horizons = sorted(train_df['horizon'].unique())
    models = {}
    
    params = {
        'objective': 'huber',
        'alpha': alpha,
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
    
    for horizon in horizons:
        train_h = train_df[train_df['horizon'] == horizon]
        X_train = train_h[feature_cols]
        y_train = train_h['y_target']
        train_weights = np.power(train_h['weight'].values + 1, weight_power)
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
        model = lgb.train(params, train_data, num_boost_round=1500,
                          callbacks=[lgb.log_evaluation(period=1500)])
        models[horizon] = model
    
    return models


def find_optimal_shrinkages(models, val_df, feature_cols):
    """Find optimal shrinkage per horizon with fine grid search."""
    horizons = sorted(val_df['horizon'].unique())
    shrinkages = {}
    
    for horizon in horizons:
        val_h = val_df[val_df['horizon'] == horizon]
        X_val = val_h[feature_cols]
        y_val = val_h['y_target'].values
        w_val = val_h['weight'].values
        
        raw_preds = models[horizon].predict(X_val)
        
        # Fine grid search (0.01 steps)
        best_s, best_r = 0, float('inf')
        for s in np.arange(0.0, 1.01, 0.01):
            r = get_ratio(y_val, raw_preds * s, w_val)
            if r < best_r:
                best_r = r
                best_s = s
        
        shrinkages[int(horizon)] = round(best_s, 2)
        print(f"  H{horizon}: shrinkage={best_s:.2f}, ratio={best_r:.6f}")
    
    return shrinkages


def evaluate_per_horizon(models, shrinkages, val_df, feature_cols):
    """Evaluate per-horizon models with given shrinkages."""
    all_preds = np.zeros(len(val_df))
    
    for horizon in sorted(val_df['horizon'].unique()):
        mask = val_df['horizon'] == horizon
        X_val_h = val_df.loc[mask, feature_cols]
        raw_preds = models[horizon].predict(X_val_h)
        all_preds[mask.values] = raw_preds * shrinkages[horizon]
    
    ratio = get_ratio(val_df['y_target'].values, all_preds, val_df['weight'].values)
    score = weighted_rmse_score(val_df['y_target'].values, all_preds, val_df['weight'].values)
    return ratio, score


def generate_submission(models, shrinkages, test_df, feature_cols, suffix, config):
    """Generate submission file."""
    test_preds = np.zeros(len(test_df))
    
    for horizon in sorted(test_df['horizon'].unique()):
        mask = test_df['horizon'] == horizon
        X_test_h = test_df.loc[mask, feature_cols]
        raw_preds = models[horizon].predict(X_test_h)
        test_preds[mask.values] = raw_preds * shrinkages[horizon]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'prediction': test_preds
    })
    
    output_path = OUTPUT_DIR / f'submission_{suffix}_{timestamp}.csv'
    submission.to_csv(output_path, index=False)
    
    # Save config
    config['timestamp'] = timestamp
    config_path = OUTPUT_DIR / f'submission_{suffix}_config_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Saved: {output_path}")
    print(f"  Pred stats: mean={test_preds.mean():.6f}, std={test_preds.std():.6f}")
    
    return output_path


def main():
    print("=" * 80)
    print("HEDGE FUND FORECASTING - IMPROVED SUBMISSION")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    feature_cols = [c for c in train.columns if c.startswith('feature_')]
    print(f"Train: {len(train):,} rows, Test: {len(test):,} rows, Features: {len(feature_cols)}")
    
    # Split for validation
    train_df, val_df = time_based_split(train, val_ratio=0.2)
    print(f"Train split: {len(train_df):,}, Val split: {len(val_df):,}")
    
    results = {}
    
    # =========================================================================
    # STRATEGY 1: Restore Original Shrinkage Values (Known Good)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STRATEGY 1: Restore Original Shrinkage Values")
    print("=" * 60)
    
    original_shrinkages = {1: 0.12, 3: 0.06, 10: 0.27, 25: 0.29}
    print(f"Shrinkages: {original_shrinkages}")
    
    print("\nTraining models (alpha=0.5)...")
    models_s1 = train_per_horizon_models(train_df, feature_cols, alpha=0.5)
    ratio_s1, score_s1 = evaluate_per_horizon(models_s1, original_shrinkages, val_df, feature_cols)
    print(f"Validation: ratio={ratio_s1:.6f}, score={score_s1:.6f}")
    results['original'] = {'ratio': ratio_s1, 'score': score_s1}
    
    # =========================================================================
    # STRATEGY 2: Alpha=0.1 + Optimized Shrinkage
    # =========================================================================
    print("\n" + "=" * 60)
    print("STRATEGY 2: Alpha=0.1 with Optimized Shrinkage")
    print("=" * 60)
    
    print("\nTraining models (alpha=0.1)...")
    models_s2 = train_per_horizon_models(train_df, feature_cols, alpha=0.1)
    
    print("Finding optimal shrinkages...")
    shrinkages_s2 = find_optimal_shrinkages(models_s2, val_df, feature_cols)
    ratio_s2, score_s2 = evaluate_per_horizon(models_s2, shrinkages_s2, val_df, feature_cols)
    print(f"Validation: ratio={ratio_s2:.6f}, score={score_s2:.6f}")
    results['alpha_0.1'] = {'ratio': ratio_s2, 'score': score_s2, 'shrinkages': shrinkages_s2}
    
    # =========================================================================
    # STRATEGY 3: Alpha=0.5 + Re-Optimized Shrinkage
    # =========================================================================
    print("\n" + "=" * 60)
    print("STRATEGY 3: Alpha=0.5 with Re-Optimized Shrinkage")
    print("=" * 60)
    
    print("Finding optimal shrinkages for alpha=0.5...")
    shrinkages_s3 = find_optimal_shrinkages(models_s1, val_df, feature_cols)
    ratio_s3, score_s3 = evaluate_per_horizon(models_s1, shrinkages_s3, val_df, feature_cols)
    print(f"Validation: ratio={ratio_s3:.6f}, score={score_s3:.6f}")
    results['alpha_0.5_opt'] = {'ratio': ratio_s3, 'score': score_s3, 'shrinkages': shrinkages_s3}
    
    # =========================================================================
    # COMPARISON & BEST STRATEGY SELECTION
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    for name, r in results.items():
        print(f"  {name:20s}: ratio={r['ratio']:.6f}, score={r['score']:.6f}")
    
    # Find best strategy
    best_strategy = min(results.keys(), key=lambda k: results[k]['ratio'])
    print(f"\n  BEST: {best_strategy}")
    
    # =========================================================================
    # GENERATE SUBMISSIONS (Retrain on full data)
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING SUBMISSIONS (Training on Full Data)")
    print("=" * 60)
    
    # Submission 1: Original shrinkages (known good)
    print("\n1. Original Shrinkage Submission:")
    models_full_05 = train_per_horizon_models(train, feature_cols, alpha=0.5)
    generate_submission(models_full_05, original_shrinkages, test, feature_cols, 
                       "original_shrinkage",
                       {'strategy': 'original_shrinkage', 'alpha': 0.5, 
                        'shrinkages': original_shrinkages})
    
    # Submission 2: Alpha=0.1 optimized (if different from best)
    print("\n2. Alpha=0.1 Optimized Submission:")
    models_full_01 = train_per_horizon_models(train, feature_cols, alpha=0.1)
    generate_submission(models_full_01, shrinkages_s2, test, feature_cols,
                       "alpha01_optimized",
                       {'strategy': 'alpha_0.1_optimized', 'alpha': 0.1,
                        'shrinkages': shrinkages_s2})
    
    # Submission 3: Alpha=0.5 re-optimized
    print("\n3. Alpha=0.5 Re-Optimized Submission:")
    generate_submission(models_full_05, shrinkages_s3, test, feature_cols,
                       "alpha05_reoptimized",
                       {'strategy': 'alpha_0.5_reoptimized', 'alpha': 0.5,
                        'shrinkages': shrinkages_s3})
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Generated 3 submission files. Recommend starting with 'original_shrinkage'")
    print("since it matches the first submission that scored 0.053.")
    
    return results


if __name__ == "__main__":
    results = main()
