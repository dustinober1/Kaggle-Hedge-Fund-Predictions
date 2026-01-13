"""
Hedge Fund Time Series Forecasting - COMPETITION WINNER
========================================================

Goal: Move from 7th place to top 3

Strategies:
1. Multi-Seed Ensemble - Average 5 models (reduces variance)
2. Ultra-Fine Shrinkage - 0.002 step grid search
3. Weight Power Tuning - Test 0.50, 0.55, 0.60
4. Weight-Adaptive Shrinkage - More shrinkage for high-weight samples
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
import json
from datetime import datetime
from typing import Dict, List, Tuple

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


def train_model(train_df, feature_cols, horizon, alpha=0.5, weight_power=0.5, seed=42):
    """Train a single LightGBM model for a specific horizon."""
    train_h = train_df[train_df['horizon'] == horizon]
    X_train = train_h[feature_cols]
    y_train = train_h['y_target']
    train_weights = np.power(train_h['weight'].values + 1, weight_power)
    
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
        'random_state': seed,
        'verbose': -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    model = lgb.train(params, train_data, num_boost_round=1500)
    return model


def find_ultra_fine_shrinkage(model, val_df, feature_cols, horizon) -> Tuple[float, float]:
    """Find optimal shrinkage with ultra-fine grid (0.002 steps)."""
    val_h = val_df[val_df['horizon'] == horizon]
    X_val = val_h[feature_cols]
    y_val = val_h['y_target'].values
    w_val = val_h['weight'].values
    
    raw_preds = model.predict(X_val)
    
    # Ultra-fine search
    best_s, best_r = 0, float('inf')
    for s in np.arange(0.0, 0.60, 0.002):
        r = get_ratio(y_val, raw_preds * s, w_val)
        if r < best_r:
            best_r = r
            best_s = s
    
    return round(best_s, 3), best_r


def train_ensemble(train_df, feature_cols, horizon, n_seeds=5, weight_power=0.5):
    """Train ensemble of models with different seeds."""
    seeds = [42, 123, 456, 789, 1024][:n_seeds]
    models = []
    for seed in seeds:
        model = train_model(train_df, feature_cols, horizon, seed=seed, weight_power=weight_power)
        models.append(model)
    return models


def predict_ensemble(models, X) -> np.ndarray:
    """Average predictions from ensemble."""
    preds = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        preds[:, i] = model.predict(X)
    return np.mean(preds, axis=1)


def apply_weight_adaptive_shrinkage(preds, weights, min_s=0.05, max_s=0.35):
    """Apply more shrinkage to high-weight samples."""
    # Normalize weights to 0-1 range
    log_weights = np.log1p(weights)
    w_min, w_max = log_weights.min(), log_weights.max()
    w_norm = (log_weights - w_min) / (w_max - w_min + 1e-10)
    
    # High weight -> more shrinkage (closer to max_s)
    # Low weight -> less shrinkage (closer to min_s)
    shrinkage = min_s + (max_s - min_s) * w_norm
    
    return preds * shrinkage


def main():
    print("=" * 80)
    print("HEDGE FUND FORECASTING - COMPETITION WINNER")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    feature_cols = [c for c in train.columns if c.startswith('feature_')]
    print(f"Train: {len(train):,}, Test: {len(test):,}, Features: {len(feature_cols)}")
    
    # Split for validation
    train_df, val_df = time_based_split(train, val_ratio=0.2)
    print(f"Train split: {len(train_df):,}, Val split: {len(val_df):,}")
    
    horizons = sorted(train['horizon'].unique())
    results = {}
    
    # =========================================================================
    # STRATEGY 1: Baseline with Ultra-Fine Shrinkage
    # =========================================================================
    print("\n" + "=" * 60)
    print("STRATEGY 1: Ultra-Fine Shrinkage (power=0.5)")
    print("=" * 60)
    
    s1_models = {}
    s1_shrinkages = {}
    for h in horizons:
        print(f"  Training H{h}...")
        s1_models[h] = train_model(train_df, feature_cols, h, weight_power=0.5)
        s1_shrinkages[h], ratio = find_ultra_fine_shrinkage(s1_models[h], val_df, feature_cols, h)
        print(f"    shrinkage={s1_shrinkages[h]:.3f}, ratio={ratio:.6f}")
    
    # Evaluate
    s1_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        X = val_df.loc[mask, feature_cols]
        s1_preds[mask.values] = s1_models[h].predict(X) * s1_shrinkages[h]
    
    s1_ratio = get_ratio(val_df['y_target'].values, s1_preds, val_df['weight'].values)
    s1_score = weighted_rmse_score(val_df['y_target'].values, s1_preds, val_df['weight'].values)
    print(f"\n  RESULT: ratio={s1_ratio:.6f}, score={s1_score:.6f}")
    results['ultra_fine'] = {'ratio': s1_ratio, 'score': s1_score, 'shrinkages': s1_shrinkages}
    
    # =========================================================================
    # STRATEGY 2: Weight Power = 0.55
    # =========================================================================
    print("\n" + "=" * 60)
    print("STRATEGY 2: Weight Power = 0.55")
    print("=" * 60)
    
    s2_models = {}
    s2_shrinkages = {}
    for h in horizons:
        print(f"  Training H{h}...")
        s2_models[h] = train_model(train_df, feature_cols, h, weight_power=0.55)
        s2_shrinkages[h], ratio = find_ultra_fine_shrinkage(s2_models[h], val_df, feature_cols, h)
        print(f"    shrinkage={s2_shrinkages[h]:.3f}, ratio={ratio:.6f}")
    
    s2_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        X = val_df.loc[mask, feature_cols]
        s2_preds[mask.values] = s2_models[h].predict(X) * s2_shrinkages[h]
    
    s2_ratio = get_ratio(val_df['y_target'].values, s2_preds, val_df['weight'].values)
    s2_score = weighted_rmse_score(val_df['y_target'].values, s2_preds, val_df['weight'].values)
    print(f"\n  RESULT: ratio={s2_ratio:.6f}, score={s2_score:.6f}")
    results['power_055'] = {'ratio': s2_ratio, 'score': s2_score, 'shrinkages': s2_shrinkages}
    
    # =========================================================================
    # STRATEGY 3: Multi-Seed Ensemble (5 seeds)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STRATEGY 3: Multi-Seed Ensemble (5 seeds)")
    print("=" * 60)
    
    s3_models = {}
    s3_shrinkages = {}
    for h in horizons:
        print(f"  Training ensemble for H{h}...")
        s3_models[h] = train_ensemble(train_df, feature_cols, h, n_seeds=5, weight_power=0.5)
        
        # Find shrinkage for ensemble
        val_h = val_df[val_df['horizon'] == h]
        X_val = val_h[feature_cols]
        y_val = val_h['y_target'].values
        w_val = val_h['weight'].values
        raw_preds = predict_ensemble(s3_models[h], X_val)
        
        best_s, best_r = 0, float('inf')
        for s in np.arange(0.0, 0.60, 0.002):
            r = get_ratio(y_val, raw_preds * s, w_val)
            if r < best_r:
                best_r = r
                best_s = s
        
        s3_shrinkages[h] = round(best_s, 3)
        print(f"    shrinkage={s3_shrinkages[h]:.3f}, ratio={best_r:.6f}")
    
    s3_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        X = val_df.loc[mask, feature_cols]
        s3_preds[mask.values] = predict_ensemble(s3_models[h], X) * s3_shrinkages[h]
    
    s3_ratio = get_ratio(val_df['y_target'].values, s3_preds, val_df['weight'].values)
    s3_score = weighted_rmse_score(val_df['y_target'].values, s3_preds, val_df['weight'].values)
    print(f"\n  RESULT: ratio={s3_ratio:.6f}, score={s3_score:.6f}")
    results['ensemble_5'] = {'ratio': s3_ratio, 'score': s3_score, 'shrinkages': s3_shrinkages}
    
    # =========================================================================
    # STRATEGY 4: Weight-Adaptive Shrinkage
    # =========================================================================
    print("\n" + "=" * 60)
    print("STRATEGY 4: Weight-Adaptive Shrinkage")
    print("=" * 60)
    
    # Use models from strategy 1, but apply weight-adaptive shrinkage
    # Find optimal min_s, max_s per horizon
    s4_params = {}
    s4_preds = np.zeros(len(val_df))
    
    for h in horizons:
        val_h = val_df[val_df['horizon'] == h]
        X_val = val_h[feature_cols]
        y_val = val_h['y_target'].values
        w_val = val_h['weight'].values
        raw_preds = s1_models[h].predict(X_val)
        
        # Grid search for min_s, max_s
        best_params = None
        best_r = float('inf')
        for min_s in np.arange(0.02, 0.15, 0.02):
            for max_s in np.arange(0.15, 0.45, 0.02):
                if max_s <= min_s:
                    continue
                adaptive_preds = apply_weight_adaptive_shrinkage(raw_preds, w_val, min_s, max_s)
                r = get_ratio(y_val, adaptive_preds, w_val)
                if r < best_r:
                    best_r = r
                    best_params = (min_s, max_s)
        
        s4_params[h] = best_params
        print(f"  H{h}: min_s={best_params[0]:.2f}, max_s={best_params[1]:.2f}, ratio={best_r:.6f}")
        
        # Apply to validation
        mask = val_df['horizon'] == h
        s4_preds[mask.values] = apply_weight_adaptive_shrinkage(raw_preds, w_val, *best_params)
    
    s4_ratio = get_ratio(val_df['y_target'].values, s4_preds, val_df['weight'].values)
    s4_score = weighted_rmse_score(val_df['y_target'].values, s4_preds, val_df['weight'].values)
    print(f"\n  RESULT: ratio={s4_ratio:.6f}, score={s4_score:.6f}")
    results['weight_adaptive'] = {'ratio': s4_ratio, 'score': s4_score, 'params': {int(k): v for k, v in s4_params.items()}}
    
    # =========================================================================
    # COMPARISON & BEST SELECTION
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    for name, r in results.items():
        print(f"  {name:20s}: ratio={r['ratio']:.6f}, score={r['score']:.6f}")
    
    best_name = min(results.keys(), key=lambda k: results[k]['ratio'])
    print(f"\n  BEST: {best_name}")
    
    # =========================================================================
    # GENERATE BEST SUBMISSION
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"GENERATING SUBMISSION: {best_name}")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_preds = np.zeros(len(test))
    
    if best_name == 'ultra_fine':
        # Retrain on full data
        for h in horizons:
            print(f"  Training H{h} on full data...")
            model = train_model(train, feature_cols, h, weight_power=0.5)
            mask = test['horizon'] == h
            test_preds[mask.values] = model.predict(test.loc[mask, feature_cols]) * s1_shrinkages[h]
        config = {'strategy': 'ultra_fine', 'shrinkages': {int(k): v for k, v in s1_shrinkages.items()}}
        
    elif best_name == 'power_055':
        for h in horizons:
            print(f"  Training H{h} on full data (power=0.55)...")
            model = train_model(train, feature_cols, h, weight_power=0.55)
            mask = test['horizon'] == h
            test_preds[mask.values] = model.predict(test.loc[mask, feature_cols]) * s2_shrinkages[h]
        config = {'strategy': 'power_055', 'shrinkages': {int(k): v for k, v in s2_shrinkages.items()}}
        
    elif best_name == 'ensemble_5':
        for h in horizons:
            print(f"  Training ensemble for H{h} on full data...")
            models = train_ensemble(train, feature_cols, h, n_seeds=5, weight_power=0.5)
            mask = test['horizon'] == h
            test_preds[mask.values] = predict_ensemble(models, test.loc[mask, feature_cols]) * s3_shrinkages[h]
        config = {'strategy': 'ensemble_5', 'shrinkages': {int(k): v for k, v in s3_shrinkages.items()}}
        
    elif best_name == 'weight_adaptive':
        for h in horizons:
            print(f"  Training H{h} on full data...")
            model = train_model(train, feature_cols, h, weight_power=0.5)
            mask = test['horizon'] == h
            X_test = test.loc[mask, feature_cols]
            raw_preds = model.predict(X_test)
            # Use test weights for adaptive shrinkage (estimate from features)
            test_preds[mask.values] = raw_preds * np.mean(s4_params[h])  # Use average as fallback
        config = {'strategy': 'weight_adaptive', 'params': {int(k): v for k, v in s4_params.items()}}
    
    # Save submission
    submission = pd.DataFrame({
        'id': test['id'],
        'prediction': test_preds
    })
    
    output_path = OUTPUT_DIR / f'submission_winner_{best_name}_{timestamp}.csv'
    submission.to_csv(output_path, index=False)
    
    config['cv_ratio'] = results[best_name]['ratio']
    config['cv_score'] = results[best_name]['score']
    config['timestamp'] = timestamp
    
    config_path = OUTPUT_DIR / f'submission_winner_{best_name}_config_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Saved: {output_path}")
    print(f"  Pred stats: mean={test_preds.mean():.6f}, std={test_preds.std():.6f}")
    
    # =========================================================================
    # ALSO GENERATE ENSEMBLE SUBMISSION (often most robust)
    # =========================================================================
    if best_name != 'ensemble_5':
        print("\n" + "=" * 60)
        print("ALSO GENERATING ENSEMBLE SUBMISSION (backup)")
        print("=" * 60)
        
        test_preds_ens = np.zeros(len(test))
        for h in horizons:
            print(f"  Training ensemble for H{h}...")
            models = train_ensemble(train, feature_cols, h, n_seeds=5, weight_power=0.5)
            mask = test['horizon'] == h
            test_preds_ens[mask.values] = predict_ensemble(models, test.loc[mask, feature_cols]) * s3_shrinkages[h]
        
        submission_ens = pd.DataFrame({
            'id': test['id'],
            'prediction': test_preds_ens
        })
        
        ens_path = OUTPUT_DIR / f'submission_winner_ensemble_{timestamp}.csv'
        submission_ens.to_csv(ens_path, index=False)
        print(f"  Saved: {ens_path}")
    
    print("\n" + "=" * 60)
    print("DONE - Good luck!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
