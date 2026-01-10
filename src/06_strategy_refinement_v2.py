"""
Hedge Fund Time Series Forecasting - Strategy Refinement v2
=============================================================

Key finding from v1: Strategy 6 (sqrt weights) achieved ratio = 1.019,
which is VERY close to beating zero baseline (need ratio < 1).

This script:
1. Deep-dive into why predictions are hurting the metric
2. Try conservative prediction approaches
3. Focus on making fewer but more accurate predictions
4. Explore different weight transformations
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
    """Competition metric - returns score"""
    denom = np.sum(weights * y_true ** 2)
    if denom == 0:
        return 0.0
    ratio = np.sum(weights * (y_true - y_pred) ** 2) / denom
    clipped = np.clip(ratio, 0.0, 1.0)
    return float(np.sqrt(1.0 - clipped))


def get_ratio(y_true, y_pred, weights):
    """Get the ratio component of the metric"""
    denom = np.sum(weights * y_true ** 2)
    if denom == 0:
        return float('inf')
    numer = np.sum(weights * (y_true - y_pred) ** 2)
    return numer / denom


# =============================================================================
# Data Loading
# =============================================================================
def load_data():
    """Load and prepare the data"""
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test


def get_feature_columns(df):
    """Get list of feature columns"""
    return [col for col in df.columns if col.startswith('feature_')]


def time_based_split(train, val_ratio=0.2):
    """Split data based on ts_index"""
    ts_max = train['ts_index'].max()
    ts_min = train['ts_index'].min()
    ts_range = ts_max - ts_min
    
    val_threshold = ts_max - int(ts_range * val_ratio)
    
    train_mask = train['ts_index'] <= val_threshold
    val_mask = train['ts_index'] > val_threshold
    
    train_df = train[train_mask].copy()
    val_df = train[val_mask].copy()
    
    print(f"\nTime-based split:")
    print(f"  Train: {len(train_df):,} samples (ts_index {train_df['ts_index'].min()}-{train_df['ts_index'].max()})")
    print(f"  Val: {len(val_df):,} samples (ts_index {val_df['ts_index'].min()}-{val_df['ts_index'].max()})")
    
    return train_df, val_df


# =============================================================================
# Analysis: Why are predictions hurting?
# =============================================================================
def analyze_prediction_impact(val_df, preds):
    """
    Analyze which samples the predictions are helping vs hurting.
    
    For each sample:
    - Zero prediction error = w * y^2
    - Model prediction error = w * (y - pred)^2
    - If model error < zero error, we're helping
    """
    y = val_df['y_target'].values
    w = val_df['weight'].values
    
    zero_error = w * y ** 2
    model_error = w * (y - preds) ** 2
    
    # Where are we helping?
    helping_mask = model_error < zero_error
    hurting_mask = model_error > zero_error
    
    help_reduction = np.sum(zero_error[helping_mask] - model_error[helping_mask])
    hurt_increase = np.sum(model_error[hurting_mask] - zero_error[hurting_mask])
    
    print(f"\n=== Prediction Impact Analysis ===")
    print(f"  Samples helping: {helping_mask.sum():,} ({helping_mask.sum()/len(y)*100:.1f}%)")
    print(f"  Samples hurting: {hurting_mask.sum():,} ({hurting_mask.sum()/len(y)*100:.1f}%)")
    print(f"  Error reduction (helping): {help_reduction:.4e}")
    print(f"  Error increase (hurting): {hurt_increase:.4e}")
    print(f"  Net effect: {hurt_increase - help_reduction:.4e} (positive = hurting)")
    
    # Analyze characteristics of helped vs hurt samples
    helped_y_mean = np.mean(np.abs(y[helping_mask])) if helping_mask.any() else 0
    hurt_y_mean = np.mean(np.abs(y[hurting_mask])) if hurting_mask.any() else 0
    
    helped_w_mean = np.mean(w[helping_mask]) if helping_mask.any() else 0
    hurt_w_mean = np.mean(w[hurting_mask]) if hurting_mask.any() else 0
    
    print(f"\n  Helped samples: avg |y|={helped_y_mean:.4f}, avg weight={helped_w_mean:.2e}")
    print(f"  Hurt samples: avg |y|={hurt_y_mean:.4f}, avg weight={hurt_w_mean:.2e}")
    
    return helping_mask, hurting_mask


# =============================================================================
# Strategy: Conservative Predictions
# =============================================================================
def train_conservative(train_df, val_df, feature_cols, shrinkage=0.1):
    """
    Make very conservative predictions - shrink towards zero.
    
    The idea: If our predictions have high variance but low signal,
    shrinking them towards zero might help.
    """
    print(f"\n{'='*60}")
    print(f"CONSERVATIVE STRATEGY (shrinkage={shrinkage})")
    print("="*60)
    
    # Transform weights with varying power
    weight_power = 0.25  # Even gentler than sqrt
    train_weights = np.power(train_df['weight'].values + 1, weight_power)
    
    print(f"  Weight transformation: w^{weight_power}")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,  # Slower learning
        'num_leaves': 31,  # Simpler model
        'max_depth': 5,  # Shallower
        'min_child_samples': 200,  # More regularization
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,  # More L1
        'reg_lambda': 1.0,  # More L2
        'n_estimators': 5000,
        'random_state': 42,
        'verbose': -1,
    }
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target']
    w_val = val_df['weight'].values
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=params.get('n_estimators', 5000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=500)
        ]
    )
    
    raw_preds = model.predict(X_val)
    
    # Apply shrinkage
    preds = raw_preds * shrinkage
    
    ratio = get_ratio(y_val.values, preds, w_val)
    score = weighted_rmse_score(y_val.values, preds, w_val)
    
    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Ratio: {ratio:.6f}")
    print(f"  Score: {score:.6f}")
    
    # Sweep shrinkage values
    print("\n  Shrinkage sweep:")
    best_shrinkage = 0
    best_ratio = float('inf')
    for s in np.arange(0.0, 1.1, 0.05):
        s_preds = raw_preds * s
        s_ratio = get_ratio(y_val.values, s_preds, w_val)
        marker = " *" if s_ratio < best_ratio else ""
        print(f"    shrinkage={s:.2f}: ratio={s_ratio:.6f}{marker}")
        if s_ratio < best_ratio:
            best_ratio = s_ratio
            best_shrinkage = s
    
    print(f"\n  Best shrinkage: {best_shrinkage}, ratio: {best_ratio:.6f}")
    
    return model, best_shrinkage, best_ratio


# =============================================================================
# Strategy: Weight power sweep
# =============================================================================
def sweep_weight_power(train_df, val_df, feature_cols):
    """
    Try different weight transformations: w^power
    """
    print(f"\n{'='*60}")
    print("WEIGHT POWER SWEEP")
    print("="*60)
    
    params = {
        'objective': 'regression',
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
        'n_estimators': 1000,
        'random_state': 42,
        'verbose': -1,
    }
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    w_train = train_df['weight'].values
    
    X_val = val_df[feature_cols]
    y_val_arr = val_df['y_target'].values
    w_val = val_df['weight'].values
    
    results = []
    
    for power in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]:
        print(f"\n  Power = {power}...")
        
        # Transform weights
        if power == 0:
            train_weights = np.ones_like(w_train)
        else:
            train_weights = np.power(w_train + 1, power)
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
        val_data = lgb.Dataset(X_val, label=y_val_arr, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['val'],
            num_boost_round=params.get('n_estimators', 1000),
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=1000)
            ]
        )
        
        preds = model.predict(X_val)
        ratio = get_ratio(y_val_arr, preds, w_val)
        score = weighted_rmse_score(y_val_arr, preds, w_val)
        
        print(f"    iter={model.best_iteration}, ratio={ratio:.6f}, score={score:.6f}")
        
        # Also try applying shrinkage
        best_shrink_ratio = float('inf')
        best_shrink = 0
        for s in np.arange(0.0, 1.1, 0.1):
            s_ratio = get_ratio(y_val_arr, preds * s, w_val)
            if s_ratio < best_shrink_ratio:
                best_shrink_ratio = s_ratio
                best_shrink = s
        
        print(f"    with shrinkage={best_shrink:.1f}: ratio={best_shrink_ratio:.6f}")
        
        results.append({
            'power': power,
            'raw_ratio': ratio,
            'best_shrink': best_shrink,
            'shrunk_ratio': best_shrink_ratio,
        })
    
    # Find best
    best = min(results, key=lambda x: x['shrunk_ratio'])
    print(f"\n  Best: power={best['power']}, shrinkage={best['best_shrink']}, ratio={best['shrunk_ratio']:.6f}")
    
    return results


# =============================================================================
# Strategy: Huber/MAE loss (more robust to outliers)
# =============================================================================
def train_robust_loss(train_df, val_df, feature_cols):
    """
    Try robust loss functions that are less sensitive to outliers.
    """
    print(f"\n{'='*60}")
    print("ROBUST LOSS FUNCTIONS")
    print("="*60)
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    
    X_val = val_df[feature_cols]
    y_val_arr = val_df['y_target'].values
    w_val = val_df['weight'].values
    
    # Weight transformation (use sqrt based on earlier findings)
    train_weights = np.sqrt(train_df['weight'].values + 1)
    
    losses = {
        'regression': 'RMSE',
        'regression_l1': 'MAE',
        'huber': 'Huber',
    }
    
    results = {}
    
    for objective, name in losses.items():
        print(f"\n  {name} Loss:")
        
        params = {
            'objective': objective,
            'metric': 'rmse',  # Always evaluate with RMSE
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'n_estimators': 1000,
            'random_state': 42,
            'verbose': -1,
        }
        
        if objective == 'huber':
            params['alpha'] = 0.5  # Huber delta parameter
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
        val_data = lgb.Dataset(X_val, label=y_val_arr, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['val'],
            num_boost_round=params.get('n_estimators', 1000),
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=1000)
            ]
        )
        
        preds = model.predict(X_val)
        ratio = get_ratio(y_val_arr, preds, w_val)
        
        # Try shrinkage
        best_s, best_r = 0, float('inf')
        for s in np.arange(0.0, 1.1, 0.1):
            r = get_ratio(y_val_arr, preds * s, w_val)
            if r < best_r:
                best_r = r
                best_s = s
        
        print(f"    Raw ratio: {ratio:.6f}")
        print(f"    Best shrinkage={best_s:.1f}: ratio={best_r:.6f}")
        
        results[name] = {'model': model, 'ratio': best_r, 'shrinkage': best_s, 'preds': preds}
    
    return results


# =============================================================================
# Strategy: Quantile Regression
# =============================================================================
def train_quantile(train_df, val_df, feature_cols, quantile=0.5):
    """
    Try quantile regression to predict median instead of mean.
    """
    print(f"\n{'='*60}")
    print(f"QUANTILE REGRESSION (q={quantile})")
    print("="*60)
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    
    X_val = val_df[feature_cols]
    y_val_arr = val_df['y_target'].values
    w_val = val_df['weight'].values
    
    # Weight transformation
    train_weights = np.sqrt(train_df['weight'].values + 1)
    
    params = {
        'objective': 'quantile',
        'alpha': quantile,
        'metric': 'quantile',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'n_estimators': 1000,
        'random_state': 42,
        'verbose': -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    val_data = lgb.Dataset(X_val, label=y_val_arr, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['val'],
        num_boost_round=params.get('n_estimators', 1000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=1000)
        ]
    )
    
    preds = model.predict(X_val)
    ratio = get_ratio(y_val_arr, preds, w_val)
    
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Raw ratio: {ratio:.6f}")
    
    # Try shrinkage
    best_s, best_r = 0, float('inf')
    for s in np.arange(0.0, 1.1, 0.1):
        r = get_ratio(y_val_arr, preds * s, w_val)
        if r < best_r:
            best_r = r
            best_s = s
    
    print(f"  Best shrinkage={best_s:.1f}: ratio={best_r:.6f}")
    
    return model, best_r, best_s


# =============================================================================
# Strategy: Per-horizon with optimal shrinkage
# =============================================================================
def train_per_horizon_with_shrinkage(train_df, val_df, feature_cols):
    """
    Train per-horizon models and find optimal shrinkage for each.
    """
    print(f"\n{'='*60}")
    print("PER-HORIZON MODELS WITH OPTIMAL SHRINKAGE")
    print("="*60)
    
    horizons = sorted(train_df['horizon'].unique())
    models = {}
    shrinkages = {}
    all_preds = np.zeros(len(val_df))
    
    params = {
        'objective': 'regression',
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
        'n_estimators': 1000,
        'random_state': 42,
        'verbose': -1,
    }
    
    for horizon in horizons:
        print(f"\n  Horizon {horizon}:")
        
        train_h = train_df[train_df['horizon'] == horizon]
        val_h = val_df[val_df['horizon'] == horizon]
        
        X_train = train_h[feature_cols]
        y_train = train_h['y_target']
        train_weights = np.sqrt(train_h['weight'].values + 1)
        
        X_val = val_h[feature_cols]
        y_val = val_h['y_target'].values
        w_val = val_h['weight'].values
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['val'],
            num_boost_round=params.get('n_estimators', 1000),
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=1000)
            ]
        )
        
        raw_preds = model.predict(X_val)
        
        # Find optimal shrinkage for this horizon
        best_s, best_r = 0, float('inf')
        for s in np.arange(0.0, 1.1, 0.05):
            r = get_ratio(y_val, raw_preds * s, w_val)
            if r < best_r:
                best_r = r
                best_s = s
        
        models[horizon] = model
        shrinkages[horizon] = best_s
        
        mask = val_df['horizon'] == horizon
        all_preds[mask.values] = raw_preds * best_s
        
        print(f"    iter={model.best_iteration}, best_shrink={best_s:.2f}, ratio={best_r:.6f}")
    
    # Overall score
    y_val = val_df['y_target'].values
    w_val = val_df['weight'].values
    ratio = get_ratio(y_val, all_preds, w_val)
    score = weighted_rmse_score(y_val, all_preds, w_val)
    
    print(f"\n  Overall: ratio={ratio:.6f}, score={score:.6f}")
    
    return models, shrinkages, ratio, all_preds


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 80)
    print("HEDGE FUND TIME SERIES FORECASTING - STRATEGY REFINEMENT v2")
    print("=" * 80)
    
    # Load data
    train, test = load_data()
    feature_cols = get_feature_columns(train)
    
    # Split
    train_df, val_df = time_based_split(train, val_ratio=0.2)
    
    # Run strategies
    results = {}
    
    # 1. Conservative approach
    model_cons, shrink_cons, ratio_cons = train_conservative(train_df, val_df, feature_cols, shrinkage=0.1)
    results['Conservative'] = ratio_cons
    
    # 2. Weight power sweep
    power_results = sweep_weight_power(train_df, val_df, feature_cols)
    best_power = min(power_results, key=lambda x: x['shrunk_ratio'])
    results['Weight Power Sweep'] = best_power['shrunk_ratio']
    
    # 3. Robust losses
    robust_results = train_robust_loss(train_df, val_df, feature_cols)
    best_robust = min(robust_results.items(), key=lambda x: x[1]['ratio'])
    results[f'Robust ({best_robust[0]})'] = best_robust[1]['ratio']
    
    # 4. Quantile regression
    model_q, ratio_q, shrink_q = train_quantile(train_df, val_df, feature_cols, quantile=0.5)
    results['Quantile (0.5)'] = ratio_q
    
    # 5. Per-horizon with shrinkage
    models_ph, shrinkages_ph, ratio_ph, preds_ph = train_per_horizon_with_shrinkage(train_df, val_df, feature_cols)
    results['Per-Horizon + Shrinkage'] = ratio_ph
    
    # Zero baseline
    results['Zero Baseline'] = 1.0  # By definition
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - ALL STRATEGIES")
    print("=" * 80)
    print("\nAll ratios (lower is better, need < 1.0 to beat zero):")
    for name, ratio in sorted(results.items(), key=lambda x: x[1]):
        beats_zero = "✓ BEATS ZERO!" if ratio < 1.0 else ""
        print(f"  {name}: {ratio:.6f} {beats_zero}")
    
    best_strategy = min(results, key=results.get)
    best_ratio = results[best_strategy]
    
    print(f"\nBest strategy: {best_strategy} with ratio={best_ratio:.6f}")
    
    if best_ratio < 1.0:
        score = np.sqrt(1.0 - best_ratio)
        print(f"  This would score: {score:.6f}")
    else:
        print(f"  Gap to beat zero: {best_ratio - 1.0:.6f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dict = {
        'timestamp': timestamp,
        'best_strategy': best_strategy,
        'best_ratio': best_ratio,
        'all_ratios': {k: float(v) for k, v in results.items()},
        'power_sweep': power_results,
        'per_horizon_shrinkages': {int(k): float(v) for k, v in shrinkages_ph.items()},
    }
    
    results_path = OUTPUT_DIR / f'refinement_v2_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results_dict


if __name__ == "__main__":
    results = main()
