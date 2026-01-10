"""
Hedge Fund Time Series Forecasting - Advanced Tuning
=====================================================

Building on top of the working baseline (score ~0.053), this script explores:
1. Fine-tune Huber loss parameters (alpha/delta values)
2. Per-sample shrinkage based on prediction confidence
3. Cross-validate shrinkage on multiple time folds

Current best config:
- Huber loss with alpha=0.5
- sqrt(weight + 1) transformation
- Per-horizon models with shrinkage: H1=0.12, H3=0.06, H10=0.27, H25=0.29
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
import json
from datetime import datetime
from collections import defaultdict

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
# Data Loading
# =============================================================================
def load_data():
    """Load data"""
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test


def get_feature_columns(df):
    """Get feature columns"""
    return [col for col in df.columns if col.startswith('feature_')]


# =============================================================================
# Time Series Cross-Validation
# =============================================================================
def time_series_cv_splits(train, n_folds=3, val_ratio=0.15):
    """
    Create time-series cross-validation splits.
    
    For each fold, we use a progressively larger training window and
    validate on the next chunk of time.
    
    Returns list of (train_indices, val_indices) tuples.
    """
    ts_max = train['ts_index'].max()
    ts_min = train['ts_index'].min()
    ts_range = ts_max - ts_min
    
    # Calculate validation window size
    val_size = int(ts_range * val_ratio)
    
    # Reserve space for all validation folds + buffer
    total_val_space = val_size * n_folds
    train_end = ts_max - total_val_space
    
    splits = []
    for fold in range(n_folds):
        # Validation window for this fold
        val_start = train_end + fold * val_size + 1
        val_end = val_start + val_size - 1
        
        # Training data: everything before validation
        train_mask = train['ts_index'] <= (val_start - 1)
        val_mask = (train['ts_index'] >= val_start) & (train['ts_index'] <= val_end)
        
        train_idx = train.index[train_mask].tolist()
        val_idx = train.index[val_mask].tolist()
        
        splits.append((train_idx, val_idx))
        
        print(f"  Fold {fold + 1}: Train ts_index up to {val_start - 1}, "
              f"Val ts_index {val_start}-{val_end} "
              f"(Train: {len(train_idx):,}, Val: {len(val_idx):,})")
    
    return splits


# =============================================================================
# EXPERIMENT 1: Huber Alpha (Delta) Tuning
# =============================================================================
def experiment_huber_alpha(train, feature_cols, n_folds=3):
    """
    Test different Huber loss alpha (delta) parameters.
    
    In LightGBM's Huber loss:
    - alpha controls the transition point between quadratic and linear loss
    - Smaller alpha = more robust to outliers (more like MAE)
    - Larger alpha = more like MSE
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Huber Alpha (Delta) Tuning")
    print("=" * 80)
    
    # Alpha values to test (covering wide range)
    alphas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    print(f"\nTesting alpha values: {alphas}")
    print(f"Using {n_folds}-fold time-series CV\n")
    
    splits = time_series_cv_splits(train, n_folds=n_folds)
    
    results = []
    
    for alpha in alphas:
        print(f"\nAlpha = {alpha}:")
        fold_ratios = []
        fold_shrinkages = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            train_fold = train.iloc[train_idx]
            val_fold = train.iloc[val_idx]
            
            X_train = train_fold[feature_cols]
            y_train = train_fold['y_target']
            train_weights = np.sqrt(train_fold['weight'].values + 1)
            
            X_val = val_fold[feature_cols]
            y_val = val_fold['y_target'].values
            w_val = val_fold['weight'].values
            
            params = {
                'objective': 'huber',
                'alpha': alpha,  # This is the delta parameter
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
            
            train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=params['n_estimators'],
                callbacks=[lgb.log_evaluation(period=99999)]  # Suppress
            )
            
            raw_preds = model.predict(X_val)
            
            # Find optimal shrinkage
            best_s, best_r = 0, float('inf')
            for s in np.arange(0.0, 0.6, 0.02):  # Focus on smaller shrinkages
                r = get_ratio(y_val, raw_preds * s, w_val)
                if r < best_r:
                    best_r = r
                    best_s = s
            
            fold_ratios.append(best_r)
            fold_shrinkages.append(best_s)
            
            print(f"  Fold {fold + 1}: ratio={best_r:.6f}, shrinkage={best_s:.2f}")
        
        avg_ratio = np.mean(fold_ratios)
        std_ratio = np.std(fold_ratios)
        avg_shrinkage = np.mean(fold_shrinkages)
        
        results.append({
            'alpha': alpha,
            'avg_ratio': avg_ratio,
            'std_ratio': std_ratio,
            'avg_shrinkage': avg_shrinkage,
            'fold_ratios': fold_ratios,
            'fold_shrinkages': fold_shrinkages,
        })
        
        beats_zero = "✓ BEATS ZERO!" if avg_ratio < 1.0 else ""
        print(f"  --> Avg ratio: {avg_ratio:.6f} ± {std_ratio:.6f} {beats_zero}")
    
    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY - Huber Alpha Results:")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x['avg_ratio']):
        beats = "✓" if r['avg_ratio'] < 1.0 else " "
        print(f"  {beats} alpha={r['alpha']:.2f}: ratio={r['avg_ratio']:.6f} ± {r['std_ratio']:.6f}, "
              f"shrinkage={r['avg_shrinkage']:.2f}")
    
    best = min(results, key=lambda x: x['avg_ratio'])
    print(f"\nBest alpha: {best['alpha']}")
    
    return results


# =============================================================================
# EXPERIMENT 2: Per-Sample Shrinkage
# =============================================================================
def experiment_sample_shrinkage(train, feature_cols, n_folds=3):
    """
    Apply different shrinkage based on prediction characteristics.
    
    Ideas tested:
    1. Shrink more for extreme predictions
    2. Shrink more for low-weight samples  
    3. Shrink based on model uncertainty (using prediction intervals)
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Per-Sample Shrinkage")
    print("=" * 80)
    
    print(f"\nUsing {n_folds}-fold time-series CV\n")
    
    splits = time_series_cv_splits(train, n_folds=n_folds)
    
    strategies = [
        'uniform',           # Baseline: same shrinkage for all
        'magnitude_adaptive', # More shrinkage for extreme predictions
        'weight_adaptive',    # More shrinkage for low-weight samples
        'combined',           # Combine magnitude and weight
    ]
    
    results = {s: {'ratios': [], 'params': None} for s in strategies}
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold + 1} ---")
        
        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]
        
        X_train = train_fold[feature_cols]
        y_train = train_fold['y_target']
        train_weights = np.sqrt(train_fold['weight'].values + 1)
        
        X_val = val_fold[feature_cols]
        y_val = val_fold['y_target'].values
        w_val = val_fold['weight'].values
        
        # Train model
        params = {
            'objective': 'huber',
            'alpha': 0.5,
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
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
        model = lgb.train(
            params, train_data, num_boost_round=params['n_estimators'],
            callbacks=[lgb.log_evaluation(period=99999)]
        )
        
        raw_preds = model.predict(X_val)
        
        # Strategy 1: Uniform shrinkage (baseline)
        best_s_uniform, best_r_uniform = 0, float('inf')
        for s in np.arange(0.0, 0.6, 0.02):
            r = get_ratio(y_val, raw_preds * s, w_val)
            if r < best_r_uniform:
                best_r_uniform = r
                best_s_uniform = s
        
        results['uniform']['ratios'].append(best_r_uniform)
        if fold == 0:
            results['uniform']['params'] = {'shrinkage': best_s_uniform}
        print(f"  Uniform: ratio={best_r_uniform:.6f}, shrinkage={best_s_uniform:.2f}")
        
        # Strategy 2: Magnitude-adaptive shrinkage
        # Shrink more for predictions far from zero
        pred_abs = np.abs(raw_preds)
        pred_percentiles = np.percentile(pred_abs, [50, 90, 99])
        
        best_r_mag = float('inf')
        best_params_mag = None
        for base_s in np.arange(0.1, 0.4, 0.05):
            for decay in np.arange(0.01, 0.1, 0.02):
                # Shrinkage decreases for extreme predictions
                shrinkage_factors = base_s * np.exp(-decay * pred_abs)
                preds_shrunk = raw_preds * shrinkage_factors
                r = get_ratio(y_val, preds_shrunk, w_val)
                if r < best_r_mag:
                    best_r_mag = r
                    best_params_mag = {'base': base_s, 'decay': decay}
        
        results['magnitude_adaptive']['ratios'].append(best_r_mag)
        if fold == 0:
            results['magnitude_adaptive']['params'] = best_params_mag
        print(f"  Magnitude-adaptive: ratio={best_r_mag:.6f}")
        
        # Strategy 3: Weight-adaptive shrinkage
        # Higher shrinkage for low-weight samples (less confident)
        w_normalized = w_val / (w_val.max() + 1e-10)
        
        best_r_weight = float('inf')
        best_params_weight = None
        for min_s in np.arange(0.05, 0.2, 0.03):
            for max_s in np.arange(0.2, 0.5, 0.05):
                if max_s <= min_s:
                    continue
                # Higher weight = lower shrinkage (more confident)
                shrinkage_factors = max_s - (max_s - min_s) * np.sqrt(w_normalized)
                preds_shrunk = raw_preds * shrinkage_factors
                r = get_ratio(y_val, preds_shrunk, w_val)
                if r < best_r_weight:
                    best_r_weight = r
                    best_params_weight = {'min_s': min_s, 'max_s': max_s}
        
        results['weight_adaptive']['ratios'].append(best_r_weight)
        if fold == 0:
            results['weight_adaptive']['params'] = best_params_weight
        print(f"  Weight-adaptive: ratio={best_r_weight:.6f}")
        
        # Strategy 4: Combined
        best_r_combined = float('inf')
        best_params_combined = None
        for base_s in np.arange(0.1, 0.3, 0.05):
            for mag_decay in [0.02, 0.05]:
                for weight_factor in [0.5, 1.0]:
                    # Combine magnitude and weight effects
                    mag_effect = np.exp(-mag_decay * pred_abs)
                    weight_effect = np.sqrt(w_normalized) * weight_factor + (1 - weight_factor)
                    shrinkage_factors = base_s * mag_effect * weight_effect
                    preds_shrunk = raw_preds * shrinkage_factors
                    r = get_ratio(y_val, preds_shrunk, w_val)
                    if r < best_r_combined:
                        best_r_combined = r
                        best_params_combined = {
                            'base': base_s, 
                            'mag_decay': mag_decay,
                            'weight_factor': weight_factor
                        }
        
        results['combined']['ratios'].append(best_r_combined)
        if fold == 0:
            results['combined']['params'] = best_params_combined
        print(f"  Combined: ratio={best_r_combined:.6f}")
    
    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY - Per-Sample Shrinkage Results:")
    print("-" * 60)
    
    final_results = []
    for strategy, data in results.items():
        avg_ratio = np.mean(data['ratios'])
        std_ratio = np.std(data['ratios'])
        beats = "✓" if avg_ratio < 1.0 else " "
        print(f"  {beats} {strategy}: ratio={avg_ratio:.6f} ± {std_ratio:.6f}")
        final_results.append({
            'strategy': strategy,
            'avg_ratio': avg_ratio,
            'std_ratio': std_ratio,
            'params': data['params']
        })
    
    best = min(final_results, key=lambda x: x['avg_ratio'])
    print(f"\nBest strategy: {best['strategy']}")
    
    return final_results


# =============================================================================
# EXPERIMENT 3: Cross-Validated Per-Horizon Shrinkage
# =============================================================================
def experiment_cv_shrinkage(train, feature_cols, n_folds=3):
    """
    Use multiple time folds to find robust per-horizon shrinkage values.
    
    This reduces the risk of overfitting shrinkage to a single validation split.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: CV Per-Horizon Shrinkage")
    print("=" * 80)
    
    horizons = sorted(train['horizon'].unique())
    print(f"\nHorizons: {horizons}")
    print(f"Using {n_folds}-fold time-series CV\n")
    
    splits = time_series_cv_splits(train, n_folds=n_folds)
    
    # Collect shrinkage values across folds
    horizon_shrinkages = {h: [] for h in horizons}
    horizon_ratios = {h: [] for h in horizons}
    overall_ratios = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold + 1} ---")
        
        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]
        
        all_preds = np.zeros(len(val_fold))
        
        for horizon in horizons:
            train_h = train_fold[train_fold['horizon'] == horizon]
            val_h = val_fold[val_fold['horizon'] == horizon]
            
            if len(val_h) == 0:
                print(f"  Horizon {horizon}: No validation samples")
                continue
            
            X_train = train_h[feature_cols]
            y_train = train_h['y_target']
            train_weights = np.sqrt(train_h['weight'].values + 1)
            
            X_val = val_h[feature_cols]
            y_val = val_h['y_target'].values
            w_val = val_h['weight'].values
            
            params = {
                'objective': 'huber',
                'alpha': 0.5,
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
            
            train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
            model = lgb.train(
                params, train_data, num_boost_round=params['n_estimators'],
                callbacks=[lgb.log_evaluation(period=99999)]
            )
            
            raw_preds = model.predict(X_val)
            
            # Find optimal shrinkage with finer grid
            best_s, best_r = 0, float('inf')
            for s in np.arange(0.0, 0.6, 0.01):
                r = get_ratio(y_val, raw_preds * s, w_val)
                if r < best_r:
                    best_r = r
                    best_s = s
            
            horizon_shrinkages[horizon].append(best_s)
            horizon_ratios[horizon].append(best_r)
            
            # Store predictions for overall metric
            mask = val_fold['horizon'] == horizon
            all_preds[mask.values] = raw_preds * best_s
            
            print(f"  Horizon {horizon}: ratio={best_r:.6f}, shrinkage={best_s:.2f}")
        
        # Overall fold ratio
        y_val_full = val_fold['y_target'].values
        w_val_full = val_fold['weight'].values
        fold_ratio = get_ratio(y_val_full, all_preds, w_val_full)
        overall_ratios.append(fold_ratio)
        
        beats = "✓" if fold_ratio < 1.0 else ""
        print(f"  --> Overall fold ratio: {fold_ratio:.6f} {beats}")
    
    # Summary with robust shrinkage estimates
    print("\n" + "-" * 60)
    print("SUMMARY - Robust Per-Horizon Shrinkages:")
    print("-" * 60)
    
    robust_shrinkages = {}
    for h in horizons:
        if horizon_shrinkages[h]:
            mean_s = np.mean(horizon_shrinkages[h])
            std_s = np.std(horizon_shrinkages[h])
            median_s = np.median(horizon_shrinkages[h])
            mean_r = np.mean(horizon_ratios[h])
            
            # Use median for robustness, slightly conservative
            robust_s = min(mean_s, median_s)  # Take more conservative estimate
            robust_shrinkages[h] = robust_s
            
            print(f"  Horizon {h}: mean_s={mean_s:.3f} ± {std_s:.3f}, "
                  f"median_s={median_s:.3f}, robust_s={robust_s:.3f}")
    
    avg_overall = np.mean(overall_ratios)
    std_overall = np.std(overall_ratios)
    
    print(f"\nOverall CV ratio: {avg_overall:.6f} ± {std_overall:.6f}")
    
    if avg_overall < 1.0:
        score = np.sqrt(1 - avg_overall)
        print(f"Expected score: {score:.6f} ✓")
    
    return {
        'robust_shrinkages': robust_shrinkages,
        'horizon_shrinkages': {h: list(s) for h, s in horizon_shrinkages.items()},
        'horizon_ratios': {h: list(r) for h, r in horizon_ratios.items()},
        'overall_ratios': overall_ratios,
        'avg_ratio': float(avg_overall),
        'std_ratio': float(std_overall),
    }


# =============================================================================
# FINAL SUBMISSION with best settings
# =============================================================================
def create_optimized_submission(train, test, feature_cols, config):
    """
    Create final submission using the best configuration found.
    """
    print("\n" + "=" * 80)
    print("CREATING FINAL SUBMISSION")
    print("=" * 80)
    
    horizons = sorted(train['horizon'].unique())
    
    # Use robust shrinkages from CV
    shrinkages = config.get('robust_shrinkages', {})
    huber_alpha = config.get('best_alpha', 0.5)
    
    print(f"\nConfiguration:")
    print(f"  Huber alpha: {huber_alpha}")
    print(f"  Shrinkages: {shrinkages}")
    
    # Train on full training data
    all_test_preds = np.zeros(len(test))
    
    for horizon in horizons:
        print(f"\nTraining Horizon {horizon}...")
        
        train_h = train[train['horizon'] == horizon]
        
        X_train = train_h[feature_cols]
        y_train = train_h['y_target']
        train_weights = np.sqrt(train_h['weight'].values + 1)
        
        params = {
            'objective': 'huber',
            'alpha': huber_alpha,
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
            'n_estimators': 2000,  # More trees for final model
            'random_state': 42,
            'verbose': -1,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
        model = lgb.train(
            params, train_data, num_boost_round=params['n_estimators'],
            callbacks=[lgb.log_evaluation(period=500)]
        )
        
        # Predict test
        test_h = test[test['horizon'] == horizon]
        X_test = test_h[feature_cols]
        raw_preds = model.predict(X_test)
        
        # Apply shrinkage
        shrinkage = shrinkages.get(horizon, 0.15)  # Default conservative shrinkage
        shrunk_preds = raw_preds * shrinkage
        
        mask = test['horizon'] == horizon
        all_test_preds[mask.values] = shrunk_preds
        
        print(f"  Shrinkage: {shrinkage:.3f}, pred mean: {shrunk_preds.mean():.4f}, "
              f"pred std: {shrunk_preds.std():.4f}")
    
    # Create submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({
        'id': test['id'],
        'prediction': all_test_preds
    })
    
    output_path = OUTPUT_DIR / f'submission_advanced_{timestamp}.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Prediction stats:")
    print(f"    Mean: {all_test_preds.mean():.6f}")
    print(f"    Std: {all_test_preds.std():.6f}")
    print(f"    Min: {all_test_preds.min():.6f}")
    print(f"    Max: {all_test_preds.max():.6f}")
    
    return output_path, submission


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 80)
    print("HEDGE FUND TIME SERIES FORECASTING - ADVANCED TUNING")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    train, test = load_data()
    feature_cols = get_feature_columns(train)
    
    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"Number of horizons: {train['horizon'].nunique()}")
    
    all_results = {}
    
    # Experiment 1: Huber alpha tuning
    print("\n" + "#" * 80)
    print("RUNNING EXPERIMENT 1: Huber Alpha Tuning")
    print("#" * 80)
    
    alpha_results = experiment_huber_alpha(train, feature_cols, n_folds=3)
    all_results['huber_alpha'] = alpha_results
    
    best_alpha = min(alpha_results, key=lambda x: x['avg_ratio'])['alpha']
    
    # Experiment 2: Per-sample shrinkage
    print("\n" + "#" * 80)
    print("RUNNING EXPERIMENT 2: Per-Sample Shrinkage")
    print("#" * 80)
    
    sample_results = experiment_sample_shrinkage(train, feature_cols, n_folds=3)
    all_results['sample_shrinkage'] = sample_results
    
    # Experiment 3: CV per-horizon shrinkage
    print("\n" + "#" * 80)
    print("RUNNING EXPERIMENT 3: CV Per-Horizon Shrinkage")
    print("#" * 80)
    
    cv_results = experiment_cv_shrinkage(train, feature_cols, n_folds=3)
    all_results['cv_shrinkage'] = cv_results
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print("\n1. Best Huber Alpha:")
    print(f"   alpha = {best_alpha} (ratio = {min(alpha_results, key=lambda x: x['avg_ratio'])['avg_ratio']:.6f})")
    
    print("\n2. Best Per-Sample Shrinkage Strategy:")
    best_sample = min(sample_results, key=lambda x: x['avg_ratio'])
    print(f"   {best_sample['strategy']} (ratio = {best_sample['avg_ratio']:.6f})")
    
    print("\n3. Robust Per-Horizon Shrinkages:")
    for h, s in cv_results['robust_shrinkages'].items():
        print(f"   Horizon {h}: {s:.3f}")
    print(f"   Overall CV ratio: {cv_results['avg_ratio']:.6f}")
    
    # Create final config
    final_config = {
        'best_alpha': best_alpha,
        'robust_shrinkages': cv_results['robust_shrinkages'],
        'sample_strategy': best_sample['strategy'],
    }
    
    # Create submission
    submission_path, submission = create_optimized_submission(
        train, test, feature_cols, final_config
    )
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_output = {
        'timestamp': timestamp,
        'final_config': {k: (float(v) if isinstance(v, (np.float32, np.float64)) else 
                            ({int(kk): float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v))
                        for k, v in final_config.items()},
        'alpha_results': [{k: (float(v) if isinstance(v, (np.float32, np.float64)) 
                              else ([float(x) for x in v] if isinstance(v, list) else v))
                          for k, v in r.items()} 
                         for r in alpha_results],
        'sample_results': [{k: (float(v) if isinstance(v, (np.float32, np.float64)) else v)
                           for k, v in r.items()} 
                          for r in sample_results],
        'cv_results': {k: (float(v) if isinstance(v, (np.float32, np.float64)) else 
                          ({int(kk): ([float(x) for x in vv] if isinstance(vv, list) else float(vv)) 
                            for kk, vv in v.items()} if isinstance(v, dict) else 
                           ([float(x) for x in v] if isinstance(v, list) else v)))
                      for k, v in cv_results.items()},
    }
    
    results_path = OUTPUT_DIR / f'advanced_tuning_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results_output, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return all_results, final_config


if __name__ == "__main__":
    results, config = main()
