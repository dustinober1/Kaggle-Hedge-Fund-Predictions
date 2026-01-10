"""
Hedge Fund Time Series Forecasting - Optimized Submission
===========================================================

Based on refinement findings:
- Huber loss with sqrt weights achieved ratio=0.998 (score ~0.044)
- Key insight: Very conservative shrinkage (0.3) is needed
- Use sqrt of weights during training for best balance

This script creates an optimized submission.
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


def time_based_split(train, val_ratio=0.2):
    """Split data based on ts_index"""
    ts_max = train['ts_index'].max()
    ts_min = train['ts_index'].min()
    ts_range = ts_max - ts_min
    val_threshold = ts_max - int(ts_range * val_ratio)
    
    train_df = train[train['ts_index'] <= val_threshold].copy()
    val_df = train[train['ts_index'] > val_threshold].copy()
    
    return train_df, val_df


# =============================================================================
# Model Training
# =============================================================================
def train_huber_model(train_df, feature_cols, weight_power=0.5):
    """
    Train Huber loss model with weight transformation.
    """
    print("\n=== Training Huber Loss Model ===")
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    
    # Weight transformation
    train_weights = np.power(train_df['weight'].values + 1, weight_power)
    
    params = {
        'objective': 'huber',
        'alpha': 0.5,  # Huber delta
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
        'n_estimators': 2000,
        'random_state': 42,
        'verbose': -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=params.get('n_estimators', 2000),
        callbacks=[
            lgb.log_evaluation(period=500)
        ]
    )
    
    print(f"  Trained with {model.num_trees()} trees")
    
    return model


def find_optimal_shrinkage(model, val_df, feature_cols):
    """
    Find optimal shrinkage on validation set.
    """
    print("\n=== Finding Optimal Shrinkage ===")
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target'].values
    w_val = val_df['weight'].values
    
    raw_preds = model.predict(X_val)
    
    best_shrinkage = 0
    best_ratio = float('inf')
    
    for s in np.arange(0.0, 1.05, 0.05):
        ratio = get_ratio(y_val, raw_preds * s, w_val)
        if ratio < best_ratio:
            best_ratio = ratio
            best_shrinkage = s
    
    # Fine-tune around best
    for s in np.arange(max(0, best_shrinkage - 0.05), min(1, best_shrinkage + 0.05), 0.01):
        ratio = get_ratio(y_val, raw_preds * s, w_val)
        if ratio < best_ratio:
            best_ratio = ratio
            best_shrinkage = s
    
    score = weighted_rmse_score(y_val, raw_preds * best_shrinkage, w_val)
    
    print(f"  Best shrinkage: {best_shrinkage:.3f}")
    print(f"  Best ratio: {best_ratio:.6f}")
    print(f"  Score: {score:.6f}")
    
    return best_shrinkage, best_ratio, score


def train_per_horizon_models(train_df, feature_cols, weight_power=0.5):
    """
    Train separate Huber models per horizon.
    """
    print("\n=== Training Per-Horizon Huber Models ===")
    
    horizons = sorted(train_df['horizon'].unique())
    models = {}
    
    for horizon in horizons:
        print(f"\n  Horizon {horizon}...")
        train_h = train_df[train_df['horizon'] == horizon]
        
        X_train = train_h[feature_cols]
        y_train = train_h['y_target']
        train_weights = np.power(train_h['weight'].values + 1, weight_power)
        
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
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 1500),
            callbacks=[lgb.log_evaluation(period=1500)]
        )
        
        models[horizon] = model
        print(f"    Trained with {model.num_trees()} trees")
    
    return models


def find_per_horizon_shrinkages(models, val_df, feature_cols):
    """
    Find optimal shrinkage for each horizon.
    """
    print("\n=== Finding Per-Horizon Shrinkages ===")
    
    horizons = sorted(val_df['horizon'].unique())
    shrinkages = {}
    
    for horizon in horizons:
        val_h = val_df[val_df['horizon'] == horizon]
        
        X_val = val_h[feature_cols]
        y_val = val_h['y_target'].values
        w_val = val_h['weight'].values
        
        raw_preds = models[horizon].predict(X_val)
        
        best_s, best_r = 0, float('inf')
        for s in np.arange(0.0, 1.05, 0.05):
            r = get_ratio(y_val, raw_preds * s, w_val)
            if r < best_r:
                best_r = r
                best_s = s
        
        # Fine-tune
        for s in np.arange(max(0, best_s - 0.05), min(1, best_s + 0.05), 0.01):
            r = get_ratio(y_val, raw_preds * s, w_val)
            if r < best_r:
                best_r = r
                best_s = s
        
        shrinkages[horizon] = best_s
        print(f"  Horizon {horizon}: shrinkage={best_s:.3f}, ratio={best_r:.6f}")
    
    return shrinkages


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 80)
    print("HEDGE FUND TIME SERIES FORECASTING - OPTIMIZED SUBMISSION")
    print("=" * 80)
    
    # Load data
    train, test = load_data()
    feature_cols = get_feature_columns(train)
    
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Split for validation
    train_df, val_df = time_based_split(train, val_ratio=0.2)
    print(f"Train: {len(train_df):,} samples")
    print(f"Val: {len(val_df):,} samples")
    
    # Approach 1: Single Huber model
    print("\n" + "=" * 60)
    print("APPROACH 1: Single Huber Model")
    print("=" * 60)
    
    model_single = train_huber_model(train_df, feature_cols, weight_power=0.5)
    shrinkage_single, ratio_single, score_single = find_optimal_shrinkage(
        model_single, val_df, feature_cols
    )
    
    # Approach 2: Per-horizon Huber models
    print("\n" + "=" * 60)
    print("APPROACH 2: Per-Horizon Huber Models")
    print("=" * 60)
    
    models_ph = train_per_horizon_models(train_df, feature_cols, weight_power=0.5)
    shrinkages_ph = find_per_horizon_shrinkages(models_ph, val_df, feature_cols)
    
    # Evaluate per-horizon approach on full validation
    all_preds = np.zeros(len(val_df))
    for horizon in sorted(val_df['horizon'].unique()):
        mask = val_df['horizon'] == horizon
        X_val_h = val_df.loc[mask, feature_cols]
        raw_preds = models_ph[horizon].predict(X_val_h)
        all_preds[mask.values] = raw_preds * shrinkages_ph[horizon]
    
    ratio_ph = get_ratio(val_df['y_target'].values, all_preds, val_df['weight'].values)
    score_ph = weighted_rmse_score(val_df['y_target'].values, all_preds, val_df['weight'].values)
    
    print(f"\n  Overall per-horizon: ratio={ratio_ph:.6f}, score={score_ph:.6f}")
    
    # Decide which approach to use
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Single Model:     ratio={ratio_single:.6f}, score={score_single:.6f}")
    print(f"Per-Horizon:      ratio={ratio_ph:.6f}, score={score_ph:.6f}")
    
    use_per_horizon = ratio_ph < ratio_single
    print(f"\nUsing: {'Per-Horizon' if use_per_horizon else 'Single Model'}")
    
    # Retrain on FULL training data for final submission
    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING (on full data)")
    print("=" * 60)
    
    if use_per_horizon:
        final_models = train_per_horizon_models(train, feature_cols, weight_power=0.5)
        final_shrinkages = shrinkages_ph  # Use shrinkages from validation
        
        # Generate test predictions
        test_preds = np.zeros(len(test))
        for horizon in sorted(test['horizon'].unique()):
            mask = test['horizon'] == horizon
            X_test_h = test.loc[mask, feature_cols]
            raw_preds = final_models[horizon].predict(X_test_h)
            test_preds[mask.values] = raw_preds * final_shrinkages[horizon]
    else:
        final_model = train_huber_model(train, feature_cols, weight_power=0.5)
        raw_preds = final_model.predict(test[feature_cols])
        test_preds = raw_preds * shrinkage_single
    
    # Create submission
    print("\n" + "=" * 60)
    print("CREATING SUBMISSION")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({
        'id': test['id'],
        'prediction': test_preds
    })
    
    output_path = OUTPUT_DIR / f'submission_optimized_{timestamp}.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Prediction stats:")
    print(f"    Mean: {test_preds.mean():.6f}")
    print(f"    Std: {test_preds.std():.6f}")
    print(f"    Min: {test_preds.min():.6f}")
    print(f"    Max: {test_preds.max():.6f}")
    
    # Save configuration
    config = {
        'timestamp': timestamp,
        'approach': 'per_horizon' if use_per_horizon else 'single',
        'weight_power': 0.5,
        'single_shrinkage': float(shrinkage_single),
        'single_ratio': float(ratio_single),
        'single_score': float(score_single),
        'per_horizon_shrinkages': {int(k): float(v) for k, v in shrinkages_ph.items()},
        'per_horizon_ratio': float(ratio_ph),
        'per_horizon_score': float(score_ph),
    }
    
    config_path = OUTPUT_DIR / f'submission_optimized_config_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to: {config_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_score = score_ph if use_per_horizon else score_single
    print(f"Expected validation score: {best_score:.6f}")
    print(f"This would rank above zero baseline submissions!")
    
    return config


if __name__ == "__main__":
    config = main()
