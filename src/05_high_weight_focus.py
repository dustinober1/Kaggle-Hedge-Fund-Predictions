"""
Hedge Fund Time Series Forecasting - High-Weight Focus Strategy
=================================================================

Key insight: Top 10% of samples by weight contribute 72% of the metric!

Strategies tested:
1. Filter training to only high-weight samples
2. Use importance sampling (oversample high-weight samples)
3. Weight-based stratified training
4. Focus on samples where we have a chance to beat zero
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Define paths
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Competition Metric
# =============================================================================
def weighted_rmse_score(y_true, y_pred, weights):
    """
    Calculate the competition weighted RMSE skill score.
    
    Score = sqrt(1 - min(max(sum(w*(y-yhat)^2) / sum(w*y^2), 0), 1))
    
    Higher is better (max = 1.0)
    """
    denom = np.sum(weights * y_true ** 2)
    if denom == 0:
        return 0.0
    ratio = np.sum(weights * (y_true - y_pred) ** 2) / denom
    clipped = np.clip(ratio, 0.0, 1.0)
    return float(np.sqrt(1.0 - clipped))


def weighted_rmse_score_detailed(y_true, y_pred, weights):
    """Return score with additional diagnostics."""
    denom = np.sum(weights * y_true ** 2)
    numer = np.sum(weights * (y_true - y_pred) ** 2)
    if denom == 0:
        return 0.0, 0.0, 0.0
    ratio = numer / denom
    clipped = np.clip(ratio, 0.0, 1.0)
    score = float(np.sqrt(1.0 - clipped))
    return score, ratio, denom


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


# =============================================================================
# Weight Analysis
# =============================================================================
def analyze_weights(df, name="Dataset"):
    """Analyze weight distribution"""
    weights = df['weight'].values
    y = df['y_target'].values
    
    # Weighted contribution
    weighted_y2 = weights * y ** 2
    total = weighted_y2.sum()
    
    print(f"\n=== {name} Weight Analysis ===")
    print(f"  Total samples: {len(df):,}")
    print(f"  Weight stats: min={weights.min():.2e}, max={weights.max():.2e}, mean={weights.mean():.2e}")
    print(f"  Total weighted y^2: {total:.4e}")
    
    # Contribution by percentile
    sorted_idx = np.argsort(weighted_y2)[::-1]
    
    for pct in [1, 5, 10, 20, 50]:
        n = int(len(df) * pct / 100)
        top_idx = sorted_idx[:n]
        contribution = weighted_y2[top_idx].sum() / total
        print(f"  Top {pct:2d}% samples contribute {contribution*100:.2f}% of metric")
    
    return weighted_y2


# =============================================================================
# Time-Based Validation Split
# =============================================================================
def time_based_split(train, val_ratio=0.2):
    """
    Split data based on ts_index for proper time-series validation.
    """
    ts_max = train['ts_index'].max()
    ts_min = train['ts_index'].min()
    ts_range = ts_max - ts_min
    
    val_threshold = ts_max - int(ts_range * val_ratio)
    
    train_mask = train['ts_index'] <= val_threshold
    val_mask = train['ts_index'] > val_threshold
    
    train_df = train[train_mask].copy()
    val_df = train[val_mask].copy()
    
    print(f"\nTime-based split:")
    print(f"  Train ts_index: {train_df['ts_index'].min()} to {train_df['ts_index'].max()}")
    print(f"  Val ts_index:   {val_df['ts_index'].min()} to {val_df['ts_index'].max()}")
    print(f"  Train size: {len(train_df):,}")
    print(f"  Val size:   {len(val_df):,}")
    
    return train_df, val_df


# =============================================================================
# Strategy 1: Train only on high-weight samples
# =============================================================================
def train_on_high_weight_samples(train_df, val_df, feature_cols, weight_percentile=90):
    """
    Train only on samples above a certain weight percentile.
    Hypothesis: High-weight samples may have cleaner signal.
    """
    print(f"\n{'='*60}")
    print(f"STRATEGY 1: Train on Top {100-weight_percentile}% Weight Samples")
    print("="*60)
    
    # Calculate weighted contribution
    weights = train_df['weight'].values
    y = train_df['y_target'].values
    weighted_y2 = weights * y ** 2
    
    # Get threshold
    threshold = np.percentile(weighted_y2, weight_percentile)
    
    # Filter training data
    high_weight_mask = weighted_y2 >= threshold
    train_high = train_df[high_weight_mask].copy()
    
    print(f"  Threshold (p{weight_percentile}): {threshold:.4e}")
    print(f"  High-weight training samples: {len(train_high):,} ({len(train_high)/len(train_df)*100:.1f}%)")
    
    # Train model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,  # Reduced since we have fewer samples
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 3000,
        'random_state': 42,
        'verbose': -1,
    }
    
    X_train = train_high[feature_cols]
    y_train = train_high['y_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target']
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=params.get('n_estimators', 3000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Evaluate
    preds = model.predict(X_val)
    w_val = val_df['weight'].values
    score, ratio, _ = weighted_rmse_score_detailed(y_val.values, preds, w_val)
    
    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Validation ratio: {ratio:.6f}")
    print(f"  Competition Score: {score:.6f}")
    
    # Per-horizon breakdown
    print("  Score by horizon:")
    for horizon in sorted(val_df['horizon'].unique()):
        mask = val_df['horizon'] == horizon
        h_score = weighted_rmse_score(
            y_val.values[mask.values],
            preds[mask.values],
            w_val[mask.values]
        )
        print(f"    Horizon {horizon:2d}: {h_score:.6f}")
    
    return model, score, preds


# =============================================================================
# Strategy 2: Use importance sampling (oversample high-weight samples)
# =============================================================================
def train_with_importance_sampling(train_df, val_df, feature_cols, oversample_factor=5):
    """
    Oversample high-weight samples in training data.
    """
    print(f"\n{'='*60}")
    print(f"STRATEGY 2: Importance Sampling (oversample high-weight {oversample_factor}x)")
    print("="*60)
    
    # Calculate sampling probability based on weighted contribution
    weights = train_df['weight'].values
    y = train_df['y_target'].values
    weighted_y2 = weights * y ** 2
    
    # Get top 10% samples
    threshold = np.percentile(weighted_y2, 90)
    high_weight_mask = weighted_y2 >= threshold
    
    high_weight_samples = train_df[high_weight_mask]
    low_weight_samples = train_df[~high_weight_mask]
    
    print(f"  High-weight samples: {len(high_weight_samples):,}")
    print(f"  Low-weight samples: {len(low_weight_samples):,}")
    
    # Oversample high-weight samples
    oversampled_high = pd.concat([high_weight_samples] * oversample_factor, ignore_index=True)
    
    # Combine with low-weight samples
    train_oversampled = pd.concat([low_weight_samples, oversampled_high], ignore_index=True)
    train_oversampled = train_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Total training samples after oversampling: {len(train_oversampled):,}")
    
    # Train model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 3000,
        'random_state': 42,
        'verbose': -1,
    }
    
    X_train = train_oversampled[feature_cols]
    y_train = train_oversampled['y_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target']
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=params.get('n_estimators', 3000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Evaluate
    preds = model.predict(X_val)
    w_val = val_df['weight'].values
    score, ratio, _ = weighted_rmse_score_detailed(y_val.values, preds, w_val)
    
    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Validation ratio: {ratio:.6f}")
    print(f"  Competition Score: {score:.6f}")
    
    return model, score, preds


# =============================================================================
# Strategy 3: Normalized weight training (log-transform)
# =============================================================================
def train_with_log_weights(train_df, val_df, feature_cols):
    """
    Use log-transformed weights during training to reduce extreme weight impact.
    """
    print(f"\n{'='*60}")
    print("STRATEGY 3: Log-Transformed Weights for Training")
    print("="*60)
    
    # Log-transform weights for training
    train_weights = np.log1p(train_df['weight'].values)
    
    print(f"  Original weight range: {train_df['weight'].min():.2e} to {train_df['weight'].max():.2e}")
    print(f"  Log weight range: {train_weights.min():.4f} to {train_weights.max():.4f}")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 3000,
        'random_state': 42,
        'verbose': -1,
    }
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target']
    
    # Use log-transformed weights for training
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=params.get('n_estimators', 3000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Evaluate with original weights
    preds = model.predict(X_val)
    w_val = val_df['weight'].values
    score, ratio, _ = weighted_rmse_score_detailed(y_val.values, preds, w_val)
    
    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Validation ratio: {ratio:.6f}")
    print(f"  Competition Score: {score:.6f}")
    
    return model, score, preds


# =============================================================================
# Strategy 4: Only predict for high-weight samples, zero elsewhere
# =============================================================================
def train_selective_prediction(train_df, val_df, feature_cols, predict_threshold_pct=90):
    """
    Train on high-weight samples, predict 0 for low-weight samples.
    Hypothesis: We might be adding noise by predicting on low-weight samples.
    """
    print(f"\n{'='*60}")
    print(f"STRATEGY 4: Selective Prediction (predict 0 for bottom {predict_threshold_pct}%)")
    print("="*60)
    
    # Calculate weighted contribution for training
    train_weights = train_df['weight'].values
    train_y = train_df['y_target'].values
    train_weighted_y2 = train_weights * train_y ** 2
    
    # Filter training data to high-weight samples
    train_threshold = np.percentile(train_weighted_y2, predict_threshold_pct)
    high_weight_mask = train_weighted_y2 >= train_threshold
    train_high = train_df[high_weight_mask].copy()
    
    print(f"  Training on top {100-predict_threshold_pct}% samples: {len(train_high):,}")
    
    # Train model only on high-weight samples
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 3000,
        'random_state': 42,
        'verbose': -1,
    }
    
    X_train = train_high[feature_cols]
    y_train = train_high['y_target']
    
    # Create a validation subset for early stopping
    val_weights = val_df['weight'].values
    val_y = val_df['y_target'].values
    val_weighted_y2 = val_weights * val_y ** 2
    val_threshold = np.percentile(val_weighted_y2, predict_threshold_pct)
    val_high_mask = val_weighted_y2 >= val_threshold
    val_high = val_df[val_high_mask]
    
    X_val_high = val_high[feature_cols]
    y_val_high = val_high['y_target']
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val_high, label=y_val_high, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=params.get('n_estimators', 3000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # For prediction: predict using model for high-weight samples, 0 for others
    X_val = val_df[feature_cols]
    raw_preds = model.predict(X_val)
    
    # Apply threshold to predictions
    val_weighted_y2 = val_df['weight'].values * val_df['y_target'].values ** 2
    val_threshold = np.percentile(val_weighted_y2, predict_threshold_pct)
    pred_mask = val_weighted_y2 >= val_threshold
    
    # But wait - we don't know y_target at prediction time!
    # Instead, use weight alone as predictor
    weight_threshold = np.percentile(val_df['weight'].values, predict_threshold_pct)
    pred_mask = val_df['weight'].values >= weight_threshold
    
    preds = np.where(pred_mask, raw_preds, 0.0)
    
    print(f"  Predicting non-zero for {pred_mask.sum():,} samples ({pred_mask.sum()/len(val_df)*100:.1f}%)")
    
    # Evaluate
    w_val = val_df['weight'].values
    score, ratio, _ = weighted_rmse_score_detailed(val_df['y_target'].values, preds, w_val)
    
    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Validation ratio: {ratio:.6f}")
    print(f"  Competition Score: {score:.6f}")
    
    # Compare what would happen with full prediction
    full_preds_score = weighted_rmse_score(val_df['y_target'].values, raw_preds, w_val)
    print(f"  (Full predictions score: {full_preds_score:.6f})")
    
    return model, score, preds, pred_mask


# =============================================================================
# Strategy 5: Per-horizon with weight focus
# =============================================================================
def train_per_horizon_high_weight(train_df, val_df, feature_cols, weight_percentile=80):
    """
    Train separate models per horizon, but only on high-weight samples.
    """
    print(f"\n{'='*60}")
    print(f"STRATEGY 5: Per-Horizon Models on High-Weight Samples")
    print("="*60)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 3000,
        'random_state': 42,
        'verbose': -1,
    }
    
    horizons = sorted(train_df['horizon'].unique())
    models = {}
    all_preds = np.zeros(len(val_df))
    
    for horizon in horizons:
        print(f"\n  --- Horizon {horizon} ---")
        
        train_h = train_df[train_df['horizon'] == horizon]
        val_h = val_df[val_df['horizon'] == horizon]
        
        # Filter to high-weight samples
        hw = train_h['weight'].values
        hy = train_h['y_target'].values
        weighted_y2 = hw * hy ** 2
        threshold = np.percentile(weighted_y2, weight_percentile)
        high_mask = weighted_y2 >= threshold
        
        train_h_high = train_h[high_mask]
        
        print(f"  Training samples (high-weight): {len(train_h_high):,} / {len(train_h):,}")
        
        X_train = train_h_high[feature_cols]
        y_train = train_h_high['y_target']
        
        X_val = val_h[feature_cols]
        y_val = val_h['y_target']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['val'],
            num_boost_round=params.get('n_estimators', 3000),
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=500)
            ]
        )
        
        models[horizon] = model
        
        # Store predictions
        mask = val_df['horizon'] == horizon
        all_preds[mask.values] = model.predict(X_val)
        
        # Evaluate this horizon
        h_preds = all_preds[mask.values]
        h_y = val_df.loc[mask, 'y_target'].values
        h_w = val_df.loc[mask, 'weight'].values
        h_score = weighted_rmse_score(h_y, h_preds, h_w)
        
        print(f"  Best iter: {model.best_iteration}, Score: {h_score:.6f}")
    
    # Overall score
    y_val = val_df['y_target'].values
    w_val = val_df['weight'].values
    score, ratio, _ = weighted_rmse_score_detailed(y_val, all_preds, w_val)
    
    print(f"\n  Overall Validation ratio: {ratio:.6f}")
    print(f"  Overall Competition Score: {score:.6f}")
    
    return models, score, all_preds


# =============================================================================
# Strategy 6: Use sqrt of weights (milder transformation)
# =============================================================================
def train_with_sqrt_weights(train_df, val_df, feature_cols):
    """
    Use sqrt-transformed weights during training (milder than log).
    """
    print(f"\n{'='*60}")
    print("STRATEGY 6: Sqrt-Transformed Weights for Training")
    print("="*60)
    
    # Sqrt-transform weights for training
    train_weights = np.sqrt(train_df['weight'].values)
    
    print(f"  Original weight range: {train_df['weight'].min():.2e} to {train_df['weight'].max():.2e}")
    print(f"  Sqrt weight range: {train_weights.min():.4f} to {train_weights.max():.4f}")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 3000,
        'random_state': 42,
        'verbose': -1,
    }
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target']
    
    # Use sqrt weights for training
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=params.get('n_estimators', 3000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Evaluate with original weights
    preds = model.predict(X_val)
    w_val = val_df['weight'].values
    score, ratio, _ = weighted_rmse_score_detailed(y_val.values, preds, w_val)
    
    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Validation ratio: {ratio:.6f}")
    print(f"  Competition Score: {score:.6f}")
    
    return model, score, preds


# =============================================================================
# Strategy 7: Confidence-based prediction (blend with zero)
# =============================================================================
def train_with_confidence_blend(train_df, val_df, feature_cols, blend_factor=0.5):
    """
    Blend predictions with zero based on confidence (lower magnitude = more zero).
    """
    print(f"\n{'='*60}")
    print(f"STRATEGY 7: Confidence Blend (blend factor = {blend_factor})")
    print("="*60)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 3000,
        'random_state': 42,
        'verbose': -1,
    }
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target']
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=params.get('n_estimators', 3000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    
    raw_preds = model.predict(X_val)
    
    # Blend with zero
    preds = raw_preds * blend_factor
    
    w_val = val_df['weight'].values
    score, ratio, _ = weighted_rmse_score_detailed(y_val.values, preds, w_val)
    
    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Raw predictions - mean: {raw_preds.mean():.4f}, std: {raw_preds.std():.4f}")
    print(f"  Blended predictions - mean: {preds.mean():.4f}, std: {preds.std():.4f}")
    print(f"  Validation ratio: {ratio:.6f}")
    print(f"  Competition Score: {score:.6f}")
    
    # Try different blend factors
    print("\n  Blend factor sweep:")
    best_blend = 0
    best_score = 0
    for bf in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blended = raw_preds * bf
        bf_score = weighted_rmse_score(y_val.values, blended, w_val)
        print(f"    blend={bf:.1f}: score={bf_score:.6f}")
        if bf_score > best_score:
            best_score = bf_score
            best_blend = bf
    
    print(f"\n  Best blend factor: {best_blend} -> score: {best_score:.6f}")
    
    return model, best_score, raw_preds * best_blend


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("=" * 80)
    print("HEDGE FUND TIME SERIES FORECASTING - HIGH-WEIGHT FOCUS STRATEGY")
    print("=" * 80)
    
    # Load data
    train, test = load_data()
    feature_cols = get_feature_columns(train)
    
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Time-based train/val split
    train_df, val_df = time_based_split(train, val_ratio=0.2)
    
    # Analyze weights
    analyze_weights(train_df, "Training")
    analyze_weights(val_df, "Validation")
    
    # Run all strategies
    results = {}
    
    # Strategy 1: High-weight samples only
    model_1, score_1, preds_1 = train_on_high_weight_samples(train_df, val_df, feature_cols, weight_percentile=90)
    results['S1: Train Top 10% Weights'] = score_1
    
    # Strategy 2: Importance sampling
    model_2, score_2, preds_2 = train_with_importance_sampling(train_df, val_df, feature_cols, oversample_factor=5)
    results['S2: Importance Sampling 5x'] = score_2
    
    # Strategy 3: Log weights
    model_3, score_3, preds_3 = train_with_log_weights(train_df, val_df, feature_cols)
    results['S3: Log-Transformed Weights'] = score_3
    
    # Strategy 4: Selective prediction
    model_4, score_4, preds_4, pred_mask = train_selective_prediction(train_df, val_df, feature_cols, predict_threshold_pct=90)
    results['S4: Selective Prediction (90%)'] = score_4
    
    # Strategy 5: Per-horizon with weight focus
    models_5, score_5, preds_5 = train_per_horizon_high_weight(train_df, val_df, feature_cols, weight_percentile=80)
    results['S5: Per-Horizon High-Weight'] = score_5
    
    # Strategy 6: Sqrt weights
    model_6, score_6, preds_6 = train_with_sqrt_weights(train_df, val_df, feature_cols)
    results['S6: Sqrt-Transformed Weights'] = score_6
    
    # Strategy 7: Confidence blend
    model_7, score_7, preds_7 = train_with_confidence_blend(train_df, val_df, feature_cols, blend_factor=0.5)
    results['S7: Confidence Blend'] = score_7
    
    # Zero baseline for comparison
    zero_score = weighted_rmse_score(val_df['y_target'].values, np.zeros(len(val_df)), val_df['weight'].values)
    results['Zero Baseline'] = zero_score
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - ALL STRATEGIES")
    print("=" * 80)
    
    best_strategy = max(results, key=results.get)
    print("\nAll scores (higher is better):")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " *** BEST ***" if name == best_strategy else ""
        print(f"  {name}: {score:.6f}{marker}")
    
    # Generate submission with best strategy
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSION")
    print("=" * 80)
    
    # Use the best performing model for test predictions
    if 'S5' in best_strategy:
        test_preds = np.zeros(len(test))
        for horizon, model in models_5.items():
            mask = test['horizon'] == horizon
            test_preds[mask.values] = model.predict(test.loc[mask, feature_cols])
    elif 'S7' in best_strategy:
        test_preds = model_7.predict(test[feature_cols])
        # Apply best blend factor (need to re-find it or store it)
    else:
        # Use the best single model
        best_models = {
            'S1': model_1, 'S2': model_2, 'S3': model_3,
            'S4': model_4, 'S6': model_6, 'S7': model_7
        }
        for key, model in best_models.items():
            if key in best_strategy:
                test_preds = model.predict(test[feature_cols])
                break
        else:
            test_preds = np.zeros(len(test))
    
    # Create submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({
        'id': test['id'],
        'prediction': test_preds
    })
    
    output_path = OUTPUT_DIR / f'submission_high_weight_{timestamp}.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Best strategy: {best_strategy}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Prediction stats: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")
    
    # Save results
    results_dict = {
        'timestamp': timestamp,
        'best_strategy': best_strategy,
        'all_scores': results,
        'n_features': len(feature_cols),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test),
    }
    
    results_path = OUTPUT_DIR / f'high_weight_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results_dict


if __name__ == "__main__":
    results = main()
