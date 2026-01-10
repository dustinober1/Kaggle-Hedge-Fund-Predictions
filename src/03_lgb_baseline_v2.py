"""
Hedge Fund Time Series Forecasting - LightGBM Baseline v2
==========================================================

Improvements over v1:
- Better handling of extreme weights (clipping/normalization)
- Per-horizon models for better prediction
- Weight normalization to prevent extreme values dominating
- More robust hyperparameters
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


def weighted_rmse(y_true, y_pred, weights):
    """Calculate weighted RMSE (lower is better)"""
    return np.sqrt(np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights))


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


def preprocess_data(train, test):
    """
    Preprocess data with weight handling and feature additions.
    """
    # Clip extreme weights (use log transform for training weights)
    # This prevents a few extreme weights from dominating the loss
    train = train.copy()
    
    # Normalize weights per-sample using log1p to reduce extreme skew
    # This helps training stability while preserving relative ordering
    train['weight_normalized'] = np.log1p(train['weight'])
    
    # Add horizon as feature
    # (LightGBM can handle this even though we might also train per-horizon)
    
    print(f"\nWeight normalization:")
    print(f"  Original weight range: {train['weight'].min():.2e} to {train['weight'].max():.2e}")
    print(f"  Normalized weight range: {train['weight_normalized'].min():.4f} to {train['weight_normalized'].max():.4f}")
    
    return train, test


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
# Model Training
# =============================================================================
def get_lgb_params():
    """Get LightGBM parameters"""
    return {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 127,
        'max_depth': 10,
        'min_child_samples': 200,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'n_estimators': 2000,
        'random_state': 42,
        'verbose': -1,
        'force_row_wise': True,  # Better for large datasets
    }


def train_single_model(train_df, val_df, feature_cols, use_normalized_weights=True):
    """
    Train a single LightGBM model on all data.
    """
    params = get_lgb_params()
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target']
    
    # Use normalized weights for training
    if use_normalized_weights and 'weight_normalized' in train_df.columns:
        w_train = train_df['weight_normalized'].values
        w_val = val_df['weight_normalized'].values
        print("\n  Using normalized weights for training")
    else:
        w_train = train_df['weight'].values
        w_val = val_df['weight'].values
        print("\n  Using original weights for training")
    
    print(f"\nTraining single LightGBM model...")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=params.get('n_estimators', 2000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def train_horizon_models(train_df, val_df, feature_cols, use_normalized_weights=True):
    """
    Train separate models for each horizon.
    This can help capture horizon-specific patterns.
    """
    horizons = sorted(train_df['horizon'].unique())
    models = {}
    
    params = get_lgb_params()
    
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"Training model for Horizon {horizon}")
        print(f"{'='*60}")
        
        train_h = train_df[train_df['horizon'] == horizon]
        val_h = val_df[val_df['horizon'] == horizon]
        
        X_train = train_h[feature_cols]
        y_train = train_h['y_target']
        
        X_val = val_h[feature_cols]
        y_val = val_h['y_target']
        
        if use_normalized_weights:
            w_train = train_h['weight_normalized'].values
            w_val = val_h['weight_normalized'].values
        else:
            w_train = train_h['weight'].values
            w_val = val_h['weight'].values
        
        print(f"  Train samples: {len(X_train):,}")
        print(f"  Val samples: {len(X_val):,}")
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=params.get('n_estimators', 2000),
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )
        
        models[horizon] = model
        print(f"  Best iteration: {model.best_iteration}")
    
    return models


def evaluate_models(models, val_df, feature_cols, is_horizon_models=True):
    """Evaluate models on validation set using competition metric (with original weights)"""
    
    if is_horizon_models:
        # Per-horizon predictions
        preds = np.zeros(len(val_df))
        for horizon, model in models.items():
            mask = val_df['horizon'] == horizon
            X_h = val_df.loc[mask, feature_cols]
            preds[mask.values] = model.predict(X_h)
    else:
        # Single model
        preds = models.predict(val_df[feature_cols])
    
    # Evaluate with ORIGINAL weights (not normalized)
    y_val = val_df['y_target'].values
    w_val = val_df['weight'].values
    
    score = weighted_rmse_score(y_val, preds, w_val)
    rmse = weighted_rmse(y_val, preds, w_val)
    
    print(f"\n" + "=" * 60)
    print(f"VALIDATION RESULTS (using original weights)")
    print(f"=" * 60)
    print(f"  Weighted RMSE Score (higher is better): {score:.6f}")
    print(f"  Weighted RMSE (lower is better):        {rmse:.6f}")
    print(f"=" * 60)
    
    # Evaluate per horizon
    print("\nScore by horizon:")
    for horizon in sorted(val_df['horizon'].unique()):
        mask = val_df['horizon'] == horizon
        h_preds = preds[mask.values]
        h_y = y_val[mask.values]
        h_w = w_val[mask.values]
        h_score = weighted_rmse_score(h_y, h_preds, h_w)
        print(f"  Horizon {horizon:2d}: {h_score:.6f}")
    
    return score, preds


def make_predictions(models, test, feature_cols, is_horizon_models=True):
    """Make predictions on test set"""
    print(f"\nMaking predictions on test set ({len(test):,} rows)...")
    
    if is_horizon_models:
        preds = np.zeros(len(test))
        for horizon, model in models.items():
            mask = test['horizon'] == horizon
            X_h = test.loc[mask, feature_cols]
            preds[mask.values] = model.predict(X_h)
    else:
        preds = models.predict(test[feature_cols])
    
    return preds


def create_submission(test, predictions, filename='submission.csv'):
    """Create a submission file"""
    submission = pd.DataFrame({
        'id': test['id'],
        'prediction': predictions
    })
    
    output_path = OUTPUT_DIR / filename
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Prediction stats: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    
    return submission


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("=" * 80)
    print("HEDGE FUND TIME SERIES FORECASTING - LIGHTGBM BASELINE v2")
    print("=" * 80)
    
    # Load data
    train, test = load_data()
    feature_cols = get_feature_columns(train)
    
    print(f"\nTarget column: y_target")
    print(f"Number of features: {len(feature_cols)}")
    
    # Preprocess
    train, test = preprocess_data(train, test)
    
    # Time-based train/val split
    train_df, val_df = time_based_split(train, val_ratio=0.2)
    
    # Option 1: Train single model
    print("\n" + "=" * 80)
    print("APPROACH: Per-Horizon Models")
    print("=" * 80)
    
    # Train per-horizon models (typically better for this type of problem)
    horizon_models = train_horizon_models(train_df, val_df, feature_cols, use_normalized_weights=True)
    
    # Evaluate
    val_score, val_preds = evaluate_models(horizon_models, val_df, feature_cols, is_horizon_models=True)
    
    # Make test predictions
    test_preds = make_predictions(horizon_models, test, feature_cols, is_horizon_models=True)
    
    # Create submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = create_submission(test, test_preds, f'submission_lgb_v2_{timestamp}.csv')
    
    # Save results
    results = {
        'timestamp': timestamp,
        'model': 'LightGBM Baseline v2 (per-horizon)',
        'val_score': float(val_score),
        'n_features': len(feature_cols),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test),
        'best_iterations': {str(h): m.best_iteration for h, m in horizon_models.items()},
    }
    
    results_path = OUTPUT_DIR / f'lgb_v2_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Validation Score: {val_score:.6f}")
    print(f"Submission file ready for upload!")
    
    return horizon_models, results


if __name__ == "__main__":
    models, results = main()
