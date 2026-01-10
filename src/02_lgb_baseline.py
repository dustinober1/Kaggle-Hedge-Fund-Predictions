"""
Hedge Fund Time Series Forecasting - LightGBM Baseline
=======================================================

This script builds a LightGBM baseline model with:
- Time-based validation (train on early ts_index, validate on later)
- Competition-specific weighted RMSE metric
- Proper handling of the no look-ahead constraint
- Submission file generation
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


# =============================================================================
# Time-Based Validation Split
# =============================================================================
def time_based_split(train, val_ratio=0.2):
    """
    Split data based on ts_index for proper time-series validation.
    
    The validation set uses the LATER ts_index values to simulate
    the test scenario where we predict future data.
    """
    ts_max = train['ts_index'].max()
    ts_min = train['ts_index'].min()
    ts_range = ts_max - ts_min
    
    # Use the last val_ratio of ts_index for validation
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
def train_lightgbm(train_df, val_df, feature_cols, params=None):
    """
    Train a LightGBM model with early stopping.
    """
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 8,
            'min_child_samples': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_estimators': 1000,
            'random_state': 42,
            'verbose': -1,
        }
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    w_train = train_df['weight']
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target']
    w_val = val_df['weight']
    
    print(f"\nTraining LightGBM...")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)
    
    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=params.get('n_estimators', 1000),
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def evaluate_model(model, val_df, feature_cols):
    """Evaluate model on validation set using competition metric"""
    X_val = val_df[feature_cols]
    y_val = val_df['y_target'].values
    w_val = val_df['weight'].values
    
    preds = model.predict(X_val)
    
    score = weighted_rmse_score(y_val, preds, w_val)
    rmse = weighted_rmse(y_val, preds, w_val)
    
    print(f"\n" + "=" * 60)
    print(f"VALIDATION RESULTS")
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


def get_feature_importance(model, feature_cols, top_n=20):
    """Get and display feature importances"""
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Important Features:")
    for i, row in importance.head(top_n).iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")
    
    return importance


# =============================================================================
# Prediction and Submission
# =============================================================================
def make_predictions(model, test, feature_cols):
    """Make predictions on test set"""
    print(f"\nMaking predictions on test set ({len(test):,} rows)...")
    
    X_test = test[feature_cols]
    preds = model.predict(X_test)
    
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
    print("HEDGE FUND TIME SERIES FORECASTING - LIGHTGBM BASELINE")
    print("=" * 80)
    
    # Load data
    train, test = load_data()
    feature_cols = get_feature_columns(train)
    
    print(f"\nTarget column: y_target")
    print(f"Number of features: {len(feature_cols)}")
    
    # Time-based train/val split
    train_df, val_df = time_based_split(train, val_ratio=0.2)
    
    # Train model
    model = train_lightgbm(train_df, val_df, feature_cols)
    
    # Evaluate
    val_score, val_preds = evaluate_model(model, val_df, feature_cols)
    
    # Feature importance
    importance = get_feature_importance(model, feature_cols)
    
    # Make test predictions
    test_preds = make_predictions(model, test, feature_cols)
    
    # Create submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = create_submission(test, test_preds, f'submission_lgb_baseline_{timestamp}.csv')
    
    # Save model info
    results = {
        'timestamp': timestamp,
        'model': 'LightGBM Baseline',
        'val_score': float(val_score),
        'n_features': len(feature_cols),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test),
        'best_iteration': model.best_iteration,
        'top_10_features': importance.head(10)['feature'].tolist(),
    }
    
    results_path = OUTPUT_DIR / f'lgb_baseline_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Validation Score: {val_score:.6f}")
    print(f"Best Iteration: {model.best_iteration}")
    print(f"Submission file ready for upload!")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
