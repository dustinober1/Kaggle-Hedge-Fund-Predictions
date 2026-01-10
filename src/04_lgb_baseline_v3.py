"""
Hedge Fund Time Series Forecasting - LightGBM Baseline v3
==========================================================

Key changes:
- Don't use weights during training (weights are for evaluation only)
- Try predicting 0 as baseline (since target is centered around 0)
- Compare different approaches
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
# Baselines
# =============================================================================
def evaluate_zero_baseline(val_df):
    """
    Evaluate the trivial baseline of predicting 0 for everything.
    Since target is centered around 0, this might actually be decent.
    """
    y_val = val_df['y_target'].values
    w_val = val_df['weight'].values
    preds = np.zeros(len(val_df))
    
    score = weighted_rmse_score(y_val, preds, w_val)
    
    print(f"\n=== ZERO BASELINE ===")
    print(f"  Predicting 0 for all samples")
    print(f"  Competition Score: {score:.6f}")
    
    return score


def evaluate_mean_baseline(train_df, val_df):
    """
    Evaluate predicting the training mean.
    """
    train_mean = train_df['y_target'].mean()
    
    y_val = val_df['y_target'].values
    w_val = val_df['weight'].values
    preds = np.full(len(val_df), train_mean)
    
    score = weighted_rmse_score(y_val, preds, w_val)
    
    print(f"\n=== MEAN BASELINE ===")
    print(f"  Predicting train mean = {train_mean:.6f}")
    print(f"  Competition Score: {score:.6f}")
    
    return score


def evaluate_horizon_mean_baseline(train_df, val_df):
    """
    Evaluate predicting per-horizon mean.
    """
    horizon_means = train_df.groupby('horizon')['y_target'].mean()
    
    y_val = val_df['y_target'].values
    w_val = val_df['weight'].values
    preds = val_df['horizon'].map(horizon_means).values
    
    score = weighted_rmse_score(y_val, preds, w_val)
    
    print(f"\n=== HORIZON MEAN BASELINE ===")
    print(f"  Per-horizon means: {horizon_means.to_dict()}")
    print(f"  Competition Score: {score:.6f}")
    
    return score


# =============================================================================
# Model Training (WITHOUT sample weights)
# =============================================================================
def train_lgb_no_weights(train_df, val_df, feature_cols):
    """
    Train LightGBM WITHOUT sample weights.
    Weights are used only for evaluation, not training.
    """
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
        'n_estimators': 2000,
        'random_state': 42,
        'verbose': -1,
    }
    
    X_train = train_df[feature_cols]
    y_train = train_df['y_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['y_target']
    
    print(f"\n=== LIGHTGBM (NO SAMPLE WEIGHTS) ===")
    print(f"  Training without sample weights...")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    
    # NO weights in training
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
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
    
    # Evaluate with competition metric (using weights)
    preds = model.predict(X_val)
    w_val = val_df['weight'].values
    score = weighted_rmse_score(y_val.values, preds, w_val)
    
    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Competition Score: {score:.6f}")
    
    # Per-horizon scores
    print("\n  Score by horizon:")
    for horizon in sorted(val_df['horizon'].unique()):
        mask = val_df['horizon'] == horizon
        h_preds = preds[mask.values]
        h_y = y_val.values[mask.values]
        h_w = w_val[mask.values]
        h_score = weighted_rmse_score(h_y, h_preds, h_w)
        print(f"    Horizon {horizon:2d}: {h_score:.6f}")
    
    return model, score, preds


def train_lgb_per_horizon_no_weights(train_df, val_df, feature_cols):
    """
    Train separate LightGBM models per horizon WITHOUT sample weights.
    """
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
        'n_estimators': 2000,
        'random_state': 42,
        'verbose': -1,
    }
    
    print(f"\n=== LIGHTGBM PER-HORIZON (NO SAMPLE WEIGHTS) ===")
    
    horizons = sorted(train_df['horizon'].unique())
    models = {}
    all_preds = np.zeros(len(val_df))
    
    for horizon in horizons:
        train_h = train_df[train_df['horizon'] == horizon]
        val_h = val_df[val_df['horizon'] == horizon]
        
        X_train = train_h[feature_cols]
        y_train = train_h['y_target']
        
        X_val = val_h[feature_cols]
        y_val = val_h['y_target']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['val'],
            num_boost_round=params.get('n_estimators', 2000),
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
        
        print(f"  Horizon {horizon:2d}: iter={model.best_iteration:3d}, score={h_score:.6f}")
    
    # Overall score
    y_val = val_df['y_target'].values
    w_val = val_df['weight'].values
    score = weighted_rmse_score(y_val, all_preds, w_val)
    
    print(f"\n  Overall Competition Score: {score:.6f}")
    
    return models, score, all_preds


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    print("=" * 80)
    print("HEDGE FUND TIME SERIES FORECASTING - BASELINE ANALYSIS v3")
    print("=" * 80)
    
    # Load data
    train, test = load_data()
    feature_cols = get_feature_columns(train)
    
    print(f"\nTarget column: y_target")
    print(f"Number of features: {len(feature_cols)}")
    
    # Target analysis
    print(f"\nTarget statistics:")
    print(f"  Mean: {train['y_target'].mean():.6f}")
    print(f"  Median: {train['y_target'].median():.6f}")
    print(f"  Std: {train['y_target'].std():.6f}")
    
    # Time-based train/val split
    train_df, val_df = time_based_split(train, val_ratio=0.2)
    
    # Compare baselines
    print("\n" + "=" * 80)
    print("BASELINE COMPARISONS")
    print("=" * 80)
    
    zero_score = evaluate_zero_baseline(val_df)
    mean_score = evaluate_mean_baseline(train_df, val_df)
    horizon_mean_score = evaluate_horizon_mean_baseline(train_df, val_df)
    
    # LightGBM without weights
    print("\n" + "=" * 80)
    print("LIGHTGBM MODELS")
    print("=" * 80)
    
    single_model, single_score, single_preds = train_lgb_no_weights(train_df, val_df, feature_cols)
    horizon_models, horizon_score, horizon_preds = train_lgb_per_horizon_no_weights(train_df, val_df, feature_cols)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    results = {
        'Zero Baseline': zero_score,
        'Mean Baseline': mean_score,
        'Horizon Mean Baseline': horizon_mean_score,
        'LightGBM Single': single_score,
        'LightGBM Per-Horizon': horizon_score,
    }
    
    best_approach = max(results, key=results.get)
    print("\nAll scores (higher is better):")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " *** BEST ***" if name == best_approach else ""
        print(f"  {name}: {score:.6f}{marker}")
    
    # Generate submission with best approach
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSION")
    print("=" * 80)
    
    if best_approach == 'Zero Baseline':
        test_preds = np.zeros(len(test))
    elif best_approach == 'Mean Baseline':
        test_preds = np.full(len(test), train_df['y_target'].mean())
    elif best_approach == 'Horizon Mean Baseline':
        horizon_means = train_df.groupby('horizon')['y_target'].mean()
        test_preds = test['horizon'].map(horizon_means).values
    elif best_approach == 'LightGBM Single':
        test_preds = single_model.predict(test[feature_cols])
    else:  # Per-Horizon
        test_preds = np.zeros(len(test))
        for horizon, model in horizon_models.items():
            mask = test['horizon'] == horizon
            test_preds[mask.values] = model.predict(test.loc[mask, feature_cols])
    
    # Create submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({
        'id': test['id'],
        'prediction': test_preds
    })
    
    output_path = OUTPUT_DIR / f'submission_v3_{timestamp}.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Best approach: {best_approach}")
    print(f"  Rows: {len(submission):,}")
    print(f"  Prediction stats: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")
    
    # Save results
    results_dict = {
        'timestamp': timestamp,
        'best_approach': best_approach,
        'all_scores': results,
        'n_features': len(feature_cols),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test),
    }
    
    results_path = OUTPUT_DIR / f'v3_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results_dict


if __name__ == "__main__":
    results = main()
