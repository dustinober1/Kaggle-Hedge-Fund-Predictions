"""
Hedge Fund Time Series Forecasting - Hybrid Submission
======================================================

Hybrid Strategy:
- Horizon 1 (Short): Use Base Features (Original). Market/Sector aggregates add noise.
- Horizon 3, 10, 25 (Med/Long): Use Enhanced Features (Original + Market/Sector/Rolling). 
  Macro trends significantly improve predictability.

Winning Configuration:
- Model: LightGBM Huber (alpha=0.1)
- Weights: sqrt(w+1)
- Features: Hybrid (Base for H1, FE for H3+)
- Shrinkage:
  - H1: 0.29 (Base model CV)
  - H3: 0.27 (FE model CV)
  - H10: 0.32 (FE model CV)
  - H25: 0.34 (FE model CV)
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
# Feature Engineering
# =============================================================================
def engineer_features(train, test):
    """
    Create new features for both train and test.
    """
    print("\n=== Feature Engineering ===")
    
    key_features = ['feature_bz', 'feature_cd', 'feature_u', 'feature_af']
    agg_cols = ['ts_index', 'code'] + key_features
    
    combined = pd.concat([train[agg_cols], test[agg_cols]], axis=0).reset_index(drop=True)
    
    # 1. Market Features
    print("  Creating Market Features...")
    market_stats = combined.groupby('ts_index')[key_features].agg(['mean', 'std'])
    market_stats.columns = [f'market_{col}_{stat}' for col, stat in market_stats.columns]
    
    # 2. Sector Features
    print("  Creating Sector Features...")
    sector_stats = combined.groupby(['ts_index', 'code'])[key_features].mean()
    sector_stats.columns = [f'sector_{col}_mean' for col in sector_stats.columns]
    
    # 3. Rolling Trends
    print("  Creating Rolling Trends...")
    rolling_windows = [5, 20]
    rolling_feats = []
    
    for window in rolling_windows:
        rolled = market_stats[[f'market_{f}_mean' for f in key_features]].rolling(window=window).mean()
        rolled.columns = [f'market_rolling{window}_{f}' for f in key_features]
        rolling_feats.append(rolled)
        
    all_market_features = pd.concat([market_stats] + rolling_feats, axis=1)
    
    # Merge
    print("  Merging...")
    train = train.merge(all_market_features, on='ts_index', how='left')
    train = train.merge(sector_stats, on=['ts_index', 'code'], how='left')
    
    test = test.merge(all_market_features, on='ts_index', how='left')
    test = test.merge(sector_stats, on=['ts_index', 'code'], how='left')
    
    # Fill NAs
    new_cols = [c for c in train.columns if c.startswith('market_') or c.startswith('sector_')]
    train[new_cols] = train[new_cols].fillna(0)
    test[new_cols] = test[new_cols].fillna(0)
    
    print(f"  Added {len(new_cols)} features")
    return train, test, new_cols


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 80)
    print("HEDGE FUND FORECASTING - HYBRID SUBMISSION")
    print("=" * 80)
    
    # 1. Load Data
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    
    # 2. Feature Engineering
    train_eng, test_eng, new_features = engineer_features(train, test)
    
    original_features = [c for c in train.columns if c.startswith('feature_')]
    all_features = original_features + new_features
    
    print(f"Features: Base={len(original_features)}, Extended={len(all_features)}")
    
    # 3. Training & Prediction
    print("\n" + "=" * 60)
    print("TRAINING HYBRID MODELS")
    print("=" * 60)
    
    # Configuration
    # H1 uses Base, others use Extended
    feature_config = {
        1: original_features,
        3: all_features,
        10: all_features,
        25: all_features
    }
    
    # Shrinkages (Verified by Time-Series CV)
    shrinkage_config = {
        1: 0.293,  # Base CV
        3: 0.270,  # FE CV
        10: 0.320, # FE CV
        25: 0.340  # FE CV (Big improvement from 0.18!)
    }
    
    params = {
        'objective': 'huber',
        'alpha': 0.1,
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
    
    all_test_preds = np.zeros(len(test))
    horizons = sorted(train['horizon'].unique())
    
    for horizon in horizons:
        print(f"\nHorizon {horizon}...")
        
        # Select Features
        features_h = feature_config.get(horizon, all_features)
        print(f"  Using {len(features_h)} features ({'Base' if len(features_h)==86 else 'Extended'})")
        
        # Data
        train_h = train_eng[train_eng['horizon'] == horizon]
        X_train = train_h[features_h]
        y_train = train_h['y_target']
        train_weights = np.sqrt(train_h['weight'].values + 1)
        
        # Train
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
        model = lgb.train(
            params, train_data, num_boost_round=params['n_estimators'],
            callbacks=[lgb.log_evaluation(period=500)]
        )
        
        # Predict
        test_h = test_eng[test_eng['horizon'] == horizon]
        X_test = test_h[features_h]
        raw_preds = model.predict(X_test)
        
        # Shrink
        shrinkage = shrinkage_config.get(horizon, 0.25)
        shrunk_preds = raw_preds * shrinkage
        
        # Store
        mask = test_eng['horizon'] == horizon
        all_test_preds[mask.values] = shrunk_preds
        
        print(f"  Shrinkage: {shrinkage:.3f}")
        print(f"  Pred stats: mean={shrunk_preds.mean():.4f}, std={shrunk_preds.std():.4f}")
        
    # 4. Create Submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({
        'id': test['id'],
        'prediction': all_test_preds
    })
    
    output_path = OUTPUT_DIR / f'submission_hybrid_{timestamp}.csv'
    submission.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("SUBMISSION CREATED")
    print("=" * 60)
    print(f"File: {output_path}")
    print(f"Rows: {len(submission):,}")
    print(f"Overall Mean: {all_test_preds.mean():.6f}")
    
    # Save config
    config = {
        'timestamp': timestamp,
        'strategy': 'Hybrid (Base for H1, FE for H3+)',
        'huber_alpha': 0.1,
        'shrinkages': shrinkage_config
    }
    with open(OUTPUT_DIR / f'submission_hybrid_config_{timestamp}.json', 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()
