"""
Hedge Fund Time Series Forecasting - FINAL SUBMISSION (Safe Baseline)
===================================================================

STRATEGY: REVERT TO ROBUST BASELINE
-----------------------------------
The "Hybrid" strategy with Market features failed on LB (Likely regime shift).
The "Baseline" (Score 0.053) was superior.

Key Characteristics of the 0.053 Baseline:
1. Original Features ONLY (No aggregate noise).
2. Huber Alpha = 0.5 (Standard robust).
3. Aggressive Shrinkage for Short Horizons (H1~0.12, H3~0.06).

This script reproduces that robust interactions with hardcoded conservative shrinkages.
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

def main():
    print("=" * 80)
    print("HEDGE FUND FORECASTING - FINAL SUBMISSION (ROBUST BASELINE)")
    print("=" * 80)
    
    # 1. Load Data
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    
    # 2. Features: ORIGINAL ONLY
    feature_cols = [c for c in train.columns if c.startswith('feature_')]
    print(f"Using {len(feature_cols)} Original Features (Dropping all experimental features)")
    
    # 3. Parameters (Matching the 0.053 Baseline)
    params = {
        'objective': 'huber',
        'alpha': 0.5,            # Revert to 0.5 (Baseline value)
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
    
    # 4. Shrinkage Configuration (Conservative/Aggressive)
    # H1/H3 are dominated by noise -> Shrink heavily
    # H10/H25 have signal -> Shrink moderately
    shrinkage_config = {
        1: 0.15,
        3: 0.15,
        10: 0.28,
        25: 0.30
    }
    
    print("\n" + "=" * 60)
    print("TRAINING PER-HORIZON MODELS")
    print("=" * 60)
    print(f"Shrinkage Config: {shrinkage_config}")
    
    all_test_preds = np.zeros(len(test))
    horizons = sorted(train['horizon'].unique())
    
    for horizon in horizons:
        print(f"\nHorizon {horizon}...")
        
        # Filter Data
        train_h = train[train['horizon'] == horizon]
        X_train = train_h[feature_cols]
        y_train = train_h['y_target']
        
        # Weights: Sqrt (Proven robust)
        w_train = np.sqrt(train_h['weight'].values + 1)
        
        # Train
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        model = lgb.train(
            params, train_data, num_boost_round=params['n_estimators'],
            callbacks=[lgb.log_evaluation(period=500)]
        )
        
        # Predict Test
        test_h = test[test['horizon'] == horizon]
        X_test = test_h[feature_cols]
        raw_preds = model.predict(X_test)
        
        # Apply Shrinkage
        s = shrinkage_config.get(horizon, 0.25)
        final_preds = raw_preds * s
        
        # Store
        mask = test['horizon'] == horizon
        all_test_preds[mask.values] = final_preds
        
        print(f"  Shrinkage Applied: {s}")
        print(f"  Pred stats: mean={final_preds.mean():.6f}, std={final_preds.std():.6f}")

    # 5. Create Submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({
        'id': test['id'],
        'prediction': all_test_preds
    })
    
    output_path = OUTPUT_DIR / f'submission_final_robust_{timestamp}.csv'
    submission.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("FINAL SUBMISSION CREATED")
    print("=" * 60)
    print(f"File: {output_path}")
    print(f"Strategy: Original Features + Huber(0.5) + Conservative Shrinkage")
    
    # Save config
    config = {
        'strategy': 'Final Robust Baseline',
        'features': 'Original Only',
        'alpha': 0.5,
        'shrinkage': shrinkage_config
    }
    with open(OUTPUT_DIR / f'submission_final_config_{timestamp}.json', 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()
