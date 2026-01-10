"""
Hedge Fund Time Series Forecasting - Data Exploration
======================================================

Competition Overview:
- Evaluation: Weighted RMSE Skill Score
- Key Rule: Predict ts_index t using only data from 0 to t (no look-ahead)
- Features: 86 anonymized features (feature_a to feature_ch)
- Horizons: 1 (short), 3 (medium), 10 (long), 25 (extra-long)
- Prize: $10,000 total
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json

# Configure display options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.4f}'.format)
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [14, 6]
plt.rcParams['font.size'] = 12
sns.set_palette('husl')

# Define paths
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 80)
    print("HEDGE FUND TIME SERIES FORECASTING - DATA EXPLORATION")
    print("=" * 80)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. LOADING DATA")
    print("=" * 80)
    
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"\nTrain memory usage: {train.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"Test memory usage: {test.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    # =========================================================================
    # 2. DATA STRUCTURE
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. DATA STRUCTURE")
    print("=" * 80)
    
    print("\nTrain columns:")
    print(train.columns.tolist())
    
    print("\nTest columns:")
    print(test.columns.tolist())
    
    # Identify column groups
    train_columns = train.columns.tolist()
    feature_cols = [col for col in train_columns if col.startswith('feature_')]
    
    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:5]} ... {feature_cols[-5:]}")
    
    # Check column differences
    train_cols_set = set(train.columns)
    test_cols_set = set(test.columns)
    
    print("\nColumns in train but NOT in test:")
    print(train_cols_set - test_cols_set)
    
    print("\nColumns in test but NOT in train:")
    print(test_cols_set - train_cols_set)
    
    # =========================================================================
    # 3. CATEGORICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. CATEGORICAL ANALYSIS")
    print("=" * 80)
    
    for col in ['code', 'sub_code', 'sub_category']:
        print(f"\n--- {col.upper()} ---")
        train_vals = set(train[col].unique())
        test_vals = set(test[col].unique())
        print(f"Unique in train: {len(train_vals)}")
        print(f"Unique in test: {len(test_vals)}")
        print(f"Only in train: {len(train_vals - test_vals)}")
        print(f"Only in test: {len(test_vals - train_vals)}")
        print(f"In both: {len(train_vals & test_vals)}")
    
    # =========================================================================
    # 4. HORIZON ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. HORIZON ANALYSIS")
    print("=" * 80)
    
    print("\nTrain horizon distribution:")
    print(train['horizon'].value_counts().sort_index())
    
    print("\nTest horizon distribution:")
    print(test['horizon'].value_counts().sort_index())
    
    # =========================================================================
    # 5. TEMPORAL ANALYSIS (ts_index)
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. TEMPORAL ANALYSIS (ts_index)")
    print("=" * 80)
    
    print(f"\nTrain ts_index range: {train['ts_index'].min()} to {train['ts_index'].max()}")
    print(f"Train ts_index unique: {train['ts_index'].nunique()}")
    
    print(f"\nTest ts_index range: {test['ts_index'].min()} to {test['ts_index'].max()}")
    print(f"Test ts_index unique: {test['ts_index'].nunique()}")
    
    train_ts = set(train['ts_index'].unique())
    test_ts = set(test['ts_index'].unique())
    print(f"\nts_index overlap: {len(train_ts & test_ts)}")
    print(f"IMPORTANT: Test data comes AFTER training data!" if test['ts_index'].min() > train['ts_index'].max() else "Note: Some overlap between train and test ts_index")
    
    # =========================================================================
    # 6. TARGET ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. TARGET ANALYSIS")
    print("=" * 80)
    
    if 'y_target' in train.columns:
        print("\nTarget (y_target) statistics:")
        print(train['y_target'].describe())
        
        print(f"\nTarget missing: {train['y_target'].isna().sum()} ({100*train['y_target'].isna().mean():.2f}%)")
        print(f"Target zeros: {(train['y_target'] == 0).sum()} ({100*(train['y_target'] == 0).mean():.2f}%)")
        
        print("\nTarget by horizon:")
        print(train.groupby('horizon')['y_target'].describe())
        
        print("\nTarget by sub_category:")
        print(train.groupby('sub_category')['y_target'].agg(['mean', 'std', 'median', 'count']).sort_values('mean', ascending=False))
    else:
        print("No 'y_target' column in train data")
    
    # =========================================================================
    # 7. WEIGHT ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. WEIGHT ANALYSIS (Critical for evaluation metric!)")
    print("=" * 80)
    
    if 'weight' in train.columns:
        print("\nTrain weight statistics:")
        print(train['weight'].describe())
        
        print("\nWeight by horizon:")
        print(train.groupby('horizon')['weight'].describe())
        
        print("\nWeight by sub_category:")
        print(train.groupby('sub_category')['weight'].agg(['mean', 'std', 'min', 'max']))
    else:
        print("No 'weight' column in train data")
    
    if 'weight' in test.columns:
        print("\nTest weight statistics:")
        print(test['weight'].describe())
    else:
        print("\nNote: Test data does NOT have 'weight' column (weights are used for scoring only)")
    
    # =========================================================================
    # 8. FEATURE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. FEATURE ANALYSIS")
    print("=" * 80)
    
    # Feature statistics
    feature_stats = train[feature_cols].describe().T
    feature_stats['missing'] = train[feature_cols].isna().sum()
    feature_stats['missing_pct'] = 100 * train[feature_cols].isna().mean()
    feature_stats['zeros'] = (train[feature_cols] == 0).sum()
    feature_stats['zeros_pct'] = 100 * (train[feature_cols] == 0).mean()
    
    print("\nFeature statistics summary (first 20):")
    print(feature_stats.head(20))
    
    print("\nFeatures with most missing values:")
    print(feature_stats.sort_values('missing_pct', ascending=False)[['missing', 'missing_pct']].head(10))
    
    # Feature-target correlations
    if 'y_target' in train.columns:
        print("\nComputing feature-target correlations...")
        correlations = train[feature_cols + ['y_target']].corr()['y_target'].drop('y_target').sort_values(key=abs, ascending=False)
        
        print("\nTop 20 most correlated features with target:")
        print(correlations.head(20))
        
        print("\nTop 20 least correlated features with target:")
        print(correlations.tail(20))
    
    # =========================================================================
    # 9. ENTITY-LEVEL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("9. ENTITY-LEVEL ANALYSIS")
    print("=" * 80)
    
    entity_combo_train = train.groupby(['code', 'sub_code', 'sub_category']).size().reset_index(name='count')
    entity_combo_test = test.groupby(['code', 'sub_code', 'sub_category']).size().reset_index(name='count')
    
    print(f"\nUnique entity combinations in train: {len(entity_combo_train)}")
    print(f"Unique entity combinations in test: {len(entity_combo_test)}")
    
    train_entities = set(zip(entity_combo_train['code'], entity_combo_train['sub_code'], entity_combo_train['sub_category']))
    test_entities = set(zip(entity_combo_test['code'], entity_combo_test['sub_code'], entity_combo_test['sub_category']))
    
    print(f"\nEntity combinations only in train: {len(train_entities - test_entities)}")
    print(f"Entity combinations only in test: {len(test_entities - train_entities)}")
    print(f"Entity combinations in both: {len(train_entities & test_entities)}")
    
    # Time series length per entity
    entity_ts_length = train.groupby(['code', 'sub_code', 'sub_category'])['ts_index'].nunique()
    print("\nTime series length per entity:")
    print(entity_ts_length.describe())
    
    # =========================================================================
    # 10. HIGHLY CORRELATED FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("10. HIGHLY CORRELATED FEATURES")
    print("=" * 80)
    
    print("\nComputing feature correlation matrix...")
    feature_corr = train[feature_cols].corr()
    
    # Find highly correlated pairs
    upper_triangle = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    high_corr_pairs = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            corr_val = upper_triangle.loc[idx, col]
            if pd.notna(corr_val) and abs(corr_val) > 0.9:
                high_corr_pairs.append((idx, col, corr_val))
    
    print(f"\nHighly correlated feature pairs (|r| > 0.9): {len(high_corr_pairs)}")
    if high_corr_pairs:
        for f1, f2, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:20]:
            print(f"  {f1} <-> {f2}: {r:.4f}")
    
    # =========================================================================
    # 11. SAVE EXPLORATION RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("11. SAVING EXPLORATION RESULTS")
    print("=" * 80)
    
    exploration_results = {
        'train_shape': list(train.shape),
        'test_shape': list(test.shape),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'n_codes': int(train['code'].nunique()),
        'n_sub_codes': int(train['sub_code'].nunique()),
        'n_sub_categories': int(train['sub_category'].nunique()),
        'horizons': sorted([int(x) for x in train['horizon'].unique()]),
        'train_ts_range': [int(train['ts_index'].min()), int(train['ts_index'].max())],
        'test_ts_range': [int(test['ts_index'].min()), int(test['ts_index'].max())],
        'target_stats': train['y_target'].describe().to_dict() if 'y_target' in train.columns else None,
        'weight_stats': train['weight'].describe().to_dict() if 'weight' in train.columns else None,
        'entities_only_in_test': len(test_entities - train_entities),
        'high_corr_pairs': len(high_corr_pairs),
    }
    
    output_path = OUTPUT_DIR / 'exploration_results.json'
    with open(output_path, 'w') as f:
        json.dump(exploration_results, f, indent=2, default=str)
    
    print(f"\nExploration results saved to {output_path}")
    
    # =========================================================================
    # 12. KEY FINDINGS SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("12. KEY FINDINGS SUMMARY")
    print("=" * 80)
    
    print(f"""
    DATA STRUCTURE:
    - Training: {train.shape[0]:,} rows, {train.shape[1]} columns
    - Test: {test.shape[0]:,} rows, {test.shape[1]} columns
    - Features: {len(feature_cols)} anonymized features
    - Horizons: {sorted(train['horizon'].unique().tolist())}

    TEMPORAL STRUCTURE:
    - Train ts_index: {train['ts_index'].min()} to {train['ts_index'].max()}
    - Test ts_index: {test['ts_index'].min()} to {test['ts_index'].max()}
    - Test is AFTER train: {test['ts_index'].min() > train['ts_index'].max()}

    ENTITY STRUCTURE:
    - Codes: {train['code'].nunique()}
    - Sub-codes: {train['sub_code'].nunique()}
    - Sub-categories: {train['sub_category'].nunique()}
    - New entities in test: {len(test_entities - train_entities)}

    CRITICAL INSIGHTS:
    - Weight column only in train (used for evaluation)
    - Test data is strictly AFTER training data
    - Must use sequential prediction (no look-ahead)
    - {len(high_corr_pairs)} highly correlated feature pairs (consider dropping)

    RECOMMENDATIONS:
    1. Time-based validation (train early, validate late ts_index)
    2. Weight recent data more heavily
    3. Build horizon-specific models or features
    4. Handle new entities in test with global model
    5. Use weighted loss function matching competition metric
    """)
    
    return train, test, feature_cols, exploration_results


if __name__ == "__main__":
    train, test, feature_cols, results = main()
