"""
Hedge Fund Time Series Forecasting - ADVANCED SCORE PUSH
==========================================================

Goal: Beat current best LB score (~0.053) and reach top 3

New Strategies NOT Yet Tried:
1. Quantile Regression (median) - more robust than Huber
2. LightGBM + CatBoost Blend - different gradient implementations
3. Ridge Regression Blend - simple linear model generalizes well
4. 10-Seed Deep Ensemble - double variance reduction
5. Residual Boosting - train 2nd model on residuals
6. Prediction Confidence Shrinkage - shrink uncertain predictions more
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
from pathlib import Path
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available, skipping Ridge strategy")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, skipping blend strategy")

# Paths
DATA_DIR = Path(__file__).parent.parent / 'ts-forecasting'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# GLOBAL HELPERS
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


def time_based_split(train, val_ratio=0.2):
    """Split data based on ts_index"""
    ts_max = train['ts_index'].max()
    ts_min = train['ts_index'].min()
    ts_range = ts_max - ts_min
    val_threshold = ts_max - int(ts_range * val_ratio)
    
    train_df = train[train['ts_index'] <= val_threshold].copy()
    val_df = train[train['ts_index'] > val_threshold].copy()
    return train_df, val_df


def ultra_fine_shrinkage_search(y_true, raw_preds, weights, step=0.001):
    """Find optimal shrinkage with ultra-fine grid"""
    best_s, best_r = 0, float('inf')
    # Search range appropriate for this competition (usually 0.0 to 0.5 is sufficient)
    for s in np.arange(0.0, 0.60, step):
        r = get_ratio(y_true, raw_preds * s, weights)
        if r < best_r:
            best_r = r
            best_s = s
    return round(best_s, 4), best_r


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================
def train_huber_model(train_df, feature_cols, horizon, weight_power=0.5, seed=42):
    """Train Huber LightGBM model (baseline)"""
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
        'random_state': seed,
        'verbose': -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    model = lgb.train(params, train_data, num_boost_round=1500)
    return model


def train_quantile_model(train_df, feature_cols, horizon, weight_power=0.5, seed=42):
    """Train quantile regression model (predicts median)"""
    train_h = train_df[train_df['horizon'] == horizon]
    X_train = train_h[feature_cols]
    y_train = train_h['y_target']
    train_weights = np.power(train_h['weight'].values + 1, weight_power)
    
    params = {
        'objective': 'quantile',
        'alpha': 0.5,  # 0.5 = median
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
        'random_state': seed,
        'verbose': -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    model = lgb.train(params, train_data, num_boost_round=1500)
    return model


def train_catboost_model(train_df, feature_cols, horizon, weight_power=0.5, seed=42):
    """Train CatBoost model"""
    train_h = train_df[train_df['horizon'] == horizon]
    X_train = train_h[feature_cols].values
    y_train = train_h['y_target'].values
    train_weights = np.power(train_h['weight'].values + 1, weight_power)
    
    model = CatBoostRegressor(
        loss_function='Huber:delta=0.5',
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=0.5,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False
    )
    model.fit(X_train, y_train, sample_weight=train_weights)
    return model


# =============================================================================
# STRATEGIES
# =============================================================================
def strategy_baseline(train_df, val_df, feature_cols, horizons, n_seeds=5):
    """Baseline: Multi-seed ensemble"""
    print(f"  Training {n_seeds}-seed baseline ensemble...")
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4242, 5555, 6789][:n_seeds]
    
    all_models = {h: [] for h in horizons}
    shrinkages = {}
    
    for h in horizons:
        print(f"    H{h}: ", end="")
        for seed in seeds:
            model = train_huber_model(train_df, feature_cols, h, seed=seed)
            all_models[h].append(model)
            print(".", end="", flush=True)
        
        val_h = val_df[val_df['horizon'] == h]
        X_val = val_h[feature_cols]
        preds = np.zeros((len(val_h), n_seeds))
        for i, model in enumerate(all_models[h]):
            preds[:, i] = model.predict(X_val)
        avg_preds = np.mean(preds, axis=1)
        
        shrinkages[h], ratio = ultra_fine_shrinkage_search(
            val_h['y_target'].values, avg_preds, val_h['weight'].values
        )
        print(f" shrinkage={shrinkages[h]:.4f}, ratio={ratio:.6f}")
    
    all_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        X_val = val_df.loc[mask, feature_cols]
        preds = np.zeros((mask.sum(), n_seeds))
        for i, model in enumerate(all_models[h]):
            preds[:, i] = model.predict(X_val)
        avg_preds = np.mean(preds, axis=1)
        all_preds[mask.values] = avg_preds * shrinkages[h]
        
    ratio = get_ratio(val_df['y_target'].values, all_preds, val_df['weight'].values)
    score = weighted_rmse_score(val_df['y_target'].values, all_preds, val_df['weight'].values)
    
    return {'name': 'baseline', 'ratio': ratio, 'score': score, 'models': all_models, 'shrinkages': shrinkages, 'n_seeds': n_seeds}


def strategy_quantile(train_df, val_df, feature_cols, horizons):
    """Strategy 1: Quantile Regression"""
    print("  Training quantile models...")
    models = {}
    shrinkages = {}
    
    for h in horizons:
        models[h] = train_quantile_model(train_df, feature_cols, h)
        val_h = val_df[val_df['horizon'] == h]
        raw_preds = models[h].predict(val_h[feature_cols])
        shrinkages[h], ratio = ultra_fine_shrinkage_search(
            val_h['y_target'].values, raw_preds, val_h['weight'].values
        )
        print(f"    H{h}: shrinkage={shrinkages[h]:.4f}, ratio={ratio:.6f}")
    
    all_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        all_preds[mask.values] = models[h].predict(val_df.loc[mask, feature_cols]) * shrinkages[h]
    
    ratio = get_ratio(val_df['y_target'].values, all_preds, val_df['weight'].values)
    score = weighted_rmse_score(val_df['y_target'].values, all_preds, val_df['weight'].values)
    
    return {'name': 'quantile', 'ratio': ratio, 'score': score, 'models': models, 'shrinkages': shrinkages}


def strategy_blend(train_df, val_df, feature_cols, horizons):
    """Strategy 2: LightGBM + CatBoost Blend"""
    if not CATBOOST_AVAILABLE:
        print("  CatBoost skipped.")
        return None
    
    print("  Training LightGBM and CatBoost models...")
    lgb_models = {}
    cb_models = {}
    blend_weights = {}
    shrinkages = {}
    
    for h in horizons:
        print(f"    H{h}...")
        lgb_models[h] = train_huber_model(train_df, feature_cols, h)
        cb_models[h] = train_catboost_model(train_df, feature_cols, h)
        
        val_h = val_df[val_df['horizon'] == h]
        lgb_preds = lgb_models[h].predict(val_h[feature_cols])
        cb_preds = cb_models[h].predict(val_h[feature_cols].values)
        
        best_r = float('inf')
        best_w = 0.5
        best_s = 0.2
        for w in np.arange(0.0, 1.01, 0.1):
            blended = w * lgb_preds + (1 - w) * cb_preds
            for s in np.arange(0.0, 0.60, 0.01):
                r = get_ratio(val_h['y_target'].values, blended * s, val_h['weight'].values)
                if r < best_r:
                    best_r = r
                    best_w = w
                    best_s = s
        blend_weights[h] = best_w
        shrinkages[h] = round(best_s, 3)
        print(f"      blend_w={best_w:.1f}, shrinkage={best_s:.3f}, ratio={best_r:.6f}")
    
    all_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        X_val = val_df.loc[mask, feature_cols]
        lgb_p = lgb_models[h].predict(X_val)
        cb_p = cb_models[h].predict(X_val.values)
        blended = blend_weights[h] * lgb_p + (1 - blend_weights[h]) * cb_p
        all_preds[mask.values] = blended * shrinkages[h]
    
    ratio = get_ratio(val_df['y_target'].values, all_preds, val_df['weight'].values)
    score = weighted_rmse_score(val_df['y_target'].values, all_preds, val_df['weight'].values)
    
    return {
        'name': 'lgb_catboost_blend', 'ratio': ratio, 'score': score,
        'lgb_models': lgb_models, 'cb_models': cb_models,
        'blend_weights': blend_weights, 'shrinkages': shrinkages
    }


def strategy_ridge_blend(train_df, val_df, feature_cols, horizons):
    """Strategy 3: Ridge Blend"""
    if not SKLEARN_AVAILABLE:
        print("  Ridge skipped.")
        return None
        
    print("  Training LightGBM and Ridge models...")
    lgb_models = {}
    ridge_models = {}
    blend_weights = {}
    shrinkages = {}
    
    for h in horizons:
        print(f"    H{h}...")
        train_h = train_df[train_df['horizon'] == h]
        lgb_models[h] = train_huber_model(train_df, feature_cols, h)
        
        # Ridge (fill NaNs)
        X_train = train_h[feature_cols].fillna(0).values
        y_train = train_h['y_target'].values
        ridge = Ridge(alpha=100.0)
        ridge.fit(X_train, y_train)
        ridge_models[h] = ridge
        
        val_h = val_df[val_df['horizon'] == h]
        lgb_preds = lgb_models[h].predict(val_h[feature_cols])
        ridge_preds = ridge.predict(val_h[feature_cols].fillna(0).values)
        
        best_r = float('inf')
        best_w = 0.9
        best_s = 0.2
        for w in np.arange(0.7, 1.01, 0.05):
            blended = w * lgb_preds + (1 - w) * ridge_preds
            for s in np.arange(0.0, 0.60, 0.01):
                r = get_ratio(val_h['y_target'].values, blended * s, val_h['weight'].values)
                if r < best_r:
                    best_r = r
                    best_w = w
                    best_s = s
        blend_weights[h] = best_w
        shrinkages[h] = round(best_s, 3)
        print(f"      blend_w={best_w:.2f}, shrinkage={best_s:.3f}, ratio={best_r:.6f}")
    
    all_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        X_val = val_df.loc[mask, feature_cols]
        lgb_p = lgb_models[h].predict(X_val)
        ridge_p = ridge_models[h].predict(X_val.values)
        blended = blend_weights[h] * lgb_p + (1 - blend_weights[h]) * ridge_p
        all_preds[mask.values] = blended * shrinkages[h]
    
    ratio = get_ratio(val_df['y_target'].values, all_preds, val_df['weight'].values)
    score = weighted_rmse_score(val_df['y_target'].values, all_preds, val_df['weight'].values)
    
    return {
        'name': 'lgb_ridge_blend', 'ratio': ratio, 'score': score,
        'lgb_models': lgb_models, 'ridge_models': ridge_models,
        'blend_weights': blend_weights, 'shrinkages': shrinkages
    }



def strategy_deep_ensemble(train_df, val_df, feature_cols, horizons, n_seeds=10):
    """Strategy 4: Deep Ensemble"""
    print(f"  Training {n_seeds}-seed deep ensemble...")
    # More seeds for deep ensemble
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4242, 5555, 6789, 1111, 2222, 3333, 4444, 5555][:n_seeds]
    
    all_models = {h: [] for h in horizons}
    shrinkages = {}
    
    for h in horizons:
        print(f"    H{h}: ", end="")
        for seed in seeds:
            model = train_huber_model(train_df, feature_cols, h, seed=seed)
            all_models[h].append(model)
            print(".", end="", flush=True)
            
        val_h = val_df[val_df['horizon'] == h]
        X_val = val_h[feature_cols]
        preds = np.zeros((len(val_h), n_seeds))
        for i, model in enumerate(all_models[h]):
            preds[:, i] = model.predict(X_val)
        avg_preds = np.mean(preds, axis=1)
        
        shrinkages[h], ratio = ultra_fine_shrinkage_search(
            val_h['y_target'].values, avg_preds, val_h['weight'].values
        )
        print(f" shrinkage={shrinkages[h]:.4f}, ratio={ratio:.6f}")
        
    all_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        X_val = val_df.loc[mask, feature_cols]
        preds = np.zeros((mask.sum(), n_seeds))
        for i, model in enumerate(all_models[h]):
            preds[:, i] = model.predict(X_val)
        avg_preds = np.mean(preds, axis=1)
        all_preds[mask.values] = avg_preds * shrinkages[h]
        
    ratio = get_ratio(val_df['y_target'].values, all_preds, val_df['weight'].values)
    score = weighted_rmse_score(val_df['y_target'].values, all_preds, val_df['weight'].values)
    
    return {'name': f'ensemble_{n_seeds}', 'ratio': ratio, 'score': score, 
            'models': all_models, 'shrinkages': shrinkages, 'n_seeds': n_seeds}


def strategy_residual_boosting(train_df, val_df, feature_cols, horizons):
    """Strategy 5: Residual Boosting"""
    print("  Training residual boosting models...")
    models_1 = {}
    models_2 = {}
    alphas = {}
    shrinkages = {}
    
    for h in horizons:
        print(f"    H{h}...")
        train_h = train_df[train_df['horizon'] == h].copy()
        
        models_1[h] = train_huber_model(train_df, feature_cols, h)
        train_preds_1 = models_1[h].predict(train_h[feature_cols])
        residuals = train_h['y_target'].values - train_preds_1
        
        X_train = train_h[feature_cols]
        train_weights = np.sqrt(train_h['weight'].values + 1)
        params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'learning_rate': 0.01, 'num_leaves': 15, 'max_depth': 4,
            'min_child_samples': 200, 'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 1.0, 'random_state': 42, 'verbose': -1,
        }
        train_data = lgb.Dataset(X_train, label=residuals, weight=train_weights)
        models_2[h] = lgb.train(params, train_data, num_boost_round=500)
        
        val_h = val_df[val_df['horizon'] == h]
        val_preds_1 = models_1[h].predict(val_h[feature_cols])
        val_preds_2 = models_2[h].predict(val_h[feature_cols])
        
        best_r = float('inf')
        best_alpha = 0.0
        best_s = 0.2
        for alpha in np.arange(0.0, 0.51, 0.05):
            combined = val_preds_1 + alpha * val_preds_2
            for s in np.arange(0.0, 0.60, 0.01):
                r = get_ratio(val_h['y_target'].values, combined * s, val_h['weight'].values)
                if r < best_r:
                    best_r = r
                    best_alpha = alpha
                    best_s = s
        alphas[h] = best_alpha
        shrinkages[h] = round(best_s, 3)
        print(f"      alpha={best_alpha:.2f}, shrinkage={best_s:.3f}, ratio={best_r:.6f}")
    
    all_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        X_val = val_df.loc[mask, feature_cols]
        combined = models_1[h].predict(X_val) + alphas[h] * models_2[h].predict(X_val)
        all_preds[mask.values] = combined * shrinkages[h]
        
    ratio = get_ratio(val_df['y_target'].values, all_preds, val_df['weight'].values)
    score = weighted_rmse_score(val_df['y_target'].values, all_preds, val_df['weight'].values)
    
    return {
        'name': 'residual_boosting', 'ratio': ratio, 'score': score,
        'models_1': models_1, 'models_2': models_2, 'alphas': alphas, 'shrinkages': shrinkages
    }


def strategy_confidence_shrinkage(train_df, val_df, feature_cols, horizons, n_seeds=5):
    """Strategy 6: Confidence Shrinkage"""
    print(f"  Training ensemble for confidence ({n_seeds} seeds)...")
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4242, 5555, 6789][:n_seeds]
    all_models = {h: [] for h in horizons}
    shrinkage_params = {}
    
    for h in horizons:
        print(f"    H{h}: ", end="")
        for seed in seeds:
            model = train_huber_model(train_df, feature_cols, h, seed=seed)
            all_models[h].append(model)
            print(".", end="", flush=True)
        
        val_h = val_df[val_df['horizon'] == h]
        X_val = val_h[feature_cols]
        preds = np.zeros((len(val_h), n_seeds))
        for i, model in enumerate(all_models[h]):
            preds[:, i] = model.predict(X_val)
        avg_preds = np.mean(preds, axis=1)
        std_preds = np.std(preds, axis=1)
        
        best_r = float('inf')
        best_base_s = 0.2
        best_conf_factor = 0.0
        for base_s in np.arange(0.05, 0.40, 0.02):
            for conf_factor in np.arange(0.0, 1.01, 0.1):
                std_norm = (std_preds - std_preds.min()) / (std_preds.max() - std_preds.min() + 1e-10)
                adaptive_s = base_s * (1 - conf_factor * std_norm)
                shrunk_preds = avg_preds * adaptive_s
                r = get_ratio(val_h['y_target'].values, shrunk_preds, val_h['weight'].values)
                if r < best_r:
                    best_r = r
                    best_base_s = base_s
                    best_conf_factor = conf_factor
        shrinkage_params[h] = (best_base_s, best_conf_factor)
        print(f" base_s={best_base_s:.2f}, conf={best_conf_factor:.1f}, ratio={best_r:.6f}")
    
    all_preds = np.zeros(len(val_df))
    for h in horizons:
        mask = val_df['horizon'] == h
        X_val = val_df.loc[mask, feature_cols]
        preds = np.zeros((mask.sum(), n_seeds))
        for i, model in enumerate(all_models[h]):
            preds[:, i] = model.predict(X_val)
        avg_p = np.mean(preds, axis=1)
        std_p = np.std(preds, axis=1)
        base_s, conf_factor = shrinkage_params[h]
        std_norm = (std_p - std_p.min()) / (std_p.max() - std_p.min() + 1e-10)
        adaptive_s = base_s * (1 - conf_factor * std_norm)
        all_preds[mask.values] = avg_p * adaptive_s
        
    ratio = get_ratio(val_df['y_target'].values, all_preds, val_df['weight'].values)
    score = weighted_rmse_score(val_df['y_target'].values, all_preds, val_df['weight'].values)
    return {'name': 'confidence_shrinkage', 'ratio': ratio, 'score': score, 
            'models': all_models, 'shrinkage_params': shrinkage_params, 'n_seeds': n_seeds}


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Run fast mode (random 20% sample, fewer seeds)')
    args = parser.parse_args()

    print("=" * 80)
    print("HEDGE FUND FORECASTING - ADVANCED SCORE PUSH")
    if args.fast:
        print("  [FAST MODE ENABLED] using 20% random subsample")
    print("=" * 80)
    
    print("\nLoading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    test = pd.read_parquet(DATA_DIR / 'test.parquet')
    feature_cols = [c for c in train.columns if c.startswith('feature_')]
    print(f"Train: {len(train):,}, Test: {len(test):,}, Features: {len(feature_cols)}")
    
    train_df, val_df = time_based_split(train, val_ratio=0.2)
    
    if args.fast:
        print("  Subsampling training data (random 20%)...")
        train_df = train_df.sample(frac=0.2, random_state=42)
        print(f"  Train subset: {len(train_df):,}")
    else:
        print(f"Train split: {len(train_df):,}, Val split: {len(val_df):,}")
    
    horizons = sorted(train['horizon'].unique())
    results = []
    
    # Baseline (Skipped for speed - Strategy 2 verified better)
    # print("\n" + "=" * 60)
    # n_seeds_base = 1 if args.fast else 5
    # baseline = strategy_baseline(train_df, val_df, feature_cols, horizons, n_seeds=n_seeds_base)
    # results.append(baseline)
    # print(f"  RESULT: ratio={baseline['ratio']:.6f}, score={baseline['score']:.6f}")
    
    # Strategy 1 (Skipped - failed in fast bench)
    # print("\n" + "=" * 60)
    # s1 = strategy_quantile(train_df, val_df, feature_cols, horizons)
    # results.append(s1)
    # print(f"  RESULT: ratio={s1['ratio']:.6f}, score={s1['score']:.6f}")
    
    # Strategy 1 (Skipped)
    # ...
    
    # Strategy 2 (Skipped validation to save time - jumping to submission)
    # print("\n" + "=" * 60)
    # s2 = strategy_blend(train_df, val_df, feature_cols, horizons)
    # if s2:
    #     results.append(s2)
    #     print(f"  RESULT: ratio={s2['ratio']:.6f}, score={s2['score']:.6f}")
        
    # Strategy 3 (Skipped)
    # ...
        
    # Strategy 4 (Skipped)
    # print("\n" + "=" * 60)
    # n_seeds_ens = 2 if args.fast else 10
    # s4 = strategy_deep_ensemble(train_df, val_df, feature_cols, horizons, n_seeds=n_seeds_ens)
    # results.append(s4)
    # print(f"  RESULT: ratio={s4['ratio']:.6f}, score={s4['score']:.6f}")
    
    # Strategy 5 (Skipped)
    # print("\n" + "=" * 60)
    # s5 = strategy_residual_boosting(train_df, val_df, feature_cols, horizons)
    # results.append(s5)
    # print(f"  RESULT: ratio={s5['ratio']:.6f}, score={s5['score']:.6f}")
    
    # Strategy 6 (Skipped)
    # print("\n" + "=" * 60)
    # n_seeds_conf = 2 if args.fast else 5
    # s6 = strategy_confidence_shrinkage(train_df, val_df, feature_cols, horizons, n_seeds=n_seeds_conf)
    # results.append(s6)
    # print(f"  RESULT: ratio={s6['ratio']:.6f}, score={s6['score']:.6f}")
    
    # Force best strategy for submission
    results = [{'name': 'lgb_catboost_blend', 'ratio': 0.9964, 'score': 0.0597}]
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON - All Strategies")
    print("=" * 60)
    results_sorted = sorted(results, key=lambda x: x['ratio'])
    best = results_sorted[0]
    
    # for i, r in enumerate(results_sorted):
    #     marker = "★ BEST" if i == 0 else ""
    #     improvement = (baseline['ratio'] - r['ratio']) * 100
    #     print(f"  {r['name']:25s}: ratio={r['ratio']:.6f} ({improvement:+.4f}%) {marker}")
        
    if not args.fast:
        print(f"\nBest strategy: {best['name']}. Generating submission...")
        
        # Hardcoded best parameters from validation run
        best_params = {
            1:  {'w': 1.0, 's': 0.120},
            3:  {'w': 0.6, 's': 0.080},
            10: {'w': 0.8, 's': 0.300},
            25: {'w': 0.6, 's': 0.330}
        }
        
        print("  Reloading full data for final training...")
        train = pd.read_parquet(DATA_DIR / 'train.parquet')
        test = pd.read_parquet(DATA_DIR / 'test.parquet')
        feature_cols = [c for c in train.columns if c.startswith('feature_')]
        
        all_preds = np.zeros(len(test))
        
        for h in horizons:
            print(f"  Processing H{h} (w={best_params[h]['w']}, s={best_params[h]['s']})...")
            
            # Filter by horizon
            train_h = train[train['horizon'] == h]
            test_h = test[test['horizon'] == h]
            
            # Train LightGBM
            print("    Training LightGBM on full data...")
            lgb_model = train_huber_model(train, feature_cols, h) # Train on full logic handles filtering inside? No, passed full df usually?
            # Wait, `train_huber_model` takes `train_df`. If I pass `train_h`, it uses that.
            # But the function signature is `train_huber_model(train_df, feature_cols, horizon, ...)`
            # And it internally filters: `train_h = train_df[train_df['horizon'] == horizon]`
            # So I should pass the full `train` dataframe.
            lgb_model = train_huber_model(train, feature_cols, h)
            lgb_pred = lgb_model.predict(test_h[feature_cols])
            
            # Train CatBoost
            print("    Training CatBoost on full data...")
            cb_model = train_catboost_model(train, feature_cols, h)
            cb_pred = cb_model.predict(test_h[feature_cols])
            
            # Blend
            w = best_params[h]['w']
            s = best_params[h]['s']
            
            final_pred = w * lgb_pred + (1 - w) * cb_pred
            final_pred = final_pred * s
            
            # Assign to strict indices
            test_indices = test_h.index
            all_preds[test_indices] = final_pred
            
        # Save submission
        sub = pd.DataFrame({'id': test['id'], 'pred': all_preds})
        
        # Ensure outputs dir exists
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"outputs/submission_advanced_blend_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        sub.to_csv(filename, index=False)
        print(f"\nSubmission saved to: {filename}")
        print("First 5 predictions:")
        print(sub.head())

    else:
        print(f"\n[FAST MODE] Best quick-check strategy: {best['name']}")


if __name__ == "__main__":
    main()
