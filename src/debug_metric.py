"""
Debug script to understand the competition metric better.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'


def weighted_rmse_score(y_true, y_pred, weights):
    """Competition metric"""
    denom = np.sum(weights * y_true ** 2)
    numer = np.sum(weights * (y_true - y_pred) ** 2)
    
    if denom == 0:
        return 0.0
    
    ratio = numer / denom
    clipped = np.clip(ratio, 0.0, 1.0)
    score = float(np.sqrt(1.0 - clipped))
    
    return score, ratio, numer, denom


def main():
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / 'train.parquet')
    
    print(f"Train shape: {train.shape}")
    
    # Split for analysis
    val_mask = train['ts_index'] > 2881
    val = train[val_mask].copy()
    
    print(f"\nValidation set: {len(val):,} rows")
    
    y = val['y_target'].values
    w = val['weight'].values
    
    # Analyze the components of the metric
    print("\n" + "=" * 60)
    print("METRIC ANALYSIS")
    print("=" * 60)
    
    print(f"\nWeight statistics:")
    print(f"  Min: {w.min():.2e}")
    print(f"  Max: {w.max():.2e}")
    print(f"  Mean: {w.mean():.2e}")
    print(f"  Sum: {w.sum():.2e}")
    
    print(f"\nTarget statistics:")
    print(f"  Min: {y.min():.4f}")
    print(f"  Max: {y.max():.4f}")
    print(f"  Mean: {y.mean():.4f}")
    print(f"  Std: {y.std():.4f}")
    
    # Denominator: sum(w * y^2)
    denom = np.sum(w * y ** 2)
    print(f"\nDenominator (sum(w * y^2)): {denom:.4e}")
    
    # If we predict 0, numerator = sum(w * y^2) = denominator
    # So ratio = 1, score = 0
    pred_zero = np.zeros_like(y)
    score_zero, ratio_zero, numer_zero, _ = weighted_rmse_score(y, pred_zero, w)
    print(f"\nPredicting ZERO:")
    print(f"  Numerator (sum(w * (y-0)^2)): {numer_zero:.4e}")
    print(f"  Ratio: {ratio_zero:.6f}")
    print(f"  Score: {score_zero:.6f}")
    
    # If we predict y perfectly, numerator = 0
    # So ratio = 0, score = 1
    pred_perfect = y.copy()
    score_perfect, ratio_perfect, numer_perfect, _ = weighted_rmse_score(y, pred_perfect, w)
    print(f"\nPredicting PERFECTLY:")
    print(f"  Numerator (sum(w * (y-y)^2)): {numer_perfect:.4e}")
    print(f"  Ratio: {ratio_perfect:.6f}")
    print(f"  Score: {score_perfect:.6f}")
    
    # To get a positive score, we need ratio < 1
    # That means sum(w * (y-pred)^2) < sum(w * y^2)
    # Which means our predictions need to be BETTER than predicting zero
    
    # Let's see what happens with mean prediction
    pred_mean = np.full_like(y, y.mean())
    score_mean, ratio_mean, numer_mean, _ = weighted_rmse_score(y, pred_mean, w)
    print(f"\nPredicting MEAN ({y.mean():.4f}):")
    print(f"  Numerator: {numer_mean:.4e}")
    print(f"  Ratio: {ratio_mean:.6f}")
    print(f"  Score: {score_mean:.6f}")
    
    # The issue might be that extreme weights are dominating
    # Let's check which samples have the highest weighted contribution
    print("\n" + "=" * 60)
    print("EXTREME WEIGHT ANALYSIS")
    print("=" * 60)
    
    weighted_y2 = w * y ** 2
    top_indices = np.argsort(weighted_y2)[-10:]
    
    print(f"\nTop 10 samples by weighted y^2:")
    for idx in reversed(top_indices):
        print(f"  idx={idx}: w={w[idx]:.2e}, y={y[idx]:.4f}, w*y^2={weighted_y2[idx]:.4e}")
    
    # What % of denominator comes from top samples?
    top_1pct = int(len(val) * 0.01)
    top_indices_1pct = np.argsort(weighted_y2)[-top_1pct:]
    contribution = weighted_y2[top_indices_1pct].sum() / denom
    print(f"\nTop 1% of samples contribute {contribution*100:.2f}% of denominator!")
    
    top_10pct = int(len(val) * 0.10)
    top_indices_10pct = np.argsort(weighted_y2)[-top_10pct:]
    contribution_10 = weighted_y2[top_indices_10pct].sum() / denom
    print(f"Top 10% of samples contribute {contribution_10*100:.2f}% of denominator!")
    
    # What if we only focus on predicting these high-weight samples well?
    print("\n" + "=" * 60)
    print("TARGETED PREDICTION STRATEGY")
    print("=" * 60)
    
    # Try median prediction (more robust to outliers)
    pred_median = np.full_like(y, np.median(y))
    score_median, ratio_median, _, _ = weighted_rmse_score(y, pred_median, w)
    print(f"\nPredicting MEDIAN ({np.median(y):.6f}):")
    print(f"  Ratio: {ratio_median:.6f}")
    print(f"  Score: {score_median:.6f}")
    
    # Very small prediction (close to median which is near 0)
    for pred_val in [0.0, -0.001, -0.01, -0.1, -0.5, -1.0]:
        pred_const = np.full_like(y, pred_val)
        score_const, ratio_const, _, _ = weighted_rmse_score(y, pred_const, w)
        print(f"  Predicting {pred_val:7.3f}: ratio={ratio_const:.6f}, score={score_const:.6f}")
    
    print("\n" + "=" * 60)
    print("INSIGHTS")
    print("=" * 60)
    print("""
    The metric is a "skill score" that compares your predictions to predicting zero.
    
    - Score = 0 means you're no better than predicting zero
    - Score > 0 means you're making predictions that are closer to true values
    - The extreme weights make this very challenging
    
    Strategy needed:
    1. Focus on samples with high w*y^2 (they dominate the metric)
    2. Need to predict the direction and magnitude correctly
    3. The signal-to-noise ratio is very low
    """)


if __name__ == "__main__":
    main()
