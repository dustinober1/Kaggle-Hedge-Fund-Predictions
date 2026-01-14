# Project Context and History

## 2026-01-13 - Advanced Score Push
- Implemented `src/15_advanced_score_push.py` to benchmark 6 advanced strategies.
- Strategies tested:
    1. Quantile Regression (Failed)
    2. LightGBM + CatBoost Blend (Winner)
    3. Ridge Blend (Failed)
    4. Deep Ensemble (Failed)
    5. Residual Boosting (Failed)
    6. Confidence Shrinkage (Mixed results)
- **Winning Strategy**: Strategy 2 (LightGBM + CatBoost Blend).
    - Improvement: +0.10% CV Ratio (0.9962 vs Baseline 0.9972).
    - Significant gains on H25 horizon.
- Generated final submission using Strategy 2 trained on full dataset.
