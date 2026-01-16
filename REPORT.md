# Experiment Report: Hedge Fund Time Series Forecasting

This report is a process write-up. It focuses on the exploration loop (hypothesis → test → takeaway), not on presenting a single “best submission.”

## 1) Problem Framing

- Task: predict a continuous return-like target for many entities across multiple forecast horizons (1/3/10/25).
- Practical difficulty: extremely low signal-to-noise. Under the competition metric, “predict zero” is a strong baseline and many models can underperform it.
- Constraint: predictions should be produced without look-ahead (predict `ts_index = t` using only data available up to `t`).

## 2) Metric + Validation Discipline

- Metric: a weighted, clipped, RMSE-style “skill score” (`Competition_Rules/overview.md` provides the formula and reference code).
- Public/private split: only part of the test set is visible during development; treat “small CV gains” as fragile.
- Validation approach used in this repo: time-series cross-validation (no shuffling), plus horizon-aware reporting (horizons behave differently).

## 3) Experiment Timeline

### Phase A — EDA and baseline scaffolding

Goal: understand the data shape, horizon differences, and set up a reliable training/validation loop.

- `src/01_data_exploration.py`: initial EDA / sanity checks.
- `src/02_lgb_baseline.py`, `src/03_lgb_baseline_v2.py`, `src/04_lgb_baseline_v3.py`: establish baseline LightGBM pipelines and iterate on validation plumbing.

Takeaway: the baseline modeling problem is dominated by outliers and noise; naive loss choices are unstable.

### Phase B — Weighting experiments

Hypothesis: because the metric is weighted, training should respect `weight` (without using it as a feature).

- `src/05_high_weight_focus.py`: tested “focus on high-weight rows” approaches.

Takeaway: over-focusing on high-weight samples can harm generalization. A more stable compromise used later is a monotone transform of the provided weights (e.g., `sqrt(weight + 1)`), which reduces extreme weight dominance while still respecting the metric.

### Phase C — Robust objectives (outlier resistance)

Hypothesis: outliers cause MSE-trained models to swing too hard; a robust loss should improve stability.

- `src/06_strategy_refinement_v2.py`, `src/08_advanced_tuning.py`: explored robust objectives (Huber variants) and tuned their aggressiveness (the `alpha` / “box width”).

Takeaway: robust losses tend to reduce variance and produce predictions that are easier to calibrate under the skill-score metric.

### Phase D — Calibration via shrinkage (variance control)

Hypothesis: even robust models remain overconfident; shrinking predictions toward zero can improve the metric by reducing variance.

- Implemented global/per-horizon shrinkage factors and tuned them alongside the model.

Postmortem lesson: shrinkage is sensitive, especially by horizon. A single wrong factor can dominate results. Example of a shrinkage regression that motivated tighter config tracking:

| Horizon | Example “correct” factors | Example “incorrect” factors |
|---------|---------------------------|-----------------------------|
| H1 | 0.12 | 0.15 |
| H3 | 0.06 | 0.15 (too high) |
| H10 | 0.27 | 0.28 |
| H25 | 0.29 | 0.30 |

Takeaway: treat shrinkage like a first-class hyperparameter and log it per run, per horizon.

### Phase E — Feature set ablations (macro vs. micro)

Hypothesis: aggregated “market/sector” features can help longer horizons by adding macro context.

- `src/09_feature_engineering.py`, `src/10_hybrid_submission.py`: tested adding aggregated features and compared against using only the provided anonymized features.

Takeaway: feature engineering can improve cross-validation while failing out-of-sample if the test period is a different regime. This pushed the workflow toward simpler, more robust feature sets and tighter ablation checks.

### Phase F — Ensembling and strategy benchmarks

Goal: test whether modest complexity (blends) can add robustness without overfitting.

- `src/11_ensemble_strategy.py`: ensemble experiments.
- `src/15_advanced_score_push.py` + notes in `GEMINI.md`: benchmarked multiple advanced strategies; several failed outright, while a LightGBM+CatBoost blend was recorded as the most promising of that batch (notably on the long horizon).

Takeaway: the bar for “complexity that helps” is high; improvements, when found, tend to be horizon-specific and easy to overfit without careful validation.

## 4) Key Takeaways (So Far)

- In low SNR settings, robust objectives + careful calibration can matter more than model class.
- Weighting needs moderation: respect `weight` without letting it collapse generalization.
- Horizon behavior differs: “one size fits all” tends to hide failures.
- CV is necessary but not sufficient: use strict leakage controls and avoid over-trusting small deltas.

## 5) How to Use This Repo as a Process Artifact

- Use `src/01_data_exploration.py` and baseline scripts to show the initial framing and scaffolding.
- Use `src/06_strategy_refinement_v2.py` / `src/08_advanced_tuning.py` to illustrate the “robust loss + calibration” turn.
- Use `src/09_feature_engineering.py` to show ablation practice (and the regime-shift lesson).
- Use `src/15_advanced_score_push.py` + `GEMINI.md` to show structured benchmarking of ideas.
