# Kaggle Hedge Fund Time Series Forecasting — Experiment Log

This repository is an experiment-driven walk-through of modeling for the Kaggle competition “Hedge fund - Time series forecasting.” The goal is to predict future returns (multiple horizons) and evaluate with a weighted “skill score” based on squared error.

This repo is intentionally organized to show the process: baselines → ablations → postmortems → refined ideas. If you’re looking for the full narrative and lessons learned, start with `REPORT.md`.

## What’s Included

- `src/`: numbered scripts that reflect the experiment timeline (EDA, baselines, tuning, feature ablations, ensembling).
- `outputs/`: generated submissions and logs.
- `Competition_Rules/`: copied competition overview, dataset description, and official rules.
- `GEMINI.md`: lightweight project notes / history.

## Competition Constraints (Quick Reference)

- **Metric**: weighted, clipped RMSE-style “skill score” (see `Competition_Rules/overview.md`).
- **Leakage control**: for `ts_index = t`, predict using only data up to `t` and process sequentially (see `Competition_Rules/overview.md`).
- **Submission format**: CSV with `id,prediction` generated from `test.parquet` (see `Competition_Rules/overview.md`).

## Data (Quick Reference)

- Files: `train.parquet` and `test.parquet` (see `Competition_Rules/data_descripition.md`).
- Columns include identifiers (`code`, `sub_code`, `sub_category`), time (`ts_index`), `horizon` (1/3/10/25), `weight`, and 86 anonymized features.
- `weight` is part of the metric; it’s treated as a training weight, not a predictive feature.

## Experiment Workflow

- Use time-series splits (no shuffling) and compare changes on the competition metric.
- Prefer small, one-change-at-a-time ablations (loss function, weighting, feature sets, calibration).
- Treat prediction calibration (e.g., shrinkage) as a first-class tuning knob because the metric can heavily penalize variance.

## Experiment Map (Scripts)

| Stage | Script(s) | What it explores |
|------:|-----------|------------------|
| EDA | `src/01_data_exploration.py` | Data sanity checks, distributions, horizon behavior |
| Baselines | `src/02_lgb_baseline.py`, `src/03_lgb_baseline_v2.py`, `src/04_lgb_baseline_v3.py` | Basic LightGBM setups and validation scaffolding |
| Weighting | `src/05_high_weight_focus.py` | Weighting schemes / tradeoffs in generalization |
| Robust loss + calibration | `src/06_strategy_refinement_v2.py`, `src/08_advanced_tuning.py` | Robust objectives (e.g., Huber) and shrinkage tuning |
| Feature ablations | `src/09_feature_engineering.py`, `src/10_hybrid_submission.py` | Aggregate/macro features vs. raw anonymized features |
| Ensembling | `src/11_ensemble_strategy.py` | Blends / alternative learners |
| Iterations | `src/12_final_submission.py`, `src/13_improved_submission.py`, `src/14_competition_winner.py` | Iterative refinements and consolidation |
| Benchmarking | `src/15_advanced_score_push.py` | Side-by-side strategy benchmarks (see `GEMINI.md`) |

## Usage

### Prerequisites
- Python 3.9+
- Enough RAM to load parquet data for training/validation (size depends on your workflow).

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Download Data
```bash
kaggle competitions download -c ts-forecasting
```

### Run Key Experiments
```bash
python src/01_data_exploration.py
python src/02_lgb_baseline.py
python src/06_strategy_refinement_v2.py
python src/09_feature_engineering.py
python src/15_advanced_score_push.py
```

### Generate a Submission File
Several scripts emit a `outputs/submission_*.csv` artifact. Pick the script/config you want to evaluate and run it (for example, `src/10_hybrid_submission.py`).

## Read the Narrative
See `REPORT.md` for a “hypothesis → experiment → takeaway” timeline and the key postmortems that drove later iterations.
