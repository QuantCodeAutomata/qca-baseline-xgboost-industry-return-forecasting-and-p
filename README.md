# XGBoost Industry-Return Forecasting with Entropy Features

## Overview

This project reproduces and extends a pooled-panel dual-model XGBoost framework for
forecasting 5-, 21-, and 63-trading-day U.S. industry-group returns and constructing
long-only non-overlapping maximum-profit and minimum-risk portfolios.

Three experiments are implemented:

| Experiment | Description |
|---|---|
| **exp_1** | Baseline — technical + cross-sectional features only |
| **exp_2** | Shannon entropy — adds 20-day rolling Shannon entropy (10 bins) |
| **exp_3** | Fuzzy entropy — adds fuzzy sample entropy (m=2, Chebyshev, exponential kernel) |

Each experiment produces 6 strategies: 3 horizons (H = 5, 21, 63 days) × 2 portfolio types
(maximum-profit, minimum-risk).

## Repository Structure

```
├── data/
│   └── load_data.py          # Massive API data loader for 25 industry ETFs
├── src/
│   ├── config.py             # All configuration constants and hyperparameters
│   ├── features.py           # Causal technical + cross-sectional feature engineering
│   ├── entropy.py            # Shannon and fuzzy entropy implementations
│   ├── backtest.py           # Rolling XGBoost backtest with non-overlapping rebalancing
│   ├── portfolio.py          # Max-profit and min-risk portfolio construction
│   └── metrics.py            # Performance metric computation
├── exp/
│   ├── run_experiment.py     # Unified CLI runner (--exp {all,1,2,3}, --horizon, --dry-run)
│   ├── exp_1_baseline.py     # Baseline experiment entry point
│   ├── exp_2_shannon.py      # Shannon entropy experiment entry point
│   └── exp_3_fuzzy.py        # Fuzzy entropy experiment entry point
├── tests/
│   ├── test_features.py      # Feature engineering unit tests
│   ├── test_entropy.py       # Entropy computation unit tests
│   ├── test_portfolio.py     # Portfolio construction unit tests
│   └── test_metrics.py       # Performance metrics unit tests
├── results/
│   └── RESULTS.md            # Full experiment results and analysis
└── AGENTS.md                 # Agent memory and project context
```

## Data

Daily closing levels for 25 U.S. industry ETFs (GICS-aligned proxies for the paper's MSCI
industry-group series) are fetched via the Massive REST API. Data spans 1992–2024 where
available (ETF availability limits some series to post-1993/1996).

**Important:** Raw market data files (`data/*.parquet`) are excluded from version control
via `.gitignore`. Run `data/load_data.py` with a valid `MASSIVE_API_KEY` environment
variable to regenerate.

## Installation

```bash
pip install pandas numpy xgboost scikit-learn scipy matplotlib quantstats massive
```

## Usage

```bash
# Set API key
export MASSIVE_API_KEY="your_key_here"

# Download data
python data/load_data.py

# Dry-run exp_1 (H=5 only, fast smoke test)
python exp/run_experiment.py --exp 1 --horizon 5 --dry-run

# Run all experiments, all horizons (full backtest ~20 min)
python exp/run_experiment.py --exp all

# Run specific experiment
python exp/run_experiment.py --exp 2 --horizon 21
```

## Tests

```bash
pytest tests/ -v
```

All 31 tests should pass with no live data required.

## Key Configuration (src/config.py)

| Parameter | Value | Note |
|---|---|---|
| Rolling window | 504 days | ~2 years; paper is silent on exact length |
| Train/val split | 80/20 | Assumption |
| n_estimators | 50 | Paper does not disclose |
| Shannon window | 20 days, 10 bins | Paper-specified |
| Fuzzy params | L=30, m=2, r=0.2σ, n=2 | Paper underspecified — assumptions |
| Empty eligible set | Equal-weight fallback | Paper is silent |

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full metrics tables, plots, and
comparison against the paper's published benchmark values.

### Quick Summary (out-of-sample 2012–2024, 13 years)

| Best Strategy | Experiment | Sharpe | Avg Ann. Return |
|---|---|---:|---:|
| H=5 Max-Profit | Fuzzy (exp_3) | **0.913** | 14.25% |
| H=63 Min-Risk | Fuzzy (exp_3) | **1.094** | 11.07% |
| H=21 Max-Profit | Shannon (exp_2) | 0.814 | 12.71% |
| H=5 Min-Risk | Baseline (exp_1) | 0.957 | 11.49% |

## Replication Notes

- Paper uses proprietary MSCI Institute series; this project uses ETF proxies.
- OOS period is 13 years (2012–2024) vs paper's 32 years due to ETF availability.
- All unresolved methodological choices are exposed as named configuration variables and
  logged in `results/RESULTS.md`.

## License

Research use only. This is an academic replication project.
