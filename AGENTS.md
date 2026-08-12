# Project Agent Memory

## Project Overview
Replication/extension of a finance paper using MSCI GICS industry-group returns.
Predicts forward returns (5/21/63-day horizons) using XGBoost classifiers/regressors.
Universe: 25 industries (see `src/config.py`), 1992-2024.

## Repository Layout
```
src/config.py    — central config: all assumptions surfaced as named constants
src/entropy.py   — Shannon + Fuzzy entropy feature computation (causal, rolling)
src/features.py  — individual feature functions + compute_baseline_features
src/backtest.py  — BacktestConfig dataclass, RebalanceRecord, run_backtest
src/portfolio.py — max_profit_portfolio, min_risk_portfolio, estimate_covariance
src/metrics.py   — compute_annual_returns, compute_cumulative_wealth,
                   compute_performance_metrics, compute_average_weights, save_results
data/load_data.py — download_industry_prices, compute_daily_returns, compute_forward_returns
exp/run_experiment.py   — unified runner: run_full_experiment(exp_id, horizons, dry_run)
exp/exp_1_baseline.py   — calls run_full_experiment('exp_1', ...) + paper comparison
exp/exp_2_shannon.py    — calls run_full_experiment('exp_2', ...) + paper comparison
exp/exp_3_fuzzy.py      — calls run_full_experiment('exp_3', ...) + paper comparison
tests/test_features.py  — pytest tests for src/features.py (9 tests)
tests/test_entropy.py   — pytest tests for src/entropy.py scalar functions (8 tests)
tests/test_portfolio.py — pytest tests for src/portfolio.py (8 tests)
tests/test_metrics.py   — pytest tests for src/metrics.py (6 tests)
results/         — experiment outputs (PNG plots, CSV metrics)
```

## exp/run_experiment.py — Design Decisions

### run_full_experiment(exp_id, horizons, dry_run=False)
Returns `dict[strategy_name → metrics_dict]`.
Strategy names: `"H=5 Max-Profit"`, `"H=21 Max-Profit"`, ..., `"H=63 Min-Risk"`.
Metrics dict keys: `total_return`, `avg_annual_return`, `std_annual_return`,
`sharpe_ratio`, `n_years`, `pct_positive_years`.

### Dry-run mode
- start_date overridden to "2010-01-01"
- effective horizons: [5, 21] only
- rolling_window overridden to 252 (1 year)

### Plot output (matplotlib only)
- Cumulative wealth: `results/{exp_id}_cumulative_wealth.png`
- Annual returns bar: `results/{exp_id}_annual_returns.png`
- Average weights bar: `results/{exp_id}_weights.png`
- dpi=150, distinct colors per horizon (steelblue/darkorange/forestgreen),
  linestyles '-' for max_profit, '--' for min_risk.

### BacktestConfig constructor arguments
All sourced from src/config.py:
`horizon, rolling_window, train_frac, min_train_obs, portfolio_fn,
xgb_classifier_params, xgb_regressor_params, ensemble_weight_rule,
cov_lookback, cov_fallback, cov_shrinkage_alpha, empty_eligible_fallback,
clip_return_lb, clip_return_ub`

---

## tests/ — Design Decisions

### RebalanceRecord stub (test_metrics.py)
Field names match AGENTS.md src/backtest.py documentation:
- `decision_date`: pd.Timestamp
- `realized_return`: float (holding-period portfolio return)
- `weights`: dict[str, float]
- `fallback_used`: bool

### Metric units (test_metrics.py)
AGENTS.md confirms metrics are stored as **percent** (not decimal):
- `avg_annual_return`: e.g., 18.9 (not 0.189)
- `std_annual_return`: e.g., 21.5
- `total_return`: e.g., 15220 (not 152.20)
- `sharpe_ratio`: dimensionless (≈ avg_annual_pct / std_annual_pct)
- `pct_positive_years`: 0.60 (decimal fraction, not percent)
  → Tests accept both decimal and percent form for robustness.

### test_features.py — helper conventions
- `_return_series(n, seed)`: pd.Series of daily returns via rng.normal(0.0005, 0.015)
- All feature functions assumed to take a daily returns pd.Series (not prices).
- Functions that need prices reconstruct internally via cumprod(1+returns).

### test_entropy.py — boundary constants
- Uses `SHANNON_WINDOW=20` and `SHANNON_BINS=10` from src/config.py directly.
- `fuzzy_entropy_scalar(data, r=...)` accepts explicit r for test_fuzzy_larger_r_lower_entropy.

### test_portfolio.py — helper conventions
- `_scores(n, seed)`: uniform random pd.Series in [−0.04, 0.04]
- `_returns(n_obs, n_ind, seed)`: multivariate_normal with diagonal cov
- Singular covariance test: duplicates all n_ind columns → rank-1 matrix

---

## src/entropy.py — Key Design Decisions

### Shannon Entropy
- Paper-specified: 20-day trailing window, 10 equal-width bins, natural log.
- Bin edges: `np.linspace(min, max, 11)` — bins span exactly [min, max].
- Degenerate window (min == max): returns 0.0; logged per industry.
- Column names returned by `compute_shannon_entropy`: `'shannon_ent_<industry>'`.

### Fuzzy Entropy
- Standard FuzzyEn per Chen et al. (2007).
- Paper omits r, n, window length, phi averaging formula — all assumptions in `assumption_log()`.
- phi_m = sum_{i≠j} mu_ij / (N*(N-1))  [FUZZY_EXCLUDE_SELF=True]
- r = FUZZY_R_FRAC * std(window, ddof=1) when FUZZY_R_RULE='std_frac'
- r==0 or phi_m==0 or ratio<=0 → NaN (not 0).
- Core implementation: numpy broadcasting over (N,N,m) tensor for pairwise Chebyshev distance; no Python pair loops.
- `np.lib.stride_tricks.sliding_window_view` used for embedding matrix (numpy ≥ 1.20 required; project has 1.26.4).
- Column names returned by `compute_fuzzy_entropy`: `'fuzzy_ent_<industry>'`.

### add_entropy_features
- Takes long-format panel + wide returns; merges on (date, industry).
- Renames wide entropy columns to industry names before melt — avoids fragile prefix-stripping.
- Shannon uses `config.SHANNON_WINDOW=20` and `config.SHANNON_BINS=10`.
- Fuzzy uses all `config.FUZZY_*` defaults.

## Config Constants (relevant to entropy)
| Constant | Value | Meaning |
|---|---|---|
| SHANNON_WINDOW | 20 | days in Shannon rolling window |
| SHANNON_BINS | 10 | histogram bins |
| FUZZY_WINDOW | 63 | days in fuzzy rolling window |
| FUZZY_M | 2 | embedding dimension |
| FUZZY_TAU | 1 | delay (fixed, no subsampling) |
| FUZZY_R_RULE | 'std_frac' | r derived from window std |
| FUZZY_R_FRAC | 0.2 | r = 0.2 * std(window) |
| FUZZY_N | 2.0 | kernel exponent |
| FUZZY_EXCLUDE_SELF | True | exclude i==j pairs |

## Causality Rule
Entropy at date t (integer index t) uses `series[t-window+1 : t+1]`.
First `window-1` rows are NaN.

---

## data/load_data.py — Key Design Decisions

### INDUSTRY_TICKER_MAP
- 25 **distinct** ETF tickers (one per MSCI GICS industry group).
- Authoritative mapping for data loading; differs from `config.INDUSTRY_TICKERS` / `config.YAHOO_TICKERS` (those have duplicates).
- `IDHD` (Household and Personal Products) is delisted → all-NaN column; this is a documented replication deviation.

### download_industry_prices(start, end, cache_path)
- Uses `yfinance.download(..., auto_adjust=True)` → MultiIndex columns `('Close', ticker)`.
- `yfinance` end is **exclusive**; function adds 1 calendar day internally so `end` date is included.
- Returns wide DataFrame: DatetimeIndex named `"date"`, 25 industry-name columns, float64.
- **No forward-fill** for short-history ETFs; NaN is preserved and logged.
- Saves/loads Parquet cache (requires `pyarrow`).

### compute_daily_returns(prices)
- Simple `pct_change()`, float64. First row NaN. NaN prices propagate to NaN returns.

### compute_forward_returns(prices, horizon, clip_lb=-0.80, clip_ub=0.80)
- `shift(-horizon)` for forward price. Clips to `[clip_lb, clip_ub]`.
- Returns `(fwd_returns, binary_targets)`. Last `horizon` rows NaN. Binary target NaN where fwd return is NaN.

---

## src/features.py — Key Design Decisions

All features implemented from scratch (no `ta` library). All outputs float64, causal.

### compute_baseline_features(prices, returns) → long-format panel
Output schema (23 columns):
`date, industry, mom_1, mom_5, mom_21, mom_63, vol_21, vol_63, roll_max_21, roll_min_21, ma_ratio_10, ma_ratio_21, ma_ratio_50, rsi_14, macd, macd_signal, bb_zscore, bb_width, stoch_k_14, cs_zscore_ret, cs_rank_ret, cs_zscore_mom21, cs_rank_mom21`

Rows = len(prices) × 25 (one row per date–industry pair).

### Per-industry time-series features
| Feature | Key params | Causality mechanism |
|---|---|---|
| momentum_features | windows=[1,5,21,63] | rolling sum, min_periods=W |
| volatility_features | windows=[21,63] | rolling std, min_periods=W |
| rolling_max_min_features | window=21 | rolling max/min, min_periods=W |
| ma_ratio_features | windows=[10,21,50] | `prices.shift(1).rolling(m).mean()` (t-1 lag) |
| rsi_feature | window=14 | `ewm(com=window-1, adjust=False)` = Wilder's smoothing |
| macd_features | fast=12, slow=26, signal=9 | EWM span, adjust=False |
| bollinger_features | window=20 | includes P_t in rolling window (standard) |
| stochastic_k_feature | window=14 | uses close as high/low proxy; `returns` arg unused |

### Cross-sectional features (cross_sectional_features)
- Computed row-wise across all 25 industries per date.
- z-score: `(x - mean) / std`; when std=0, z-score=0 (logged).
- Fractional rank: `(rank - 1) / (n_valid - 1)` → [0, 1]; average ties; NaN when n_valid ≤ 1.
- Uses `pd.DataFrame.stack(future_stack=True)` (pandas 2.1+ API, avoids FutureWarning).
- Returns long-format DataFrame with [date, industry, cs_zscore_ret, cs_rank_ret, cs_zscore_mom21, cs_rank_mom21].

### Environment notes
- pandas 2.2.2, numpy 1.26.4, yfinance 1.5.2, pyarrow required for Parquet.
- yfinance 1.5.x multi-ticker download returns `raw["Close"]` with ticker columns named by `Ticker` index level; reset `columns.name = None` after extraction.

---

## src/portfolio.py — Key Design Decisions

### max_profit_portfolio
- q_s = max(S_s, 0) for s in eligible; w_s = q_s / sum(q_j).
- Fallback when eligible empty OR sum(q)==0: 'equal_weight' (1/N over all_industries), 'all_cash' (zero), 'prev_holdings' (raises – caller must handle).
- Returns pd.Series indexed by all_industries; non-eligible industries always have weight=0.

### min_risk_portfolio
- Unconstrained MV: w̃ = Σ⁻¹1 / (1ᵀΣ⁻¹1), then truncate negatives and renormalise.
- Singularity handled by `_safe_inv`: direct `scipy.linalg.inv`, then `cov_fallback` ('shrinkage'/'diag_load'/'pinv'), then `scipy.linalg.pinv` as last resort.
- Fallback applied when eligible empty OR renorm denominator = 0.

### estimate_covariance
- `returns_history[eligible].tail(lookback).cov()` (pairwise-complete NaN handling).
- PD check via `np.linalg.eigvalsh`; if fails: shrinkage (Σ_reg = (1-α)Σ + α·diag(Σ)) or diag_load.
- Missing columns filled with 0 rather than raising KeyError.

---

## src/backtest.py — Key Design Decisions

### Decision date enumeration
- `all_panel_dates = sorted(panel['date'].unique())`
- First valid index = `rolling_window`; step = `horizon` (non-overlapping holdings).
- A date is valid if it exists in `prices.index` and `prices.index[idx + horizon]` exists.

### Estimation window
- Strictly before decision date t: `all_panel_dates[t_idx - rolling_window : t_idx]`.
- Split by unique dates: first `int(n_window * train_frac)` dates = training.
- `n_train_obs` = total (date × industry) rows in training block; skipped if < `min_train_obs`.

### Fast panel indexing
- `panel.groupby('date').groups` → dict `{date: Index}` converted to numpy arrays.
- Row indices concatenated per date range → `panel.iloc[row_idx]` for O(n_window * 25) access.

### Tau selection
- Grid: `np.arange(0.30, 0.76, 0.05)` → 10 values [0.30 … 0.75].
- Initialise `best_f1 = 0.0` (not -1.0) so best_tau stays at 0.50 if ALL taus yield F1=0.

### Ensemble weights (auc_mse rule)
- `w_cls = AUC / (AUC + 1/(MSE + 1e-8))`; `w_reg = 1 - w_cls`.
- Special case: if `val_mse < 1e-8` → `w_cls = 1 - 1e-8`, `w_reg = 1e-8` (overfitting guard).
- AUC fallback: 0.5 when validation set has only one class.

### Z-standardisation
- `mu_val`, `sigma_val` = mean/std of regressor predictions on ALL validation rows.
- If `sigma_val == 0`: Z = 0 everywhere; `fallback_used = True`.

### Combined score and eligibility
- `S_s = w_cls * p_s + w_reg * Z_s`
- Eligible: `{s : p_s > tau}` (uses raw classifier probability, not combined score).

### Realized return
- Computed from `prices.iloc[t_idx + H] / prices.iloc[t_idx] - 1` (NOT clipped; raw prices).
- NaN if decision_date not in prices or t+H beyond prices end.

### fallback_used flag
- True if: eligible_set is empty, OR (max_profit) all positive-clipped scores sum to 0, OR sigma_val=0.

---

## src/metrics.py — Key Design Decisions

### compute_annual_returns
- Method='calendar_year': compound all HP returns with decision_date.year == Y.
- `annual_return_Y = prod(1 + R_t for t in year Y) - 1`.

### compute_performance_metrics
- Risk-free rate = 0 (ASSUMPTION; paper silent).
- Sharpe = avg_annual / std_annual (both in %, ratio is dimensionless).
- `std_annual` uses ddof=1; returns NaN Sharpe when n_years ≤ 1.

### save_results
- Writes 4 files: `{prefix}metrics.csv`, `{prefix}annual_returns.csv`, `{prefix}weights.csv`, `{prefix}hp_returns.csv`.
- Creates `results_dir` if absent.
