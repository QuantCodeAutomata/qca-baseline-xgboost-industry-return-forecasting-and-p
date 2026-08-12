"""
Central configuration for all experiments.

All under-specified parameters from the paper are surfaced here as explicit
configuration variables and must be logged with every experiment run.
"""

from __future__ import annotations

# ─────────────────────────── Universe ────────────────────────────────────────
INDUSTRY_NAMES: list[str] = [
    "REITs",
    "Utilities",
    "Food, Beverage and Tobacco",
    "Semiconductors",
    "Consumer Staples",
    "Automobiles",
    "Telecommunication",
    "Household and Personal Products",
    "Banks",
    "Energy",
    "Technology Hardware and Equipment",
    "Consumer Services",
    "Software and Services",
    "Pharmaceuticals, Biotechnology and Life Sciences",
    "Consumer Durables and Apparel",
    "Materials",
    "Health Care Equipment and Services",
    "Media and Entertainment",
    "Commercial and Professional Services",
    "Consumer Discretionary Distribution and Retail",
    "Financial Services",
    "Transportation",
    "Capital Goods",
    "Insurance",
    "Real Estate Management and Development",
]

# Ticker proxies mapped to industry names (ETF-based proxies for MSCI GICS industry groups)
# ASSUMPTION: exact MSCI series unavailable; using U.S. sector/industry ETF proxies.
# Replication deviation documented here.
INDUSTRY_TICKERS: dict[str, str] = {
    "REITs": "VNQ",
    "Utilities": "XLU",
    "Food, Beverage and Tobacco": "PBJ",
    "Semiconductors": "SOXX",
    "Consumer Staples": "XLP",
    "Automobiles": "CARZ",
    "Telecommunication": "IYZ",
    "Household and Personal Products": "EZA",       # proxy (limited coverage)
    "Banks": "KBE",
    "Energy": "XLE",
    "Technology Hardware and Equipment": "IGV",
    "Consumer Services": "IYC",
    "Software and Services": "IGV",
    "Pharmaceuticals, Biotechnology and Life Sciences": "XBI",
    "Consumer Durables and Apparel": "XRT",
    "Materials": "XLB",
    "Health Care Equipment and Services": "XLV",
    "Media and Entertainment": "PBS",
    "Commercial and Professional Services": "IYJ",
    "Consumer Discretionary Distribution and Retail": "XRT",
    "Financial Services": "XLF",
    "Transportation": "IYT",
    "Capital Goods": "XLI",
    "Insurance": "KIE",
    "Real Estate Management and Development": "XLRE",
}

# Use Yahoo Finance proxies for long history from 1992 (Massive ETF data starts later)
# We map to broad SPDR/iShares sector ETFs or indices with sufficient history
YAHOO_TICKERS: dict[str, str] = {
    "REITs": "IYR",
    "Utilities": "XLU",
    "Food, Beverage and Tobacco": "XLP",
    "Semiconductors": "SOXX",
    "Consumer Staples": "XLP",
    "Automobiles": "F",
    "Telecommunication": "IYZ",
    "Household and Personal Products": "XLP",
    "Banks": "XLF",
    "Energy": "XLE",
    "Technology Hardware and Equipment": "XLK",
    "Consumer Services": "XLY",
    "Software and Services": "XLK",
    "Pharmaceuticals, Biotechnology and Life Sciences": "XLV",
    "Consumer Durables and Apparel": "XLY",
    "Materials": "XLB",
    "Health Care Equipment and Services": "XLV",
    "Media and Entertainment": "XLC",
    "Commercial and Professional Services": "XLI",
    "Consumer Discretionary Distribution and Retail": "XLY",
    "Financial Services": "XLF",
    "Transportation": "XLI",
    "Capital Goods": "XLI",
    "Insurance": "XLF",
    "Real Estate Management and Development": "IYR",
}

# ─────────────────────────── Date range ──────────────────────────────────────
START_DATE: str = "1992-01-02"
END_DATE: str = "2024-12-31"
EXPECTED_YEARS: int = 32  # per paper: positive-year share granularity 1/32

# ─────────────────────────── Horizons ────────────────────────────────────────
HORIZONS: list[int] = [5, 21, 63]  # trading days

# ─────────────────────────── Feature parameters ──────────────────────────────
MOMENTUM_WINDOWS: list[int] = [1, 5, 21, 63]
VOL_WINDOWS: list[int] = [21, 63]
MA_WINDOWS: list[int] = [10, 21, 50]
RSI_WINDOW: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BOLLINGER_WINDOW: int = 20
STOCH_WINDOW: int = 14
ROLLING_RETURN_MIN_MAX_WINDOW: int = 21

# ─────────────────────────── Rolling backtest ────────────────────────────────
# ASSUMPTION: paper says window is proportional to horizon but does not give exact values.
ROLLING_WINDOW_MULTIPLIER: int = 252 * 2  # base lookback in trading days (default)
ROLLING_WINDOWS: dict[int, int] = {
    5: 252 * 2,    # ~2 years for weekly  (ASSUMPTION: paper silent; reduced for compute)
    21: 252 * 2,   # ~2 years for monthly
    63: 252 * 2,   # ~2 years for quarterly
}
# Fraction of rolling window used for training (rest for validation)
TRAIN_FRAC: float = 0.80
MIN_TRAIN_OBS: int = 500  # minimum observations per industry per window

# ─────────────────────────── XGBoost hyperparameters ─────────────────────────
# ASSUMPTION: paper does not give hyperparameters; all exposed here.
XGB_CLASSIFIER_PARAMS: dict = {
    "objective": "binary:logistic",
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "gamma": 0.0,
    "random_state": 42,
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs": -1,
}

XGB_REGRESSOR_PARAMS: dict = {
    "objective": "reg:squarederror",
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "gamma": 0.0,
    "random_state": 42,
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs": -1,
}

# ─────────────────────────── Ensemble weight mapping ─────────────────────────
# ASSUMPTION: paper does not give the exact validation-metric → weight mapping.
# We use AUC-based classification weight and MSE-based regression weight.
# w_cls = AUC_val / (AUC_val + 1/(MSE_val + 1e-8))
# Configurable: "auc_mse" (default) or "equal"
ENSEMBLE_WEIGHT_RULE: str = "auc_mse"

# ─────────────────────────── Covariance estimation ───────────────────────────
# ASSUMPTION: paper does not specify covariance lookback.
COV_LOOKBACK: dict[int, int] = {
    5: 252 * 3,   # 3 years of daily returns for weekly
    21: 252 * 3,  # 3 years for monthly
    63: 252 * 3,  # 3 years for quarterly
}
# Singular covariance fallback: "pinv" | "diag_load" | "shrinkage"
COV_FALLBACK: str = "shrinkage"
COV_SHRINKAGE_ALPHA: float = 0.1  # diagonal loading coefficient when needed

# ─────────────────────────── Portfolio ───────────────────────────────────────
# Fallback when eligible set is empty
# ASSUMPTION: paper is silent. Using equal-weight over full universe.
EMPTY_ELIGIBLE_FALLBACK: str = "equal_weight"  # "all_cash" | "equal_weight" | "prev_holdings"

# Forward return clip bounds
CLIP_RETURN_LB: float = -0.80
CLIP_RETURN_UB: float = 0.80

# ─────────────────────────── Shannon entropy ─────────────────────────────────
SHANNON_WINDOW: int = 20
SHANNON_BINS: int = 10

# ─────────────────────────── Fuzzy entropy ───────────────────────────────────
# ASSUMPTION: paper does not fully specify the following:
FUZZY_WINDOW: int = 63         # rolling window length (trading days)
FUZZY_M: int = 2               # embedding dimension
FUZZY_TAU: int = 1             # sampling step
FUZZY_R_RULE: str = "std_frac" # "std_frac" → r = r_frac * std(window); "fixed"
FUZZY_R_FRAC: float = 0.2      # fraction of std for scale parameter r
FUZZY_N: float = 2.0           # exponent in exp(-(d/r)^n)
FUZZY_EXCLUDE_SELF: bool = True  # exclude self-matches (standard sample entropy style)

# ─────────────────────────── Annual aggregation ──────────────────────────────
# ASSUMPTION: derive annual returns by compounding non-overlapping holding-period returns
# that fall within each calendar year. Partial first/last holdings attributed to year of
# decision date.
ANNUAL_AGGR_METHOD: str = "calendar_year"  # "calendar_year" | "rolling_12m"

# ─────────────────────────── Paths ───────────────────────────────────────────
DATA_DIR: str = "data"
RESULTS_DIR: str = "results"
