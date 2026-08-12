"""
Tests for src/features.py individual feature engineering functions.

Verifies causality (no lookahead), numeric bounds, NaN propagation,
and mathematical correctness for each feature group.
"""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, "/workspace/project")

from src.features import (
    bollinger_features,
    ma_ratio_features,
    macd_features,
    momentum_features,
    rsi_feature,
    rolling_max_min_features,
    stochastic_k_feature,
    volatility_features,
)

# ── Shared test fixtures ───────────────────────────────────────────────────────

def _return_series(n: int = 200, seed: int = 42) -> pd.Series:
    """Random-walk daily return series with realistic mean and volatility."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, size=n)
    return pd.Series(
        returns,
        index=pd.date_range("2000-01-03", periods=n, freq="B"),
        name="returns",
    )


def _price_from_returns(returns: pd.Series, start: float = 100.0) -> pd.Series:
    """Reconstruct a price level series from a returns series."""
    return start * (1 + returns).cumprod()


# ── Momentum ───────────────────────────────────────────────────────────────────

def test_momentum_no_lookahead():
    """mom_1 at index t must equal r_t (the current return), not r_{t+1}."""
    returns = _return_series(n=50)
    feat = momentum_features(returns)

    col = "mom_1"
    assert col in feat.columns, f"Column '{col}' missing from momentum_features output"

    valid = feat[col].dropna()
    assert len(valid) > 0, "mom_1 has no valid values"

    for idx in valid.index:
        assert np.isclose(feat.loc[idx, col], returns.loc[idx], rtol=1e-9), (
            f"Lookahead at {idx}: mom_1={feat.loc[idx, col]:.8f}, "
            f"r_t={returns.loc[idx]:.8f}"
        )


# ── Volatility ────────────────────────────────────────────────────────────────

def test_volatility_positive():
    """Rolling volatility estimates must be non-negative everywhere defined."""
    returns = _return_series(n=150)
    feat = volatility_features(returns)

    assert len(feat.columns) > 0, "volatility_features returned empty DataFrame"
    for col in feat.columns:
        valid = feat[col].dropna()
        assert (valid >= 0).all(), (
            f"Negative values in {col}: min={valid.min():.6f}"
        )


# ── MA ratio ──────────────────────────────────────────────────────────────────

def test_ma_ratio_causal():
    """MA ratio at t must not incorporate any price information beyond t.

    Construct a price series that is flat (=100) for 60 steps, then spikes
    to 200 at step 61.  At the last stable index (t=59 in returns space, one
    step before the spike return), all MA ratios should be ≈ 1.0 because
    every price through that point is identical.
    """
    n_flat = 60
    prices = pd.Series(
        [100.0] * n_flat + [200.0],
        index=pd.date_range("2000-01-03", periods=n_flat + 1, freq="B"),
    )
    returns = prices.pct_change().dropna()  # length = n_flat
    feat = ma_ratio_features(returns)

    assert len(feat.columns) > 0, "ma_ratio_features returned empty DataFrame"

    # At the second-to-last return index (before spike enters the window),
    # all prices seen up to that point are 100 → ratio must be ≈ 1.0.
    last_stable_idx = returns.index[-2]
    for col in feat.columns:
        val = feat.loc[last_stable_idx, col]
        if not np.isnan(val):
            assert abs(val - 1.0) < 0.05, (
                f"MA ratio '{col}' at {last_stable_idx} = {val:.4f}; "
                "expected ≈ 1.0 (possible lookahead or wrong computation)"
            )


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_bounds():
    """RSI must lie in [0, 100] for every valid (non-NaN) row."""
    returns = _return_series(n=150)
    feat = rsi_feature(returns)

    col = "rsi_14"
    assert col in feat.columns, f"Column '{col}' missing from rsi_feature output"

    valid = feat[col].dropna()
    assert len(valid) > 0, "rsi_14 has no valid values"
    assert (valid >= 0).all() and (valid <= 100).all(), (
        f"RSI out of [0, 100]: min={valid.min():.4f}, max={valid.max():.4f}"
    )


# ── MACD ──────────────────────────────────────────────────────────────────────

def test_macd_sign():
    """MACD line must be positive when the price trend is consistently upward.

    For a linearly increasing price series fast EMA converges faster than slow
    EMA, so (fast EMA − slow EMA) > 0 at the tail of the series.
    """
    returns = _return_series(n=150)
    feat = macd_features(returns)

    macd_col = "macd_line"
    assert macd_col in feat.columns, f"Column '{macd_col}' missing from macd_features output"

    # Construct a strict upward-trending price series with enough history.
    n = 80
    prices_up = pd.Series(
        np.linspace(100.0, 180.0, n),
        index=pd.date_range("2000-01-03", periods=n, freq="B"),
    )
    rets_up = prices_up.pct_change().dropna()
    feat_up = macd_features(rets_up)

    valid_macd = feat_up[macd_col].dropna()
    assert len(valid_macd) > 0, "macd_line has no valid values for uptrend series"

    last_macd = valid_macd.iloc[-1]
    assert last_macd > 0, (
        f"Expected positive MACD for an uptrend series, got {last_macd:.6f}"
    )


# ── Bollinger ─────────────────────────────────────────────────────────────────

def test_bollinger_zscore():
    """Bollinger z-score must be 0 when price equals the rolling mean.

    A series of zero returns reconstructs a flat price level.  The rolling
    mean equals the constant price, so the z-score is 0 (or NaN when std=0
    and the implementation avoids division by zero differently).
    """
    n = 60
    prices_const = pd.Series(
        [100.0] * n,
        index=pd.date_range("2000-01-03", periods=n, freq="B"),
    )
    rets_const = prices_const.pct_change().fillna(0.0)
    feat = bollinger_features(rets_const)

    zscore_col = "bollinger_zscore"
    assert zscore_col in feat.columns, (
        f"Column '{zscore_col}' missing from bollinger_features output"
    )

    valid = feat[zscore_col].dropna()
    # Implementations may produce NaN for zero-std windows; that is acceptable.
    # Non-NaN values must be exactly zero (price == MA throughout).
    if len(valid) > 0:
        assert (np.abs(valid) < 1e-8).all(), (
            f"Non-zero Bollinger z-score for flat price series: {valid.values}"
        )


# ── Stochastic %K ─────────────────────────────────────────────────────────────

def test_stoch_k_bounds():
    """%K must lie in [0, 100] for every valid (non-NaN) row."""
    returns = _return_series(n=120)
    feat = stochastic_k_feature(returns)

    col = "stoch_k"
    assert col in feat.columns, f"Column '{col}' missing from stochastic_k_feature output"

    valid = feat[col].dropna()
    assert len(valid) > 0, "stoch_k has no valid values"
    assert (valid >= 0).all() and (valid <= 100).all(), (
        f"%K out of [0, 100]: min={valid.min():.4f}, max={valid.max():.4f}"
    )


# ── NaN propagation ────────────────────────────────────────────────────────────

def test_nan_propagation():
    """Long-window features must be NaN for the first (window−1) rows.

    vol_63 uses a 63-day rolling window; the first 62 rows must be NaN
    because fewer than 63 observations are available.
    """
    returns = _return_series(n=250)
    feat = volatility_features(returns)

    col = "vol_63"
    assert col in feat.columns, f"Column '{col}' missing from volatility_features output"

    warmup = 62  # window=63 → first 62 rows must be NaN
    first_rows = feat[col].iloc[:warmup]
    assert first_rows.isna().all(), (
        f"Expected all-NaN in first {warmup} rows of '{col}'; "
        f"found {first_rows.notna().sum()} non-NaN value(s): "
        f"{first_rows.dropna().values}"
    )


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_edge_single_value():
    """Feature functions must not raise an exception for a 1-element series."""
    tiny = pd.Series(
        [0.01],
        index=pd.date_range("2000-01-03", periods=1, freq="B"),
    )
    try:
        feat = momentum_features(tiny)
        assert isinstance(feat, pd.DataFrame), (
            f"Expected pd.DataFrame, got {type(feat)}"
        )
    except Exception as exc:
        pytest.fail(
            f"momentum_features raised on a single-element series: {exc}"
        )
