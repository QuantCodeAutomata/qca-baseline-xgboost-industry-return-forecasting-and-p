"""
Tests for src/portfolio.py portfolio construction functions.

Verifies the simplex constraint (weights sum to 1, all >= 0),
eligibility filtering, and graceful fallback behaviour for degenerate
inputs (empty eligible set, singular covariance matrix).
"""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, "/workspace/project")

from src.portfolio import estimate_covariance, max_profit_portfolio, min_risk_portfolio

# ── Shared helpers ────────────────────────────────────────────────────────────

def _scores(n: int = 10, seed: int = 42) -> pd.Series:
    """Synthetic ensemble scores (predicted forward returns)."""
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.uniform(-0.04, 0.04, size=n),
        index=[f"ind_{i}" for i in range(n)],
    )


def _returns(n_obs: int = 300, n_ind: int = 10, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily returns DataFrame for covariance estimation."""
    rng = np.random.default_rng(seed)
    data = rng.multivariate_normal(
        mean=np.zeros(n_ind),
        cov=np.eye(n_ind) * 0.0004,  # daily vol ≈ 2 %
        size=n_obs,
    )
    return pd.DataFrame(
        data,
        index=pd.date_range("2010-01-04", periods=n_obs, freq="B"),
        columns=[f"ind_{i}" for i in range(n_ind)],
    )


def _eligible(n: int = 5) -> list:
    return [f"ind_{i}" for i in range(n)]


# ── max_profit_portfolio ──────────────────────────────────────────────────────

def test_max_profit_weights_sum_to_one():
    """Portfolio weights must sum to exactly 1.0 (fully invested)."""
    scores = _scores()
    eligible = _eligible()
    daily_returns = _returns()
    weights = max_profit_portfolio(scores, eligible, daily_returns)
    assert np.isclose(weights.sum(), 1.0, atol=1e-8), (
        f"max_profit weights sum to {weights.sum():.10f}, expected 1.0"
    )


def test_max_profit_nonnegative():
    """All weights must be >= 0 (long-only portfolio)."""
    scores = _scores()
    eligible = _eligible()
    daily_returns = _returns()
    weights = max_profit_portfolio(scores, eligible, daily_returns)
    neg = weights[weights < -1e-9]
    assert len(neg) == 0, (
        f"max_profit has {len(neg)} negative weight(s): {neg.to_dict()}"
    )


def test_max_profit_eligible_only():
    """Industries outside the eligible set must receive zero weight."""
    n_ind = 10
    n_eligible = 3
    scores = _scores(n=n_ind)
    eligible = _eligible(n=n_eligible)
    daily_returns = _returns(n_ind=n_ind)
    weights = max_profit_portfolio(scores, eligible, daily_returns)

    non_eligible = [ind for ind in scores.index if ind not in eligible]
    for ind in non_eligible:
        w = weights.get(ind, 0.0) if hasattr(weights, "get") else float(weights[ind])
        assert np.isclose(w, 0.0, atol=1e-9), (
            f"Non-eligible industry '{ind}' has weight {w:.8f} (expected 0)"
        )


def test_max_profit_empty_eligible():
    """With an empty eligible set the fallback must return a valid distribution."""
    scores = _scores()
    daily_returns = _returns()
    try:
        weights = max_profit_portfolio(scores, [], daily_returns)
        assert np.isclose(weights.sum(), 1.0, atol=1e-8), (
            f"Fallback weights sum to {weights.sum():.8f}, expected 1.0"
        )
        assert (weights >= -1e-9).all(), "Fallback weights contain negatives"
    except Exception as exc:
        pytest.fail(
            f"max_profit_portfolio raised on empty eligible set: {exc}"
        )


# ── min_risk_portfolio ────────────────────────────────────────────────────────

def test_min_risk_weights_sum_to_one():
    """Portfolio weights must sum to exactly 1.0 (fully invested)."""
    scores = _scores()
    eligible = _eligible()
    daily_returns = _returns()
    weights = min_risk_portfolio(scores, eligible, daily_returns)
    assert np.isclose(weights.sum(), 1.0, atol=1e-8), (
        f"min_risk weights sum to {weights.sum():.10f}, expected 1.0"
    )


def test_min_risk_nonnegative():
    """All weights must be >= 0 (long-only constraint satisfied)."""
    scores = _scores()
    eligible = _eligible()
    daily_returns = _returns()
    weights = min_risk_portfolio(scores, eligible, daily_returns)
    neg = weights[weights < -1e-9]
    assert len(neg) == 0, (
        f"min_risk has {len(neg)} negative weight(s) (short position): {neg.to_dict()}"
    )


def test_min_risk_singular_cov():
    """min_risk_portfolio must not crash when the covariance matrix is singular.

    Perfectly correlated returns (all columns identical) produce a rank-1
    covariance matrix.  The implementation must apply its configured fallback
    (shrinkage / pseudo-inverse / diagonal load) and still return weights
    that sum to 1.
    """
    n_obs, n_ind = 150, 5
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.02, size=(n_obs, 1))
    data = np.hstack([base] * n_ind)          # rank-1 → singular covariance
    daily_returns = pd.DataFrame(
        data,
        index=pd.date_range("2015-01-02", periods=n_obs, freq="B"),
        columns=[f"ind_{i}" for i in range(n_ind)],
    )
    scores = _scores(n=n_ind, seed=5)
    eligible = [f"ind_{i}" for i in range(n_ind)]

    try:
        weights = min_risk_portfolio(scores, eligible, daily_returns)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6), (
            f"Singular-cov fallback: weights sum to {weights.sum():.8f}"
        )
        assert (weights >= -1e-9).all(), (
            "Singular-cov fallback returned negative weights"
        )
    except Exception as exc:
        pytest.fail(
            f"min_risk_portfolio raised on a singular covariance matrix: {exc}"
        )


def test_min_risk_empty_eligible():
    """With an empty eligible set the fallback must return a valid distribution."""
    scores = _scores()
    daily_returns = _returns()
    try:
        weights = min_risk_portfolio(scores, [], daily_returns)
        assert np.isclose(weights.sum(), 1.0, atol=1e-8), (
            f"Fallback weights sum to {weights.sum():.8f}, expected 1.0"
        )
        assert (weights >= -1e-9).all(), "Fallback weights contain negatives"
    except Exception as exc:
        pytest.fail(
            f"min_risk_portfolio raised on empty eligible set: {exc}"
        )
