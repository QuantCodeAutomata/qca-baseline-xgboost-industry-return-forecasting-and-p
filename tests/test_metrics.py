"""
Tests for src/metrics.py performance metric functions.

Uses a minimal stub for RebalanceRecord so this test module does not depend
on src/backtest.py.  The stub mirrors the RebalanceRecord interface documented
in AGENTS.md:
    .decision_date   – pd.Timestamp of the rebalancing decision
    .realized_return – float, realised holding-period portfolio return
    .weights         – dict[str, float], portfolio weights (may be unused by metrics)

If the real RebalanceRecord uses different field names, update the stub below.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from dataclasses import dataclass, field
from typing import Dict

sys.path.insert(0, "/workspace/project")

from src.metrics import (
    compute_annual_returns,
    compute_cumulative_wealth,
    compute_performance_metrics,
)


# ── Stub matching the expected RebalanceRecord interface ──────────────────────

@dataclass
class _Record:
    """Minimal stand-in for src.backtest.RebalanceRecord.

    Field names match AGENTS.md:
      decision_date   → used by compute_annual_returns to group by calendar year
      realized_return → holding-period portfolio return used by all metrics
    """
    decision_date: pd.Timestamp
    realized_return: float
    weights: Dict[str, float] = field(default_factory=dict)
    fallback_used: bool = False


def _make_records(dates_returns: list) -> list:
    """Build a list of _Record from [(date_str, return), ...] pairs."""
    return [
        _Record(decision_date=pd.Timestamp(d), realized_return=r)
        for d, r in dates_returns
    ]


def _daily_returns_for(records: list) -> pd.DataFrame:
    """Minimal daily-returns DataFrame covering the record date range."""
    if not records:
        return pd.DataFrame()
    start = min(r.decision_date for r in records) - pd.offsets.BDay(5)
    end = max(r.decision_date for r in records) + pd.offsets.BDay(5)
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"mkt": np.zeros(len(idx))}, index=idx)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_annual_returns_correct():
    """Intra-year returns must be compounded; distinct years must be independent.

    Year 2020: two records with returns 0.10 and 0.20
      → compounded annual return = 1.10 × 1.20 − 1 = 0.32

    Year 2021: one record with return 0.15
      → annual return = 0.15
    """
    records = _make_records([
        ("2020-03-02", 0.10),
        ("2020-09-01", 0.20),
        ("2021-06-01", 0.15),
    ])
    daily_returns = _daily_returns_for(records)
    annual = compute_annual_returns(records, daily_returns)

    assert 2020 in annual, "Year 2020 missing from compute_annual_returns output"
    assert 2021 in annual, "Year 2021 missing from compute_annual_returns output"
    assert np.isclose(annual[2020], 1.10 * 1.20 - 1, atol=1e-8), (
        f"Year-2020 compounded return: expected {1.10*1.20-1:.6f}, got {annual[2020]:.6f}"
    )
    assert np.isclose(annual[2021], 0.15, atol=1e-8), (
        f"Year-2021 return: expected 0.15, got {annual[2021]:.6f}"
    )


def test_cumulative_wealth_monotone_positive():
    """All-positive returns must produce a strictly increasing wealth path."""
    records = _make_records([
        ("2018-01-02", 0.05),
        ("2018-04-02", 0.03),
        ("2018-07-02", 0.08),
        ("2018-10-01", 0.02),
        ("2019-01-02", 0.06),
    ])
    daily_returns = _daily_returns_for(records)
    wealth = compute_cumulative_wealth(records, daily_returns)

    assert isinstance(wealth, pd.Series), (
        f"Expected pd.Series, got {type(wealth)}"
    )
    assert len(wealth) > 1, "Wealth path has fewer than 2 points"
    diffs = np.diff(wealth.values)
    assert (diffs > 0).all(), (
        f"Wealth path not strictly increasing for all-positive returns; "
        f"non-positive increments at positions: {np.where(diffs <= 0)[0]}"
    )


def test_sharpe_ratio():
    """Sharpe ratio must equal mean_annual_return / std_annual_return (risk-free = 0).

    Three records, one per year with returns 0.10, 0.20, 0.30:
      mean = 0.20 ,  std (ddof=1) = 0.10  →  Sharpe = 2.0
    """
    records = _make_records([
        ("2017-06-01", 0.10),
        ("2018-06-01", 0.20),
        ("2019-06-01", 0.30),
    ])
    daily_returns = _daily_returns_for(records)
    metrics = compute_performance_metrics(records, daily_returns)

    assert "sharpe_ratio" in metrics, (
        "Key 'sharpe_ratio' missing from compute_performance_metrics output"
    )
    # Sharpe = mean(annual_returns) / std(annual_returns, ddof=1), risk-free = 0.
    # Regardless of whether metrics are stored as % or decimal, the ratio is the same.
    expected_sharpe = 2.0
    assert np.isclose(metrics["sharpe_ratio"], expected_sharpe, atol=0.05), (
        f"Sharpe ratio: expected ≈ {expected_sharpe:.3f}, "
        f"got {metrics['sharpe_ratio']:.4f}"
    )


def test_total_return_calculation():
    """Total return must equal the product of all (1 + r_i) minus 1.

    Returns: 0.10, 0.20, −0.05  →  total = 1.10 × 1.20 × 0.95 − 1 = 0.254
    """
    returns_seq = [0.10, 0.20, -0.05]
    records = _make_records([
        ("2016-01-04", returns_seq[0]),
        ("2016-05-02", returns_seq[1]),
        ("2016-09-01", returns_seq[2]),
    ])
    daily_returns = _daily_returns_for(records)
    metrics = compute_performance_metrics(records, daily_returns)

    assert "total_return" in metrics, (
        "Key 'total_return' missing from compute_performance_metrics output"
    )
    expected_total = np.prod([1 + r for r in returns_seq]) - 1  # = 0.254
    # total_return may be stored as a percentage (25.4) or decimal (0.254);
    # test the decimal form first, then the percentage form.
    stored = metrics["total_return"]
    matches_decimal = np.isclose(stored, expected_total, atol=1e-6)
    matches_pct = np.isclose(stored, expected_total * 100, atol=1e-4)
    assert matches_decimal or matches_pct, (
        f"total_return: expected {expected_total:.6f} (or {expected_total*100:.4f}%), "
        f"got {stored}"
    )


def test_n_years_consistent():
    """n_years must reflect the number of calendar years covered by the records.

    Five records spanning 2015–2019 (one per year) → n_years should be 5.
    """
    records = _make_records([
        ("2015-06-01", 0.10),
        ("2016-06-01", 0.08),
        ("2017-06-01", 0.12),
        ("2018-06-01", -0.03),
        ("2019-06-01", 0.09),
    ])
    daily_returns = _daily_returns_for(records)
    metrics = compute_performance_metrics(records, daily_returns)

    assert "n_years" in metrics, (
        "Key 'n_years' missing from compute_performance_metrics output"
    )
    # Allow ±1 year tolerance to accommodate different counting conventions
    # (e.g., distinct calendar years vs. continuous time in years).
    assert abs(metrics["n_years"] - 5) <= 1, (
        f"n_years: expected 5 (±1), got {metrics['n_years']}"
    )


def test_pct_positive_years():
    """pct_positive_years must equal the fraction of years with positive annual return.

    Five years: returns 0.20, −0.05, 0.10, −0.02, 0.15 → 3 positive out of 5 = 0.60.
    """
    records = _make_records([
        ("2015-06-01",  0.20),
        ("2016-06-01", -0.05),
        ("2017-06-01",  0.10),
        ("2018-06-01", -0.02),
        ("2019-06-01",  0.15),
    ])
    daily_returns = _daily_returns_for(records)
    metrics = compute_performance_metrics(records, daily_returns)

    assert "pct_positive_years" in metrics, (
        "Key 'pct_positive_years' missing from compute_performance_metrics output"
    )
    stored = metrics["pct_positive_years"]
    # Accept both decimal (0.60) and percentage (60.0) representations.
    matches_decimal = np.isclose(stored, 0.60, atol=1e-6)
    matches_pct = np.isclose(stored, 60.0, atol=1e-4)
    assert matches_decimal or matches_pct, (
        f"pct_positive_years: expected 0.60 (or 60.0%), got {stored}"
    )
