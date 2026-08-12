"""
Performance reporting and result persistence for the rolling backtest.

All metrics follow the paper specification; undocumented aggregation choices
(annual compounding method, risk-free rate) are noted as ASSUMPTIONS.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtest import RebalanceRecord


# ---------------------------------------------------------------------------
# Return aggregation
# ---------------------------------------------------------------------------

def compute_annual_returns(
    records: list,
    daily_returns: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Aggregate holding-period returns into annual calendar returns.

    ASSUMPTION (paper-consistent): compound all holding-period returns whose
    *decision date* falls within calendar year Y.

        annual_return_Y = ∏(1 + R_t  for t ∈ year Y) − 1

    Parameters
    ----------
    records : list
        Rebalance records; each must expose ``.decision_date`` (pd.Timestamp)
        and ``.realized_return`` (float).
    daily_returns : pd.DataFrame or None
        Unused; accepted for API compatibility with callers that pass it.

    Returns
    -------
    pd.Series indexed by year (int), float64 annual returns.
    """
    if not records:
        return pd.Series(dtype=np.float64)

    dates = pd.DatetimeIndex([r.decision_date for r in records])
    rets = np.array([r.realized_return for r in records], dtype=np.float64)
    rets = np.where(np.isfinite(rets), rets, 0.0)

    annual: dict[int, float] = {}
    for yr in sorted(set(dates.year)):
        mask = dates.year == yr
        annual[yr] = float(np.prod(1.0 + rets[mask]) - 1.0)

    return pd.Series(annual, dtype=np.float64)


def compute_cumulative_wealth(
    records: list,
    daily_returns: pd.DataFrame | None = None,
    initial: float = 1.0,
) -> pd.Series:
    """
    Compound holding-period returns multiplicatively.

        wealth_t = initial × ∏_{k ≤ t}(1 + R_k)

    Parameters
    ----------
    records : list
        Rebalance records; each must expose ``.decision_date`` (pd.Timestamp)
        and ``.realized_return`` (float).
    daily_returns : pd.DataFrame or None
        Unused; accepted for API compatibility.
    initial : float
        Starting wealth (default 1.0).

    Returns
    -------
    pd.Series of cumulative wealth indexed by decision_date.
    """
    if not records:
        return pd.Series(dtype=np.float64)

    dates = pd.DatetimeIndex([r.decision_date for r in records])
    rets = np.array([r.realized_return for r in records], dtype=np.float64)
    rets = np.where(np.isfinite(rets), rets, 0.0)

    cum = initial * np.cumprod(1.0 + rets)
    return pd.Series(cum, index=dates, dtype=np.float64)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    records: list,
    daily_returns: pd.DataFrame | None = None,
    strategy_name: str = "",
) -> dict:
    """
    Compute all paper-specified performance metrics.

    Metrics (all percentages where indicated):
    - total_return        : (final_wealth − 1) × 100  [%]
    - avg_annual_return   : mean of annual returns × 100  [%]
    - std_annual_return   : std of annual returns × 100  [%]
    - sharpe_ratio        : avg_annual / std_annual  (risk-free rate = 0,
                            ASSUMPTION: paper does not specify risk-free rate)
    - best_year           : max(annual_returns) × 100  [%]
    - worst_year          : min(annual_returns) × 100  [%]
    - pct_positive_years  : fraction of years with positive return × 100  [%]
    - n_years             : number of calendar years
    - strategy            : strategy_name

    Parameters
    ----------
    records : list
        Rebalance records; each must expose ``.decision_date`` (pd.Timestamp)
        and ``.realized_return`` (float).
    daily_returns : pd.DataFrame or None
        Unused; accepted for API compatibility with callers that pass it.
    strategy_name : str
        Label for the strategy (stored in output dict).

    Returns
    -------
    dict with keys matching the metric names above.
    """
    annual_returns = compute_annual_returns(records, daily_returns)
    wealth = compute_cumulative_wealth(records, daily_returns)
    total_return = float((wealth.iloc[-1] - 1.0) * 100.0) if not wealth.empty else 0.0

    ar = annual_returns.values.astype(np.float64)
    n_years = len(ar)

    if n_years == 0:
        return {
            "strategy": strategy_name,
            "total_return": total_return,
            "avg_annual_return": np.nan,
            "std_annual_return": np.nan,
            "sharpe_ratio": np.nan,
            "best_year": np.nan,
            "worst_year": np.nan,
            "pct_positive_years": np.nan,
            "n_years": 0,
        }

    avg_ann = float(np.mean(ar) * 100.0)
    std_ann = float(np.std(ar, ddof=1) * 100.0) if n_years > 1 else 0.0
    sharpe = avg_ann / std_ann if std_ann != 0.0 else np.nan
    best = float(np.max(ar) * 100.0)
    worst = float(np.min(ar) * 100.0)
    pct_pos = float(np.mean(ar > 0) * 100.0)

    return {
        "strategy": strategy_name,
        "total_return": total_return,
        "avg_annual_return": avg_ann,
        "std_annual_return": std_ann,
        "sharpe_ratio": sharpe,
        "best_year": best,
        "worst_year": worst,
        "pct_positive_years": pct_pos,
        "n_years": n_years,
    }


# ---------------------------------------------------------------------------
# Weight analysis
# ---------------------------------------------------------------------------

def compute_average_weights(
    records: list,  # list[RebalanceRecord]
    industry_names: list[str] | None = None,
) -> pd.Series:
    """
    Average portfolio weight per industry across all rebalance dates.

    Parameters
    ----------
    records : list[RebalanceRecord]
        All diagnostic records from a backtest run.
    industry_names : list[str] or None
        Full universe of industry names (defines the Series index).
        If None, derived from the union of weight keys across all records.

    Returns
    -------
    pd.Series indexed by industry name, float64 average weights.
    """
    if not records:
        idx = industry_names if industry_names is not None else []
        return pd.Series(0.0, index=idx, dtype=np.float64)

    if industry_names is None:
        names_set: set[str] = set()
        for r in records:
            names_set.update(r.weights.keys())
        industry_names = sorted(names_set)

    weight_matrix = pd.DataFrame(
        [{s: r.weights.get(s, 0.0) for s in industry_names} for r in records],
        dtype=np.float64,
    )
    return weight_matrix.mean(axis=0).reindex(industry_names).fillna(0.0)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_performance_table(metrics_dict: dict[str, dict]) -> None:
    """
    Print a formatted table of performance metrics for multiple strategies.

    Parameters
    ----------
    metrics_dict : dict[str, dict]
        Mapping of strategy name → metrics dict (from compute_performance_metrics).
    """
    if not metrics_dict:
        print("No metrics to display.")
        return

    col_order = [
        "total_return",
        "avg_annual_return",
        "std_annual_return",
        "sharpe_ratio",
        "best_year",
        "worst_year",
        "pct_positive_years",
        "n_years",
    ]
    col_labels = {
        "total_return": "Total Ret. (%)",
        "avg_annual_return": "Avg Ann. (%)",
        "std_annual_return": "Std Ann. (%)",
        "sharpe_ratio": "Sharpe",
        "best_year": "Best Yr (%)",
        "worst_year": "Worst Yr (%)",
        "pct_positive_years": "% Pos. Yrs",
        "n_years": "N Years",
    }

    strategies = list(metrics_dict.keys())
    col_w = 16
    name_w = max(max(len(s) for s in strategies), 20)

    # Header
    header = f"{'Strategy':<{name_w}}" + "".join(
        f"{col_labels[c]:>{col_w}}" for c in col_order
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for strat, m in metrics_dict.items():
        row = f"{strat:<{name_w}}"
        for c in col_order:
            val = m.get(c, np.nan)
            if c == "n_years":
                row += f"{int(val) if not np.isnan(float(val)) else 'N/A':>{col_w}}"
            elif isinstance(val, float) and np.isnan(val):
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{val:>{col_w}.2f}"
        print(row)

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(
    metrics_by_strategy: dict,
    exp_id: str = "",
    out_dir: str = "results",
) -> None:
    """
    Save strategy performance metrics to CSV in ``out_dir``.

    File written:
    - ``{exp_id}_metrics.csv`` – one row per strategy, all metrics.

    Parameters
    ----------
    metrics_by_strategy : dict[str, dict]
        Summary metrics per strategy (strategy → metrics dict).
    exp_id : str
        Experiment identifier used as filename prefix (e.g., 'exp_1').
    out_dir : str
        Output directory (created if absent).
    """
    os.makedirs(out_dir, exist_ok=True)

    metrics_rows = list(metrics_by_strategy.values())
    if not metrics_rows:
        return

    prefix = f"{exp_id}_" if exp_id else ""
    df = pd.DataFrame(metrics_rows)
    if "strategy" in df.columns:
        df = df.set_index("strategy")
    df.to_csv(os.path.join(out_dir, f"{prefix}metrics.csv"))
