"""
Unified experiment runner for all three experiments:
- exp_1: baseline (no entropy)
- exp_2: shannon entropy augmented
- exp_3: fuzzy entropy augmented

Usage:
    python exp/run_experiment.py --exp exp_1 --horizon 5
    python exp/run_experiment.py --exp all
"""

import argparse
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import (
    CLIP_RETURN_LB,
    CLIP_RETURN_UB,
    COV_FALLBACK,
    COV_LOOKBACK,
    COV_SHRINKAGE_ALPHA,
    EMPTY_ELIGIBLE_FALLBACK,
    ENSEMBLE_WEIGHT_RULE,
    HORIZONS,
    MIN_TRAIN_OBS,
    RESULTS_DIR,
    ROLLING_WINDOWS,
    TRAIN_FRAC,
    XGB_CLASSIFIER_PARAMS,
    XGB_REGRESSOR_PARAMS,
)
from data.load_data import (
    compute_daily_returns,
    compute_forward_returns,
    download_industry_prices,
)
from src.backtest import BacktestConfig, run_backtest, run_backtest_multi_portfolio
from src.entropy import add_entropy_features, assumption_log
from src.features import compute_baseline_features
from src.metrics import (
    compute_annual_returns,
    compute_average_weights,
    compute_cumulative_wealth,
    compute_performance_metrics,
    save_results,
)

# ── Dry-run overrides ─────────────────────────────────────────────────────────
DRY_RUN_START = "2010-01-01"
DRY_RUN_HORIZONS = [5, 21]
DRY_RUN_ROLLING_WINDOW = 252  # ~1 year; keeps each backtest fast

# ── Plot styling ──────────────────────────────────────────────────────────────
_HORIZON_COLORS = {5: "steelblue", 21: "darkorange", 63: "forestgreen"}
_PTYPE_LINESTYLES = {"max_profit": "-", "min_risk": "--"}
_PTYPE_LABELS = {"max_profit": "Max-Profit", "min_risk": "Min-Risk"}
_PORTFOLIO_TYPES = ["max_profit", "min_risk"]


def _strategy_name(horizon: int, ptype: str) -> str:
    return f"H={horizon} {_PTYPE_LABELS[ptype]}"


def _make_backtest_config(
    horizon: int,
    ptype: str,
    rolling_window: int,
    feature_set: str = "baseline",
) -> BacktestConfig:
    """Construct a BacktestConfig from global config.py settings."""
    return BacktestConfig(
        horizon=horizon,
        feature_set=feature_set,
        portfolio_type=ptype,
        rolling_window=rolling_window,
        train_frac=TRAIN_FRAC,
        min_train_obs=MIN_TRAIN_OBS,
        xgb_clf_params=XGB_CLASSIFIER_PARAMS,
        xgb_reg_params=XGB_REGRESSOR_PARAMS,
        ensemble_weight_rule=ENSEMBLE_WEIGHT_RULE,
        cov_lookback=COV_LOOKBACK[horizon],
        cov_fallback=COV_FALLBACK,
        cov_shrinkage_alpha=COV_SHRINKAGE_ALPHA,
        empty_eligible_fallback=EMPTY_ELIGIBLE_FALLBACK,
        clip_lb=CLIP_RETURN_LB,
        clip_ub=CLIP_RETURN_UB,
    )


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_cumulative_wealth(
    wealth_paths: dict,
    exp_id: str,
    out_dir: str,
) -> None:
    """
    Plot cumulative wealth paths for all strategies on one figure.

    Parameters
    ----------
    wealth_paths : dict[str, pd.Series]
        Strategy name → cumulative wealth Series (indexed by date, starting at 1.0).
    exp_id : str
        Used in the title and output filename.
    out_dir : str
        Directory to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy, wealth in wealth_paths.items():
        h = int(strategy.split("=")[1].split(" ")[0])
        ptype = "max_profit" if "Max-Profit" in strategy else "min_risk"
        ax.plot(
            wealth.index,
            wealth.values,
            color=_HORIZON_COLORS.get(h, "gray"),
            linestyle=_PTYPE_LINESTYLES[ptype],
            label=strategy,
            linewidth=1.5,
        )
    ax.set_title(f"{exp_id} – Cumulative Wealth Paths", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Wealth (starting at 1.0)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{exp_id}_cumulative_wealth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_annual_returns(
    annual_returns: dict,
    exp_id: str,
    out_dir: str,
) -> None:
    """
    Grouped bar chart of annual returns (%) across all strategies.

    Parameters
    ----------
    annual_returns : dict[str, dict[int, float]]
        Strategy name → {year: annual_return (decimal)}.
    """
    strategies = list(annual_returns.keys())
    all_years = sorted({yr for ret_dict in annual_returns.values() for yr in ret_dict})
    x = np.arange(len(all_years))
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, strategy in enumerate(strategies):
        h = int(strategy.split("=")[1].split(" ")[0])
        ptype = "max_profit" if "Max-Profit" in strategy else "min_risk"
        vals = [annual_returns[strategy].get(yr, np.nan) * 100 for yr in all_years]
        ax.bar(
            x + i * width,
            vals,
            width=width,
            label=strategy,
            color=_HORIZON_COLORS.get(h, "gray"),
            alpha=0.75 if ptype == "max_profit" else 0.45,
        )
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels([str(y) for y in all_years], rotation=45, ha="right")
    ax.set_title(f"{exp_id} – Annual Returns (%)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Return (%)")
    ax.legend(loc="upper left", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{exp_id}_annual_returns.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_weights(
    avg_weights: dict,
    exp_id: str,
    out_dir: str,
) -> None:
    """
    Grouped bar chart of average industry weights (%) across all strategies.

    Parameters
    ----------
    avg_weights : dict[str, dict[str, float]]
        Strategy name → {industry: avg_weight (decimal)}.
    """
    strategies = list(avg_weights.keys())
    all_industries = sorted({ind for w in avg_weights.values() for ind in w})
    x = np.arange(len(all_industries))
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, strategy in enumerate(strategies):
        h = int(strategy.split("=")[1].split(" ")[0])
        ptype = "max_profit" if "Max-Profit" in strategy else "min_risk"
        vals = [avg_weights[strategy].get(ind, 0.0) * 100 for ind in all_industries]
        ax.bar(
            x + i * width,
            vals,
            width=width,
            label=strategy,
            color=_HORIZON_COLORS.get(h, "gray"),
            alpha=0.75 if ptype == "max_profit" else 0.45,
        )
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(all_industries, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{exp_id} – Average Industry Weights (%)", fontsize=14)
    ax.set_xlabel("Industry")
    ax.set_ylabel("Average Weight (%)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{exp_id}_weights.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _print_performance_table(results: dict, exp_id: str) -> None:
    """Print a formatted performance summary table to stdout."""
    header = f"\n{'='*72}\n  {exp_id} – Performance Summary\n{'='*72}"
    print(header)
    fmt = "{:<28} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Strategy", "Total%", "Ann%", "Std%", "Sharpe"))
    print("-" * 72)
    for strategy, m in results.items():
        total = m.get("total_return", float("nan"))
        ann = m.get("avg_annual_return", float("nan"))
        std = m.get("std_annual_return", float("nan"))
        sharpe = m.get("sharpe_ratio", float("nan"))
        print(fmt.format(strategy, f"{total:.1f}", f"{ann:.1f}", f"{std:.1f}", f"{sharpe:.3f}"))
    print("=" * 72)


# ── Core runner ───────────────────────────────────────────────────────────────

def run_full_experiment(
    exp_id: str,
    horizons: list,
    dry_run: bool = False,
) -> dict:
    """
    Run one full experiment across all horizons and portfolio types.

    Parameters
    ----------
    exp_id : str
        Experiment identifier: 'exp_1', 'exp_2', or 'exp_3'.
    horizons : list[int]
        Holding-period horizons in trading days; subset of [5, 21, 63].
    dry_run : bool
        If True, uses date range 2010–2024, horizons [5, 21], and a small
        rolling window (~1 year) so the full pipeline runs in minutes.

    Returns
    -------
    dict
        Mapping of strategy name → performance metrics dict.
        Each metrics dict contains: total_return (%), avg_annual_return (%),
        std_annual_return (%), sharpe_ratio, n_years, pct_positive_years.
    """
    print(f"\n{'#'*72}")
    print(f"  Starting {exp_id}  |  dry_run={dry_run}")
    print(f"{'#'*72}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Resolve runtime parameters ────────────────────────────────────────────
    effective_horizons = DRY_RUN_HORIZONS if dry_run else horizons
    start_date = DRY_RUN_START if dry_run else None  # None → load_data uses START_DATE

    # Map exp_id to feature_set label used by BacktestConfig
    _FEATURE_SET_MAP = {"exp_1": "baseline", "exp_2": "shannon", "exp_3": "fuzzy"}
    feature_set = _FEATURE_SET_MAP.get(exp_id, "baseline")

    # ── 1. Download prices ────────────────────────────────────────────────────
    print("Loading industry price data …")
    if start_date is None:
        prices = download_industry_prices()
    else:
        prices = download_industry_prices(start=start_date)

    # ── 2. Daily returns ──────────────────────────────────────────────────────
    print("Computing daily returns …")
    daily_returns = compute_daily_returns(prices)

    # ── 3. Baseline features ──────────────────────────────────────────────────
    print("Computing baseline features …")
    features = compute_baseline_features(prices, daily_returns)

    # ── 4. Entropy augmentation ───────────────────────────────────────────────
    if exp_id == "exp_2":
        print("Adding Shannon entropy features …")
        features = add_entropy_features(
            features, daily_returns, include_shannon=True, include_fuzzy=False
        )
    elif exp_id == "exp_3":
        print("Adding fuzzy entropy features …")
        features = add_entropy_features(
            features, daily_returns, include_shannon=False, include_fuzzy=True
        )

    industry_names = list(prices.columns)

    # ── 5. Backtest each (horizon, portfolio_type) combination ────────────────
    all_records: dict = {}
    for horizon in effective_horizons:
        rolling_window = DRY_RUN_ROLLING_WINDOW if dry_run else ROLLING_WINDOWS[horizon]
        print(f"\nHorizon={horizon} d  |  rolling_window={rolling_window} d")

        # Compute clipped forward returns and binary targets from PRICES
        fwd_ret_wide, binary_tgt_wide = compute_forward_returns(prices, horizon)

        # Melt to long format and merge targets into the feature panel
        fwd_ret_long = fwd_ret_wide.reset_index().melt(
            id_vars="date", var_name="industry", value_name="target_reg"
        )
        binary_tgt_long = binary_tgt_wide.reset_index().melt(
            id_vars="date", var_name="industry", value_name="target_cls"
        )
        panel = features.merge(fwd_ret_long, on=["date", "industry"], how="left")
        panel = panel.merge(binary_tgt_long, on=["date", "industry"], how="left")

        feature_cols = [
            c for c in panel.columns
            if c not in ("date", "industry", "target_cls", "target_reg")
        ]

        # Train XGBoost once per horizon, apply both portfolio types in one pass
        print(f"  Backtesting both portfolios (shared training) …")
        config = _make_backtest_config(horizon, "max_profit", rolling_window, feature_set)
        multi_results = run_backtest_multi_portfolio(
            panel, prices, daily_returns, config, feature_cols, industry_names,
            portfolio_types=_PORTFOLIO_TYPES, verbose=True,
        )
        for ptype in _PORTFOLIO_TYPES:
            strategy = _strategy_name(horizon, ptype)
            records, _hp_returns = multi_results[ptype]
            all_records[strategy] = records
            print(f"    → {strategy}: {len(records)} rebalance records")

    # ── 6. Metrics and plot data ──────────────────────────────────────────────
    print("\nComputing performance metrics …")
    results: dict = {}
    wealth_paths: dict = {}
    annual_rets: dict = {}
    avg_wts: dict = {}

    for strategy, records in all_records.items():
        results[strategy] = compute_performance_metrics(records, daily_returns, strategy)
        wealth_paths[strategy] = compute_cumulative_wealth(records, daily_returns)
        annual_rets[strategy] = compute_annual_returns(records, daily_returns)
        avg_wts[strategy] = compute_average_weights(records)

    # ── 7. Persist numeric results ────────────────────────────────────────────
    save_results(results, exp_id, out_dir=RESULTS_DIR)

    # ── 8. Generate plots ─────────────────────────────────────────────────────
    print("\nGenerating plots …")
    _plot_cumulative_wealth(wealth_paths, exp_id, RESULTS_DIR)
    _plot_annual_returns(annual_rets, exp_id, RESULTS_DIR)
    _plot_weights(avg_wts, exp_id, RESULTS_DIR)

    # ── 9. Console summary ────────────────────────────────────────────────────
    _print_performance_table(results, exp_id)

    # ── 10. Log all config assumptions ───────────────────────────────────────
    print("\nConfiguration assumptions:")
    assumption_log()

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run entropy-augmented XGBoost portfolio experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python exp/run_experiment.py --exp exp_1 --horizon 5
              python exp/run_experiment.py --exp all
              python exp/run_experiment.py --exp exp_2 --dry-run
        """),
    )
    parser.add_argument(
        "--exp",
        choices=["exp_1", "exp_2", "exp_3", "all"],
        default="all",
        help="Experiment to run (default: all)",
    )
    parser.add_argument(
        "--horizon",
        choices=["5", "21", "63", "all"],
        default="all",
        help="Holding-period horizon in trading days (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short date range + fast settings for quick smoke-testing",
    )
    return parser.parse_args()


def main() -> dict:
    args = _parse_args()
    exp_ids = ["exp_1", "exp_2", "exp_3"] if args.exp == "all" else [args.exp]
    horizons = HORIZONS if args.horizon == "all" else [int(args.horizon)]

    all_results: dict = {}
    for exp_id in exp_ids:
        all_results[exp_id] = run_full_experiment(exp_id, horizons, dry_run=args.dry_run)

    print("\nAll experiments complete.")
    return all_results


if __name__ == "__main__":
    main()
