"""
Experiment 1: Baseline XGBoost (no entropy features)

Runs the 6-strategy baseline:
- H=5,21,63 x max_profit,min_risk

Usage:
    python exp/exp_1_baseline.py
    python exp/exp_1_baseline.py --dry-run
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from exp.run_experiment import run_full_experiment

# Paper target values for Experiment 1 (Table results from the paper).
# total: cumulative total return (%); avg_annual: mean annual return (%);
# std: std-dev of annual returns (%); sharpe: annualised Sharpe ratio.
_PAPER_TARGETS = {
    "H=5 Max-Profit":  {"total": 15220, "avg_annual": 18.9, "std": 21.5, "sharpe": 0.876},
    "H=21 Max-Profit": {"total": 3931,  "avg_annual": 14.5, "std": 21.7, "sharpe": 0.669},
    "H=63 Max-Profit": {"total": 1569,  "avg_annual": 11.5, "std": 23.1, "sharpe": 0.498},
    "H=5 Min-Risk":    {"total": 1993,  "avg_annual": 10.7, "std": 12.9, "sharpe": 0.831},
    "H=21 Min-Risk":   {"total": 941,   "avg_annual": 8.5,  "std": 13.6, "sharpe": 0.626},
    "H=63 Min-Risk":   {"total": 1332,  "avg_annual": 9.4,  "std": 12.9, "sharpe": 0.730},
}


def _print_comparison(results: dict) -> None:
    """Print a side-by-side comparison of obtained vs. paper target values."""
    print("\n" + "=" * 84)
    print("  Exp 1 – Obtained vs. Paper Target Values")
    print("=" * 84)
    fmt = "{:<28}  {:>9} {:>9}  |  {:>9} {:>9}  |  {:>8} {:>8}"
    print(fmt.format(
        "Strategy",
        "Got Ann%", "Tgt Ann%",
        "Got Std%", "Tgt Std%",
        "Got Shrp", "Tgt Shrp",
    ))
    print("-" * 84)
    for strategy, target in _PAPER_TARGETS.items():
        got = results.get(strategy, {})
        print(fmt.format(
            strategy,
            f"{got.get('avg_annual_return', float('nan')):.1f}",
            f"{target['avg_annual']:.1f}",
            f"{got.get('std_annual_return', float('nan')):.1f}",
            f"{target['std']:.1f}",
            f"{got.get('sharpe_ratio', float('nan')):.3f}",
            f"{target['sharpe']:.3f}",
        ))
    print("=" * 84)
    print(
        "Note: differences are expected — the paper uses proprietary MSCI series;\n"
        "      this replication uses U.S. sector ETF proxies (see src/config.py).\n"
        "      Unspecified hyperparameters (XGBoost, rolling window, covariance\n"
        "      lookback) are documented as ASSUMPTION entries in src/config.py."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1: Baseline XGBoost (no entropy)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use short date range and fast settings for smoke-testing.",
    )
    args = parser.parse_args()

    results = run_full_experiment("exp_1", [5, 21, 63], dry_run=args.dry_run)
    _print_comparison(results)


if __name__ == "__main__":
    main()
