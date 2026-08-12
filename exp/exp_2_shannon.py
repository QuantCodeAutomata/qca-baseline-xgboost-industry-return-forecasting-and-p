"""
Experiment 2: XGBoost + Shannon Entropy

Augments the baseline feature set with rolling Shannon entropy computed from
the last SHANNON_WINDOW (=20) daily returns using SHANNON_BINS (=10) bins
(see src/config.py for all entropy parameters).

Usage:
    python exp/exp_2_shannon.py
    python exp/exp_2_shannon.py --dry-run
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from exp.run_experiment import run_full_experiment

# Paper target values for Experiment 2 (XGBoost + Shannon Entropy).
_PAPER_TARGETS = {
    "H=5 Max-Profit":  {"total": 30268, "avg_annual": 21.2, "std": 21.0, "sharpe": 1.008},
    "H=21 Max-Profit": {"total": 3972,  "avg_annual": 15.0, "std": 24.1, "sharpe": 0.621},
    "H=63 Max-Profit": {"total": 1609,  "avg_annual": 11.6, "std": 23.4, "sharpe": 0.496},
    "H=5 Min-Risk":    {"total": 2334,  "avg_annual": 11.3, "std": 13.6, "sharpe": 0.832},
    "H=21 Min-Risk":   {"total": 1068,  "avg_annual": 9.1,  "std": 15.5, "sharpe": 0.591},
    "H=63 Min-Risk":   {"total": 1676,  "avg_annual": 10.0, "std": 11.4, "sharpe": 0.877},
}


def _print_comparison(results: dict) -> None:
    """Print a side-by-side comparison of obtained vs. paper target values."""
    print("\n" + "=" * 84)
    print("  Exp 2 (Shannon) – Obtained vs. Paper Target Values")
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
        "      Shannon window=20, bins=10 (ASSUMPTION — see src/config.py)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 2: XGBoost + Shannon Entropy."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use short date range and fast settings for smoke-testing.",
    )
    args = parser.parse_args()

    results = run_full_experiment("exp_2", [5, 21, 63], dry_run=args.dry_run)
    _print_comparison(results)


if __name__ == "__main__":
    main()
