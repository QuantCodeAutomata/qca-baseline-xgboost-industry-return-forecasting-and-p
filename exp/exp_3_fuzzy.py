"""
Experiment 3: XGBoost + Fuzzy Entropy

Augments the baseline feature set with rolling Fuzzy Entropy computed using
a 63-day window, embedding dimension m=2, tolerance r = 0.2 * std(window),
and exponent n=2 (see src/config.py for all FUZZY_* parameters).

Usage:
    python exp/exp_3_fuzzy.py
    python exp/exp_3_fuzzy.py --dry-run
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from exp.run_experiment import run_full_experiment

# Paper target values for Experiment 3 (XGBoost + Fuzzy Entropy).
_PAPER_TARGETS = {
    "H=5 Max-Profit":  {"total": 12049, "avg_annual": 17.6, "std": 18.9, "sharpe": 0.929},
    "H=21 Max-Profit": {"total": 4143,  "avg_annual": 14.2, "std": 19.1, "sharpe": 0.742},
    "H=63 Max-Profit": {"total": 1914,  "avg_annual": 12.3, "std": 23.6, "sharpe": 0.521},
    "H=5 Min-Risk":    {"total": 1990,  "avg_annual": 10.8, "std": 13.0, "sharpe": 0.829},
    "H=21 Min-Risk":   {"total": 1180,  "avg_annual": 9.1,  "std": 12.6, "sharpe": 0.716},
    "H=63 Min-Risk":   {"total": 1742,  "avg_annual": 10.1, "std": 11.6, "sharpe": 0.872},
}


def _print_comparison(results: dict) -> None:
    """Print a side-by-side comparison of obtained vs. paper target values."""
    print("\n" + "=" * 84)
    print("  Exp 3 (Fuzzy) – Obtained vs. Paper Target Values")
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
        "      Fuzzy entropy params: window=63, m=2, r_frac=0.2, n=2.0 (ASSUMPTION)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 3: XGBoost + Fuzzy Entropy."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use short date range and fast settings for smoke-testing.",
    )
    args = parser.parse_args()

    results = run_full_experiment("exp_3", [5, 21, 63], dry_run=args.dry_run)
    _print_comparison(results)


if __name__ == "__main__":
    main()
