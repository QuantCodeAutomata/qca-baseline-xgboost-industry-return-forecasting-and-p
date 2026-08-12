"""
Entropy computation module: Shannon entropy and Fuzzy entropy.

All computations are causal (no look-ahead bias): entropy at date t uses
only returns[t-window+1 : t+1].

References
----------
Shannon entropy: paper-specified (20-day window, 10 equal-width bins, natural log).
Fuzzy entropy: Chen et al. (2007), standard FuzzyEn definition; paper omits r, n,
    window length, and averaging formula — assumptions documented in assumption_log().
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# ─────────────────────────── Assumption log ──────────────────────────────────

def assumption_log() -> dict:
    """
    Return dict of all assumption choices for fuzzy entropy.
    This is logged alongside every experiment run.
    """
    return {
        "fuzzy_window": config.FUZZY_WINDOW,
        "fuzzy_m": config.FUZZY_M,
        "fuzzy_tau": config.FUZZY_TAU,
        "fuzzy_r_rule": config.FUZZY_R_RULE,
        "fuzzy_r_frac": config.FUZZY_R_FRAC,
        "fuzzy_n": config.FUZZY_N,
        "fuzzy_exclude_self": config.FUZZY_EXCLUDE_SELF,
        "fuzzy_r_derivation": (
            "r = FUZZY_R_FRAC * std(window_returns, ddof=1) when FUZZY_R_RULE='std_frac'. "
            "r=0 (zero-variance window) yields NaN."
        ),
        "fuzzy_kernel": (
            "mu_ij = exp(-(d_ij / r)^n) where d_ij is Chebyshev distance "
            "(Chen et al. 2007, standard FuzzyEn)."
        ),
        "fuzzy_phi_formula": (
            "phi_m = (1/N) * sum_i [ (1/(N-1)) * sum_{j!=i} mu_ij ] "
            "= sum_{i!=j} mu_ij / (N*(N-1))  [FUZZY_EXCLUDE_SELF=True]"
        ),
        "fuzzy_self_exclusion": (
            "Self-pairs excluded from phi average (FUZZY_EXCLUDE_SELF=True); "
            "denominator N*(N-1). Consistent with sample-entropy convention."
        ),
        "fuzzy_tau_note": (
            "tau=1 (consecutive delay vectors, no subsampling); fixed per paper."
        ),
    }


# ─────────────────────────── Shannon entropy ─────────────────────────────────

def shannon_entropy_scalar(returns_window: np.ndarray, bins: int = 10) -> float:
    """
    Compute Shannon entropy for a single 20-day return window.

    # Custom — paper-specified formula (20-day window, 10 equal-width bins, natural log)

    Parameters
    ----------
    returns_window : array of shape (W,) where W=20
    bins : number of histogram bins (default 10)

    Returns
    -------
    float: Shannon entropy, 0.0 if all returns are identical
    """
    lo = returns_window.min()
    hi = returns_window.max()
    if lo == hi:  # degenerate: zero range → all mass in one bin → entropy = 0
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(returns_window, bins=edges)
    probs = counts / len(returns_window)
    nonzero = probs[probs > 0.0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def compute_shannon_entropy(
    returns: pd.DataFrame,
    window: int = 20,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling Shannon entropy for all industries.
    Returns DataFrame same shape as returns with column names prefixed 'shannon_ent_'.

    Causal: entropy at t uses returns[t-window+1 : t+1] (inclusive).
    NaN for first (window-1) rows.
    Logs count of degenerate windows (min==max) per industry.
    """
    out_cols = ["shannon_ent_" + c for c in returns.columns]
    result = pd.DataFrame(
        np.nan, index=returns.index, columns=out_cols, dtype=np.float64
    )

    for col in returns.columns:
        series = returns[col].to_numpy(dtype=np.float64)
        n = len(series)
        vals = np.full(n, np.nan, dtype=np.float64)
        degenerate_count = 0

        for t in range(window - 1, n):
            w = series[t - window + 1 : t + 1]
            if np.any(np.isnan(w)):
                continue
            if w.min() == w.max():
                degenerate_count += 1
            vals[t] = shannon_entropy_scalar(w, n_bins)

        result["shannon_ent_" + col] = vals

        if degenerate_count > 0:
            logger.info(
                "Shannon entropy: %d degenerate window(s) (min==max) for industry '%s'",
                degenerate_count,
                col,
            )

    return result


# ─────────────────────────── Fuzzy entropy ───────────────────────────────────

def _compute_phi(
    x: np.ndarray, m: int, r: float, n: float, exclude_self: bool
) -> float:
    """
    Compute phi_m: average fuzzy membership across all template pairs of dimension m.

    Uses numpy broadcasting to build the full (N x N) Chebyshev distance matrix
    and apply the fuzzy kernel, avoiding Python-level pair loops.
    """
    N = len(x) - m + 1
    if N <= 1:
        return np.nan
    # Embedding matrix: row i = [x[i], x[i+1], ..., x[i+m-1]], shape (N, m)
    U = np.lib.stride_tricks.sliding_window_view(x, m)  # (N, m), view not copy
    # Pairwise element-wise differences: (N, N, m)
    diff = U[:, np.newaxis, :] - U[np.newaxis, :, :]
    # Chebyshev distance = max absolute difference across dimension: (N, N)
    dist = np.max(np.abs(diff), axis=2)
    # Fuzzy similarity: mu_ij = exp(-(d_ij / r)^n)
    mu = np.exp(-((dist / r) ** n))

    if exclude_self:
        np.fill_diagonal(mu, 0.0)
        return float(mu.sum() / (N * (N - 1)))
    else:
        # Include self-pair: d_ii = 0 → mu_ii = 1; denominator = N*N
        return float(mu.sum() / (N * N))


def fuzzy_entropy_scalar(
    x: np.ndarray,
    m: int = 2,
    r: float = None,
    n: float = 2.0,
    exclude_self: bool = True,
) -> float:
    """
    Compute Fuzzy Entropy for a single window.

    # Custom — Context7 found no library equivalent (standard FuzzyEn, Chen et al. 2007)
    # Paper specifies: m=2, tau=1, mu=exp(-(d/r)^n), but omits r, n, window length,
    # averaging formula. Using standard FuzzyEn with documented assumptions.

    Parameters
    ----------
    x : 1D array of returns for the rolling window
    m : embedding dimension (default 2)
    r : scale parameter. If None, uses 0.2 * std(x)
    n : exponent in fuzzy kernel (default 2.0)
    exclude_self : whether to exclude self-pairs (default True)

    Returns
    -------
    float: fuzzy entropy, or NaN if undefined (e.g. r=0, phi_m=0)
    """
    # Assumption log: r=0.2*std rule from common SampEn/FuzzyEn practice;
    # n=2 from Chen et al.; excluding self-matches per standard practice.
    x = np.asarray(x, dtype=np.float64)
    if r is None:
        r = config.FUZZY_R_FRAC * np.std(x, ddof=1)
    if r == 0.0:
        return np.nan

    phi_m = _compute_phi(x, m, r, n, exclude_self)
    phi_m1 = _compute_phi(x, m + 1, r, n, exclude_self)

    if np.isnan(phi_m) or np.isnan(phi_m1) or phi_m == 0.0:
        return np.nan
    ratio = phi_m1 / phi_m
    if ratio <= 0.0:
        return np.nan
    return float(-np.log(ratio))


def compute_fuzzy_entropy(
    returns: pd.DataFrame,
    window: int = None,
    m: int = None,
    r_rule: str = None,
    r_frac: float = None,
    n: float = None,
    exclude_self: bool = None,
) -> pd.DataFrame:
    """
    Compute rolling Fuzzy Entropy for all industries.
    Returns DataFrame same shape as returns with column names prefixed 'fuzzy_ent_'.
    NaN for first (window-1) rows.
    Logs assumption choices and count of NaN windows per industry.
    """
    window = config.FUZZY_WINDOW if window is None else window
    m = config.FUZZY_M if m is None else m
    r_rule = config.FUZZY_R_RULE if r_rule is None else r_rule
    r_frac = config.FUZZY_R_FRAC if r_frac is None else r_frac
    n = config.FUZZY_N if n is None else n
    exclude_self = config.FUZZY_EXCLUDE_SELF if exclude_self is None else exclude_self

    logger.info("Fuzzy entropy assumptions: %s", assumption_log())
    logger.info(
        "compute_fuzzy_entropy: window=%d, m=%d, r_rule='%s', r_frac=%.3f, "
        "n=%.1f, exclude_self=%s",
        window, m, r_rule, r_frac, n, exclude_self,
    )

    out_cols = ["fuzzy_ent_" + c for c in returns.columns]
    result = pd.DataFrame(
        np.nan, index=returns.index, columns=out_cols, dtype=np.float64
    )

    for col in returns.columns:
        series = returns[col].to_numpy(dtype=np.float64)
        nt = len(series)
        vals = np.full(nt, np.nan, dtype=np.float64)
        nan_count = 0

        for t in range(window - 1, nt):
            w = series[t - window + 1 : t + 1]
            if np.any(np.isnan(w)):
                nan_count += 1
                continue

            if r_rule == "std_frac":
                r_val = r_frac * np.std(w, ddof=1)
            else:
                r_val = float(r_frac)  # treat r_frac as a fixed absolute r

            fe = fuzzy_entropy_scalar(w, m=m, r=r_val, n=n, exclude_self=exclude_self)
            vals[t] = fe
            if np.isnan(fe):
                nan_count += 1

        result["fuzzy_ent_" + col] = vals

        if nan_count > 0:
            logger.info(
                "Fuzzy entropy: %d NaN result(s) for industry '%s'",
                nan_count,
                col,
            )

    return result


# ─────────────────────────── Combined feature adder ──────────────────────────

def add_entropy_features(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    include_shannon: bool = True,
    include_fuzzy: bool = False,
) -> pd.DataFrame:
    """
    Add entropy feature(s) to the long-format panel DataFrame.
    Shannon: column 'shannon_ent'
    Fuzzy: column 'fuzzy_ent'
    Merges on (date, industry).
    """
    panel = panel.copy()

    if include_shannon:
        shannon_wide = compute_shannon_entropy(
            returns_wide,
            window=config.SHANNON_WINDOW,
            n_bins=config.SHANNON_BINS,
        )
        # Strip 'shannon_ent_' prefix so columns match returns_wide industry names
        shannon_wide.columns = returns_wide.columns
        shannon_wide.index.name = "date"
        shannon_long = (
            shannon_wide.reset_index()
            .melt(id_vars="date", var_name="industry", value_name="shannon_ent")
        )
        panel = panel.merge(shannon_long, on=["date", "industry"], how="left")

    if include_fuzzy:
        fuzzy_wide = compute_fuzzy_entropy(returns_wide)
        # Strip 'fuzzy_ent_' prefix so columns match returns_wide industry names
        fuzzy_wide.columns = returns_wide.columns
        fuzzy_wide.index.name = "date"
        fuzzy_long = (
            fuzzy_wide.reset_index()
            .melt(id_vars="date", var_name="industry", value_name="fuzzy_ent")
        )
        panel = panel.merge(fuzzy_long, on=["date", "industry"], how="left")

    return panel
