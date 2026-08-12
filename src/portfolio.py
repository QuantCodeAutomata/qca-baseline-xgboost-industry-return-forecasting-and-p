"""
Portfolio construction functions for the rolling backtest engine.

Implements maximum-profit and minimum-risk (minimum-variance) portfolio
construction, plus covariance estimation with singularity fallbacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, inv, pinv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_fallback(fallback: str, all_industries: list[str]) -> pd.Series:
    """
    Return a weight Series for the fallback allocation rule.

    Parameters
    ----------
    fallback : str
        'equal_weight' – uniform 1/N over all industries.
        'all_cash'     – zero allocation everywhere.
        'prev_holdings'– not supported here; must be resolved by the caller.
    all_industries : list[str]
        Full universe of industry names (defines index of returned Series).

    Returns
    -------
    pd.Series indexed by all_industries, dtype float64.
    """
    if fallback == "equal_weight":
        n = len(all_industries)
        return pd.Series(1.0 / n, index=all_industries, dtype=np.float64)
    elif fallback == "all_cash":
        return pd.Series(0.0, index=all_industries, dtype=np.float64)
    elif fallback == "prev_holdings":
        raise ValueError(
            "'prev_holdings' fallback must be resolved by the caller before "
            "invoking portfolio construction functions."
        )
    else:
        raise ValueError(f"Unknown fallback rule: {fallback!r}")


def _regularise_matrix(
    sigma: np.ndarray,
    method: str,
    alpha: float,
) -> np.ndarray:
    """
    Apply regularisation to a potentially singular covariance matrix.

    Parameters
    ----------
    sigma  : (n, n) numpy array – covariance matrix to regularise.
    method : str – one of 'shrinkage', 'pinv', 'diag_load'.
    alpha  : float – regularisation coefficient (used by shrinkage/diag_load).

    Returns
    -------
    Regularised (n, n) numpy array (for 'shrinkage' and 'diag_load').
    For 'pinv', returns the pseudo-inverse directly.
    """
    n = sigma.shape[0]
    if method == "shrinkage":
        # Ledoit–Wolf-style convex combination: shrink toward diagonal
        return (1.0 - alpha) * sigma + alpha * np.diag(np.diag(sigma))
    elif method == "diag_load":
        # Add alpha * trace(Sigma)/n to each diagonal element
        load = alpha * np.trace(sigma) / n
        return sigma + load * np.eye(n)
    elif method == "pinv":
        # Return pseudo-inverse directly; caller should skip a second inv()
        return sigma  # signal: will be handled in _safe_inv
    else:
        raise ValueError(f"Unknown cov_fallback method: {method!r}")


def _safe_inv(
    sigma: np.ndarray,
    cov_fallback: str,
    alpha: float,
) -> np.ndarray:
    """
    Invert a covariance matrix with automatic fallback on singularity.

    Returns the inverse (or pseudo-inverse) of sigma.
    """
    # First attempt: direct inverse
    try:
        sigma_inv = inv(sigma)
        if not np.all(np.isfinite(sigma_inv)):
            raise LinAlgError("Non-finite values after inversion.")
        return sigma_inv
    except LinAlgError:
        pass

    # Fallback
    if cov_fallback == "pinv":
        return pinv(sigma)
    else:
        sigma_reg = _regularise_matrix(sigma, cov_fallback, alpha)
        try:
            return inv(sigma_reg)
        except LinAlgError:
            # Last resort: pseudo-inverse of regularised matrix
            return pinv(sigma_reg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def max_profit_portfolio(
    scores: pd.Series,
    eligible: list[str],
    daily_returns: pd.DataFrame | None = None,
    fallback: str = "equal_weight",
    all_industries: list[str] | None = None,
) -> pd.Series:
    """
    Maximum-profit portfolio.

    Constructs weights by thresholding the combined score at zero and
    normalising over the eligible set:

        q_s = max(S_s, 0)          for s in eligible
        w_s = q_s / sum(q_j)

    Fallback is applied when ``eligible`` is empty **or** all q-values are
    zero (ASSUMPTION: documented in config – paper is silent on this case).

    Parameters
    ----------
    scores : pd.Series
        Combined score S_{s,t}^{(H)} indexed by industry name.
    eligible : list[str]
        Industry names that passed the probability threshold filter.
    fallback : str
        Rule to use when portfolio cannot be formed: 'equal_weight',
        'all_cash', or 'prev_holdings' (must be resolved by caller).
    all_industries : list[str] or None
        Full universe; defines the index of the returned Series.
        Defaults to scores.index if None.

    Returns
    -------
    pd.Series indexed by all_industries with float64 weights summing to 1
    (or 0 for 'all_cash'), zeros for non-eligible industries.
    """
    if all_industries is None:
        all_industries = list(scores.index)

    weights = pd.Series(0.0, index=all_industries, dtype=np.float64)

    if not eligible:
        return _apply_fallback(fallback, all_industries)

    q = scores.reindex(eligible).clip(lower=0.0).fillna(0.0).astype(np.float64)
    total = float(q.sum())

    if total == 0.0:
        return _apply_fallback(fallback, all_industries)

    for s in eligible:
        weights[s] = q[s] / total

    return weights


def min_risk_portfolio(
    scores: pd.Series,
    eligible: list[str],
    daily_returns: pd.DataFrame | None = None,
    fallback: str = "equal_weight",
    all_industries: list[str] | None = None,
    lookback: int = 252,
    cov_fallback: str = "shrinkage",
    cov_shrinkage_alpha: float = 0.1,
) -> pd.Series:
    """
    Minimum-risk (minimum-variance) long-only portfolio over the eligible set.

    Steps:
    1. Estimate covariance from ``daily_returns`` (or accept a pre-built matrix
       via the ``cov_matrix`` keyword kept for backward compatibility).
    2. Compute unconstrained MV weights:
           w̃ = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)
    3. Truncate negatives: w_s = max(w̃_s, 0).
    4. Renormalise: w_s = w_s / Σ w_j.

    Singularity fallbacks for Σ (selected via ``cov_fallback``):
    - 'shrinkage' : Σ_reg = (1-α)Σ + α·diag(Σ)
    - 'pinv'      : use NumPy pseudoinverse
    - 'diag_load' : Σ_reg = Σ + α·trace(Σ)/n · I

    The portfolio-level ``fallback`` is applied when ``eligible`` is empty
    or after truncation/renormalisation leaves all weights at zero.

    Parameters
    ----------
    scores : pd.Series
        Combined score S_{s,t}^{(H)} indexed by industry name (used only
        to derive ``all_industries`` when the parameter is None).
    eligible : list[str]
        Industry names in the eligible set.
    daily_returns : pd.DataFrame or None
        Wide-format daily returns (index=date, cols=industries).  The last
        ``lookback`` rows are used for covariance estimation.  If None,
        falls back to an identity covariance (equal-weight MV solution).
    fallback : str
        Portfolio fallback rule ('equal_weight' | 'all_cash' | 'prev_holdings').
    all_industries : list[str] or None
        Full universe index for the returned Series.
    lookback : int
        Number of trailing rows used to estimate Σ.
    cov_fallback : str
        Matrix regularisation method when Σ is singular.
    cov_shrinkage_alpha : float
        Regularisation coefficient α.

    Returns
    -------
    pd.Series indexed by all_industries with float64 weights, zeros for
    non-eligible industries.
    """
    if all_industries is None:
        all_industries = list(scores.index)

    weights = pd.Series(0.0, index=all_industries, dtype=np.float64)

    if not eligible:
        return _apply_fallback(fallback, all_industries)

    # Build covariance matrix from returns history
    if daily_returns is not None and not daily_returns.empty:
        cov_matrix = estimate_covariance(
            returns_history=daily_returns,
            eligible=eligible,
            lookback=lookback,
            fallback=cov_fallback,
            shrinkage_alpha=cov_shrinkage_alpha,
        )
    else:
        # Fallback: identity (→ equal-weight minimum-variance)
        n_el = len(eligible)
        cov_matrix = pd.DataFrame(
            np.eye(n_el, dtype=np.float64),
            index=eligible,
            columns=eligible,
        )

    sigma = cov_matrix.loc[eligible, eligible].values.astype(np.float64)
    n = len(eligible)
    ones = np.ones(n, dtype=np.float64)

    sigma_inv = _safe_inv(sigma, cov_fallback, cov_shrinkage_alpha)
    denom = float(ones @ sigma_inv @ ones)

    if denom == 0.0 or not np.isfinite(denom):
        return _apply_fallback(fallback, all_industries)

    wtilda = (sigma_inv @ ones) / denom
    w_raw = np.maximum(wtilda, 0.0)
    renorm_denom = float(w_raw.sum())

    if renorm_denom == 0.0:
        return _apply_fallback(fallback, all_industries)

    w_norm = w_raw / renorm_denom

    for i, s in enumerate(eligible):
        weights[s] = float(w_norm[i])

    return weights


def estimate_covariance(
    returns_history: pd.DataFrame,
    eligible: list[str],
    lookback: int,
    fallback: str = "shrinkage",
    shrinkage_alpha: float = 0.1,
) -> pd.DataFrame:
    """
    Estimate the covariance matrix of returns for the eligible industries.

    Uses the trailing ``lookback`` rows of ``returns_history`` (daily returns,
    wide format: index = date, columns = industries).  When the resulting
    sample covariance is not positive definite a regularisation fallback is
    applied.

    Parameters
    ----------
    returns_history : pd.DataFrame
        Wide-format daily returns (index=date, cols=industries).  Only the
        last ``lookback`` rows are used.
    eligible : list[str]
        Subset of column names to include in the covariance matrix.
    lookback : int
        Number of trailing trading days used for estimation.
    fallback : str
        Regularisation method if matrix is singular/non-PD:
        'shrinkage', 'diag_load', or 'pinv' (applied via diagonal loading
        before recomputing the covariance).
    shrinkage_alpha : float
        Regularisation coefficient.

    Returns
    -------
    pd.DataFrame of shape (len(eligible), len(eligible)), float64.
    """
    if not eligible:
        return pd.DataFrame(dtype=np.float64)

    # Select eligible columns and take trailing lookback rows
    avail = [c for c in eligible if c in returns_history.columns]
    subset = (
        returns_history[avail]
        .tail(lookback)
        .astype(np.float64)
    )

    cov = subset.cov()  # pandas handles NaN via pairwise complete obs
    # Reindex to ensure all eligible appear (missing → NaN → fill 0)
    cov = cov.reindex(index=eligible, columns=eligible).fillna(0.0)

    sigma = cov.values.copy()

    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(sigma)
    if np.all(eigenvalues > 0):
        return pd.DataFrame(sigma, index=eligible, columns=eligible, dtype=np.float64)

    # Apply fallback regularisation
    if fallback == "shrinkage":
        sigma_reg = _regularise_matrix(sigma, "shrinkage", shrinkage_alpha)
    elif fallback == "diag_load":
        sigma_reg = _regularise_matrix(sigma, "diag_load", shrinkage_alpha)
    else:
        # For 'pinv', still return a regularised PD matrix (pinv is for inversion)
        sigma_reg = _regularise_matrix(sigma, "shrinkage", shrinkage_alpha)

    return pd.DataFrame(sigma_reg, index=eligible, columns=eligible, dtype=np.float64)
