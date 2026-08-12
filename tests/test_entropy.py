"""
Tests for src/entropy.py scalar entropy functions.

Verifies boundary conditions, mathematical properties, and the
monotonicity of fuzzy entropy with respect to its tolerance parameter r.
"""

import numpy as np
import pytest
import sys

sys.path.insert(0, "/workspace/project")

from src.entropy import fuzzy_entropy_scalar, shannon_entropy_scalar
from src.config import SHANNON_BINS, SHANNON_WINDOW

# ── Shannon entropy ────────────────────────────────────────────────────────────

def test_shannon_uniform():
    """Uniform distribution across all bins → entropy = log(SHANNON_BINS).

    np.linspace over [−0.05, 0.05] with SHANNON_WINDOW points guarantees
    each of the SHANNON_BINS equal-width bins receives the same count, giving
    the maximum discrete entropy log(K).
    """
    data = np.linspace(-0.05, 0.05, SHANNON_WINDOW)
    h = shannon_entropy_scalar(data, bins=SHANNON_BINS)
    expected = np.log(SHANNON_BINS)
    assert np.isclose(h, expected, atol=0.05), (
        f"Uniform entropy expected ≈ {expected:.4f} (log({SHANNON_BINS})), "
        f"got {h:.4f}"
    )


def test_shannon_degenerate():
    """All identical values → entropy = 0 (single occupied bin)."""
    data = np.ones(SHANNON_WINDOW) * 0.01
    h = shannon_entropy_scalar(data, bins=SHANNON_BINS)
    assert np.isclose(h, 0.0, atol=1e-10), (
        f"Degenerate (constant) series: expected entropy = 0, got {h:.8f}"
    )


def test_shannon_bounds():
    """Shannon entropy must be >= 0 for any input (non-negativity axiom)."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        data = rng.normal(0.0, 0.02, size=SHANNON_WINDOW)
        h = shannon_entropy_scalar(data, bins=SHANNON_BINS)
        assert h >= 0, (
            f"Negative entropy on trial {trial}: {h:.8f}"
        )


def test_shannon_exact():
    """Two equally-probable bins → entropy = log(2).

    Splitting SHANNON_WINDOW observations exactly in half into negative and
    positive returns (bins=2) gives p = 0.5 for each bin, so
    H = −2 × 0.5 × log(0.5) = log(2).
    """
    half = SHANNON_WINDOW // 2
    data = np.array([-0.01] * half + [0.01] * (SHANNON_WINDOW - half))
    h = shannon_entropy_scalar(data, bins=2)
    expected = np.log(2)
    assert np.isclose(h, expected, atol=1e-10), (
        f"Two-bin uniform entropy expected {expected:.8f} (log(2)), got {h:.8f}"
    )


def test_shannon_20_observations():
    """shannon_entropy_scalar must accept and process exactly SHANNON_WINDOW observations.

    This confirms the function is designed for the rolling window size used in
    add_entropy_features and raises no index errors with that exact length.
    """
    data = np.linspace(-0.04, 0.04, SHANNON_WINDOW)  # SHANNON_WINDOW = 20
    h = shannon_entropy_scalar(data, bins=SHANNON_BINS)
    assert np.isfinite(h), (
        f"Expected finite entropy for a {SHANNON_WINDOW}-observation window, got {h}"
    )


# ── Fuzzy entropy ─────────────────────────────────────────────────────────────

def test_fuzzy_positive():
    """Fuzzy entropy must be strictly positive for a varied (non-constant) series."""
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 0.02, size=100)
    fe = fuzzy_entropy_scalar(data)
    assert fe > 0, f"Expected FuzzyEn > 0 for random data, got {fe}"


def test_fuzzy_degenerate():
    """Constant series → FuzzyEn should be 0 or NaN (no complexity).

    With a perfectly flat signal all template-vector distances are zero;
    depending on implementation this yields FuzzyEn = 0 or NaN (log(1)=0
    or 0/0 in the Φ ratio).  Both are acceptable.
    """
    data = np.ones(63) * 0.005
    fe = fuzzy_entropy_scalar(data)
    is_zero = np.isclose(fe, 0.0, atol=1e-8)
    is_nan = np.isnan(fe)
    assert is_zero or is_nan, (
        f"Constant series: expected FuzzyEn = 0 or NaN, got {fe}"
    )


def test_fuzzy_larger_r_lower_entropy():
    """Increasing tolerance r must decrease (or preserve) FuzzyEn.

    A larger r means more template-vector pairs are classified as 'similar',
    which increases the similarity count and lowers the entropy estimate.
    """
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 0.02, size=120)
    std = np.std(data)

    fe_small = fuzzy_entropy_scalar(data, r=0.10 * std)
    fe_large = fuzzy_entropy_scalar(data, r=0.50 * std)

    # Allow a small numerical tolerance for floating-point artefacts.
    assert fe_large <= fe_small + 1e-6, (
        f"Expected FuzzyEn(r=0.5σ) ≤ FuzzyEn(r=0.1σ); "
        f"got FuzzyEn_large={fe_large:.6f} > FuzzyEn_small={fe_small:.6f}"
    )
