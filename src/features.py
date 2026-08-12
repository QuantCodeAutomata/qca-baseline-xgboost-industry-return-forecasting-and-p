"""
Baseline feature engineering for industry return prediction.

All features are causal: at date t only data observed on or before t is used.
Features that require a k-day lookback window will have NaN for the first k rows.
All outputs use float64.

Implemented from scratch with numpy / pandas — the ``ta`` library is not used.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────── Time-series features ────────────────────────────


def momentum_features(
    returns: pd.Series,
    windows: list[int] = [1, 5, 21, 63],
) -> pd.DataFrame:
    """
    Cumulative return (sum of daily log-approx returns) over past W days.

    At date t: mom_W = sum(r_{t-W+1}, …, r_t).
    Computed as a causal rolling sum with min_periods=W to avoid partial-window
    artifacts.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns for a single industry.
    windows : list[int]
        Lookback windows in trading days.

    Returns
    -------
    pd.DataFrame
        Columns ``mom_1``, ``mom_5``, ``mom_21``, ``mom_63`` (float64).
        First W rows are NaN for each window W.
    """
    result = {}
    for w in windows:
        result[f"mom_{w}"] = (
            returns.rolling(w, min_periods=w).sum().astype("float64")
        )
    return pd.DataFrame(result, index=returns.index)


def volatility_features(
    returns: pd.Series,
    windows: list[int] = [21, 63],
) -> pd.DataFrame:
    """
    Rolling standard deviation of daily returns over past W days.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns for a single industry.
    windows : list[int]
        Lookback windows in trading days.

    Returns
    -------
    pd.DataFrame
        Columns ``vol_21``, ``vol_63`` (float64).
        First W rows are NaN for each window W.
    """
    result = {}
    for w in windows:
        result[f"vol_{w}"] = (
            returns.rolling(w, min_periods=w).std().astype("float64")
        )
    return pd.DataFrame(result, index=returns.index)


def rolling_max_min_features(
    returns: pd.Series,
    window: int = 21,
) -> pd.DataFrame:
    """
    Rolling maximum and minimum of daily returns over the past *window* days.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns for a single industry.
    window : int
        Lookback window in trading days (default 21).

    Returns
    -------
    pd.DataFrame
        Columns ``roll_max_21``, ``roll_min_21`` (float64).
        First *window* rows are NaN.
    """
    roll = returns.rolling(window, min_periods=window)
    return pd.DataFrame(
        {
            f"roll_max_{window}": roll.max().astype("float64"),
            f"roll_min_{window}": roll.min().astype("float64"),
        },
        index=returns.index,
    )


def ma_ratio_features(
    prices: pd.Series,
    windows: list[int] = [10, 21, 50],
) -> pd.DataFrame:
    """
    Moving-average ratio: P_t / MA_m(t) − 1.

    The moving average is computed from prices *shifted by one day*
    (``prices.shift(1).rolling(m).mean()``) so it uses only prices
    P_{t-m}, …, P_{t-1} — strictly no look-ahead.

    Parameters
    ----------
    prices : pd.Series
        Daily adjusted close prices for a single industry.
    windows : list[int]
        MA window lengths in trading days.

    Returns
    -------
    pd.DataFrame
        Columns ``ma_ratio_10``, ``ma_ratio_21``, ``ma_ratio_50`` (float64).
        First m+1 rows are NaN for each window m.
    """
    shifted = prices.shift(1)
    result = {}
    for m in windows:
        lagged_ma = shifted.rolling(m, min_periods=m).mean()
        result[f"ma_ratio_{m}"] = (prices / lagged_ma - 1.0).astype("float64")
    return pd.DataFrame(result, index=prices.index)


def rsi_feature(
    returns: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Relative Strength Index using Wilder's exponential smoothing.

    Gains and losses are smoothed with EWM (alpha = 1 / window, adjust=False),
    which corresponds to Wilder's original specification.  When the smoothed
    average loss is zero, RSI is set to 100.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns for a single industry.
    window : int
        RSI period (default 14).

    Returns
    -------
    pd.Series
        Named ``rsi_14``, float64.  Values in [0, 100].
    """
    gains = returns.clip(lower=0.0)
    losses = (-returns).clip(lower=0.0)

    # com = window - 1 gives alpha = 1 / window (Wilder's smoothing)
    avg_gain = gains.ewm(com=window - 1, adjust=False, min_periods=1).mean()
    avg_loss = losses.ewm(com=window - 1, adjust=False, min_periods=1).mean()

    # When avg_loss == 0 all gains → RSI = 100; compute RS where possible
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # Restore RSI = 100 for rows where avg_loss was exactly 0
    rsi = rsi.where(avg_loss != 0.0, other=100.0)

    rsi.name = f"rsi_{window}"
    rsi = rsi.astype("float64")
    return rsi.to_frame()


def macd_features(
    returns: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence / Divergence indicator.

    MACD line  = EMA(fast) − EMA(slow)
    Signal line = EMA(MACD, signal_period)

    EMA computed with ``span`` and ``adjust=False`` (recursive formula).
    Prices are reconstructed from returns via (1 + r).cumprod() * 100 so
    the function is causal and self-contained.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns for a single industry.
    fast : int
        Fast EMA span (default 12).
    slow : int
        Slow EMA span (default 26).
    signal : int
        Signal EMA span applied to the MACD line (default 9).

    Returns
    -------
    pd.DataFrame
        Columns ``macd_line`` and ``macd_signal`` (float64).
    """
    prices = (1.0 + returns.fillna(0.0)).cumprod() * 100.0
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow).astype("float64")
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().astype("float64")
    return pd.DataFrame(
        {"macd_line": macd_line, "macd_signal": signal_line},
        index=returns.index,
    )


def bollinger_features(
    returns: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    Bollinger Band features based on a *window*-day rolling mean and std of
    reconstructed closing prices.

    - ``bb_zscore`` = (P_t − MA_t) / std_t  — normalised position within band.
    - ``bb_width``  = 2 × std_t / MA_t       — relative band width.

    When rolling std is zero (flat prices over the window), ``bb_zscore`` is
    set to 0 and ``bb_width`` to 0.  Prices are reconstructed from
    ``returns`` via (1 + r).cumprod() * 100 so the function is self-contained.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns for a single industry.
    window : int
        Rolling window in trading days (default 20).

    Returns
    -------
    pd.DataFrame
        Columns ``bb_zscore``, ``bb_width`` (float64).
        First *window*−1 rows are NaN.
    """
    prices = (1.0 + returns.fillna(0.0)).cumprod() * 100.0
    roll = prices.rolling(window, min_periods=window)
    ma = roll.mean()
    std = roll.std()

    flat_mask = std == 0.0
    std_safe = std.replace(0.0, np.nan)

    bb_zscore = ((prices - ma) / std_safe).where(~flat_mask, other=0.0)
    bb_width = (2.0 * std_safe / ma).where(~flat_mask, other=0.0)

    return pd.DataFrame(
        {
            "bollinger_zscore": bb_zscore.astype("float64"),
            "bb_width": bb_width.astype("float64"),
        },
        index=returns.index,
    )


def stochastic_k_feature(
    returns: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """
    Stochastic %K using the trailing *window*-day price range.

    %K = (P_t − min_{window}) / (max_{window} − min_{window}) × 100

    Prices are reconstructed from ``returns`` via cumulative product so that
    the function accepts either a returns series or can be driven from any
    numeric index series.  When max equals min (flat prices over the window)
    %K is set to 50.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns for a single industry.
    window : int
        Lookback window in trading days (default 14).

    Returns
    -------
    pd.DataFrame
        Column ``stoch_k``, float64, values in [0, 100].
        First *window*−1 rows are NaN.
    """
    # Reconstruct a price-level index from returns (base = 100)
    prices = (1.0 + returns.fillna(0.0)).cumprod() * 100.0

    roll = prices.rolling(window, min_periods=window)
    rolling_min = roll.min()
    rolling_max = roll.max()

    price_range = rolling_max - rolling_min
    flat_mask = price_range == 0.0
    range_safe = price_range.replace(0.0, np.nan)

    stoch_k = ((prices - rolling_min) / range_safe * 100.0).where(
        ~flat_mask, other=50.0
    )
    # Propagate NaN from the original returns (rows where returns was NaN)
    stoch_k = stoch_k.where(~returns.isna())
    return pd.DataFrame({"stoch_k": stoch_k.astype("float64")}, index=returns.index)


# ───────────────────────── Cross-sectional features ──────────────────────────


def cross_sectional_features(
    returns: pd.DataFrame,
    momentum_21: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cross-sectional z-score and fractional rank (scaled to [0, 1]) for the
    same-day daily return and the 21-day momentum, computed across all 25
    industries for every date.

    Z-score computation
    -------------------
    z = (x − cross_mean) / cross_std.  When cross_std == 0 for a given date
    (all industries have the same value), the z-score is set to 0 and the
    occurrence is logged.

    Rank computation
    ----------------
    Average-tie fractional rank scaled to [0, 1]:
    rank_scaled = (rank − 1) / (n_valid − 1), where rank uses average
    tie-breaking and n_valid is the number of non-NaN industries on that date.
    When only one industry is valid on a date, rank_scaled is NaN (undefined).

    Parameters
    ----------
    returns : pd.DataFrame
        Wide-format daily returns (DatetimeIndex × industries).
    momentum_21 : pd.DataFrame
        Wide-format 21-day momentum (same index × industries).

    Returns
    -------
    pd.DataFrame
        Long-format with columns:
        ``date``, ``industry``,
        ``cs_zscore_ret``, ``cs_rank_ret``,
        ``cs_zscore_mom21``, ``cs_rank_mom21``.
    """

    def _zscore(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Row-wise (cross-sectional) z-score."""
        cs_mean = df.mean(axis=1)
        cs_std = df.std(axis=1)

        zero_std = (cs_std == 0.0) & cs_std.notna()
        n_zero = int(zero_std.sum())
        if n_zero:
            logger.warning(
                "cross_sectional_features: std=0 on %d date(s) for '%s'; "
                "z-scores set to 0.",
                n_zero,
                label,
            )

        std_safe = cs_std.replace(0.0, np.nan)
        z = df.sub(cs_mean, axis=0).div(std_safe, axis=0)
        # Rows where std was 0: all z-scores become 0
        z.loc[zero_std, :] = 0.0
        return z.astype("float64")

    def _rank_scaled(df: pd.DataFrame) -> pd.DataFrame:
        """Row-wise fractional rank scaled to [0, 1]; average ties."""
        ranks = df.rank(axis=1, method="average", na_option="keep").astype("float64")
        n_valid = df.notna().sum(axis=1).astype("float64")
        # denom = n_valid - 1; replace 0 (only one valid value) with NaN
        denom = n_valid.sub(1.0).replace(0.0, np.nan)
        return ranks.sub(1.0, axis=0).div(denom, axis=0).astype("float64")

    # Compute for returns
    z_ret = _zscore(returns, "daily_return")
    r_ret = _rank_scaled(returns)

    # Compute for 21-day momentum
    z_mom = _zscore(momentum_21, "momentum_21")
    r_mom = _rank_scaled(momentum_21)

    # Stack each wide DataFrame to long format (preserve NaN rows).
    # future_stack=True adopts the pandas 2.1+ stack implementation which
    # preserves NaN values without a separate dropna argument.
    def _stack(wide: pd.DataFrame, col_name: str) -> pd.Series:
        stacked = wide.stack(future_stack=True)
        stacked.index.names = ["date", "industry"]
        stacked.name = col_name
        return stacked

    cs_long = pd.concat(
        [
            _stack(z_ret, "cs_zscore_ret"),
            _stack(r_ret, "cs_rank_ret"),
            _stack(z_mom, "cs_zscore_mom21"),
            _stack(r_mom, "cs_rank_mom21"),
        ],
        axis=1,
    )
    return cs_long.reset_index()


# ───────────────────────────── Main function ─────────────────────────────────


def compute_baseline_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all baseline features for the pooled (long-format) panel.

    Time-series features are computed independently for each of the 25
    industries and then stacked.  Cross-sectional features are computed
    row-wise across all industries after all time-series features are ready.

    All features are causal: only data available at or before date t is used.
    Features that require a k-day lookback have NaN for the first k rows of
    each industry.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format daily adjusted close prices
        (DatetimeIndex × 25 industry columns).
    returns : pd.DataFrame
        Wide-format daily simple returns, same shape as *prices*.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:
        ``date``, ``industry``,
        ``mom_1``, ``mom_5``, ``mom_21``, ``mom_63``,
        ``vol_21``, ``vol_63``,
        ``roll_max_21``, ``roll_min_21``,
        ``ma_ratio_10``, ``ma_ratio_21``, ``ma_ratio_50``,
        ``rsi_14``,
        ``macd_line``, ``macd_signal``,
        ``bollinger_zscore``, ``bb_width``,
        ``stoch_k``,
        ``cs_zscore_ret``, ``cs_rank_ret``,
        ``cs_zscore_mom21``, ``cs_rank_mom21``.

        Rows: len(prices) × 25 (one row per date-industry pair).
    """
    industry_names = prices.columns.tolist()

    ts_panels: list[pd.DataFrame] = []
    mom_21_dict: dict[str, pd.Series] = {}

    for industry in industry_names:
        p = prices[industry]
        r = returns[industry]

        mom_df = momentum_features(r)
        vol_df = volatility_features(r)
        rollmm_df = rolling_max_min_features(r)
        ma_df = ma_ratio_features(p)
        rsi_s = rsi_feature(r)
        macd_df = macd_features(r)
        bb_df = bollinger_features(r)
        stoch_s = stochastic_k_feature(r)

        # Store mom_21 for the cross-sectional step
        mom_21_dict[industry] = mom_df["mom_21"]

        industry_df = pd.concat(
            [mom_df, vol_df, rollmm_df, ma_df, rsi_s, macd_df, bb_df, stoch_s],
            axis=1,
        )
        industry_df.index.name = "date"
        industry_df = industry_df.reset_index()
        industry_df.insert(1, "industry", industry)
        ts_panels.append(industry_df)

    panel = pd.concat(ts_panels, ignore_index=True)

    # ── Cross-sectional features ───────────────────────────────────────────
    mom_21_wide = pd.DataFrame(mom_21_dict, index=prices.index)
    cs_df = cross_sectional_features(returns, mom_21_wide)

    panel = panel.merge(cs_df, on=["date", "industry"], how="left")

    # ── Log NaN counts per feature column ─────────────────────────────────
    feature_cols = [c for c in panel.columns if c not in ("date", "industry")]
    total_rows = len(panel)
    for col in feature_cols:
        n_nan = int(panel[col].isna().sum())
        if n_nan:
            logger.info(
                "Feature '%s': %d NaN rows (%.1f%% of panel).",
                col,
                n_nan,
                100.0 * n_nan / total_rows,
            )

    logger.info(
        "compute_baseline_features: panel shape %s, %d feature columns.",
        panel.shape,
        len(feature_cols),
    )
    return panel
