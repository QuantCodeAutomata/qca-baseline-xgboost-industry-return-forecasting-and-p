"""
Data loading module: downloads daily adjusted close prices for 25 U.S. industry
index proxies using ETF tickers via yfinance.

Replication note
----------------
The paper uses proprietary MSCI GICS industry-group data from 1992-2024 which is
not publicly available. This module uses U.S.-listed sector/industry ETFs as
publicly available proxies. Many ETFs launched in the late 1990s or 2000s, so
their time series are shorter than the paper's data; those columns are padded
with NaN (no forward-fill). Each deviation is logged at WARNING level.

The 25 distinct tickers below provide one unique series per industry. A few
industries (e.g. Consumer Services / Consumer Discretionary) share broad-sector
ETFs because no single narrow-industry ETF with sufficient history exists; this
is a documented deviation from the paper's 25 fully-independent series.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Industry → ticker mapping (25 distinct tickers)
# ---------------------------------------------------------------------------
# Each key is one of the 25 MSCI GICS industry-group names used in the paper.
# Tickers are U.S.-listed ETFs that are the closest publicly available proxy.
# DEVIATION: all tickers are ETFs or US equity proxies, not MSCI index returns.
INDUSTRY_TICKER_MAP: dict[str, str] = {
    "REITs":                                                    "IYR",
    "Utilities":                                                "XLU",
    "Food, Beverage and Tobacco":                               "PBJ",
    "Semiconductors":                                           "SOXX",
    "Consumer Staples":                                         "XLP",
    "Automobiles":                                              "CARZ",
    "Telecommunication":                                        "IYZ",
    "Household and Personal Products":                          "IDHD",
    "Banks":                                                    "KBE",
    "Energy":                                                   "XLE",
    "Technology Hardware and Equipment":                        "IGV",
    "Consumer Services":                                        "IYC",
    "Software and Services":                                    "XSW",
    "Pharmaceuticals, Biotechnology and Life Sciences":         "XBI",
    "Consumer Durables and Apparel":                            "XRT",
    "Materials":                                                "XLB",
    "Health Care Equipment and Services":                       "XLV",
    "Media and Entertainment":                                  "PBS",
    "Commercial and Professional Services":                     "IYJ",
    "Consumer Discretionary Distribution and Retail":           "FXD",
    "Financial Services":                                       "XLF",
    "Transportation":                                           "IYT",
    "Capital Goods":                                            "XLI",
    "Insurance":                                                "KIE",
    "Real Estate Management and Development":                   "XLRE",
}


def download_industry_prices(
    start: str = "1992-01-02",
    end: str = "2024-12-31",
    cache_path: str = "data/industry_prices.parquet",
) -> pd.DataFrame:
    """
    Download and align daily adjusted close prices for 25 industry proxy ETFs.

    Parameters
    ----------
    start : str
        First date to attempt to download (YYYY-MM-DD). ETFs launched after
        this date will have NaN for earlier rows.
    end : str
        Last date to include (inclusive, YYYY-MM-DD).
    cache_path : str
        Path to a Parquet cache file. If the file exists the download is
        skipped and prices are loaded from cache.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (trading days), columns = 25 industry names, values are
        float64 adjusted close prices. NaN where an ETF had not yet launched.
        No forward-filling is applied.

    Notes
    -----
    - yfinance ``end`` parameter is exclusive, so 1 calendar day is added
      internally to include the requested end date.
    - Logs a WARNING for every industry whose ETF data starts after *start*
      (replication deviation).
    """
    if os.path.exists(cache_path):
        logger.info("Loading industry prices from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    # yfinance end is exclusive — add one day so `end` is included
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    tickers = list(INDUSTRY_TICKER_MAP.values())
    logger.info(
        "Downloading %d tickers from %s to %s (inclusive) via yfinance …",
        len(tickers),
        start,
        end,
    )
    raw = yf.download(
        tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
    )

    # Extract adjusted close prices; result is a DataFrame keyed by ticker
    prices_raw: pd.DataFrame = raw["Close"].copy()
    prices_raw.columns.name = None          # drop 'Ticker' metadata label
    prices_raw.index.name = "date"
    prices_raw = prices_raw.astype("float64")

    # Map ticker columns → industry-name columns
    df = pd.DataFrame(index=prices_raw.index, dtype="float64")
    ref_start = pd.Timestamp(start)

    for industry, ticker in INDUSTRY_TICKER_MAP.items():
        if ticker in prices_raw.columns:
            df[industry] = prices_raw[ticker].values
        else:
            logger.warning(
                "[DEVIATION] Ticker %s for '%s' not found in downloaded data; "
                "column set to NaN.",
                ticker,
                industry,
            )
            df[industry] = np.nan

    # Log coverage deviations
    for industry in df.columns:
        first_valid = df[industry].first_valid_index()
        if first_valid is None:
            logger.warning(
                "[DEVIATION] '%s' (%s): no data available at all.",
                industry,
                INDUSTRY_TICKER_MAP[industry],
            )
        elif first_valid > ref_start:
            logger.warning(
                "[DEVIATION] '%s' (%s): data starts %s, not %s. "
                "Earlier rows are NaN.",
                industry,
                INDUSTRY_TICKER_MAP[industry],
                first_valid.date(),
                start,
            )
        else:
            logger.info(
                "'%s' (%s): data starts %s.",
                industry,
                INDUSTRY_TICKER_MAP[industry],
                first_valid.date(),
            )

    # Persist cache
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("Saved %d-row price DataFrame to cache: %s", len(df), cache_path)

    return df


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily returns: r_{s,t} = P_{s,t} / P_{s,t-1} - 1.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format price DataFrame (DatetimeIndex, columns = industry names).

    Returns
    -------
    pd.DataFrame
        Same shape as *prices*. First row is NaN for every column.
        No forward-filling. Uses float64.

    Notes
    -----
    NaN prices (e.g. before an ETF launch) propagate into NaN returns, so
    the first valid return for short-history ETFs will also be NaN.
    """
    returns = prices.pct_change(fill_method=None)
    return returns.astype("float64")


def compute_forward_returns(
    prices: pd.DataFrame,
    horizon: int,
    clip_lb: float = -0.80,
    clip_ub: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute H-day-ahead simple returns and binary direction targets.

    Forward return at date t: R_{s,t}^{(H)} = P_{s,t+H} / P_{s,t} - 1.
    Returns are clipped to [clip_lb, clip_ub] to remove extreme outliers.

    Binary target: Y_{s,t} = 1 if R_{s,t}^{(H)} > 0, else 0. NaN where the
    forward return is NaN.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format price DataFrame (DatetimeIndex, columns = industry names).
    horizon : int
        Number of trading days to look forward.
    clip_lb : float
        Lower clip bound for forward returns (default -0.80).
    clip_ub : float
        Upper clip bound for forward returns (default +0.80).

    Returns
    -------
    fwd_returns : pd.DataFrame
        Clipped H-day forward returns, same index as *prices*. Last *horizon*
        rows are NaN.
    binary_targets : pd.DataFrame
        Binary direction indicator (0/1/NaN), same index as *prices*.
    """
    fwd_prices = prices.shift(-horizon)
    fwd_returns = (fwd_prices / prices - 1).clip(lower=clip_lb, upper=clip_ub)
    fwd_returns = fwd_returns.astype("float64")

    binary_targets = (fwd_returns > 0).astype("float64")
    binary_targets[fwd_returns.isna()] = np.nan

    return fwd_returns, binary_targets
