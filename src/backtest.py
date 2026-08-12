"""
Rolling-window backtest engine.

Implements the full estimation → prediction → portfolio → realised-return
loop described in the paper.  All paper-underspecified choices are exposed
as explicit fields of BacktestConfig and documented as ASSUMPTIONS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from src.portfolio import max_profit_portfolio, min_risk_portfolio


# ---------------------------------------------------------------------------
# Configuration and diagnostic containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Complete configuration for one backtest run.

    All under-specified paper items are surfaced as explicit fields so that
    every experiment is fully reproducible from its config alone.

    Parameters
    ----------
    horizon : int
        Forecast/holding horizon H ∈ {5, 21, 63} trading days.
    feature_set : str
        Feature group used: 'baseline' | 'shannon' | 'fuzzy'.
    portfolio_type : str
        Portfolio construction rule: 'max_profit' | 'min_risk'.
    rolling_window : int
        Total estimation window length in trading days.
        ASSUMPTION: 252 × 5 ≈ 5 years (paper does not specify).
    train_frac : float
        Fraction of estimation window used for training (rest → validation).
        ASSUMPTION: 0.80 (paper is silent).
    min_train_obs : int
        Minimum number of panel rows required in training block.
        If not met, the decision date is skipped.
    xgb_clf_params : dict
        Keyword arguments forwarded to XGBClassifier.
    xgb_reg_params : dict
        Keyword arguments forwarded to XGBRegressor.
    ensemble_weight_rule : str
        How to combine classifier and regressor outputs:
        'auc_mse' – w_cls = AUC / (AUC + 1/(MSE + ε));
        'equal'   – w_cls = w_reg = 0.5.
        ASSUMPTION: 'auc_mse' (paper is silent).
    cov_lookback : int
        Trailing trading days used for covariance estimation.
        ASSUMPTION: 252 × 3 ≈ 3 years (paper is silent).
    cov_fallback : str
        Regularisation when covariance matrix is singular:
        'shrinkage' | 'pinv' | 'diag_load'.
    cov_shrinkage_alpha : float
        Regularisation coefficient α for covariance fallback.
    clip_lb : float
        Lower clip bound for target_reg (training targets).
    clip_ub : float
        Upper clip bound for target_reg (training targets).
    empty_eligible_fallback : str
        Portfolio fallback when eligible set is empty:
        'equal_weight' | 'all_cash'.
        ASSUMPTION: 'equal_weight' (paper is silent).
    random_state : int
        Master random seed used for XGBoost models if not overridden by
        xgb_clf_params / xgb_reg_params.
    """

    horizon: int
    feature_set: str
    portfolio_type: str
    rolling_window: int = 252 * 5
    train_frac: float = 0.80
    min_train_obs: int = 1000
    xgb_clf_params: dict = field(default_factory=dict)
    xgb_reg_params: dict = field(default_factory=dict)
    ensemble_weight_rule: str = "auc_mse"
    cov_lookback: int = 252 * 3
    cov_fallback: str = "shrinkage"
    cov_shrinkage_alpha: float = 0.1
    clip_lb: float = -0.80
    clip_ub: float = 0.80
    empty_eligible_fallback: str = "equal_weight"
    random_state: int = 42


@dataclass
class RebalanceRecord:
    """
    Per-rebalance-date diagnostic record.

    Captures every modelling decision and realised outcome so that the
    backtest is fully auditable post-hoc.

    Parameters
    ----------
    decision_date : pd.Timestamp
        Date on which the portfolio is formed.
    horizon : int
        Holding horizon H.
    train_start / train_end : pd.Timestamp
        First / last date of the training block.
    val_start / val_end : pd.Timestamp
        First / last date of the validation block.
    n_train_obs : int
        Number of (date × industry) rows in training block.
    n_val_obs : int
        Number of (date × industry) rows in validation block.
    tau : float
        Selected directional probability threshold (chosen to maximise
        validation F1 over the grid [0.30, 0.35, …, 0.75]).
    w_cls : float
        Ensemble weight assigned to the classifier output.
    w_reg : float
        Ensemble weight assigned to the regressor output.
    val_accuracy / val_precision / val_recall / val_f1 : float
        Validation classification metrics computed at selected tau.
    val_auc : float
        Validation ROC-AUC (0.5 fallback when only one class is present).
    val_mse : float
        Validation mean-squared-error of the regressor.
    pred_ret_mean : float
        Mean of regressor predictions on the validation set (used for Z).
    pred_ret_std : float
        Std  of regressor predictions on the validation set (used for Z).
    eligible_set : list[str]
        Industries that passed the probability threshold at decision_date.
    weights : dict[str, float]
        Portfolio weights at decision_date.
    realized_return : float
        Realised H-day holding-period portfolio return (NaN if unavailable).
    fallback_used : bool
        True if any fallback rule was triggered during portfolio construction
        or Z-standardisation.
    """

    decision_date: pd.Timestamp
    horizon: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_train_obs: int
    n_val_obs: int
    tau: float
    w_cls: float
    w_reg: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_auc: float
    val_mse: float
    pred_ret_mean: float
    pred_ret_std: float
    eligible_set: list[str]
    weights: dict[str, float]
    realized_return: float
    fallback_used: bool


# ---------------------------------------------------------------------------
# Tau grid (paper specifies scanning F-score; grid is an ASSUMPTION)
# ---------------------------------------------------------------------------

_TAU_GRID: np.ndarray = np.arange(0.30, 0.76, 0.05)   # [0.30 … 0.75], 10 points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_tau(
    y_true_cls: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """
    Select the probability threshold τ that maximises F1 on validation data.

    Scans the grid [0.30, 0.35, …, 0.75].  Ties are broken by choosing the
    first (smallest) τ that achieves the maximum.  Returns 0.50 if no τ
    yields a positive F1 (degenerate validation fold).

    Parameters
    ----------
    y_true_cls : np.ndarray, shape (n,)
        True binary class labels.
    y_pred_proba : np.ndarray, shape (n,)
        Predicted probabilities of the positive class.

    Returns
    -------
    float – selected threshold τ.
    """
    best_tau = 0.50
    best_f1 = 0.0     # only update when F1 is strictly positive; 0.50 is the fallback
    for tau in _TAU_GRID:
        y_pred = (y_pred_proba > tau).astype(int)
        score = f1_score(y_true_cls, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)
    return best_tau


def _ensemble_weights(
    val_auc: float,
    val_mse: float,
    rule: str,
) -> tuple[float, float]:
    """
    Compute classifier / regressor ensemble weights.

    Rules:
    - 'equal':   w_cls = w_reg = 0.5
    - 'auc_mse': w_cls = AUC / (AUC + 1 / (MSE + ε))
                 w_reg = 1 − w_cls
      Special case: if MSE < ε (extremely small / overfit regressor), set
      w_reg = ε to guard against degenerate ensemble (ASSUMPTION: treating
      near-zero MSE as a reliability signal against the regressor).

    Parameters
    ----------
    val_auc : float
        Validation AUC of the classifier.
    val_mse : float
        Validation MSE of the regressor.
    rule : str
        'equal' or 'auc_mse'.

    Returns
    -------
    (w_cls, w_reg) both float64.
    """
    _EPS = 1e-8
    if rule == "equal":
        return 0.5, 0.5

    # auc_mse rule
    if val_mse < _EPS:
        # Regressor MSE is suspiciously small – cap its weight near zero
        return 1.0 - _EPS, _EPS

    reg_score = 1.0 / (val_mse + _EPS)
    denom = val_auc + reg_score
    if denom == 0.0:
        return 0.5, 0.5
    w_cls = float(val_auc / denom)
    w_reg = 1.0 - w_cls
    return w_cls, w_reg


def compute_realized_returns(
    decision_dates: list[pd.Timestamp],
    weights_by_date: dict[pd.Timestamp, pd.Series],
    prices: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """
    Compute realised holding-period portfolio returns.

    For each decision date t:
        R_port = Σ_s  w_s × (P_{s, t+H} / P_{s, t} − 1)

    Prices are looked up by integer position within the prices index so that
    *exactly* H trading days elapse regardless of calendar gaps.  Industries
    with missing prices on either leg contribute zero return.

    Parameters
    ----------
    decision_dates : list[pd.Timestamp]
        Ordered list of portfolio formation dates.
    weights_by_date : dict[pd.Timestamp, pd.Series]
        Portfolio weights (indexed by industry) per decision date.
    prices : pd.DataFrame
        Wide-format price panel (index=trading date, cols=industries).
    horizon : int
        Number of trading days in the holding period.

    Returns
    -------
    pd.Series of float64 realised returns indexed by decision date.
    """
    prices_idx = prices.index
    prices_loc: dict[pd.Timestamp, int] = {d: i for i, d in enumerate(prices_idx)}
    n_prices = len(prices_idx)

    results: dict[pd.Timestamp, float] = {}
    for t in decision_dates:
        if t not in prices_loc:
            results[t] = np.nan
            continue
        t_i = prices_loc[t]
        t_H_i = t_i + horizon
        if t_H_i >= n_prices:
            results[t] = np.nan
            continue

        P_t = prices.iloc[t_i]
        P_tH = prices.iloc[t_H_i]
        fwd_ret = (P_tH / P_t - 1.0).fillna(0.0)

        w = weights_by_date[t]
        r_port = float((w * fwd_ret.reindex(w.index).fillna(0.0)).sum())
        results[t] = r_port

    return pd.Series(results, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig,
    feature_cols: list[str],
    industry_names: list[str],
    verbose: bool = True,
) -> tuple[list[RebalanceRecord], pd.Series]:
    """
    Run the full rolling-window backtest.

    Algorithm
    ---------
    1. Sort panel by (date, industry).
    2. Build date index and step through decision dates in increments of H
       (non-overlapping holding periods).
    3. At each decision date t:
       a. Extract rolling estimation window (last ``rolling_window`` unique
          dates strictly before t in panel).
       b. Split into train (first ``train_frac``) and validation (remainder)
          by unique date.
       c. Fit XGBClassifier on training rows.
       d. Fit XGBRegressor on training rows.
       e. Predict on validation → select τ via F1 grid, compute metrics,
          derive ensemble weights (w_cls, w_reg).
       f. Predict on current cross-section at date t.
       g. Standardise predicted returns: Z_s = (ŷ_s − μ_val) / σ_val.
          If σ_val = 0 set Z = 0 (logged as fallback).
       h. Compute combined score S_s = w_cls × p_s + w_reg × Z_s.
       i. Eligibility filter: eligible = {s : p_s > τ}.
       j. Construct portfolio (max_profit or min_risk).
       k. Record realised forward return from prices.
    4. Return diagnostic records and hp-return Series.

    Data-leakage guardrails
    -----------------------
    - Estimation window uses only dates strictly before t.
    - Train block is the chronologically *earlier* portion of the window.
    - Validation block is the chronologically *later* portion (but still
      before t); its targets are known at t because they are H-day forward
      returns already computed and stored in the panel.
    - Covariance estimation uses returns up to and including t (daily returns
      on day t use prices at t and t−1, both known at t).

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel with columns:
        ['date', 'industry', *feature_cols, 'target_cls', 'target_reg'].
        Sorted by (date, industry) before use.
    prices : pd.DataFrame
        Wide-format price panel (index=trading date, cols=industry names).
        Used exclusively for realised forward-return computation (NOT clipped).
    returns : pd.DataFrame
        Wide-format daily-return panel (index=trading date, cols=industries).
        Used for covariance estimation.
    config : BacktestConfig
        Full experiment configuration.
    feature_cols : list[str]
        Names of the predictor columns in ``panel``.
    industry_names : list[str]
        Full universe of 25 industry names (defines portfolio index).
    verbose : bool
        If True, print progress every 20 decision dates.

    Returns
    -------
    records : list[RebalanceRecord]
        One record per successfully processed decision date.
    hp_returns : pd.Series
        Realised holding-period portfolio returns indexed by decision date.
    """
    # ── 0. Pre-process panel ──────────────────────────────────────────────────
    panel = panel.sort_values(["date", "industry"]).reset_index(drop=True)

    # Build fast date → row-index mapping
    # groupby preserves sort order because panel is already sorted
    _grp = panel.groupby("date", sort=True)
    date_to_rows: dict[pd.Timestamp, np.ndarray] = {
        k: v.values for k, v in _grp.groups.items()
    }
    all_panel_dates: list[pd.Timestamp] = sorted(date_to_rows.keys())
    n_panel_dates = len(all_panel_dates)

    if n_panel_dates <= config.rolling_window:
        raise ValueError(
            f"Panel has only {n_panel_dates} unique dates, but rolling_window "
            f"is {config.rolling_window}.  Insufficient data."
        )

    # Pre-extract numpy arrays for fast column access
    feature_arr = panel[feature_cols].values.astype(np.float64)
    target_cls_arr = panel["target_cls"].values.astype(np.float64)
    target_reg_arr = panel["target_reg"].values.astype(np.float64)
    industry_arr = panel["industry"].values  # object dtype (strings)

    # Prices index lookup
    prices_loc: dict[pd.Timestamp, int] = {d: i for i, d in enumerate(prices.index)}
    n_prices = len(prices.index)

    # ── 1. Enumerate decision dates ───────────────────────────────────────────
    # First valid index: rolling_window (so panel_dates[0:rolling_window] are available)
    # Step by horizon H; stop when t+H is beyond the prices index.
    decision_indices: list[int] = []
    idx = config.rolling_window
    while idx < n_panel_dates:
        t = all_panel_dates[idx]
        if t in prices_loc and (prices_loc[t] + config.horizon) < n_prices:
            decision_indices.append(idx)
        idx += config.horizon

    if not decision_indices:
        raise ValueError(
            "No valid decision dates found.  Check rolling_window, horizon, "
            "and the date coverage of panel and prices."
        )

    if verbose:
        print(
            f"[Backtest] horizon={config.horizon}, "
            f"portfolio={config.portfolio_type}, "
            f"features={config.feature_set}"
        )
        print(
            f"[Backtest] {len(decision_indices)} decision dates: "
            f"{all_panel_dates[decision_indices[0]].date()} → "
            f"{all_panel_dates[decision_indices[-1]].date()}"
        )

    # ── 2. XGBoost parameter dicts (merge user params with random_state) ──────
    clf_params: dict = {**config.xgb_clf_params}
    clf_params.setdefault("random_state", config.random_state)

    reg_params: dict = {**config.xgb_reg_params}
    reg_params.setdefault("random_state", config.random_state)

    # ── 3. Main loop ──────────────────────────────────────────────────────────
    records: list[RebalanceRecord] = []
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}

    for loop_i, t_idx in enumerate(decision_indices):
        t = all_panel_dates[t_idx]

        # ── 3a. Estimation window (strictly before t) ─────────────────────
        window_dates = all_panel_dates[
            max(0, t_idx - config.rolling_window) : t_idx
        ]
        n_window = len(window_dates)
        n_train_dates = int(n_window * config.train_frac)

        if n_train_dates == 0:
            if verbose:
                print(f"  [{t.date()}] Skipped: empty training date set.")
            continue

        train_dates = window_dates[:n_train_dates]
        val_dates = window_dates[n_train_dates:]

        if not val_dates:
            if verbose:
                print(f"  [{t.date()}] Skipped: empty validation date set.")
            continue

        # ── 3b. Row index assembly ────────────────────────────────────────
        train_row_idx = np.concatenate(
            [date_to_rows[d] for d in train_dates if d in date_to_rows]
        )
        val_row_idx = np.concatenate(
            [date_to_rows[d] for d in val_dates if d in date_to_rows]
        )
        current_row_idx = date_to_rows.get(t, np.array([], dtype=np.intp))

        if len(train_row_idx) < config.min_train_obs:
            if verbose:
                print(
                    f"  [{t.date()}] Skipped: only {len(train_row_idx)} "
                    f"training rows (min={config.min_train_obs})."
                )
            continue

        if len(val_row_idx) == 0:
            if verbose:
                print(f"  [{t.date()}] Skipped: no validation rows.")
            continue

        if len(current_row_idx) == 0:
            if verbose:
                print(f"  [{t.date()}] Skipped: no cross-section data at t.")
            continue

        # ── 3c. Slice arrays ──────────────────────────────────────────────
        X_train_raw = feature_arr[train_row_idx]
        y_train_cls_raw = target_cls_arr[train_row_idx]
        y_train_reg_raw = target_reg_arr[train_row_idx]

        X_val_raw = feature_arr[val_row_idx]
        y_val_cls_raw = target_cls_arr[val_row_idx]
        y_val_reg_raw = target_reg_arr[val_row_idx]

        X_current = feature_arr[current_row_idx]
        industries_current = industry_arr[current_row_idx]

        # Drop rows with NaN targets (last H rows of sample have no forward return).
        # XGBoost cannot train on NaN labels; feature NaN is handled natively.
        train_valid = ~(np.isnan(y_train_cls_raw) | np.isnan(y_train_reg_raw))
        X_train = X_train_raw[train_valid]
        y_train_cls = y_train_cls_raw[train_valid].astype(int)
        y_train_reg = y_train_reg_raw[train_valid]

        val_valid = ~(np.isnan(y_val_cls_raw) | np.isnan(y_val_reg_raw))
        X_val = X_val_raw[val_valid]
        y_val_cls = y_val_cls_raw[val_valid].astype(int)
        y_val_reg = y_val_reg_raw[val_valid]

        if len(X_train) < config.min_train_obs:
            if verbose:
                print(
                    f"  [{t.date()}] Skipped: only {len(X_train)} valid training "
                    f"rows after NaN-target drop (min={config.min_train_obs})."
                )
            continue

        if len(X_val) == 0:
            if verbose:
                print(f"  [{t.date()}] Skipped: no valid validation rows after NaN-target drop.")
            continue

        # ── 3d. Fit XGBClassifier ─────────────────────────────────────────
        clf = XGBClassifier(**clf_params)
        clf.fit(X_train, y_train_cls)

        # ── 3e. Fit XGBRegressor ──────────────────────────────────────────
        reg = XGBRegressor(**reg_params)
        reg.fit(X_train, y_train_reg)

        # ── 3f. Validation predictions ────────────────────────────────────
        p_val: np.ndarray = clf.predict_proba(X_val)[:, 1]
        pred_ret_val: np.ndarray = reg.predict(X_val).astype(np.float64)

        # ── 3g. Threshold τ selection via F1 grid search ──────────────────
        tau = _select_tau(y_val_cls, p_val)

        # ── 3h. Validation classification metrics at selected τ ───────────
        y_pred_cls_val = (p_val > tau).astype(int)
        val_accuracy = float(accuracy_score(y_val_cls, y_pred_cls_val))
        val_precision = float(precision_score(y_val_cls, y_pred_cls_val, zero_division=0))
        val_recall = float(recall_score(y_val_cls, y_pred_cls_val, zero_division=0))
        val_f1 = float(f1_score(y_val_cls, y_pred_cls_val, zero_division=0))

        # AUC – fallback to 0.5 when only one class present in validation
        unique_val_classes = np.unique(y_val_cls)
        if len(unique_val_classes) < 2:
            val_auc = 0.5
        else:
            val_auc = float(roc_auc_score(y_val_cls, p_val))

        val_mse = float(mean_squared_error(y_val_reg, pred_ret_val))

        # ── 3i. Ensemble weights ──────────────────────────────────────────
        w_cls, w_reg = _ensemble_weights(val_auc, val_mse, config.ensemble_weight_rule)

        # ── 3j. Predict on current cross-section ──────────────────────────
        p_current: np.ndarray = clf.predict_proba(X_current)[:, 1]
        pred_ret_current: np.ndarray = reg.predict(X_current).astype(np.float64)

        # ── 3k. Z-standardise predicted returns ───────────────────────────
        pred_ret_val_mean = float(pred_ret_val.mean())
        pred_ret_val_std = float(pred_ret_val.std())

        fallback_used = False
        if pred_ret_val_std == 0.0:
            Z_current = np.zeros(len(pred_ret_current), dtype=np.float64)
            fallback_used = True
        else:
            Z_current = (pred_ret_current - pred_ret_val_mean) / pred_ret_val_std

        # ── 3l. Combined score S_s = w_cls * p_s + w_reg * Z_s ───────────
        S_current = w_cls * p_current + w_reg * Z_current

        # Map predictions to a full-universe Series (unobserved → 0 / 0.0)
        scores = pd.Series(0.0, index=industry_names, dtype=np.float64)
        probas = pd.Series(0.0, index=industry_names, dtype=np.float64)
        for k, ind in enumerate(industries_current):
            if ind in scores.index:
                scores[ind] = float(S_current[k])
                probas[ind] = float(p_current[k])

        # ── 3m. Eligibility filter ────────────────────────────────────────
        eligible_set = [s for s in industry_names if probas[s] > tau]

        if not eligible_set:
            fallback_used = True

        # ── 3n. Portfolio construction ────────────────────────────────────
        if config.portfolio_type == "max_profit":
            # Check for pre-construction fallback condition
            if eligible_set:
                q_sum = float(
                    scores.reindex(eligible_set).clip(lower=0.0).fillna(0.0).sum()
                )
                if q_sum == 0.0:
                    fallback_used = True

            portfolio_weights = max_profit_portfolio(
                scores=scores,
                eligible=eligible_set,
                fallback=config.empty_eligible_fallback,
                all_industries=industry_names,
            )

        elif config.portfolio_type == "min_risk":
            returns_to_date = returns.loc[returns.index <= t]
            portfolio_weights = min_risk_portfolio(
                scores=scores,
                eligible=eligible_set,
                daily_returns=returns_to_date,
                fallback=config.empty_eligible_fallback,
                all_industries=industry_names,
                lookback=config.cov_lookback,
                cov_fallback=config.cov_fallback,
                cov_shrinkage_alpha=config.cov_shrinkage_alpha,
            )
        else:
            raise ValueError(
                f"Unknown portfolio_type: {config.portfolio_type!r}. "
                "Expected 'max_profit' or 'min_risk'."
            )

        # ── 3o. Realised forward return ───────────────────────────────────
        if t in prices_loc:
            t_p_idx = prices_loc[t]
            t_H_p_idx = t_p_idx + config.horizon
            if t_H_p_idx < n_prices:
                P_t = prices.iloc[t_p_idx]
                P_tH = prices.iloc[t_H_p_idx]
                fwd_ret = (P_tH / P_t - 1.0).fillna(0.0)
                realized_return = float(
                    (
                        portfolio_weights
                        * fwd_ret.reindex(portfolio_weights.index).fillna(0.0)
                    ).sum()
                )
            else:
                realized_return = np.nan
        else:
            realized_return = np.nan

        # ── 3p. Record ────────────────────────────────────────────────────
        record = RebalanceRecord(
            decision_date=t,
            horizon=config.horizon,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            val_start=val_dates[0],
            val_end=val_dates[-1],
            n_train_obs=int(len(X_train)),  # after NaN-target filter
            n_val_obs=int(len(X_val)),      # after NaN-target filter
            tau=tau,
            w_cls=w_cls,
            w_reg=w_reg,
            val_accuracy=val_accuracy,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1=val_f1,
            val_auc=val_auc,
            val_mse=val_mse,
            pred_ret_mean=pred_ret_val_mean,
            pred_ret_std=pred_ret_val_std,
            eligible_set=eligible_set,
            weights=portfolio_weights.to_dict(),
            realized_return=realized_return,
            fallback_used=fallback_used,
        )
        records.append(record)
        weights_by_date[t] = portfolio_weights

        if verbose and (loop_i + 1) % 20 == 0:
            print(
                f"  [{t.date()}] #{loop_i + 1}/{len(decision_indices)} "
                f"tau={tau:.2f} w_cls={w_cls:.3f} "
                f"eligible={len(eligible_set)} "
                f"ret={realized_return:.4f}"
            )

    # ── 4. Aggregate hp-returns ───────────────────────────────────────────────
    if records:
        hp_returns = pd.Series(
            {r.decision_date: r.realized_return for r in records},
            dtype=np.float64,
            name="hp_return",
        )
    else:
        hp_returns = pd.Series(dtype=np.float64, name="hp_return")

    if verbose:
        print(
            f"[Backtest] Done. {len(records)} rebalance records. "
            f"Mean HP return: {hp_returns.mean():.4f}"
        )

    return records, hp_returns


def run_backtest_multi_portfolio(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig,
    feature_cols: list[str],
    industry_names: list[str],
    portfolio_types: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, tuple[list[RebalanceRecord], pd.Series]]:
    """
    Efficiency-optimised variant: train XGBoost once per decision date and
    construct *all* requested portfolio types from the same predictions.

    This halves the XGBoost training cost relative to calling ``run_backtest``
    separately for each portfolio type.

    Parameters
    ----------
    panel : pd.DataFrame
        Same spec as ``run_backtest``.
    prices : pd.DataFrame
        Wide-format price panel.
    returns : pd.DataFrame
        Wide-format daily-return panel.
    config : BacktestConfig
        Configuration (``portfolio_type`` field is ignored here; all types in
        ``portfolio_types`` are used).
    feature_cols : list[str]
        Predictor column names.
    industry_names : list[str]
        Full 25-industry universe.
    portfolio_types : list[str] | None
        Portfolio types to construct.  Defaults to ['max_profit', 'min_risk'].
    verbose : bool
        Progress logging.

    Returns
    -------
    dict mapping portfolio_type → (records, hp_returns)
    """
    if portfolio_types is None:
        portfolio_types = ["max_profit", "min_risk"]

    # ── 0. Pre-process panel ──────────────────────────────────────────────────
    panel = panel.sort_values(["date", "industry"]).reset_index(drop=True)

    _grp = panel.groupby("date", sort=True)
    date_to_rows: dict[pd.Timestamp, np.ndarray] = {
        k: v.values for k, v in _grp.groups.items()
    }
    all_panel_dates: list[pd.Timestamp] = sorted(date_to_rows.keys())
    n_panel_dates = len(all_panel_dates)

    if n_panel_dates <= config.rolling_window:
        raise ValueError(
            f"Panel has only {n_panel_dates} unique dates, but rolling_window "
            f"is {config.rolling_window}.  Insufficient data."
        )

    feature_arr = panel[feature_cols].values.astype(np.float64)
    target_cls_arr = panel["target_cls"].values.astype(np.float64)
    target_reg_arr = panel["target_reg"].values.astype(np.float64)
    industry_arr = panel["industry"].values

    prices_loc: dict[pd.Timestamp, int] = {d: i for i, d in enumerate(prices.index)}
    n_prices = len(prices.index)

    # ── 1. Enumerate decision dates ───────────────────────────────────────────
    decision_indices: list[int] = []
    idx = config.rolling_window
    while idx < n_panel_dates:
        t = all_panel_dates[idx]
        if t in prices_loc and (prices_loc[t] + config.horizon) < n_prices:
            decision_indices.append(idx)
        idx += config.horizon

    if not decision_indices:
        raise ValueError("No valid decision dates found.")

    if verbose:
        print(
            f"[MultiBacktest] horizon={config.horizon}, "
            f"portfolios={portfolio_types}"
        )
        print(
            f"[MultiBacktest] {len(decision_indices)} decision dates: "
            f"{all_panel_dates[decision_indices[0]].date()} → "
            f"{all_panel_dates[decision_indices[-1]].date()}"
        )

    clf_params: dict = {**config.xgb_clf_params}
    clf_params.setdefault("random_state", config.random_state)
    reg_params: dict = {**config.xgb_reg_params}
    reg_params.setdefault("random_state", config.random_state)

    # ── 2. Per-portfolio accumulators ─────────────────────────────────────────
    all_records: dict[str, list[RebalanceRecord]] = {pt: [] for pt in portfolio_types}

    # ── 3. Main loop ──────────────────────────────────────────────────────────
    for loop_i, t_idx in enumerate(decision_indices):
        t = all_panel_dates[t_idx]

        window_dates = all_panel_dates[max(0, t_idx - config.rolling_window): t_idx]
        n_window = len(window_dates)
        n_train_dates = int(n_window * config.train_frac)

        if n_train_dates == 0:
            continue

        train_dates = window_dates[:n_train_dates]
        val_dates = window_dates[n_train_dates:]

        if not val_dates:
            continue

        train_row_idx = np.concatenate(
            [date_to_rows[d] for d in train_dates if d in date_to_rows]
        )
        val_row_idx = np.concatenate(
            [date_to_rows[d] for d in val_dates if d in date_to_rows]
        )
        current_row_idx = date_to_rows.get(t, np.array([], dtype=np.intp))

        if len(current_row_idx) == 0:
            continue

        X_train_raw = feature_arr[train_row_idx]
        y_train_cls_raw = target_cls_arr[train_row_idx]
        y_train_reg_raw = target_reg_arr[train_row_idx]
        X_val_raw = feature_arr[val_row_idx]
        y_val_cls_raw = target_cls_arr[val_row_idx]
        y_val_reg_raw = target_reg_arr[val_row_idx]
        X_current = feature_arr[current_row_idx]
        industries_current = industry_arr[current_row_idx]

        train_valid = ~(np.isnan(y_train_cls_raw) | np.isnan(y_train_reg_raw))
        X_train = X_train_raw[train_valid]
        y_train_cls = y_train_cls_raw[train_valid].astype(int)
        y_train_reg = y_train_reg_raw[train_valid]

        val_valid = ~(np.isnan(y_val_cls_raw) | np.isnan(y_val_reg_raw))
        X_val = X_val_raw[val_valid]
        y_val_cls = y_val_cls_raw[val_valid].astype(int)
        y_val_reg = y_val_reg_raw[val_valid]

        if len(X_train) < config.min_train_obs or len(X_val) == 0:
            continue

        # ── Train ONCE ────────────────────────────────────────────────────
        clf = XGBClassifier(**clf_params)
        clf.fit(X_train, y_train_cls)

        reg = XGBRegressor(**reg_params)
        reg.fit(X_train, y_train_reg)

        # ── Validation metrics ────────────────────────────────────────────
        p_val: np.ndarray = clf.predict_proba(X_val)[:, 1]
        pred_ret_val: np.ndarray = reg.predict(X_val).astype(np.float64)

        tau = _select_tau(y_val_cls, p_val)

        y_pred_cls_val = (p_val > tau).astype(int)
        val_accuracy = float(accuracy_score(y_val_cls, y_pred_cls_val))
        val_precision = float(precision_score(y_val_cls, y_pred_cls_val, zero_division=0))
        val_recall = float(recall_score(y_val_cls, y_pred_cls_val, zero_division=0))
        val_f1 = float(f1_score(y_val_cls, y_pred_cls_val, zero_division=0))

        unique_val_classes = np.unique(y_val_cls)
        val_auc = (
            float(roc_auc_score(y_val_cls, p_val))
            if len(unique_val_classes) >= 2
            else 0.5
        )
        val_mse = float(mean_squared_error(y_val_reg, pred_ret_val))

        w_cls, w_reg = _ensemble_weights(val_auc, val_mse, config.ensemble_weight_rule)

        # ── Current-date predictions ──────────────────────────────────────
        p_current: np.ndarray = clf.predict_proba(X_current)[:, 1]
        pred_ret_current: np.ndarray = reg.predict(X_current).astype(np.float64)

        pred_ret_val_mean = float(pred_ret_val.mean())
        pred_ret_val_std = float(pred_ret_val.std())

        z_fallback = pred_ret_val_std == 0.0
        if z_fallback:
            Z_current = np.zeros(len(pred_ret_current), dtype=np.float64)
        else:
            Z_current = (pred_ret_current - pred_ret_val_mean) / pred_ret_val_std

        S_current = w_cls * p_current + w_reg * Z_current

        scores = pd.Series(0.0, index=industry_names, dtype=np.float64)
        probas = pd.Series(0.0, index=industry_names, dtype=np.float64)
        for k, ind in enumerate(industries_current):
            if ind in scores.index:
                scores[ind] = float(S_current[k])
                probas[ind] = float(p_current[k])

        eligible_set = [s for s in industry_names if probas[s] > tau]

        # ── Realised return legs ──────────────────────────────────────────
        if t in prices_loc:
            t_p_idx = prices_loc[t]
            t_H_p_idx = t_p_idx + config.horizon
            if t_H_p_idx < n_prices:
                P_t = prices.iloc[t_p_idx]
                P_tH = prices.iloc[t_H_p_idx]
                fwd_ret_series = (P_tH / P_t - 1.0).fillna(0.0)
            else:
                fwd_ret_series = None
        else:
            fwd_ret_series = None

        # ── Per-portfolio construction ────────────────────────────────────
        for ptype in portfolio_types:
            fallback_used = z_fallback or (not eligible_set)

            if ptype == "max_profit":
                if eligible_set:
                    q_sum = float(
                        scores.reindex(eligible_set).clip(lower=0.0).fillna(0.0).sum()
                    )
                    if q_sum == 0.0:
                        fallback_used = True
                portfolio_weights = max_profit_portfolio(
                    scores=scores,
                    eligible=eligible_set,
                    fallback=config.empty_eligible_fallback,
                    all_industries=industry_names,
                )

            elif ptype == "min_risk":
                returns_to_date = returns.loc[returns.index <= t]
                portfolio_weights = min_risk_portfolio(
                    scores=scores,
                    eligible=eligible_set,
                    daily_returns=returns_to_date,
                    fallback=config.empty_eligible_fallback,
                    all_industries=industry_names,
                    lookback=config.cov_lookback,
                    cov_fallback=config.cov_fallback,
                    cov_shrinkage_alpha=config.cov_shrinkage_alpha,
                )
            else:
                raise ValueError(f"Unknown portfolio_type: {ptype!r}")

            if fwd_ret_series is not None:
                realized_return = float(
                    (
                        portfolio_weights
                        * fwd_ret_series.reindex(portfolio_weights.index).fillna(0.0)
                    ).sum()
                )
            else:
                realized_return = np.nan

            record = RebalanceRecord(
                decision_date=t,
                horizon=config.horizon,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                val_start=val_dates[0],
                val_end=val_dates[-1],
                n_train_obs=int(len(X_train)),
                n_val_obs=int(len(X_val)),
                tau=tau,
                w_cls=w_cls,
                w_reg=w_reg,
                val_accuracy=val_accuracy,
                val_precision=val_precision,
                val_recall=val_recall,
                val_f1=val_f1,
                val_auc=val_auc,
                val_mse=val_mse,
                pred_ret_mean=pred_ret_val_mean,
                pred_ret_std=pred_ret_val_std,
                eligible_set=eligible_set,
                weights=portfolio_weights.to_dict(),
                realized_return=realized_return,
                fallback_used=fallback_used,
            )
            all_records[ptype].append(record)

        if verbose and (loop_i + 1) % 20 == 0:
            print(
                f"  [{t.date()}] #{loop_i + 1}/{len(decision_indices)} "
                f"tau={tau:.2f} w_cls={w_cls:.3f} eligible={len(eligible_set)}"
            )

    # ── 4. Package results ────────────────────────────────────────────────────
    output: dict[str, tuple[list[RebalanceRecord], pd.Series]] = {}
    for ptype in portfolio_types:
        recs = all_records[ptype]
        if recs:
            hp_series = pd.Series(
                {r.decision_date: r.realized_return for r in recs},
                dtype=np.float64,
                name="hp_return",
            )
        else:
            hp_series = pd.Series(dtype=np.float64, name="hp_return")

        if verbose:
            print(
                f"[MultiBacktest] {ptype}: {len(recs)} records. "
                f"Mean HP return: {hp_series.mean():.4f}"
            )
        output[ptype] = (recs, hp_series)

    return output
