"""Risk model: expanding-window realised volatility (course baseline)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_volatility(history: pd.DataFrame, min_months: int) -> pd.Series:
    """Annualised sample std of every monthly return up to and including t.

    NaN for instruments with fewer than `min_months` observations
    (course code: vol_expanding(date, threshold=36)).
    """
    vol = history.std() * np.sqrt(12.0)
    vol[history.count() < min_months] = np.nan
    return vol


def ewma_volatility_monthly(daily: pd.DataFrame, halflife_days: float, min_days: int) -> pd.DataFrame:
    """Annualised EWMA volatility of daily returns, sampled at each month-end.

    Row for month t uses daily returns up to the instrument's last trading day
    in month t only (the EWMA is causal and is read at the month's last
    observation). Zero-mean: EWMA of squared returns. Index: calendar month
    periods. Each instrument's EWMA runs on its own trading days; a holiday
    is not a zero return.
    """
    out = {}
    for c in daily.columns:
        r = daily[c].dropna()
        var = (r ** 2).ewm(halflife=halflife_days, min_periods=min_days).mean()
        out[c] = var.groupby(var.index.to_period("M")).last()
    vol = np.sqrt(pd.DataFrame(out) * 252.0)
    return vol


def expanding_volatility_panel(monthly: pd.DataFrame, min_months: int) -> pd.DataFrame:
    """Course baseline vol for every month t at once (same as
    expanding_volatility(monthly.loc[:t]) row by row). Index: month periods."""
    vol = monthly.expanding().std() * np.sqrt(12.0)
    vol = vol.where(monthly.expanding().count() >= min_months)
    vol.index = monthly.index.to_period("M")
    return vol


def _nearest_correlation(c: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues and rescale to unit diagonal (PSD correlation)."""
    c = (c + c.T) / 2.0
    val, vec = np.linalg.eigh(c)
    c = (vec * np.clip(val, 1e-8, None)) @ vec.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def ewma_correlation_monthly(daily: pd.DataFrame, halflife_days: float, overlap_days: int,
                             min_days: int):
    """EWMA correlation matrices of overlapping k-day returns at each month-end.

    Runs one recursion over the union calendar: S <- lam S + (1-lam) x x',
    W <- lam W + (1-lam) m m', with x the k-day returns (a non-trading day inside
    an instrument's life counts as a zero return, i.e. the last price is carried
    forward; before its first or after its last observation the instrument is
    masked) and m the mask. Correlation_ij = (S_ij/W_ij) / sqrt(S_ii/W_ii S_jj/W_jj);
    pairs with joint weight below the equivalent of `min_days` observations get 0.
    The matrix stored for month t uses days up to the last trading day of t only.

    Returns (periods, ids, array[T, N, N]).
    """
    ids = list(daily.columns)
    first = daily.apply(lambda s: s.first_valid_index())
    last = daily.apply(lambda s: s.last_valid_index())
    alive = pd.DataFrame({c: (daily.index >= first[c]) & (daily.index <= last[c]) for c in ids},
                         index=daily.index)
    x = daily.fillna(0.0).where(alive, 0.0)
    if overlap_days > 1:
        x = x.rolling(overlap_days, min_periods=1).sum()
        alive = alive & alive.shift(overlap_days - 1, fill_value=False)
        x = x.where(alive, 0.0)
    xv, mv = x.to_numpy(), alive.to_numpy().astype(float)
    lam = 0.5 ** (1.0 / halflife_days)
    # joint weight of `min_days` equally-recent observations
    w_min = 1.0 - lam ** min_days
    n = len(ids)
    S, W = np.zeros((n, n)), np.zeros((n, n))
    month = daily.index.to_period("M")
    ends = np.r_[month[1:] != month[:-1], True]       # last calendar row of each month
    periods, mats = [], []
    for t in range(len(xv)):
        S = lam * S + (1 - lam) * np.outer(xv[t], xv[t])
        W = lam * W + (1 - lam) * np.outer(mv[t], mv[t])
        if ends[t]:
            with np.errstate(divide="ignore", invalid="ignore"):
                cov = S / W
                sd = np.sqrt(np.diag(cov))
                c = cov / np.outer(sd, sd)
            ok = (W >= w_min) & np.isfinite(c)
            c = np.where(ok, c, 0.0)
            np.fill_diagonal(c, 1.0)
            periods.append(month[t])
            mats.append(_nearest_correlation(c))
    return pd.PeriodIndex(periods), ids, np.array(mats)


def expanding_covariance_monthly(monthly: pd.DataFrame, min_months: int):
    """Course L6 risk model: expanding-window sample covariance of monthly returns
    (pairwise), annualised, for every month t. Instruments with fewer than
    `min_months` returns are NaN. Returns (periods, ids, array[T, N, N])."""
    ids = list(monthly.columns)
    x = monthly.to_numpy()
    mats = []
    for t in range(len(x)):
        h = pd.DataFrame(x[: t + 1], columns=ids)
        cov = h.cov(min_periods=min_months).to_numpy() * 12.0
        mats.append(cov)
    return monthly.index.to_period("M"), ids, np.array(mats)
