"""Monthly backtest engine with a built-in one-month execution lag.

Timeline for every rebalance:

    end of month t   : strategy sees returns[..t] and outputs weights w_t
    during month t+1 : portfolio earns sum_i w_t[i] * r_{t+1}[i]

The lag is enforced here, not by the caller: the strategy is handed a
truncated copy of the return panel (`returns.iloc[:i+1]`), so it is
physically impossible for it to read month t+1 or later. The weight row is
stored under its signal date t and is only ever multiplied with the return
row of the following month.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd


class Strategy(Protocol):
    def compute_weights(self, history: pd.DataFrame) -> pd.Series: ...


@dataclass
class BacktestResult:
    name: str
    # Target weights indexed by SIGNAL date t (decided with data up to t).
    weights: pd.DataFrame
    # Same weights re-indexed by the month in which they earn returns (t+1).
    held_weights: pd.DataFrame
    # Per-instrument P&L contribution w_t * r_{t+1}, indexed by t+1.
    contributions: pd.DataFrame
    # Net portfolio return indexed by t+1.
    returns: pd.Series
    # Two-sided turnover sum_i |w_t - drifted w_{t-1}| indexed by signal date t.
    turnover: pd.Series
    # Months in which a held instrument had no return (treated as zero).
    missing_held: pd.Series


def drift_weights(w_prev: pd.Series, r: pd.Series) -> pd.Series:
    """Weights just before rebalancing, after one month of price moves.

    Notional exposure i grows by (1 + r_i) while NAV grows by (1 + R_p).
    """
    r = r.fillna(0.0)
    port = float((w_prev * r).sum())
    return w_prev * (1.0 + r) / (1.0 + port)


def run_backtest(returns: pd.DataFrame, strategy: Strategy, name: str = "strategy",
                 cost_bps: float = 0.0, start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> BacktestResult:
    """Run the monthly backtest.

    start_date / end_date bound the months whose RETURNS are evaluated
    (e.g. "1995-01" means January 1995 is the first month earning P&L, so
    the first weights are decided at the end of December 1994). Signals and
    volatility still use all history before start_date as warm-up; that is
    data available at the time and is not look-ahead.
    """
    if end_date is not None:
        returns = returns.loc[: pd.Period(end_date, "M").end_time]
    dates = returns.index
    cols = returns.columns
    n = len(dates)
    first_ret = pd.Period(start_date, "M").start_time if start_date is not None else dates[0]

    w_rows = np.zeros((n, len(cols)))
    for i in range(n - 1):
        if dates[i + 1] < first_ret:
            # This month's weights would earn a return before the sample.
            continue
        # Point-in-time view: rows 0..i only (signal month t = dates[i]).
        history = returns.iloc[: i + 1].copy()
        assert history.index[-1] == dates[i]
        w = strategy.compute_weights(history)
        w_rows[i] = w.reindex(cols).fillna(0.0).values
    # No weights are produced on the final date: there is no t+1 to earn.

    weights = pd.DataFrame(w_rows, index=dates, columns=cols)

    # Lag by one row: weights decided at t are held during t+1.
    held = weights.shift(1).fillna(0.0)
    r_filled = returns.fillna(0.0)
    contributions = held * r_filled
    gross_ret = contributions.sum(axis=1)

    # Turnover at each signal date, measured against drifted weights.
    turnover = pd.Series(0.0, index=dates)
    prev = pd.Series(0.0, index=cols)
    for i in range(n - 1):
        t = dates[i]
        if i > 0:
            prev = drift_weights(weights.iloc[i - 1], returns.iloc[i])
        turnover.iloc[i] = float((weights.iloc[i] - prev).abs().sum())

    # Costs on the trade made at t are charged to the month they fund (t+1).
    costs = turnover.shift(1).fillna(0.0) * cost_bps / 1e4
    net_ret = gross_ret - costs

    missing = ((held != 0) & returns.isna()).sum(axis=1)

    # Trim to the live period: from the first month that holds a position.
    live = held.abs().sum(axis=1) > 0
    if not live.any():
        raise ValueError("Strategy never takes a position")
    start = live.idxmax()
    first_signal = weights.index[weights.index.get_loc(start) - 1]

    return BacktestResult(
        name=name,
        weights=weights.loc[first_signal:].iloc[:-1],
        held_weights=held.loc[start:],
        contributions=contributions.loc[start:],
        returns=net_ret.loc[start:].rename(name),
        turnover=turnover.loc[first_signal:].iloc[:-1].rename(name),
        missing_held=missing.loc[start:],
    )
