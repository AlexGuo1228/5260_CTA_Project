"""Trend continuity, the strength measure used by CTA_v2 (upgrade 4).

Based on Information Discreteness (Da, Gurun & Warachka 2014, "Frog in the Pan").
For each instrument and month-end t:

    continuity = sign(past 12m return) x (share of up days - share of down days)

over the last 252 trading days. A trend built from many small days in its direction scores
high; a trend driven by a few large jumps scores low. Everything uses data up to t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def continuity_monthly(daily: pd.DataFrame, monthly: pd.DataFrame, lookback_months: int = 12,
                       window_days: int = 252) -> pd.DataFrame:
    """Month x instrument panel of trend continuity (index: month periods).

    daily   : daily returns (date x instrument), from daily.load_daily_returns
    monthly : monthly returns (month-end x instrument); the past return is the sum of the
              last `lookback_months` of them, i.e. the same quantity as the TSMOM signal
    """
    out = {}
    for c in daily.columns:
        r = daily[c].dropna()
        up = (r > 0).astype(float).rolling(window_days, min_periods=window_days).mean()
        dn = (r < 0).astype(float).rolling(window_days, min_periods=window_days).mean()
        net = up - dn
        out[c] = net.groupby(net.index.to_period("M")).last()     # value on the last trading day of t
    net = pd.DataFrame(out).reindex(index=monthly.index.to_period("M"), columns=monthly.columns)
    past = monthly.rolling(lookback_months, min_periods=lookback_months).sum()
    past.index = net.index
    return np.sign(past) * net
