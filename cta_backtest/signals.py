"""Trend signals. Each function receives a history ending at signal date t
and may only use that history."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tsmom_sign(history: pd.DataFrame, lookback: int, aggregation: str = "sum") -> pd.Series:
    """Sign of the past return over months t-lookback+1 .. t.

    aggregation: "sum" (sum of monthly returns, the course definition) or
    "compound" (prod(1 + r) - 1). NaN when any month in the window is missing.
    """
    window = history.iloc[-lookback:]
    if aggregation == "sum":
        past = window.sum(min_count=1)
    elif aggregation == "compound":
        past = (1.0 + window).prod(min_count=1) - 1.0
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    sig = np.sign(past)
    sig[window.notna().sum() < lookback] = np.nan
    return sig

