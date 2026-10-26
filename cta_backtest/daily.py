"""Daily returns for the 58-column universe, from the raw daily files.

Returns are close-to-close on each instrument's own trading days; prices are
vendor-rolled (ratio back-adjusted), so returns are used and price levels never
are. Splices follow the monthly panel and are applied to returns, not prices.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import pandas as pd


def _instrument_returns(path: str, max_gap_days: int) -> pd.Series:
    close = pd.read_csv(path, index_col=0, parse_dates=True)["Close"].dropna().sort_index()
    ret = close.pct_change(fill_method=None)
    gap = close.index.to_series().diff().dt.days
    ret[gap > max_gap_days] = float("nan")          # data hole, not a trading day
    return ret.iloc[1:]


def load_daily_returns(directory: str, columns: Iterable[str],
                       splices: Dict[str, Tuple[str, str]], max_gap_days: int) -> pd.DataFrame:
    """Date x instrument daily returns. A column listed in `splices` uses its
    own file before the switch month and the replacement file from it on."""
    out = {}
    for c in columns:
        r = _instrument_returns(os.path.join(directory, f"{c}.csv"), max_gap_days)
        if c in splices:
            src, month = splices[c]
            switch = pd.Period(month, "M").start_time
            alt = _instrument_returns(os.path.join(directory, f"{src}.csv"), max_gap_days)
            r = pd.concat([r[r.index < switch], alt[alt.index >= switch]])
        out[c] = r
    return pd.DataFrame(out).sort_index()
