"""Monthly trend-strength panel from daily OHLC: Yanhao's hybrid_v2 trend
probability (autonomous_quant_agent/intelligence/regime_models.py), re-implemented
here with every window multiplied by `scale` so it can run at CTA horizons.

p_trend = 0.40 rank(ADX) + 0.35 rank(|close - VWAP| / ATR) + 0.25 rank(|persistence|),
ranks = rolling percentile over the last 252*scale days; floored at 0.85 / 0.65 when the
strong / moderate rules fire (same thresholds as hybrid_v2). All inputs look backwards
only; the value for month t is the one on the last trading day of t.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

W_ADX, W_GAP, W_PERSIST = 0.40, 0.35, 0.25
STRONG = (0.80, 0.80, 0.70)
MODERATE = (0.65, 0.65, 0.55)


def _rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False).mean()


def _features(o: pd.DataFrame, scale: int) -> pd.Series:
    h, l, c, v = o["high"], o["low"], o["close"], o["volume"]
    n_atr, n_vwap, n_per, lb = 14 * scale, 30 * scale, 5 * scale, 252 * scale
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, n_atr)
    up, down = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=o.index)
    mdm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=o.index)
    dip, dim = 100 * _rma(pdm, n_atr) / atr, 100 * _rma(mdm, n_atr) / atr
    adx = _rma(100 * (dip - dim).abs() / (dip + dim), n_atr)
    tp = (h + l + c) / 3.0
    vwap = (tp * v).rolling(n_vwap, min_periods=1).sum() / v.rolling(n_vwap, min_periods=1).sum()
    gap = (c - vwap) / atr
    persist = np.sign(c - vwap).replace(0, np.nan).rolling(n_per).mean()
    minp = 40
    a = adx.rolling(lb, min_periods=minp).rank(pct=True)
    g = gap.abs().rolling(lb, min_periods=minp).rank(pct=True)
    p = persist.abs().rolling(lb, min_periods=minp).rank(pct=True)
    score = (W_ADX * a + W_GAP * g + W_PERSIST * p).clip(0, 1)
    strong = (a >= STRONG[0]) & (g >= STRONG[1]) & (p >= STRONG[2])
    mod = (a >= MODERATE[0]) & (g >= MODERATE[1]) & (p >= MODERATE[2]) & ~strong
    pt = np.where(strong, np.maximum(score, 0.85), np.where(mod, np.maximum(score, 0.65), score))
    return pd.Series(pt, index=o.index).where(score.notna())


def _ohlcv(directory: str, iid: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(directory, f"{iid}.csv"), index_col=0, parse_dates=True).sort_index()
    return d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def p_trend_monthly(directory: str, ids, splices, scale: int) -> pd.DataFrame:
    """Month-period x instrument p_trend (value on the last trading day of each month)."""
    out = {}
    for i in ids:
        s = _features(_ohlcv(directory, i), scale)
        if i in splices:
            src, month = splices[i]
            sw = pd.Period(month, "M").start_time
            alt = _features(_ohlcv(directory, src), scale)
            s = pd.concat([s[s.index < sw], alt[alt.index >= sw]])
        out[i] = s.groupby(s.index.to_period("M")).last()
    return pd.DataFrame(out)


def continuity_monthly(daily: pd.DataFrame, monthly: pd.DataFrame, lookback_months: int = 12,
                       window_days: int = 252) -> pd.DataFrame:
    """Trend continuity = -Information Discreteness (Da, Gurun & Warachka 2014):
    sign(past return) x (share of up days - share of down days) over the last
    `window_days` trading days. High = the move was built from many small days in its
    direction ("frog in the pan"); low = driven by a few jumps. Past return = sum of the
    last `lookback_months` monthly returns (the TSMOM signal). Month-period index."""
    out = {}
    for c in daily.columns:
        r = daily[c].dropna()
        up = (r > 0).astype(float).rolling(window_days, min_periods=window_days).mean()
        dn = (r < 0).astype(float).rolling(window_days, min_periods=window_days).mean()
        net = (up - dn)
        out[c] = net.groupby(net.index.to_period("M")).last()
    net = pd.DataFrame(out).reindex(index=monthly.index.to_period("M"), columns=monthly.columns)
    past = monthly.rolling(lookback_months, min_periods=lookback_months).sum()
    past.index = net.index
    return np.sign(past) * net
