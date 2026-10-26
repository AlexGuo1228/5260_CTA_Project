"""Performance statistics.

Monthly returns from futures are excess returns (the margin collateral
earns the cash rate separately), so Sharpe ratios are computed without
subtracting a risk-free rate.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .engine import BacktestResult


def nav(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    v = nav(returns)
    return v / v.cummax() - 1.0


def summary_stats(res: BacktestResult) -> Dict[str, float]:
    r = res.returns
    n = len(r)
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    cagr = nav(r).iloc[-1] ** (12.0 / n) - 1.0
    dd = drawdown(r)
    # The first rebalance builds the book from cash; exclude it from turnover.
    to = res.turnover.iloc[1:]
    return {
        "start": r.index[0].strftime("%Y-%m"),
        "end": r.index[-1].strftime("%Y-%m"),
        "months": n,
        "ann_return": ann_ret,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "hit_rate": (r > 0).mean(),
        "skew": r.skew(),
        "worst_month": r.min(),
        "best_month": r.max(),
        "ann_turnover": to.mean() * 12,
        "avg_gross": res.held_weights.abs().sum(axis=1).mean(),
        "avg_n_positions": (res.held_weights != 0).sum(axis=1).mean(),
    }


def compare(results: Iterable[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({r.name: summary_stats(r) for r in results})


def by_asset_class(res: BacktestResult, asset_class: pd.Series) -> pd.DataFrame:
    """Per asset class: annualised return contribution, share of portfolio
    variance (covariance of the sleeve with the total, over total variance),
    average gross weight and average number of positions.
    """
    cls = asset_class.reindex(res.contributions.columns)
    contrib = res.contributions.T.groupby(cls).sum().T
    total = res.returns
    var_tot = total.var()
    gross = res.held_weights.abs().T.groupby(cls).sum().T
    npos = (res.held_weights != 0).T.groupby(cls).sum().T
    out = pd.DataFrame({
        "ann_return_contrib": contrib.mean() * 12,
        "risk_share": contrib.apply(lambda s: s.cov(total)) / var_tot,
        "avg_gross_weight": gross.mean(),
        "avg_n_positions": npos.mean(),
    })
    out.index.name = "asset_class"
    return out


def subperiod_stats(res: BacktestResult, periods: Dict[str, tuple]) -> pd.DataFrame:
    rows = {}
    for label, (a, b) in periods.items():
        r = res.returns.loc[a:b]
        if len(r) < 12:
            continue
        vol = r.std() * np.sqrt(12)
        rows[label] = {
            "ann_return": r.mean() * 12,
            "ann_vol": vol,
            "sharpe": r.mean() * 12 / vol if vol > 0 else np.nan,
            "max_drawdown": drawdown(r).min(),
        }
    return pd.DataFrame(rows).T
