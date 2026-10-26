"""Point-in-time risk models for the strategies: one (vol, covariance) per
month-end t, each built from data through t only (see risk.py and its tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .risk import (_nearest_correlation, ewma_correlation_monthly, ewma_volatility_monthly,
                   expanding_covariance_monthly, expanding_volatility_panel)


@dataclass
class RiskModel:
    name: str
    ids: List[str]
    periods: pd.PeriodIndex
    vol: np.ndarray          # [T, N] annualised
    cov: np.ndarray          # [T, N, N] annualised

    def at(self, period: pd.Period):
        i = self.periods.get_loc(period)
        return self.vol[i], self.cov[i]


def historical_risk(monthly: pd.DataFrame, cfg: BacktestConfig) -> RiskModel:
    """Course L6: historical to-date (expanding-window, equal-weight) volatility and
    covariance of monthly returns; each month uses all history up to that month."""
    periods, ids, cov = expanding_covariance_monthly(monthly, cfg.vol_min_months)
    vol = np.sqrt(np.einsum("tii->ti", cov))
    # pairwise estimates need not be PSD: project the correlation part
    out = np.full_like(cov, np.nan)
    for t in range(len(cov)):
        ok = np.isfinite(vol[t])
        if ok.sum() == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            c = cov[t][np.ix_(ok, ok)] / np.outer(vol[t][ok], vol[t][ok])
        c = _nearest_correlation(np.nan_to_num(c, nan=0.0) + np.diag(np.zeros(ok.sum())))
        out[t][np.ix_(ok, ok)] = c * np.outer(vol[t][ok], vol[t][ok])
    return RiskModel("historical", ids, periods, vol, out)


def ewma_risk(daily: pd.DataFrame, monthly: pd.DataFrame, cfg: BacktestConfig) -> RiskModel:
    """Upgrade 2: daily EWMA vols (half-life cfg.vol_halflife_days), floored at
    cfg.vol_floor_vs_history x the historical to-date vol, and EWMA correlation of
    overlapping k-day returns; Sigma = D C D."""
    ids = list(monthly.columns)
    periods = monthly.index.to_period("M")
    vol = ewma_volatility_monthly(daily[ids], cfg.vol_halflife_days, cfg.vol_ewma_min_days)
    vol = vol.reindex(index=periods, columns=ids).to_numpy()
    hist = expanding_volatility_panel(monthly, cfg.vol_min_months).reindex(columns=ids).to_numpy()
    vol = np.where(np.isfinite(hist), np.fmax(vol, cfg.vol_floor_vs_history * hist), vol)
    p_c, _, C = ewma_correlation_monthly(daily[ids], cfg.corr_halflife_days, cfg.corr_overlap_days,
                                         cfg.corr_min_days)
    C = pd.Series(list(C), index=p_c).reindex(periods)
    cov = np.array([np.outer(v, v) * c for v, c in zip(vol, C)])
    return RiskModel("ewma", ids, periods, vol, cov)
