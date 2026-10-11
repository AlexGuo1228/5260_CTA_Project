"""Course baseline TSMOM strategy and its static benchmark.

Both receive `history` (returns up to and including signal month t, nothing
later; the engine guarantees this) and return target weights for month t+1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .risk import expanding_volatility
from .signals import tsmom_sign


def _scaled_universe(history: pd.DataFrame, cfg: BacktestConfig):
    """Signal, vol-scaled unit position (target / vol) and the tradable set:
    instruments with both a signal and a volatility estimate at t."""
    sig = tsmom_sign(history, cfg.momentum_lookback, cfg.momentum_aggregation)
    vol = expanding_volatility(history, cfg.vol_min_months)
    scale = cfg.position_vol_target / vol
    ok = sig.notna() & (sig != 0) & scale.notna() & (vol > 0)
    return sig, scale, ok


@dataclass
class TSMOMBaseline:
    """w_i = sign_i * 40% / vol_i / N  (Lecture 6, Baseline Strategy)."""
    cfg: BacktestConfig

    def compute_weights(self, history: pd.DataFrame) -> pd.Series:
        sig, scale, ok = _scaled_universe(history, self.cfg)
        w = pd.Series(0.0, index=history.columns)
        n = int(ok.sum())
        if n >= self.cfg.min_assets:
            w[ok] = sig[ok] * scale[ok] / n
        return w


@dataclass
class StaticBenchmark:
    """Equal-weighted long position in the same vol-adjusted assets:
    w_i = +40% / vol_i / N  (Lecture 4/6, "Strategy vs Static bmk")."""
    cfg: BacktestConfig

    def compute_weights(self, history: pd.DataFrame) -> pd.Series:
        _, scale, ok = _scaled_universe(history, self.cfg)
        w = pd.Series(0.0, index=history.columns)
        n = int(ok.sum())
        if n >= self.cfg.min_assets:
            w[ok] = scale[ok] / n
        return w


def soft_cap_shares(raw: np.ndarray, cap: float, tol: float = 1e-10) -> np.ndarray:
    """Map raw risk shares (>= 0, sum 1) to capped shares:
    share_i = cap * tanh(lam * raw_i / cap), lam >= 1 solved so shares sum to 1.
    Equal raw shares are returned unchanged. Infeasible (n * cap <= 1): equal shares."""
    raw = np.asarray(raw, dtype=float)
    pos = raw > 0
    if pos.sum() * cap <= 1.0:
        out = np.zeros_like(raw)
        out[pos] = 1.0 / pos.sum()
        return out
    f = lambda lam: (cap * np.tanh(lam * raw[pos] / cap)).sum() - 1.0  # noqa: E731
    lo, hi = 1.0, 2.0
    while f(hi) < 0:
        hi *= 2.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    out = np.zeros_like(raw)
    out[pos] = cap * np.tanh(hi * raw[pos] / cap)
    return out / out.sum()


def _row(panel: pd.DataFrame, t: pd.Period, cols) -> np.ndarray:
    """Row t of a month-period panel (built from data through t), aligned to cols."""
    if t not in panel.index:
        return np.full(len(cols), np.nan)
    return panel.loc[t].reindex(cols).to_numpy(dtype=float)


@dataclass
class VolTargetTSMOM:
    """TSMOM sized with a point-in-time risk model and scaled to an ex-ante
    portfolio volatility of cfg.portfolio_vol_target.

    sizing = "baseline": course positions sign * 40% / vol / N (L6 vol-target version).
    sizing = "capped":   unit-risk positions sign / vol, then soft caps on each
                         instrument's and each asset class's share of standalone risk.
    CTA_v1 = capped sizing on the EWMA risk model. CTA_v2 = CTA_v1 with positions
    whose trend continuity (-Information Discreteness) ranks below the month's median
    halved (strength_panel = continuity rank, threshold 0.5, low weight 0.5).
    Tradable set = course baseline rule (signal + >= vol_min_months of history),
    so every version trades the same instruments in the same months.
    """
    cfg: BacktestConfig
    risk: "RiskModel"
    sizing: str = "baseline"
    asset_class: pd.Series = None
    # Optional research hook: month x instrument panel in [-1, 1] that replaces the
    # sign(12m) signal (sign = direction, |value| = position scale; NaN -> flat).
    signal_panel: pd.DataFrame = None
    # Optional strength filter (CTA_v2: trend-continuity rank): positions whose
    # strength is below `strength_threshold` are scaled by `strength_low_weight`.
    strength_panel: pd.DataFrame = None
    strength_threshold: float = 0.5
    strength_low_weight: float = 0.5

    def compute_weights(self, history: pd.DataFrame) -> pd.Series:
        cfg = self.cfg
        sig, _, ok = _scaled_universe(history, cfg)
        vol, cov = self.risk.at(history.index[-1].to_period("M"))
        ok = ok.to_numpy() & np.isfinite(vol) & (np.nan_to_num(vol) > 0)
        w = np.zeros(len(history.columns))
        n = int(ok.sum())
        if n < cfg.min_assets:
            return pd.Series(w, index=history.columns)
        s = sig.to_numpy()
        strength = np.ones(len(s))
        t = history.index[-1].to_period("M")
        if self.signal_panel is not None:
            m = _row(self.signal_panel, t, history.columns)
            # a missing or cancelled signal is a flat position inside the tradable set,
            # so every version trades the same universe in the same months
            m = np.where(np.isfinite(m), m, 0.0)
            s = np.sign(m)
            strength = np.abs(m)
        if self.strength_panel is not None:
            p = _row(self.strength_panel, t, history.columns)
            strength = strength * np.where(np.isfinite(p) & (p < self.strength_threshold),
                                           self.strength_low_weight, 1.0)
        if self.sizing == "baseline":
            w[ok] = s[ok] * cfg.position_vol_target / vol[ok] / n
        elif self.sizing == "capped":
            w[ok] = s[ok] / vol[ok]                         # unit standalone risk each
            w = self._apply_caps(w, ok, vol, cov)
        else:
            raise ValueError(self.sizing)
        S = np.nan_to_num(cov[np.ix_(ok, ok)])
        w = w * strength
        pv = float(np.sqrt(max(w[ok] @ S @ w[ok], 1e-16)))
        w *= cfg.portfolio_vol_target / pv
        return pd.Series(w, index=history.columns)

    def _apply_caps(self, w, ok, vol, cov):
        cfg = self.cfg
        idx = np.where(ok & (w != 0))[0]
        # instrument level: standalone risk |w_i| vol_i
        r = np.abs(w[idx]) * vol[idx]
        w[idx] *= soft_cap_shares(r / r.sum(), cfg.instrument_risk_cap) / (r / r.sum())
        # class level: standalone sleeve vol sqrt(w_c' Sigma_cc w_c), correlations and signs included
        cls = self.asset_class.reindex(self.risk.ids).to_numpy()[idx]
        names = sorted(set(cls))
        v = np.array([np.sqrt(max(w[idx][cls == c] @ np.nan_to_num(cov[np.ix_(idx[cls == c], idx[cls == c])])
                                  @ w[idx][cls == c], 1e-16)) for c in names])
        a = v / v.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(a > 1e-12, soft_cap_shares(a, cfg.class_risk_cap) / a, 1.0)
        for c, k in zip(names, scale):
            w[idx[cls == c]] *= k
        return w
