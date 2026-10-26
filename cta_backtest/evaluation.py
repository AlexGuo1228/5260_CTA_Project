"""Helpers for the notebook: risk-model scoring (upgrade 2), alternative signals,
ex-ante risk shares (upgrade 3) and regressions (summary section).

Risk-model scoring. A forecast made at month-end t (data through t) is scored on t+1:
    bias          std(r_{t+1} / (sigma_t / sqrt(12)))         ideal 1
    MRAD          mean |bias - 1| over instrument-years       how fast the model adapts
    QLIKE_monthly ln F + 12 r_{t+1}^2 / F,  F = sigma_t^2     lower is better
QLIKE_monthly decides the half-lives on the training sample (t+1 <= cfg.risk_train_end);
bias and MRAD are reported as checks. The test sample is reported, never used to choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .engine import run_backtest
from .risk import (ewma_correlation_monthly, ewma_volatility_monthly, expanding_covariance_monthly,
                   expanding_volatility_panel)
from .strategies import StaticBenchmark, TSMOMBaseline

GROUPS = ["TSMOM", "STATIC", "CLASS", "RANDOM"]


def _losses(r: pd.Series, F: pd.Series, year: np.ndarray, key: np.ndarray) -> dict:
    """r: realised next-month return, F: annualised variance forecast."""
    z = r / np.sqrt(F / 12.0)
    by = pd.DataFrame({"z": z.values, "k": key, "y": year}).groupby(["k", "y"])["z"].agg(["std", "count"])
    return {"n": len(r), "bias": z.std(), "mrad": (by.loc[by["count"] >= 10, "std"] - 1).abs().mean(),
            "qlike_monthly": (np.log(F) + 12 * r ** 2 / F).mean()}


def score_vol_halflives(monthly: pd.DataFrame, daily: pd.DataFrame, cfg: BacktestConfig):
    """Course expanding vol vs daily EWMA vol for each half-life in cfg.vol_halflife_grid,
    scored per instrument on the same instrument-months. Returns (scores, yearly bias)."""
    per = monthly.copy()
    per.index = per.index.to_period("M")
    forecasts = {"expanding (course)": expanding_volatility_panel(monthly, cfg.vol_min_months)}
    for h in cfg.vol_halflife_grid:
        forecasts[f"EWMA {int(h)}d"] = ewma_volatility_monthly(daily, h, cfg.vol_ewma_min_days)
    cols = {"r1": per.shift(-1).stack()}
    for k, f in forecasts.items():
        cols[k] = f.reindex(index=per.index, columns=per.columns).stack() ** 2
    df = pd.DataFrame(cols).dropna()
    df = df[(df[list(forecasts)] > 0).all(axis=1)]
    t1 = df.index.get_level_values(0) + 1
    year, ids = t1.year.to_numpy(), df.index.get_level_values(1).to_numpy()
    train = t1 <= pd.Period(cfg.risk_train_end, "M")
    rows = [{"sample": s, "model": k, **_losses(df["r1"][m], df[k][m], year[m], ids[m])}
            for s, m in [("train", train), ("test", ~train)] for k in forecasts]
    yearly = pd.DataFrame({k: (df["r1"] / np.sqrt(df[k] / 12)).groupby(year).std() for k in forecasts})
    return pd.DataFrame(rows), yearly


def score_covariance_models(monthly: pd.DataFrame, daily: pd.DataFrame, asset_class: pd.Series,
                            cfg: BacktestConfig) -> pd.DataFrame:
    """Covariance only matters through portfolios, so each model is scored on portfolios
    formed at t and held over t+1: the course TSMOM and static books, one long inverse-vol
    book per asset class, and cfg.n_random_portfolios random long/short inverse-vol books.
    Candidates: course expanding covariance, EWMA vols with zero correlation, and EWMA
    correlations for each (half-life, overlap) in the grid; vols are the chosen EWMA vols."""
    ids, per = list(monthly.columns), monthly.index.to_period("M")
    vol = ewma_volatility_monthly(daily, cfg.vol_halflife_days, cfg.vol_ewma_min_days)
    vol = vol.reindex(index=per, columns=ids).to_numpy()
    _, _, exp_cov = expanding_covariance_monthly(monthly, cfg.vol_min_months)
    diag = np.zeros_like(exp_cov)
    diag[:, np.arange(len(ids)), np.arange(len(ids))] = vol ** 2
    models = {"expanding (course)": exp_cov, "EWMA vol, zero corr": diag}
    for k in cfg.corr_overlap_grid:
        for h in cfg.corr_halflife_grid:
            p, _, C = ewma_correlation_monthly(daily, h, k, cfg.corr_min_days)
            C = pd.Series(list(C), index=p).reindex(per).to_numpy()
            models[f"EWMA corr {int(h)}d, {k}-day"] = np.array([np.outer(v, v) * c for v, c in zip(vol, C)])

    w_ts = run_backtest(monthly, TSMOMBaseline(cfg)).weights.reindex(monthly.index).fillna(0.0).to_numpy()
    w_st = run_backtest(monthly, StaticBenchmark(cfg)).weights.reindex(monthly.index).fillna(0.0).to_numpy()
    rng = np.random.default_rng(cfg.random_seed)
    cls = asset_class.reindex(ids).to_numpy()
    r_next = monthly.shift(-1).fillna(0.0).to_numpy()
    rows = []
    for t in range(len(per) - 1):
        ok = np.isfinite(vol[t]) & np.isfinite(np.diag(exp_cov[t])) & monthly.iloc[t].notna().to_numpy()
        if ok.sum() < cfg.min_assets:
            continue
        inv = np.where(ok, 1.0 / np.where(ok, vol[t], 1.0), 0.0)
        books = [("TSMOM", w_ts[t]), ("STATIC", w_st[t])]
        for c in sorted(set(cls)):
            m = ok & (cls == c)
            if m.sum() >= 2:
                books.append(("CLASS", np.where(m, inv, 0.0) / m.sum()))
        for _ in range(cfg.n_random_portfolios):
            s = rng.choice([-1.0, 1.0], len(ids)) * rng.uniform(0.5, 1.5, len(ids))
            books.append(("RANDOM", s * inv / ok.sum()))
        for g, w in books:
            if not np.any(w) or np.any(w[~ok] != 0):
                continue
            rec = {"t1": per[t + 1], "group": g, "r1": float(w @ r_next[t])}
            rec.update({k: float(w @ np.nan_to_num(S[t]) @ w) for k, S in models.items()})
            rows.append(rec)
    df = pd.DataFrame(rows)
    df = df[(df[list(models)] > 0).all(axis=1)]
    train = df["t1"] <= pd.Period(cfg.risk_train_end, "M")
    out = []
    for s, part in [("train", df[train]), ("test", df[~train])]:
        for g in GROUPS:
            sub = part[part["group"] == g]
            for k in models:
                out.append({"sample": s, "group": g, "model": k,
                            **_losses(sub["r1"], sub[k], sub["t1"].dt.year.to_numpy(), np.zeros(len(sub)))})
    return pd.DataFrame(out)


def sign_panel(monthly: pd.DataFrame, lookbacks, vol: np.ndarray = None) -> pd.DataFrame:
    """Average over lookbacks of sign(sum of the last L monthly returns). With `vol`
    (risk-model vols), each monthly return is first divided by the vol forecast made the
    month before it (risk-adjusted momentum). Row t uses months <= t only."""
    x = monthly.copy()
    x.index = x.index.to_period("M")
    if vol is not None:
        x = x / (pd.DataFrame(vol, index=x.index, columns=x.columns).shift(1) / np.sqrt(12.0))
    return sum(np.sign(x.rolling(L, min_periods=L).sum()) for L in lookbacks) / len(lookbacks)


def tstat_panel(monthly: pd.DataFrame, vol: np.ndarray, lookback: int = 12) -> pd.DataFrame:
    """Continuous signal: past `lookback`-month return divided by its standard error
    (annual vol forecast at t x sqrt(lookback/12))."""
    x = monthly.copy()
    x.index = x.index.to_period("M")
    v = pd.DataFrame(vol, index=x.index, columns=x.columns)
    return x.rolling(lookback, min_periods=lookback).sum() / (v * np.sqrt(lookback / 12.0))


def class_risk_shares(weights: pd.DataFrame, risk, asset_class: pd.Series) -> pd.DataFrame:
    """Ex-ante share of each asset class in the book's standalone risk at each month:
    sqrt(w_c' Sigma_cc w_c) / sum over classes, Sigma from the risk model at t."""
    cls = asset_class.reindex(risk.ids).to_numpy()
    names = sorted(set(cls))
    rows = {}
    for t, w in weights.iterrows():
        _, S = risk.at(t.to_period("M"))
        w = w.reindex(risk.ids).fillna(0.0).to_numpy()
        v = []
        for c in names:
            m = (cls == c) & (w != 0)
            v.append(np.sqrt(max(w[m] @ np.nan_to_num(S[np.ix_(m, m)]) @ w[m], 0.0)) if m.any() else 0.0)
        if sum(v) > 0:
            rows[t] = np.array(v) / sum(v)
    return pd.DataFrame(rows, index=names).T


def ols(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """OLS with intercept; returns annualised alpha, betas and t-stats."""
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    b, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    e = y.to_numpy() - A @ b
    se = np.sqrt(np.diag(e @ e / (len(y) - A.shape[1]) * np.linalg.inv(A.T @ A)))
    out = {"alpha (ann.)": b[0] * 12, "t(alpha)": b[0] / se[0]}
    for i, c in enumerate(X.columns, 1):
        out[f"beta {c}"], out[f"t({c})"] = b[i], b[i] / se[i]
    out["R2"] = 1 - e @ e / ((y - y.mean()) ** 2).sum()
    return pd.Series(out)
