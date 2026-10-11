"""Look-ahead tests for the engine and strategies, unit tests of the course
baseline building blocks, and a replication check against Lecture 6.

Run from cta/:  python -m pytest -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cta_backtest import BacktestConfig, StaticBenchmark, TSMOMBaseline, load_market_data, run_backtest  # noqa: E402
from cta_backtest.risk import expanding_volatility  # noqa: E402
from cta_backtest.signals import tsmom_sign  # noqa: E402

STRATEGIES = [TSMOMBaseline, StaticBenchmark]


def _cfg(**kw) -> BacktestConfig:
    return BacktestConfig(returns_path=os.path.join(ROOT, "..", "MonthlyReturns.csv"),
                          asset_map_path=os.path.join(ROOT, "..", "data", "AssetMapCsv.csv"), **kw)


@pytest.fixture(scope="module")
def full():
    cfg = _cfg()
    return cfg, load_market_data(cfg)


@pytest.fixture(scope="module")
def market(full):
    cfg, md = full
    # A 1975-2000 slice keeps the look-ahead tests fast.
    return cfg, md.returns.loc["1975":"2000"]


class RecordingStrategy:
    def __init__(self, inner):
        self.inner = inner
        self.seen = []

    def compute_weights(self, history):
        self.seen.append(history.index[-1])
        return self.inner.compute_weights(history)


# ---------------------------------------------------------------- look-ahead

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_weights_ignore_future_data(market, strategy):
    """Scramble every return after month t: weights up to t must not change
    by a single bit, and P&L of t+1 must equal w_t . r_{t+1}."""
    cfg, rets = market
    strat = strategy(cfg)
    base = run_backtest(rets, strat, name="base")
    rng = np.random.default_rng(0)
    for t in [pd.Timestamp("1985-06-28"), pd.Timestamp("1992-12-31"), pd.Timestamp("1998-08-31")]:
        t = rets.index[rets.index.get_indexer([t], method="nearest")[0]]
        fake = rets.copy()
        after = fake.index > t
        noise = rng.normal(0, 0.3, size=fake.loc[after].shape)
        fake.loc[after] = np.where(fake.loc[after].notna(), noise, np.nan)
        alt = run_backtest(fake, strat, name="alt")
        pd.testing.assert_frame_equal(base.weights.loc[:t], alt.weights.loc[:t], check_exact=True)
        t1 = rets.index[rets.index.get_loc(t) + 1]
        expected = float((alt.weights.loc[t] * fake.loc[t1].fillna(0.0)).sum())
        assert alt.returns.loc[t1] == pytest.approx(expected, abs=1e-15)


def test_engine_hands_strategy_only_past_data(market):
    cfg, rets = market
    rec = RecordingStrategy(TSMOMBaseline(cfg))
    res = run_backtest(rets, rec, name="rec")
    assert rec.seen == list(rets.index[:-1])
    for t, t1 in zip(res.weights.index[:-1], res.weights.index[1:]):
        np.testing.assert_array_equal(res.held_weights.loc[t1].values, res.weights.loc[t].values)


def test_shock_in_next_month_is_not_anticipated(market):
    """A +50% return planted in t+1 must not change the position decided at t."""
    cfg, rets = market
    strat = TSMOMBaseline(cfg)
    t, t1 = rets.index[150], rets.index[151]
    inst = rets.loc[t1].dropna().index[0]
    shocked = rets.copy()
    shocked.loc[t1, inst] = 0.5
    a = run_backtest(rets, strat, name="a")
    b = run_backtest(shocked, strat, name="b")
    assert a.weights.loc[t, inst] == b.weights.loc[t, inst]
    diff = b.returns.loc[t1] - a.returns.loc[t1]
    assert diff == pytest.approx(a.weights.loc[t, inst] * (0.5 - rets.loc[t1, inst]))


def test_start_date_trims_sample_without_dropping_warmup(market):
    cfg, rets = market
    strat = TSMOMBaseline(cfg)
    whole = run_backtest(rets, strat, name="full")
    cut = run_backtest(rets, strat, name="cut", start_date="1993-01")
    assert cut.returns.index[0] == pd.Timestamp("1993-01-29")
    pd.testing.assert_series_equal(cut.returns, whole.returns.loc["1993-01":], check_names=False)


# ------------------------------------------------------------- building blocks

def test_expanding_volatility_formula_and_threshold():
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    x = pd.DataFrame({"A": np.sin(np.arange(40)) / 20, "B": np.nan}, index=idx)
    x.loc[idx[-30:], "B"] = np.cos(np.arange(30)) / 30          # only 30 months
    vol = expanding_volatility(x, 36)
    assert vol["A"] == pytest.approx(x["A"].std() * np.sqrt(12))
    assert np.isnan(vol["B"])                                  # below the 36-month threshold


def test_tsmom_sign_sum_vs_compound():
    idx = pd.date_range("2000-01-31", periods=12, freq="ME")
    # Arithmetic sum is +1% but the compounded return is about -18%.
    r = pd.DataFrame({"A": [0.5, -0.4] + [-0.009] * 10}, index=idx)
    assert tsmom_sign(r, 12, "sum")["A"] == 1.0
    assert tsmom_sign(r, 12, "compound")["A"] == -1.0
    r.iloc[3, 0] = np.nan
    assert np.isnan(tsmom_sign(r, 12)["A"])                   # a missing month -> no signal


def test_weight_formulas(full):
    """Baseline: w = sign * 40% / vol / N. Static: |baseline| (same assets, long)."""
    cfg, md = full
    h = md.returns.loc[:"2000-12"]
    w = TSMOMBaseline(cfg).compute_weights(h)
    s = StaticBenchmark(cfg).compute_weights(h)
    sig = tsmom_sign(h, 12)
    vol = expanding_volatility(h, 36)
    on = w != 0
    n = int(on.sum())
    np.testing.assert_allclose(w[on], (sig * 0.40 / vol / n)[on])
    np.testing.assert_allclose(s, w.abs())


# --------------------------------------------------------------- replication

def test_replicates_lecture6(full):
    """Lecture 6: TSMOM 11.1% / 11.1% / SR 1.00, static 6.9% / 14.5% / 0.47."""
    cfg, md = full
    b = run_backtest(md.returns, TSMOMBaseline(cfg), name="b").returns
    s = run_backtest(md.returns, StaticBenchmark(cfg), name="s").returns
    sr = lambda r: r.mean() / r.std() * np.sqrt(12)  # noqa: E731
    assert b.mean() * 12 == pytest.approx(0.111, abs=0.005)
    assert b.std() * np.sqrt(12) == pytest.approx(0.111, abs=0.005)
    assert sr(b) == pytest.approx(1.00, abs=0.02)
    assert s.mean() * 12 == pytest.approx(0.069, abs=0.005)
    assert s.std() * np.sqrt(12) == pytest.approx(0.145, abs=0.005)
    assert sr(s) == pytest.approx(0.47, abs=0.02)
    assert b.corr(s) == pytest.approx(0.08, abs=0.02)


# ------------------------------------------------------- daily risk model

@pytest.fixture(scope="module")
def daily(full):
    from cta_backtest.daily import load_daily_returns
    cfg, md = full
    d = load_daily_returns(os.path.join(ROOT, "..", "data", "FuturesUnderlyingData"), md.returns.columns,
                           cfg.daily_splices, cfg.daily_max_gap_days)
    return d.loc["1985":"2000"]


def test_daily_risk_model_ignores_future_days(daily):
    """Scramble every daily return after month t: the EWMA vols and the EWMA
    correlation matrix stored for month t must not change."""
    from cta_backtest.risk import ewma_correlation_monthly, ewma_volatility_monthly
    t = pd.Period("1995-06", "M")
    fake = daily.copy()
    after = fake.index > t.end_time
    rng = np.random.default_rng(1)
    fake.loc[after] = np.where(fake.loc[after].notna(), rng.normal(0, 0.05, fake.loc[after].shape), np.nan)
    v0, v1 = ewma_volatility_monthly(daily, 60, 126), ewma_volatility_monthly(fake, 60, 126)
    pd.testing.assert_frame_equal(v0.loc[:t], v1.loc[:t], check_exact=True)
    p0, _, c0 = ewma_correlation_monthly(daily, 120, 3, 126)
    p1, _, c1 = ewma_correlation_monthly(fake, 120, 3, 126)
    i = list(p0).index(t)
    np.testing.assert_array_equal(c0[: i + 1], c1[: i + 1])
    assert not np.allclose(c0[i + 12], c1[i + 12])          # later months do see the change


def test_correlation_matrices_are_valid(daily):
    from cta_backtest.risk import ewma_correlation_monthly
    _, _, c = ewma_correlation_monthly(daily, 120, 3, 126)
    last = c[-1]
    np.testing.assert_allclose(np.diag(last), 1.0)
    np.testing.assert_allclose(last, last.T, atol=1e-12)
    assert np.linalg.eigvalsh(last).min() > -1e-10


# ------------------------------------------------- step 2: vol target + caps

def test_soft_cap_shares():
    from cta_backtest.strategies import soft_cap_shares
    eq = np.full(4, 0.25)
    np.testing.assert_allclose(soft_cap_shares(eq, 0.4), eq)                 # equal shares untouched
    out = soft_cap_shares(np.array([0.7, 0.1, 0.1, 0.1]), 0.4)
    assert out.sum() == pytest.approx(1.0) and out.max() <= 0.4 + 1e-9       # cap respected
    raw = np.array([0.45, 0.25, 0.2, 0.1])
    out = soft_cap_shares(raw, 0.4)
    assert np.all(np.diff(out[np.argsort(raw)]) > 0)                         # order preserved


@pytest.mark.parametrize("sizing", ["baseline", "capped"])
def test_vol_target_ignores_future_and_hits_target(full, sizing):
    """Recompute the (historical) risk model on data scrambled after t: weights up to t
    are unchanged, and every month's ex-ante vol equals the target."""
    from cta_backtest import VolTargetTSMOM, historical_risk
    cfg, md = full
    rets = md.returns.loc["1975":"2000"]
    t = pd.Timestamp("1992-12-31")
    fake = rets.copy()
    after = fake.index > t
    rng = np.random.default_rng(2)
    fake.loc[after] = np.where(fake.loc[after].notna(), rng.normal(0, 0.3, fake.loc[after].shape), np.nan)
    a = run_backtest(rets, VolTargetTSMOM(cfg, historical_risk(rets, cfg), sizing, md.asset_class), "a")
    b = run_backtest(fake, VolTargetTSMOM(cfg, historical_risk(fake, cfg), sizing, md.asset_class), "b")
    pd.testing.assert_frame_equal(a.weights.loc[:t], b.weights.loc[:t], check_exact=True)
    risk = historical_risk(rets, cfg)
    for d in a.weights.index[::24]:
        w = a.weights.loc[d].to_numpy()
        if not w.any():
            continue
        _, cov = risk.at(d.to_period("M"))
        on = w != 0
        assert np.sqrt(w[on] @ cov[np.ix_(on, on)] @ w[on]) == pytest.approx(cfg.portfolio_vol_target, rel=1e-9)




def test_continuity_and_cta_v2_no_lookahead(full, daily):
    """Continuity (-Information Discreteness) for month t and the CTA_v2 weights up to t
    are unchanged when every return after t is scrambled."""
    from cta_backtest import VolTargetTSMOM, historical_risk
    from cta_backtest.trend_strength import continuity_monthly
    cfg, md = full
    rets = md.returns.loc["1985":"2000"]
    t = pd.Timestamp("1995-06-30")
    fd, fm = daily.copy(), rets.copy()
    rng = np.random.default_rng(3)
    a_d, a_m = fd.index > t, fm.index > t
    fd.loc[a_d] = np.where(fd.loc[a_d].notna(), rng.normal(0, 0.05, fd.loc[a_d].shape), np.nan)
    fm.loc[a_m] = np.where(fm.loc[a_m].notna(), rng.normal(0, 0.3, fm.loc[a_m].shape), np.nan)
    c0, c1 = continuity_monthly(daily, rets), continuity_monthly(fd, fm)
    p = t.to_period("M")
    pd.testing.assert_frame_equal(c0.loc[:p], c1.loc[:p], check_exact=True)
    assert np.nanmax(np.abs(c0.to_numpy())) <= 1.0
    mk = lambda r, c: VolTargetTSMOM(cfg, historical_risk(r, cfg), "capped", md.asset_class,  # noqa: E731
                                     strength_panel=c.rank(axis=1, pct=True), strength_threshold=0.5,
                                     strength_low_weight=0.5)
    ra, rb = run_backtest(rets, mk(rets, c0), "a"), run_backtest(fm, mk(fm, c1), "b")
    pd.testing.assert_frame_equal(ra.weights.loc[:t], rb.weights.loc[:t], check_exact=True)
