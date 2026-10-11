"""Signal research behind CTA_v2. Every alternative is run on the CTA_v1 engine
(EWMA risk model, soft caps, 11% vol target, same universe and months) and judged
on the training sample (to 1999); 2000-2014 is reported, never used to choose.

  1. signal replacements: risk-adjusted 12m (monthly r / prior vol), multi-horizon
     signs (raw 1/3/12, risk-adjusted 3/12, 1/3/12, 3/6/12, 1/3/6/12, 6/12)
  2. hybrid_v2 p_trend filter (windows x1/x2/x4, threshold .3/.4/.5, low weight 0/.5)
  3. trend continuity (-Information Discreteness): continuity < 0 halved, below the
     monthly median halved  -> the median rule is CTA_v2

Usage (from cta/):  python research_signals.py   -> output/signal_research/
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from cta_backtest import BacktestConfig, VolTargetTSMOM, ewma_risk, load_market_data, metrics, run_backtest
from cta_backtest.daily import load_daily_returns
from cta_backtest.trend_strength import continuity_monthly, p_trend_monthly

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "signal_research")


def sign_panel(monthly: pd.DataFrame, lookbacks, vol: np.ndarray = None) -> pd.DataFrame:
    """Mean over horizons of sign(sum of the last L monthly returns); with `vol`
    (risk-model vols, row m-1 known before month m) each return is first divided by
    the monthly vol forecast made before it. Row t uses months <= t only."""
    x = monthly.copy()
    x.index = x.index.to_period("M")
    if vol is not None:
        prior = pd.DataFrame(vol, index=x.index, columns=x.columns).shift(1) / np.sqrt(12.0)
        x = x / prior
    parts = [np.sign(x.rolling(L, min_periods=L).sum()) for L in lookbacks]
    return sum(parts) / len(parts)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.chdir(HERE)
    cfg = BacktestConfig()
    md = load_market_data(cfg)
    m = md.returns
    daily = load_daily_returns(cfg.daily_dir, m.columns, cfg.daily_splices, cfg.daily_max_gap_days)
    E = ewma_risk(daily, m, cfg)
    sr = lambda y: y.mean() / y.std() * 12 ** 0.5  # noqa: E731

    def run(label, **kw):
        r = run_backtest(m, VolTargetTSMOM(cfg, E, "capped", md.asset_class, **kw), label)
        x = r.returns
        return {"sr_to_1999": sr(x[:"1999"]), "sr_2000_on": sr(x["2000":]), "sr": sr(x),
                "max_drawdown": metrics.drawdown(x).min(), "skew": x.skew(),
                "ann_turnover": r.turnover.iloc[1:].mean() * 12, "months": len(x)}

    # 1) signal replacements
    rows = {"CTA_v1 sign(12m)": run("v1"),
            "raw signs 1/3/12": run("m", signal_panel=sign_panel(m, [1, 3, 12]))}
    for lbs in ([12], [3, 12], [1, 3, 12], [3, 6, 12], [1, 3, 6, 12], [6, 12]):
        rows["risk-adj signs " + "/".join(map(str, lbs))] = run("r", signal_panel=sign_panel(m, lbs, E.vol))
    pd.DataFrame(rows).T.to_csv(os.path.join(OUT, "signal_candidates.csv"))

    # 2) p_trend filter grid
    rows = []
    for s in (1, 2, 4):
        P = p_trend_monthly(cfg.daily_dir, m.columns, cfg.daily_splices, s)
        for th in (0.3, 0.4, 0.5):
            for lw in (0.0, 0.5):
                rows.append({"window_scale": s, "threshold": th, "low_weight": lw,
                             **run("p", strength_panel=P, strength_threshold=th, strength_low_weight=lw)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "p_trend_grid.csv"), index=False)

    # 3) trend continuity (-Information Discreteness)
    C = continuity_monthly(daily, m, cfg.momentum_lookback, cfg.continuity_window_days)
    rows = {"CTA_v1": run("v1"),
            "continuity < 0 halved": run("c0", strength_panel=C, strength_threshold=0.0, strength_low_weight=0.5),
            "continuity below median halved (CTA_v2)": run("c1", strength_panel=C.rank(axis=1, pct=True),
                                                            strength_threshold=0.5, strength_low_weight=0.5)}
    pd.DataFrame(rows).T.to_csv(os.path.join(OUT, "information_discreteness.csv"))

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    for f in ("signal_candidates.csv", "information_discreteness.csv"):
        print(pd.read_csv(os.path.join(OUT, f), index_col=0)[["sr_to_1999", "sr_2000_on", "ann_turnover"]], "\n")


if __name__ == "__main__":
    main()
