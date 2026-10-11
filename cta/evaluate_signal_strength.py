"""Does trend strength at month-end t predict the next-month payoff of the TSMOM
position? Training sample only (t+1 <= risk_train_end); 2000+ is not looked at.

Payoff per instrument-month: u = sign_t * r_{t+1} / (sigma_t / sqrt(12)), with sign_t
the course 12-month signal and sigma_t the daily EWMA vol (risk model). Strength
measures (all computed from daily OHLC through t only):
  p_trend, p_trend_2w  Yanhao's hybrid_v2 trend probability (autonomous_quant_agent,
                       imported read-only, settings unchanged): rolling 252-day
                       percentile ranks of ADX(14), |close - VWAP(30)| / ATR(14) and
                       5-day persistence, weighted 0.40 / 0.35 / 0.25, floored to
                       0.85 / 0.65 by the strong / moderate rules
  adx_60               Wilder ADX on a 60-day window (monthly-horizon version)
  efficiency_60        |net 60-day change| / sum of |daily changes|
  variance_ratio       var(5-day returns) / (5 var(1-day returns)), 250 days
Also tested as a direction signal on its own: hybrid_v2 drift regime
(UPTREND +1 / DOWNTREND -1 / RANGE, MIXED 0).

Usage (from cta/):  python evaluate_signal_strength.py  -> output/signal_research/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "signal_research")
AQA_PARENT = os.environ.get("AQA_PARENT", os.path.expanduser("~/mnt"))   # folder holding autonomous_quant_agent
sys.path.insert(0, HERE)
sys.path.insert(0, AQA_PARENT)

from autonomous_quant_agent.config import settings as S  # noqa: E402
from autonomous_quant_agent.intelligence.regime_detection import adx, build_features  # noqa: E402
from cta_backtest import BacktestConfig, ewma_risk, load_market_data  # noqa: E402
from cta_backtest.daily import load_daily_returns  # noqa: E402
from cta_backtest.signals import tsmom_sign  # noqa: E402


def load_ohlcv(directory: str, iid: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(directory, f"{iid}.csv"), index_col=0, parse_dates=True).sort_index()
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return d


def hybrid_v2_features(o: pd.DataFrame) -> pd.DataFrame:
    """p_trend and drift regime exactly as run_regime_model_hybrid_v2 (same settings),
    without the per-row leverage-suitability loop. Every input is backward-looking."""
    lb = S.REGIME_HYBRID_LOOKBACK_DAYS
    minp = max(int(S.REGIME_HYBRID_MIN_PERIODS), min(lb, 40))
    f = build_features(o, atr_len=14, adx_len=14, vwap_win=30)
    rank = lambda x: x.rolling(lb, min_periods=minp).rank(pct=True)  # noqa: E731  same as the .apply rank
    a, g, p = rank(f["adx"]), rank(f["vwap_gap_norm_atr"].abs()), rank(f["vwap_sign_persist_5"].abs())
    strong = (a >= S.REGIME_HYBRID_STRONG_ADX_PCTL) & (g >= S.REGIME_HYBRID_STRONG_GAP_PCTL) & (p >= S.REGIME_HYBRID_STRONG_PERSIST_PCTL)
    mod = (a >= S.REGIME_HYBRID_MOD_ADX_PCTL) & (g >= S.REGIME_HYBRID_MOD_GAP_PCTL) & (p >= S.REGIME_HYBRID_MOD_PERSIST_PCTL) & ~strong
    score = (S.REGIME_HYBRID_W_ADX * a.clip(0, 1) + S.REGIME_HYBRID_W_GAP * g.clip(0, 1)
             + S.REGIME_HYBRID_W_PERSIST * p.clip(0, 1)).clip(0, 1)
    pt = pd.Series(np.where(strong, np.maximum(score, 0.85), np.where(mod, np.maximum(score, 0.65), score)),
                   index=f.index).where(score.notna()).clip(0, 1)
    c = f["close"]
    direction = (0.40 * c.pct_change(5) + 0.35 * c.pct_change(10) + 0.25 * c.pct_change(20)).ewm(span=3, adjust=False).mean()
    trend = pt >= S.REGIME_HYBRID_TREND_CUTOFF
    up, dn = direction > S.REGIME_HYBRID_DIR_UP_MIN, direction < S.REGIME_HYBRID_DIR_DOWN_MAX
    drift = np.where(trend & up, 1.0, np.where(trend & dn, -1.0, 0.0))     # UPTREND / DOWNTREND / RANGE or MIXED
    out = pd.DataFrame({"p_trend": pt, "p_trend_2w": pt.rolling(10, min_periods=1).mean(),
                        "drift_dir": pd.Series(drift, index=f.index).where(pt.notna())})
    out["adx_60"] = adx(o, 60)["adx"]
    ch = c.diff().abs()
    out["efficiency_60"] = (c - c.shift(60)).abs() / ch.rolling(60).sum()
    r1 = c.pct_change()
    r5 = c.pct_change(5)
    out["variance_ratio"] = r5.rolling(250).var() / (5 * r1.rolling(250).var())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = BacktestConfig()
    os.chdir(HERE)
    md = load_market_data(cfg)
    m = md.returns
    ids = list(m.columns)
    per = m.index.to_period("M")
    daily = load_daily_returns(cfg.daily_dir, ids, cfg.daily_splices, cfg.daily_max_gap_days)
    vol = pd.DataFrame(ewma_risk(daily, m, cfg).vol, index=per, columns=ids)

    feats = {}
    for i in ids:
        f = hybrid_v2_features(load_ohlcv(cfg.daily_dir, i))
        if i in cfg.daily_splices:                       # same splice as the return panel
            src, month = cfg.daily_splices[i]
            g = hybrid_v2_features(load_ohlcv(cfg.daily_dir, src))
            sw = pd.Period(month, "M").start_time
            f = pd.concat([f[f.index < sw], g[g.index >= sw]])
        feats[i] = f.groupby(f.index.to_period("M")).last()    # value on the last trading day of month t

    rows = []
    for k in range(12, len(per) - 1):
        t, t1 = per[k], per[k + 1]
        if t1 > pd.Period(cfg.risk_train_end, "M"):
            break
        sig = tsmom_sign(m.iloc[: k + 1], cfg.momentum_lookback, cfg.momentum_aggregation)
        r1 = m.iloc[k + 1]
        for i in ids:
            s, v, r = sig[i], vol.at[t, i], r1[i]
            if not (np.isfinite(s) and s != 0 and np.isfinite(v) and v > 0 and np.isfinite(r)) or t not in feats[i].index:
                continue
            x = feats[i].loc[t]
            rows.append({"t": t, "id": i, "u": s * r / (v / np.sqrt(12)), "u_dir": r / (v / np.sqrt(12)), "sign": s, **x.to_dict()})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "panel_train.csv"), index=False)

    # 1) quintiles of each strength measure within month: mean payoff, Q5-Q1 with t-stat over months
    meas = ["p_trend", "p_trend_2w", "adx_60", "efficiency_60", "variance_ratio"]
    qrows = []
    for mname in meas:
        d = df.dropna(subset=[mname]).copy()
        d["q"] = d.groupby("t")[mname].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1
                                                 if len(x) >= 10 else np.nan)
        d = d.dropna(subset=["q"])
        bym = d.groupby(["t", "q"])["u"].mean().unstack()
        spread = bym[5] - bym[1]
        row = {"measure": mname, "n": len(d), "months": len(bym)}
        for q in range(1, 6):
            row[f"Q{q}"] = bym[q].mean()
        row["Q5_minus_Q1"] = spread.mean()
        row["t_stat"] = spread.mean() / (spread.std() / np.sqrt(spread.count()))
        row["corr_with_u"] = d[[mname, "u"]].corr().iloc[0, 1]
        qrows.append(row)
    qt = pd.DataFrame(qrows).set_index("measure")
    qt.to_csv(os.path.join(OUT, "strength_quintiles_train.csv"))

    # 2) p_trend thresholds, including the 40% rule of thumb
    trows = []
    for col in ("p_trend", "p_trend_2w"):
        for th in (0.3, 0.4, 0.5, 0.55, 0.6):
            d = df.dropna(subset=[col])
            hi, lo = d[d[col] >= th], d[d[col] < th]
            sp = hi.groupby("t")["u"].mean() - lo.groupby("t")["u"].mean()
            trows.append({"measure": col, "threshold": th, "share_above": len(hi) / len(d),
                          "u_above": hi["u"].mean(), "u_below": lo["u"].mean(),
                          "diff": sp.mean(), "t_stat": sp.mean() / (sp.std() / np.sqrt(sp.count()))})
    tt = pd.DataFrame(trows)
    tt.to_csv(os.path.join(OUT, "p_trend_threshold_train.csv"), index=False)

    # 3) hybrid_v2 drift direction as a monthly direction signal, alone and vs the 12m sign
    d = df.dropna(subset=["drift_dir"])
    dd = []
    for lab, sub in [("UPTREND", d[d.drift_dir == 1]), ("DOWNTREND", d[d.drift_dir == -1]), ("RANGE/MIXED", d[d.drift_dir == 0])]:
        dd.append({"drift_regime": lab, "share": len(sub) / len(d), "next_month_r_over_sigma": sub["u_dir"].mean(),
                   "tsmom_payoff_u": sub["u"].mean(), "agrees_with_12m_sign": (sub["drift_dir"] == sub["sign"]).mean()})
    dt = pd.DataFrame(dd)
    agree = d[(d.drift_dir != 0)]
    dt.to_csv(os.path.join(OUT, "drift_direction_train.csv"), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("payoff u = sign * r / sigma_month (mean of a unit-risk TSMOM position; ~SR per month)\n")
    print(qt.to_string(), "\n")
    print(tt.to_string(index=False), "\n")
    print(dt.to_string(index=False))
    print("\nall TSMOM payoff, train:", round(df["u"].mean(), 3), " n =", len(df))
    print("drift agrees with 12m sign:", round((agree.drift_dir == agree.sign).mean(), 3),
          " u when agree:", round(agree[agree.drift_dir == agree.sign]["u"].mean(), 3),
          " u when disagree:", round(agree[agree.drift_dir != agree.sign]["u"].mean(), 3))


if __name__ == "__main__":
    main()
