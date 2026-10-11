"""Step 1b: choose the correlation model (EWMA half-life, overlap k) for the
covariance Sigma = D C D, D = daily EWMA vols (step 1a).

Covariance only matters through portfolios, so it is scored on portfolios formed
at month-end t with information through t and held over t+1:
  TSMOM    course baseline weights           STATIC   static benchmark weights
  CLASS    long inverse-vol book per asset class (4 per month)
  RANDOM   `n_random_portfolios` random long/short inverse-vol books
Forecast F = w' Sigma w (annualised). Metrics as in step 1a: bias std(r_p/sqrt(F/12)),
MRAD, QLIKE vs daily realised variance, QLIKE_monthly = ln F + 12 r_p^2 / F.
Candidates: course expanding covariance (L6), EWMA vols with zero correlation (the
course baseline assumption), and EWMA correlations over the grid. Chosen on the
training sample by QLIKE_monthly averaged over the four portfolio groups.

Usage (from cta/):  python evaluate_covariance.py   -> output/risk_model/
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from cta_backtest import BacktestConfig, StaticBenchmark, TSMOMBaseline, load_market_data, run_backtest
from cta_backtest.daily import load_daily_returns
from cta_backtest.risk import ewma_correlation_monthly, ewma_volatility_monthly, expanding_covariance_monthly

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "risk_model")
GROUPS = ["TSMOM", "STATIC", "CLASS", "RANDOM"]


def main():
    os.chdir(HERE)
    os.makedirs(OUT, exist_ok=True)
    cfg = BacktestConfig()
    md = load_market_data(cfg)
    monthly = md.returns
    ids = list(monthly.columns)
    per = monthly.index.to_period("M")
    daily = load_daily_returns(cfg.daily_dir, ids, cfg.daily_splices, cfg.daily_max_gap_days)

    # ---- risk models, all indexed by month period t ----
    vol = ewma_volatility_monthly(daily, cfg.vol_halflife_days, cfg.vol_ewma_min_days)
    vol = vol.reindex(index=per, columns=ids).to_numpy()
    p_e, _, exp_cov = expanding_covariance_monthly(monthly, cfg.vol_min_months)
    models = {"expanding_cov": exp_cov}
    diag = np.zeros_like(exp_cov)
    idx = np.arange(len(ids))
    diag[:, idx, idx] = vol ** 2
    models["ewma_diag"] = diag
    for k in cfg.corr_overlap_grid:
        for h in cfg.corr_halflife_grid:
            p_c, _, C = ewma_correlation_monthly(daily, h, k, cfg.corr_min_days)
            C = pd.Series(list(C), index=p_c).reindex(per).to_numpy()
            models[f"ewma_corr_{int(h)}d_k{k}"] = np.array([np.outer(v, v) * c for v, c in zip(vol, C)])

    # ---- portfolios at t ----
    w_ts = run_backtest(monthly, TSMOMBaseline(cfg), "ts").weights.reindex(monthly.index).fillna(0.0)
    w_st = run_backtest(monthly, StaticBenchmark(cfg), "st").weights.reindex(monthly.index).fillna(0.0)
    rng = np.random.default_rng(cfg.random_seed)
    cls = md.asset_class.reindex(ids).to_numpy()
    r_next = monthly.shift(-1).fillna(0.0).to_numpy()
    dfill = daily.fillna(0.0)
    dmonth = daily.index.to_period("M")
    day_groups = {p: dfill.to_numpy()[dmonth == p] for p in per}

    rows = []
    for t in range(len(per) - 1):
        common = np.isfinite(vol[t]) & np.isfinite(np.diag(exp_cov[t])) & ~np.isnan(monthly.iloc[t].to_numpy())
        if common.sum() < cfg.min_assets:
            continue
        books = {"TSMOM": [w_ts.iloc[t].to_numpy()], "STATIC": [w_st.iloc[t].to_numpy()], "CLASS": [], "RANDOM": []}
        inv = np.where(common, 1.0 / np.where(common, vol[t], 1.0), 0.0)
        for c in ("COMM", "EQ", "FI", "FX"):
            m = common & (cls == c)
            if m.sum() >= 2:
                books["CLASS"].append(np.where(m, inv, 0.0) / m.sum())
        for _ in range(cfg.n_random_portfolios):
            s = rng.choice([-1.0, 1.0], len(ids)) * rng.uniform(0.5, 1.5, len(ids))
            books["RANDOM"].append(np.where(common, s * inv, 0.0) / common.sum())
        dnext = day_groups[per[t + 1]]
        for g, ws in books.items():
            for w in ws:
                if not np.any(w) or np.any(w[~common] != 0):
                    continue
                rp = float(w @ r_next[t])
                rv = float(((dnext @ w) ** 2).sum() * 12.0)
                rec = {"t": per[t], "group": g, "rp": rp, "rv": rv}
                for name, S in models.items():
                    Sm = np.nan_to_num(S[t])
                    rec[name] = float(w @ Sm @ w)
                rows.append(rec)
    df = pd.DataFrame(rows)
    names = list(models)
    df = df[(df["rv"] > 0) & (df[names] > 0).all(axis=1)]
    df["train"] = (df["t"] + 1) <= pd.Period(cfg.risk_train_end, "M")
    df["year"] = df["t"].dt.year

    out = []
    for sample, part in [("train", df[df["train"]]), ("test", df[~df["train"]])]:
        for g in GROUPS + ["ALL_GROUPS"]:
            sub = part if g == "ALL_GROUPS" else part[part["group"] == g]
            for n in names:
                F = sub[n]
                z = sub["rp"] / np.sqrt(F / 12.0)
                q = sub["rv"] / F
                by = pd.DataFrame({"z": z.values, "y": sub["year"].values, "g": sub["group"].values})
                b_y = by.groupby(["g", "y"])["z"].std()
                out.append({"sample": sample, "group": g, "model": n, "n": len(sub), "bias": z.std(),
                            "mrad": (b_y - 1).abs().mean(), "qlike": (q - np.log(q) - 1).mean(),
                            "qlike_monthly": (np.log(F) + 12 * sub["rp"] ** 2 / F).mean()})
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(OUT, "covariance_selection.csv"), index=False)

    # Selection: training QLIKE_monthly averaged over the four groups (each group counts once).
    tr = res[(res["sample"] == "train") & res["group"].isin(GROUPS)]
    score = tr.groupby("model")["qlike_monthly"].mean().sort_values()
    score.rename("train_qlike_monthly_avg_groups").to_csv(os.path.join(OUT, "covariance_ranking.csv"))
    best = score.index[0]

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    show = res[res["group"].isin(GROUPS)].pivot_table(index="model", columns=["sample", "group"], values="bias")
    print("bias statistic (ideal 1)\n", show.loc[names].to_string())
    show = res[res["group"].isin(GROUPS)].pivot_table(index="model", columns=["sample", "group"], values="qlike_monthly")
    print("\nQLIKE_monthly (lower is better)\n", show.loc[names].to_string())
    print("\ntraining ranking (avg over groups):\n", score.to_string())
    print("\nchosen:", best)


if __name__ == "__main__":
    main()
