"""Step 1a: choose the half-life of the daily EWMA volatility model.

Every forecast is made at month-end t with data through t and scored against
month t+1:
  bias statistic  b = std(z), z = r_{t+1} / (sigma_t / sqrt(12))   (ideal 1)
  MRAD            mean over instrument-years of |b_year - 1|        (responsiveness)
  QLIKE           RV/F - ln(RV/F) - 1, F = sigma_t^2, RV = 12 * sum of squared
                  daily returns in t+1, both annualised             (ideal 0)
  QLIKE_monthly   ln F + 12 r_{t+1}^2 / F: same loss with the squared monthly
                  return as the proxy; scores the holding horizon itself and
                  does not favour daily-based models (lower is better)
All methods are scored on the same instrument-months. The half-life is chosen on
the training sample (t+1 <= risk_train_end) by lowest QLIKE_monthly; the test
sample is reported but never used for the choice. QLIKE against daily realised
variance is reported but not used for the choice: daily RV omits within-month
autocovariance by construction. (Tested and dropped: EWMA on overlapping k-day
returns, k = 3/5/10, did not lower the bias or the test-sample loss.)

Usage (from cta/):  python evaluate_risk_model.py   -> output/risk_model/
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cta_backtest import BacktestConfig, load_market_data  # noqa: E402
from cta_backtest.daily import load_daily_returns  # noqa: E402
from cta_backtest.risk import ewma_volatility_monthly, expanding_volatility_panel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "risk_model")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def stack(panel: pd.DataFrame, name: str) -> pd.Series:
    s = panel.stack()
    s.index.names = ["month", "id"]
    return s.rename(name)


def score(df: pd.DataFrame, f: str) -> dict:
    """df rows: instrument-month t with next-month return r1 and realised var rv1."""
    sig_m = df[f] / np.sqrt(12.0)
    z = df["r1"] / sig_m
    q = df["rv1"] / df[f] ** 2
    year = df.index.get_level_values("month").year
    by = pd.DataFrame({"z": z.values, "id": df.index.get_level_values("id"), "y": year})
    b_iy = by.groupby(["id", "y"])["z"].agg(["std", "count"])
    b_iy = b_iy.loc[b_iy["count"] >= 10, "std"]
    b_i = by.groupby("id")["z"].std()
    return {"n": len(df), "bias": z.std(), "bias_median_instr": b_i.median(),
            "mrad": (b_iy - 1).abs().mean(), "qlike": (q - np.log(q) - 1).mean(),
            "qlike_monthly": (np.log(df[f] ** 2) + 12 * df["r1"] ** 2 / df[f] ** 2).mean()}


def main():
    os.chdir(HERE)
    os.makedirs(OUT, exist_ok=True)
    cfg = BacktestConfig()
    md = load_market_data(cfg)
    monthly = md.returns
    daily = load_daily_returns(cfg.daily_dir, monthly.columns, cfg.daily_splices, cfg.daily_max_gap_days)

    per = monthly.copy()
    per.index = per.index.to_period("M")
    r1 = per.shift(-1)                                                 # return of t+1, row t
    rv = (daily ** 2).groupby(daily.index.to_period("M")).sum() * 12   # realised var of each month
    rv1 = rv.reindex(per.index).shift(-1)

    forecasts = {"expanding": expanding_volatility_panel(monthly, cfg.vol_min_months)}
    for h in cfg.vol_halflife_grid:
        forecasts[f"ewma_{int(h)}d"] = ewma_volatility_monthly(daily, h, cfg.vol_ewma_min_days)

    df = pd.concat([stack(r1, "r1"), stack(rv1, "rv1")] +
                   [stack(f.reindex(index=per.index, columns=per.columns), k) for k, f in forecasts.items()],
                   axis=1)
    df = df.dropna()
    df = df[(df["rv1"] > 0) & (df[list(forecasts)] > 0).all(axis=1)]
    t1 = df.index.get_level_values("month") + 1
    train = t1 <= pd.Period(cfg.risk_train_end, "M")

    rows = []
    for sample, mask in [("train", train), ("test", ~train)]:
        for k in forecasts:
            rows.append({"sample": sample, "method": k, **score(df[mask], k)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "vol_halflife_selection.csv"), index=False)

    tr = res[(res["sample"] == "train") & res["method"].str.startswith("ewma")]
    best = tr.loc[tr["qlike_monthly"].idxmin(), "method"]
    plain = f"ewma_{int(cfg.vol_halflife_days)}d"

    # Per asset class, chosen half-life vs the course expanding estimator (test sample).
    cls = md.asset_class.reindex(df.index.get_level_values("id")).values
    crow = []
    for sample, mask in [("train", train), ("test", ~train)]:
        for c in sorted(set(cls)):
            m = mask & (cls == c)
            for k in dict.fromkeys(("expanding", plain, best)):
                crow.append({"sample": sample, "asset_class": c, "method": k, **score(df[m], k)})
    pd.DataFrame(crow).to_csv(os.path.join(OUT, "vol_by_asset_class.csv"), index=False)

    # Yearly pooled bias statistic: does the model keep up with regime changes?
    yr = df.index.get_level_values("month").year
    yearly = pd.DataFrame({k: (df["r1"] / (df[k] / np.sqrt(12))).groupby(yr).std()
                           for k in dict.fromkeys(("expanding", plain, best))})
    n_y = df.groupby(yr).size()
    yearly.to_csv(os.path.join(OUT, "vol_bias_by_year.csv"))
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=130)
    band = np.sqrt(2.0 / n_y)
    ax.fill_between(yearly.index, 1 - band, 1 + band, color=GRID, label="±√(2/N) band")
    for k, col in zip(dict.fromkeys(("expanding", plain, best)), ("#eb6834", "#1baf7a", "#2a78d6")):
        ax.plot(yearly.index, yearly[k], color=col, linewidth=1.8, label=k)
    ax.axhline(1.0, color=MUTED, linewidth=0.8)
    ax.axvline(int(cfg.risk_train_end[:4]) + 0.5, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(int(cfg.risk_train_end[:4]) + 0.8, ax.get_ylim()[1] * 0.97, "test sample →", color=MUTED, fontsize=8, va="top")
    ax.set_title("Bias statistic by year: realised / predicted volatility (ideal = 1)", loc="left", fontsize=12, color=INK)
    ax.set_ylabel("Bias statistic", color=MUTED, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vol_bias_by_year.png"))
    plt.close(fig)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(res.to_string(index=False))
    print("\nchosen on training QLIKE_monthly:", best)


if __name__ == "__main__":
    main()
