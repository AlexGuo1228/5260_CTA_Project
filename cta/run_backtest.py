"""Course baseline TSMOM vs static benchmark (Lecture 6 replication), and the
upgrade ladder: course 11% vol target -> daily EWMA risk model -> risk caps
(= CTA_v1), then a trend-continuity filter (= CTA_v2, midterm strategy); crisis-alpha
and SG Trend comparison (2000-2014, the factor file's span).

Usage (from cta/):  python run_backtest.py     # writes tables and charts to output/
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cta_backtest import (BacktestConfig, StaticBenchmark, TSMOMBaseline, VolTargetTSMOM,  # noqa: E402
                          coverage_table, ewma_risk, historical_risk, load_market_data, metrics,
                          run_backtest)
from cta_backtest.daily import load_daily_returns  # noqa: E402
from cta_backtest.trend_strength import continuity_monthly  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FACTORS = os.path.join(HERE, "..", "output", "cleaned", "lecture3_monthly_factors.csv")

LOOKBACKS = [1, 3, 6, 12]

# Lecture 6 published figures (Avg, Std, SR, Correl with static) for reference.
COURSE = {
    "TSMOM_12": (0.111, 0.111, 1.00, 0.08),
    "TSMOM_6": (0.090, 0.114, 0.79, -0.01),
    "TSMOM_3": (0.096, 0.118, 0.81, -0.09),
    "TSMOM_1": (0.081, 0.112, 0.72, -0.11),
    "STATIC": (0.069, 0.145, 0.47, 1.00),
}

# Lecture 6 vol-targeted version (Avg, Std, SR, Correl with static).
COURSE["VT11_historical"] = (0.119, 0.110, 1.08, 0.03)

# Categorical slots in fixed order; colour follows the series across charts.
COLORS = {"TSMOM_12": "#2a78d6", "STATIC": "#eb6834", "TSMOM_1": "#1baf7a",
          "TSMOM_3": "#eda100", "TSMOM_6": "#e87ba4",
          "VT11_historical": "#1baf7a", "VT11_ewma": "#eda100", "CTA_v1": "#e87ba4",
          "CTA_v2": "#008300"}
LADDER = ["TSMOM_12", "VT11_historical", "VT11_ewma", "CTA_v1", "CTA_v2"]
# Chart legends: what each run is, not just its code name.
LABELS = {
    "TSMOM_12": "Course baseline: sign(12m) x 40%/vol, 1/N",
    "TSMOM_1": "Course baseline, 1-month lookback",
    "TSMOM_3": "Course baseline, 3-month lookback",
    "TSMOM_6": "Course baseline, 6-month lookback",
    "STATIC": "Static benchmark: long all vol-adjusted assets",
    "VT11_historical": "Course L6: + 11% vol target (historical covariance)",
    "VT11_ewma": "+ daily EWMA risk model",
    "CTA_v1": "CTA_v1: + risk caps (risk-control upgrade only)",
    "CTA_v2": "CTA_v2: + trend-continuity filter (final)",
    "SG_Trend": "SG Trend Index (live CTA benchmark, net of fees)",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_lines(series: dict, title: str, ylabel: str, path: str):
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=130)
    for label, s in series.items():
        ax.plot(s.index, s.values, color=COLORS[label], linewidth=1.6, label=LABELS.get(label, label))
        if len(series) <= 2:
            ax.annotate(label, (s.index[-1], s.values[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def course_table(rets: dict, static: pd.Series) -> pd.DataFrame:
    """Avg / Std / SR / Correl(static), ours next to the Lecture 6 numbers."""
    rows = {}
    for k, r in rets.items():
        ours = (r.mean() * 12, r.std() * 12 ** 0.5, r.mean() / r.std() * 12 ** 0.5, r.corr(static))
        c = COURSE.get(k, (None,) * 4)
        rows[k] = {"avg": ours[0], "std": ours[1], "sr": ours[2], "corr_static": ours[3],
                   "course_avg": c[0], "course_std": c[1], "course_sr": c[2], "course_corr_static": c[3]}
    return pd.DataFrame(rows).T


def ladder_table(results: dict, static: pd.Series, asset_class: pd.Series) -> pd.DataFrame:
    """Step-2 comparison: performance, vol tracking, leverage, costs, class risk."""
    rows = {}
    for k in LADDER:
        r = results[k]
        x = r.returns
        tr, te = x[:"1999"], x["2000":]
        sr = lambda y: y.mean() / y.std() * 12 ** 0.5  # noqa: E731
        g = r.held_weights.abs().sum(axis=1)
        row = {"avg": x.mean() * 12, "std": x.std() * 12 ** 0.5, "sr": sr(x), "corr_static": x.corr(static),
               "sr_to_1999": sr(tr), "sr_2000_on": sr(te), "vol_to_1999": tr.std() * 12 ** 0.5,
               "vol_2000_on": te.std() * 12 ** 0.5,
               "yearly_vol_abs_dev_from_11pct": (x.groupby(x.index.year).std() * 12 ** 0.5 - 0.11).abs().mean(),
               "max_drawdown": metrics.drawdown(x).min(), "worst_month": x.min(), "skew": x.skew(),
               "gross_avg": g.mean(), "gross_max": g.max(), "ann_turnover": r.turnover.iloc[1:].mean() * 12}
        for bps in (2, 5):
            net = x - r.turnover.shift(1).reindex(x.index).fillna(0.0) * bps / 1e4
            row[f"sr_net_{bps}bp"] = sr(net)
            row[f"sr_net_{bps}bp_to_1999"] = sr(net[:"1999"])
            row[f"sr_net_{bps}bp_2000_on"] = sr(net["2000":])
        for c, v in metrics.by_asset_class(r, asset_class)["risk_share"].items():
            row[f"risk_share_{c}"] = v
        rows[k] = row
    return pd.DataFrame(rows)


def _ols(y: pd.Series, X: pd.DataFrame) -> dict:
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    b, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    e = y.to_numpy() - A @ b
    se = np.sqrt(np.diag(e @ e / (len(y) - A.shape[1]) * np.linalg.inv(A.T @ A)))
    out = {"alpha_ann": b[0] * 12, "alpha_t": b[0] / se[0]}
    for i, c in enumerate(X.columns, 1):
        out[f"beta_{c}"], out[f"t_{c}"] = b[i], b[i] / se[i]
    return out


def crisis_alpha(rets: dict) -> None:
    """Course L4 tests on 2000-2014: r = a + b Mkt + c Mkt^2 (c > 0 = smile / crisis
    alpha), stats and correlation with SG Trend, and the smile chart."""
    f = pd.read_csv(FACTORS, encoding="utf-8-sig")
    f.index = pd.PeriodIndex(f["Month"], freq="M")
    mkt = f["Mkt-RF"].rename("Mkt")
    X = pd.concat([mkt, (mkt ** 2).rename("Mkt2")], axis=1)
    series = {k: v.set_axis(v.index.to_period("M")) for k, v in rets.items()}
    series["SG_Trend"] = f["SG Trend"]
    rows = {}
    for k, r in series.items():
        r = r.reindex(f.index).dropna()
        reg = _ols(r, X.loc[r.index])
        reg.update({"avg": r.mean() * 12, "std": r.std() * 12 ** 0.5, "sr": r.mean() / r.std() * 12 ** 0.5,
                    "corr_mkt": r.corr(mkt.loc[r.index]), "corr_sg_trend": r.corr(f["SG Trend"].loc[r.index]),
                    "ret_worst_10pct_mkt_months": r[mkt.loc[r.index] <= mkt.quantile(0.1)].mean(),
                    "months": len(r)})
        rows[k] = reg
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "crisis_alpha_2000_2014.csv"))

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=130)
    grid = np.linspace(mkt.min(), mkt.max(), 100)
    for k, col in (("CTA_v2", "#2a78d6"), ("SG_Trend", "#eb6834")):
        r = series[k].reindex(f.index).dropna()
        ax.scatter(mkt.loc[r.index], r, s=14, color=col, alpha=0.55, label=LABELS.get(k, k), edgecolors="none")
        c = np.polyfit(mkt.loc[r.index], r, 2)
        ax.plot(grid, np.polyval(c, grid), color=col, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Crisis alpha: monthly return vs equity market, 2000-2014", loc="left", fontsize=12, color=INK)
    ax.set_xlabel("US equity market excess return (Mkt-RF)", color=MUTED, fontsize=9)
    ax.set_ylabel("Strategy monthly return", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", title="dots: months; lines: quadratic fit", title_fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "crisis_alpha_smile.png"))
    plt.close(fig)


def main():
    os.chdir(HERE)
    os.makedirs(OUT, exist_ok=True)
    cfg = BacktestConfig()
    md = load_market_data(cfg)
    coverage_table(md).to_csv(os.path.join(OUT, "universe_coverage.csv"))

    def run(strategy, name):
        return run_backtest(md.returns, strategy, name=name, cost_bps=cfg.cost_bps,
                            start_date=cfg.start_date, end_date=cfg.end_date)

    results = {"STATIC": run(StaticBenchmark(cfg), "STATIC")}
    for lb in LOOKBACKS:
        results[f"TSMOM_{lb}"] = run(TSMOMBaseline(replace(cfg, momentum_lookback=lb)), f"TSMOM_{lb}")
    daily = load_daily_returns(cfg.daily_dir, md.returns.columns, cfg.daily_splices, cfg.daily_max_gap_days)
    hist, ewma = historical_risk(md.returns, cfg), ewma_risk(daily, md.returns, cfg)
    results["VT11_historical"] = run(VolTargetTSMOM(cfg, hist, "baseline"), "VT11_historical")
    results["VT11_ewma"] = run(VolTargetTSMOM(cfg, ewma, "baseline"), "VT11_ewma")
    results["CTA_v1"] = run(VolTargetTSMOM(cfg, ewma, "capped", md.asset_class), "CTA_v1")
    cont = continuity_monthly(daily, md.returns, cfg.momentum_lookback, cfg.continuity_window_days)
    results["CTA_v2"] = run(VolTargetTSMOM(cfg, ewma, "capped", md.asset_class,
                                           strength_panel=cont.rank(axis=1, pct=True), strength_threshold=0.5,
                                           strength_low_weight=cfg.continuity_low_weight), "CTA_v2")
    base, static = results["TSMOM_12"], results["STATIC"]
    assert len({tuple(r.returns.index) for r in results.values()}) == 1, "Runs cover different samples"

    rets = {k: r.returns for k, r in results.items()}
    order = ["TSMOM_12", "STATIC", "TSMOM_1", "TSMOM_3", "TSMOM_6"]
    course_table({k: rets[k] for k in order + ["VT11_historical"]}, rets["STATIC"]).to_csv(
        os.path.join(OUT, "course_comparison.csv"))
    ladder_table(results, rets["STATIC"], md.asset_class).to_csv(os.path.join(OUT, "enhancement_comparison.csv"))
    summary = metrics.compare([results[k] for k in order])
    summary.loc["corr_static"] = [rets[k].corr(rets["STATIC"]) for k in order]
    summary.to_csv(os.path.join(OUT, "summary.csv"))

    cls = []
    for k in ("TSMOM_12", "STATIC"):
        t = metrics.by_asset_class(results[k], md.asset_class)
        t.insert(0, "portfolio", k)
        cls.append(t.reset_index())
    pd.concat(cls).to_csv(os.path.join(OUT, "by_asset_class.csv"), index=False)

    pd.DataFrame(rets)[order + LADDER[1:]].to_csv(os.path.join(OUT, "monthly_returns.csv"))
    results["CTA_v2"].weights.to_csv(os.path.join(OUT, "weights_cta_v2.csv"))
    base.weights.to_csv(os.path.join(OUT, "weights_tsmom_12.csv"))
    static.weights.to_csv(os.path.join(OUT, "weights_static.csv"))

    # Course convention: monthly returns, additively cumulated.
    plot_lines({k: rets[k].cumsum() for k in ("TSMOM_12", "STATIC")},
               "TSMOM baseline vs static benchmark", "Cumulative return (additive)",
               os.path.join(OUT, "cumulative_tsmom_vs_static.png"))
    plot_lines({k: rets[k].cumsum() for k in ["TSMOM_1", "TSMOM_3", "TSMOM_6", "TSMOM_12", "STATIC"]},
               "TSMOM by lookback vs static benchmark", "Cumulative return (additive)",
               os.path.join(OUT, "cumulative_lookbacks.png"))
    plot_lines({"TSMOM_12": rets["TSMOM_12"].rolling(12).std() * 12 ** 0.5},
               "TSMOM baseline: rolling 12-month realised volatility", "Annualised volatility",
               os.path.join(OUT, "realized_vol_tsmom_12.png"))

    plot_lines({k: rets[k].cumsum() for k in LADDER},
               "Course baseline to CTA_v2", "Cumulative return (additive)",
               os.path.join(OUT, "cumulative_enhancements.png"))
    plot_lines({k: rets[k].rolling(12).std() * 12 ** 0.5 for k in LADDER[1:]},
               "Rolling 12-month realised volatility (target 11%)", "Annualised volatility",
               os.path.join(OUT, "realized_vol_vol_target.png"))

    crisis_alpha({k: rets[k] for k in ["STATIC"] + LADDER})

    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(pd.read_csv(os.path.join(OUT, "course_comparison.csv"), index_col=0))


if __name__ == "__main__":
    main()
