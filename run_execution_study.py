"""Reproduce baselines and run the Lecture 8 cost/execution study.

Run from the project root: python run_execution_study.py
All policy selection uses return dates through 1999 only. The 2000-2014 period
has already been inspected in the midterm and is a historical holdout, not a
fresh untouched test. No strategy signal or risk-model parameter is retuned.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from cta_backtest import (BacktestConfig, StaticBenchmark, TSMOMBaseline,
                          VolTargetTSMOM, ewma_risk, historical_risk,
                          load_market_data, run_backtest)
from cta_backtest.daily import load_daily_returns
from cta_backtest.execution import ExecutionConfig, performance, replay_execution
from cta_backtest.trend_strength import continuity_monthly

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "execution_study"
PERIODS = {"train": (None, "1999-12-31"),
           "holdout": ("2000-01-01", "2014-12-31"), "full": (None, "2014-12-31")}


def split_stats(ledger):
    return {label: performance(ledger.loc[a:b]) for label, (a, b) in PERIODS.items()}


def table(frame, columns):
    """Write a compact Markdown table without an optional formatting dependency."""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline").mkdir(exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    cfg = BacktestConfig(end_date="2014-12")
    md = load_market_data(cfg)
    m = md.returns.loc[:"2014-12"]
    print("Building the unchanged historical and EWMA risk models...", flush=True)
    daily = load_daily_returns(cfg.daily_dir, m.columns, cfg.daily_splices, cfg.daily_max_gap_days)
    daily = daily.loc[:m.index[-1]]
    historical, ewma = historical_risk(m, cfg), ewma_risk(daily, m, cfg)
    cont = continuity_monthly(daily, m, cfg.momentum_lookback, cfg.continuity_window_days)
    strategies = {
        "STATIC": StaticBenchmark(cfg), "TSMOM_12": TSMOMBaseline(cfg),
        "VT11_historical": VolTargetTSMOM(cfg, historical, "baseline"),
        "VT11_ewma": VolTargetTSMOM(cfg, ewma, "baseline"),
        "CTA_v1": VolTargetTSMOM(cfg, ewma, "capped", md.asset_class),
        "CTA_v2": VolTargetTSMOM(cfg, ewma, "capped", md.asset_class,
                                 strength_panel=cont.rank(axis=1, pct=True),
                                 strength_low_weight=cfg.continuity_low_weight),
    }
    baselines = {}
    for name, strategy in strategies.items():
        print(f"Freezing baseline: {name}", flush=True)
        res = run_backtest(m, strategy, name)
        baselines[name] = res
        res.weights.to_csv(OUT / "baseline" / f"{name}_targets.csv")
        pd.concat([res.returns.rename("gross_return"),
                   res.turnover.set_axis(res.returns.index).rename("turnover")], axis=1).to_csv(
                       OUT / "baseline" / f"{name}_ledger.csv")
    # Match every comparison to the latest baseline inception, without resetting
    # the ongoing book at the train/holdout boundary.
    common_start = max(r.returns.index[0] for r in baselines.values())
    common_end = min(r.returns.index[-1] for r in baselines.values())
    assumptions = ExecutionConfig()
    rows, cost_rows, policy_rows = [], [], []
    base_ledgers = {}
    for name, res in baselines.items():
        zero = replay_execution(m, res.weights, replace(assumptions, trade_bps=0, roll_roundtrip_bps=0))
        np.testing.assert_allclose(zero.ledger.net_return, res.returns, atol=1e-14)
        np.testing.assert_allclose(zero.ledger.turnover, res.turnover, atol=1e-12)
        for scenario, trade, roll in [("zero", 0, 0), ("low", 1, 1), ("base", 2, 2), ("stress", 5, 5)]:
            result = replay_execution(m, res.weights, replace(assumptions, trade_bps=trade,
                                                              roll_roundtrip_bps=roll))
            ledger = result.ledger.loc[common_start:common_end]
            ledger.to_csv(OUT / "runs" / f"{name}_{scenario}.csv")
            if scenario == "base":
                base_ledgers[name] = ledger
            for period, stats in split_stats(ledger).items():
                rows.append(dict(strategy=name, scenario=scenario, period=period, **stats))
        # A separate one-way trading-cost grid holds the rolling assumption fixed.
        for bps in [0, 1, 2, 5, 10, 20, 50, 100]:
            ledger = replay_execution(m, res.weights, replace(assumptions, trade_bps=bps)).ledger
            for period, stats in split_stats(ledger.loc[common_start:common_end]).items():
                cost_rows.append(dict(strategy=name, trade_bps=bps, period=period, **stats))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "baseline_comparison.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / "cost_sensitivity.csv", index=False)
    print("Testing reduced-turnover policies on CTA_v2...", flush=True)
    policies = {"full": assumptions}
    policies.update({f"buffer_{b:.2f}": replace(assumptions, policy="buffer", buffer=b) for b in [.05, .10, .20]})
    policies.update({f"partial_{f:.2f}": replace(assumptions, policy="partial", fraction=f) for f in [.25, .50, .75]})
    policy_ledgers = {}
    for label, execution_cfg in policies.items():
        result = replay_execution(m, baselines["CTA_v2"].weights, execution_cfg, ewma)
        ledger = result.ledger.loc[common_start:common_end]
        policy_ledgers[label] = ledger
        ledger.to_csv(OUT / "runs" / f"CTA_v2_policy_{label}.csv")
        result.positions.to_csv(OUT / "runs" / f"CTA_v2_policy_{label}_positions.csv")
        result.trades.to_csv(OUT / "runs" / f"CTA_v2_policy_{label}_trades.csv")
        for period, stats in split_stats(ledger).items():
            policy_rows.append(dict(policy=label, period=period, **stats))
    policy_summary = pd.DataFrame(policy_rows)
    policy_summary.to_csv(OUT / "policy_comparison.csv", index=False)
    # Full execution is eligible to win: an enhancement is not assumed beneficial.
    ranking = policy_summary[policy_summary.period == "train"].sort_values(
        ["sharpe", "ann_turnover"], ascending=[False, True])
    selected = ranking.iloc[0].policy
    frozen = policies[selected]
    selected_rows = []
    for bps in [0, 1, 2, 5, 10, 20]:
        for label in ["full", selected] if selected != "full" else ["full"]:
            ledger = replay_execution(m, baselines["CTA_v2"].weights,
                                      replace(policies[label], trade_bps=bps), ewma).ledger
            for period, stats in split_stats(ledger.loc[common_start:common_end]).items():
                selected_rows.append(dict(policy=label, trade_bps=bps, period=period, **stats))
    pd.DataFrame(selected_rows).to_csv(OUT / "selected_policy_cost_sensitivity.csv", index=False)
    source_paths = ([ROOT / "MonthlyReturns.csv", ROOT / "data/AssetMapCsv.csv", Path(__file__)]
                    + sorted((ROOT / "data/FuturesUnderlyingData").glob("*.csv"))
                    + sorted((ROOT / "cta_backtest").glob("*.py")))
    manifest = dict(config=cfg.to_dict(), execution=asdict(assumptions),
                    policies={k: asdict(v) for k, v in policies.items()},
                    selected_policy=selected, selected_config=asdict(frozen),
                    selection="Highest net training Sharpe through 1999; lower turnover breaks ties",
                    common_start=str(common_start.date()), common_end=str(common_end.date()),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(summary, policy_summary, selected, manifest)
    print(f"Complete. Training-selected policy: {selected}. Results: {OUT}", flush=True)


def write_report(summary, policies, selected, manifest):
    hold = summary[(summary.period == "holdout") & (summary.scenario == "base")].copy()
    zero = summary[(summary.period == "holdout") & (summary.scenario == "zero")].set_index("strategy")
    display = pd.DataFrame({"Strategy": hold.strategy,
                            "Gross Sharpe (zero cost)": [f"{zero.loc[s, 'sharpe']:.3f}" for s in hold.strategy],
                            "Net Sharpe": hold.sharpe.map(lambda v: f"{v:.3f}"),
                            "Net ann. mean": hold.ann_return.map(lambda v: f"{v:.2%}"),
                            "Cost / year": (hold.ann_trade_cost + hold.ann_roll_cost).map(lambda v: f"{v:.2%}")})
    pol = policies[policies.period == "holdout"].set_index("policy")
    full, best = pol.loc["full"], pol.loc[selected]
    train = policies[policies.period == "train"].set_index("policy")
    policy_display = pd.DataFrame([dict(Policy=label, **{
        "Train net Sharpe": f"{train.loc[label, 'sharpe']:.3f}",
        "Holdout net Sharpe": f"{row.sharpe:.3f}",
        "Turnover / year": f"{row.ann_turnover:.2f}x",
        "Realized vol": f"{row.ann_vol:.2%}",
        "Max drawdown": f"{row.max_drawdown:.2%}"}) for label, row in pol.iterrows()])
    change = best.sharpe - full.sharpe
    assessment = ("improved" if change > 0 else "did not improve")
    if selected == "full":
        selection_text = (
            "Training selected **full execution**: none of the turnover-reduction policies "
            "improved training net Sharpe. Therefore no execution-policy change is recommended "
            "from this experiment. The 20% buffer has a higher historical-holdout Sharpe, "
            "but choosing it after seeing that result would introduce selection bias.")
    else:
        selection_text = (
            f"Training selected **{selected}**. Its holdout net Sharpe is **{best.sharpe:.3f}**, "
            f"versus **{full.sharpe:.3f}** for full execution ({change:+.3f}); it {assessment} "
            f"holdout Sharpe. Annual turnover changed from **{full.ann_turnover:.2f}x** to "
            f"**{best.ann_turnover:.2f}x**. The selected policy triggered "
            f"**{int(best.forced_rebalances)}** risk fallbacks in the holdout.")
    # Report brackets, not falsely precise break-even estimates, from the cost grid.
    costs = pd.read_csv(OUT / "cost_sensitivity.csv")
    costs = costs[costs.period == "holdout"]
    cta = costs[costs.strategy == "CTA_v2"].set_index("trade_bps").sort_index()
    baseline = costs[costs.strategy == "TSMOM_12"].set_index("trade_bps").sort_index()
    def crossing_bracket(series):
        hits = series[series <= 0]
        if hits.empty:
            return f"not reached on the grid up to {series.index.max()} bps"
        upper = hits.index[0]
        earlier = series.loc[series.index < upper]
        return f"between {earlier.index[-1]} and {upper} bps" if len(earlier) else f"at or below {upper} bps"
    advantage_bracket = crossing_bracket(cta.sharpe - baseline.sharpe)
    return_bracket = crossing_bracket(cta.ann_return)
    report = f"""# Lecture 8 execution study: handoff report

## What was done
Reproduced six existing strategies without changing their signals or risk parameters.
Saved targets, returns, turnover, input hashes and configuration. A new execution layer
replays actual positions, rather than discounting the old turnover series. It adds
trading and rolling costs, a relative no-trade buffer, and partial rebalancing.
Zero-cost full execution matches the legacy engine for all six strategies.

Common comparison window: **{manifest['common_start']} to {manifest['common_end']}**.
Selection uses returns through 1999; tables below use **2000-2014 ({int(full.months)} months)**.
The latter was already examined in the midterm and is not an untouched final test.

## Cost and execution assumptions
- Base: 2 bps one-way per traded notional, plus 2 bps combined close/open cost per roll,
  four rolls per year. Roll cost is accrued uniformly monthly on gross exposure.
- Low/base/stress scenarios use (trade, round-trip roll) bps of (1,1), (2,2), (5,5).
  A separate trading-cost grid fixes roll cost at 2 bps. These are assumptions, not estimates.
- Costs are charged in the funded month; subsequent exposure drift uses net NAV.
  Initial deployment is included; terminal liquidation is not. No cash yield, taxes,
  management fees, stock borrow charge or automatic notional-leverage financing charge is added.
- Buffers are 5%, 10%, 20% of absolute target; partial steps are 25%, 50%, 75%.
  Entries, exits and reversals execute immediately. If forecast volatility exceeds
  125% of the current target book's forecast volatility, execute the full target.
  This is a monthly execution guard, not a daily risk-management system.

## Results
{table(display, list(display.columns))}

Annual mean return and cost are arithmetic annualizations of monthly values; CAGR,
drawdown, volatility, exposure and cost components are in the accompanying CSV files.
CTA_v2's Sharpe advantage over TSMOM_12 disappears **{advantage_bracket}** of one-way
trading cost; its net arithmetic mean return crosses zero **{return_bracket}**.
These grid brackets hold rolling cost fixed at 2 bps per round trip and are scenario
diagnostics, not estimated market costs or a capacity limit.

{table(policy_display, list(policy_display.columns))}

{selection_text}
These are point estimates, not evidence of statistically significant improvement.
All candidate holdout results are diagnostic; they were not used to select the policy.

## Limitations and next-week priorities
1. Replace the flat roll accrual with contract-specific roll schedules and cost inputs.
   Verify the vendor's continuous-price construction before interpreting roll effects;
   this study models execution expense, not an extra roll-yield deduction.
2. Calibrate per-market trading costs and liquidity. Volume exists in raw data, but
   contract multipliers, comparable volume and execution schedules are needed for a
   defensible AUM/market-impact model. No capacity claim is made here.
3. Validate robustness with walk-forward selection and genuinely new data if available.
   Existing risk/signal settings were inherited from the midterm. Historical selection
   and a small parameter grid do not establish out-of-sample statistical significance.
4. Check final executed-position concentration, leverage and margin constraints; then
   introduce transaction-cost-aware portfolio optimization. The current fallback only
   limits excess forecast risk relative to the target, whose constraints remain inherited.
5. Monthly fills assume month-end information can fund next-month exposure. Model
   execution delay and daily risk monitoring before describing this as production-ready.
   Missing held returns retain the legacy zero-return treatment and are counted in CSVs.

## Reproduce and continue
Run `python -m pytest -q tests`, then `python run_execution_study.py` from the project root.
Outputs are regenerated in `output/execution_study/`. Inspect `manifest.json` for exact
settings and hashes; `baseline/` holds the frozen current-run reference; `runs/` holds
monthly cost ledgers and policy positions/trades. Re-running intentionally overwrites
these files, so preserve a dated copy before changing the experiment.

Code map: `cta_backtest/execution.py` (execution, cost accounting and statistics),
`run_execution_study.py` (baselines, experiments, training-only selection and report),
`tests/test_execution.py` (accounting, timing, drift and risk-fallback checks).
All new code comments and docstrings are in English. The midterm notebook and legacy
engine are unchanged. Course reference: Lecture 8, especially pages 24, 28 and 32.
"""
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/EXECUTION_STUDY_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
