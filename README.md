# ORIE 5260 Midterm: CTA Strategy

Course baseline (12-month time-series momentum) upgraded with a portfolio vol target, a daily EWMA
risk model, risk caps and a trend-continuity filter.

## How to run

```
pip install -r requirements.txt
jupyter notebook CTA_Midterm.ipynb      # Kernel -> Restart & Run All, about one minute
```

The notebook rebuilds everything from `data/` and runs the look-ahead tests at the end.
Tests on their own: `python -m pytest -q tests`.

## Lecture 8: costs and execution (next-week handoff)

Run `python run_execution_study.py` from this directory after installing the same
requirements. This reproduces six midterm baselines, verifies zero-cost execution
against the original engine, and evaluates trading/rolling costs plus turnover
buffers and partial rebalancing. Selection uses data through 1999 only.

- [Short report](docs/EXECUTION_STUDY_REPORT.md): assumptions, measured results,
  limitations and next steps.
- `cta_backtest/execution.py`: actual-position replay and cost accounting.
- `tests/test_execution.py`: hand-computable accounting and timing tests.
- `output/execution_study/`: baseline snapshots, monthly ledgers, policy positions,
  comparison CSVs and a configuration/input-hash manifest.

Results are research scenarios, with flat cost assumptions and monthly execution.
Re-running overwrites study outputs; preserve a dated copy before changing settings.
The original notebook, strategy definitions and engine remain unchanged.

## Files

| Path | Contents |
|---|---|
| `CTA_Midterm.ipynb` | Main notebook: data, EDA, baseline, the four upgrades, summary, tests |
| `cta_backtest/config.py` | All parameters, with the reason for each value |
| `cta_backtest/data.py` | Loads the monthly panel and the asset map |
| `cta_backtest/daily.py` | Daily returns from the raw price files (used by the risk model) |
| `cta_backtest/signals.py` | Course signal: sign of the sum of the last 12 monthly returns |
| `cta_backtest/risk.py` | Volatility and correlation estimators (expanding, EWMA) |
| `cta_backtest/riskmodel.py` | Point-in-time risk models: course historical, daily EWMA |
| `cta_backtest/strategies.py` | Course baseline, static benchmark, vol-targeted strategy with caps |
| `cta_backtest/trend_strength.py` | Trend continuity (upgrade 4) |
| `cta_backtest/engine.py` | Monthly backtest: weights decided at t earn the return of t+1 |
| `cta_backtest/metrics.py` | Performance statistics |
| `cta_backtest/evaluation.py` | Risk-model scoring, alternative signals, regressions |
| `tests/test_lookahead.py` | Look-ahead and replication tests |
| `data/` | Course files (unchanged) |
| `MonthlyReturns.csv` | 58-column monthly panel rebuilt in section 1 of the notebook |
