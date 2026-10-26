# Lecture 8 execution study: handoff report

## What was done
Reproduced six existing strategies without changing their signals or risk parameters.
Saved targets, returns, turnover, input hashes and configuration. A new execution layer
replays actual positions, rather than discounting the old turnover series. It adds
trading and rolling costs, a relative no-trade buffer, and partial rebalancing.
Zero-cost full execution matches the legacy engine for all six strategies.

Common comparison window: **1976-02-27 to 2014-12-31**.
Selection uses returns through 1999; tables below use **2000-2014 (180 months)**.
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
| Strategy | Gross Sharpe (zero cost) | Net Sharpe | Net ann. mean | Cost / year |
| --- | --- | --- | --- | --- |
| STATIC | 0.723 | 0.698 | 10.75% | 0.39% |
| TSMOM_12 | 0.839 | 0.796 | 10.20% | 0.57% |
| VT11_historical | 0.959 | 0.906 | 11.45% | 0.70% |
| VT11_ewma | 1.177 | 1.101 | 12.11% | 0.86% |
| CTA_v1 | 1.161 | 1.075 | 11.85% | 0.97% |
| CTA_v2 | 1.172 | 1.092 | 12.13% | 0.91% |

Annual mean return and cost are arithmetic annualizations of monthly values; CAGR,
drawdown, volatility, exposure and cost components are in the accompanying CSV files.
CTA_v2's Sharpe advantage over TSMOM_12 disappears **between 20 and 50 bps** of one-way
trading cost; its net arithmetic mean return crosses zero **between 50 and 100 bps**.
These grid brackets hold rolling cost fixed at 2 bps per round trip and are scenario
diagnostics, not estimated market costs or a capacity limit.

| Policy | Train net Sharpe | Holdout net Sharpe | Turnover / year | Realized vol | Max drawdown |
| --- | --- | --- | --- | --- | --- |
| full | 1.392 | 1.092 | 22.32x | 11.11% | -11.97% |
| buffer_0.05 | 1.390 | 1.093 | 21.95x | 11.12% | -11.99% |
| buffer_0.10 | 1.386 | 1.096 | 21.28x | 11.08% | -11.86% |
| buffer_0.20 | 1.376 | 1.112 | 19.61x | 11.02% | -11.55% |
| partial_0.25 | 1.228 | 1.096 | 14.08x | 10.81% | -12.49% |
| partial_0.50 | 1.316 | 1.101 | 16.58x | 11.06% | -11.83% |
| partial_0.75 | 1.365 | 1.096 | 19.22x | 11.09% | -11.76% |

Training selected **full execution**: none of the turnover-reduction policies improved training net Sharpe. Therefore no execution-policy change is recommended from this experiment. The 20% buffer has a higher historical-holdout Sharpe, but choosing it after seeing that result would introduce selection bias.
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
4. Calibrate the new margin and financing scenarios and check concentration; then
   introduce transaction-cost-aware portfolio optimization. The current fallback only
   limits excess forecast risk relative to the target, whose constraints remain inherited.
5. Monthly fills assume month-end information can fund next-month exposure. Model
   execution delay and daily risk monitoring before describing this as production-ready.
   Missing held returns retain the legacy zero-return treatment and are counted in CSVs.

## Margin and financing extension
The original tables above isolate trading and rolling expense. The additional
scenarios below also model collateral capacity and incremental financing costs.
All retain 2 bps trading and 2 bps round-trip rolling costs and full execution.

Margin/NAV = margin rate x gross notional/NAV. With a 20% NAV cash reserve,
borrowing/NAV = max(margin/NAV - 80%, 0). Monthly financing expense equals
borrowing/NAV x annual incremental financing rate / 12. Margin is collateral,
not an expense. Rates of 2%, 5% and 8% are hypothetical incremental spreads net
of collateral/reference cash income, consistent with excess-return accounting.
The borrow scenarios assume credit is available; no lender limit is modeled.
The no-borrow scenario scales positions to fit the 80% NAV collateral budget,
then recomputes trades, costs and P&L. It can therefore miss the volatility target.

| Funding scenario | Avg margin / NAV | Borrowing months | Financing / year | Net Sharpe | Net ann. mean |
| --- | --- | --- | --- | --- | --- |
| margin10_rate2 | 58.5% | 27 | 0.03% | 1.090 | 12.10% |
| margin20_rate5 | 117.0% | 155 | 2.00% | 0.915 | 10.13% |
| margin30_rate8 | 175.5% | 171 | 7.72% | 0.400 | 4.41% |
| margin20_no_borrow | 76.9% | 0 | 0.00% | 0.947 | 8.14% |

These are monthly initial-margin proxies, not exchange margin schedules. They
exclude intramonth variation margin, collateral haircuts, portfolio offsets,
credit-line commitment fees and forced intraday liquidation. A zero modeled
funding gap does not establish that the strategy is safe from margin calls.
See `output/pdf/LECTURE8_RESULTS_SUMMARY.pdf` for the three-page visual summary.

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
