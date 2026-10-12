# ORIE 5260 CTA Project

Monthly time-series momentum (TSMOM) on 58 futures across COMM, EQ, FI and FX, 1969-2014.
Two stages: a data notebook that rebuilds the monthly return panel from daily prices,
and a backtest package that consumes it.

## Layout

```text
data/                               # course inputs, never modified
  FuturesUnderlyingData/*.csv       # daily OHLC / volume / open interest, one file per instrument (62)
  MonthlyReturns.csv                # course reference monthly returns (58 columns)
  AssetMapCsv.csv                   # ID, name, currency, asset class (66 rows)
  Lecture3_livedata.xlsx            # SG Trend, Fama-French factors, RF, GSCI (monthly)
MonthlyReturns.csv                  # cleaned 58-column panel = input to the backtest
ORIE5260_Futures_Analysis.ipynb     # stage 1: data cleaning, reconciliation, EDA
build_analysis_notebook.py          # rebuilds the notebook source (resets outputs)
execute_analysis_notebook.py        # optional headless execution helper
output/                             # stage 1 results (quality/, cleaned/, returns/, statistics/, figures/, report.md)
presentation/CTA_Midterm_Slides.pdf # midterm slides (10 pages)
cta/                                # stage 2: backtest
  cta_backtest/config.py            # every parameter (BacktestConfig)
  cta_backtest/data.py              # loads ../MonthlyReturns.csv and the asset map
  cta_backtest/signals.py           # TSMOM sign
  cta_backtest/risk.py              # expanding-window and daily EWMA volatility
  cta_backtest/daily.py             # daily returns for the 58 columns (same merge rules)
  cta_backtest/strategies.py        # TSMOMBaseline, StaticBenchmark, VolTargetTSMOM (CTA_v1 / CTA_v2)
  cta_backtest/trend_strength.py    # trend continuity (CTA_v2 filter), p_trend (research)
  cta_backtest/riskmodel.py         # point-in-time risk models: historical, monthly EWMA, daily EWMA
  cta_backtest/engine.py            # backtest loop, one-month lag enforced
  cta_backtest/metrics.py           # performance statistics
  run_backtest.py                   # baseline replication, upgrade ladder, crisis alpha -> cta/output/
  research_signals.py               # every signal alternative tested -> cta/output/signal_research/
  evaluate_risk_model.py            # step 1a: volatility half-life -> cta/output/risk_model/
  evaluate_covariance.py            # step 1b: correlation model    -> cta/output/risk_model/
  evaluate_signal_strength.py       # strength-measure diagnostics (imports autonomous_quant_agent read-only)
  tests/test_lookahead.py           # look-ahead, unit and replication tests
  output/                           # stage 2 results
```

## Stage 1: data

Returns are month-end close ratios from the daily files (already rolled by the vendor).
The cleaned panel keeps the reference's 58 columns, order and dates. It matches every
reference value within 1e-6 and adds 32 values the reference leaves blank where the
exchange was shut on the month-end date (reconstructed from daily closes; no imputation).
Instrument merging follows the course rule (Lecture 4, "Futures Data: Cleaning"):
ES/SC -> SP, EN -> ND, ER -> RL from 2008-12; BC, BG, ZH, FF have no data.

Open `ORIE5260_Futures_Analysis.ipynb` and run all cells. Requires pandas, numpy,
matplotlib, openpyxl, IPython.

## Stage 2: course baseline and static benchmark

Definitions follow Lecture 6 ("Baseline Strategy: Description", "Risk Model: Code snippet"):

| Step | Rule |
|---|---|
| Signal | sign of the sum of the last 12 monthly returns (t-11..t) |
| Risk model | expanding-window std of monthly returns, annualised; needs >= 36 months |
| Position | signal x 40% / vol |
| Portfolio | 1 / number of positions; trade only with >= 10 positions |
| Static benchmark | long the same vol-adjusted assets, equally weighted |
| P&L | weights decided at month-end t earn the returns of t+1 |

Replication (1976-02 to 2014-12, `cta/output/course_comparison.csv`):

| | Avg | Std | SR | Corr(static) | Lecture 6 |
|---|---|---|---|---|---|
| TSMOM 12m | 11.0% | 11.1% | 0.99 | 0.08 | 11.1% / 11.1% / 1.00 / 0.08 |
| Static | 6.9% | 14.5% | 0.48 | 1 | 6.9% / 14.5% / 0.47 |
| TSMOM 1m / 3m / 6m | 8.0 / 9.5 / 8.9% | 11.3 / 11.8 / 11.4% | 0.71 / 0.81 / 0.79 | -0.11 / -0.09 / -0.01 | SR 0.72 / 0.81 / 0.79 |

Run (from `cta/`, Python with pandas, numpy, matplotlib, pytest):

    python run_backtest.py
    python -m pytest -q

## Risk model upgrade: monthly to daily data

The upgrade ladder now includes `VT11_monthly_ewma` between `VT11_historical`
and `VT11_ewma`. It consumes only the monthly return panel: zero-mean EWMA
second moments for volatility and correlation, covariance = D C D, annualised
with 12 months/year. Missing returns are masked rather than treated as observed
zeros; pairwise correlations require effective joint history and are projected PSD.

For a fixed-decay frequency comparison, its volatility half-life is 60/21 = 2.857
months and its correlation half-life is 120/21 = 5.714 months, matching the existing
daily model at 21 trading days/month. The minimum EWMA history is 6 months
(126/21). These new monthly parameters were fixed by conversion, not selected
by maximising performance. Both versions retain the same 50%-of-historical-vol
floor, 36-month eligibility, 12-month direction signal, monthly rebalancing and
11% portfolio volatility target. The daily step also retains its existing 3-day
overlapping-return correlation estimator; this is a comparison of the two risk
model implementations, not a claim that frequency is their only difference.
Monthly EWMA weights are exported to `cta/output/weights_monthly_ewma.csv`.

Forecasts are made at month-end t from daily data through t and scored on month t+1.
Parameters are chosen on months up to 1999 (QLIKE against the squared monthly return);
2000-2014 is held out. Metrics: bias statistic std(r / sigma) (ideal 1), its yearly mean
absolute deviation from 1 (MRAD), QLIKE.

- Step 1a, volatility (`evaluate_risk_model.py`): EWMA of squared daily returns, one
  half-life, no window. Chosen: 60 trading days (20-60 are statistically indistinguishable).
  Overlapping k-day returns were tested for the daily-to-monthly aggregation and dropped.
- Step 1b, covariance (`evaluate_covariance.py`): Sigma = D C D, D the step-1a vols, C an
  EWMA correlation of overlapping 3-day returns (absorbs asynchronous closes), half-life
  120 trading days. Scored on the TSMOM book, the static book, per-class books and random
  long/short books against the course expanding covariance and a zero-correlation model.

## Midterm strategy: CTA_v2 (`cta/output/enhancement_comparison.csv`)

Trend following on 58 futures, rebalanced monthly; every input point-in-time.
1. Direction: sign of the summed last 12 monthly returns (course signal).
2. Trend-continuity filter: continuity = sign(12m return) x (share of up days - share
   of down days) over the last 252 trading days (= -Information Discreteness, Da, Gurun
   & Warachka 2014). Positions ranked below the month's median are halved.
3. Risk model: daily EWMA vol (half-life 60d, floored at 50% of the instrument's
   historical to-date vol) and EWMA correlation of 3-day returns (half-life 120d).
4. Sizing: sign / vol (equal standalone risk), soft caps on standalone risk shares
   (10% per instrument, 40% per asset class: share = cap x tanh(lambda x raw / cap)).
5. Vol target: 11% ex-ante portfolio volatility every month (course L6 level).

Same instruments and months for every version. The original strategy parameters
were selected on data to 1999; the added monthly EWMA parameters are fixed calendar
conversions of the daily settings. The 2000-2014 evaluation is not used to tune them.

| Version | SR | SR to 1999 / from 2000 | SR from 2000, net 5bp | Max DD | Skew |
|---|---|---|---|---|---|
| TSMOM_12 (course baseline) | 0.99 | 1.13 / 0.84 | 0.80 | -26% | -0.07 |
| VT11_historical (course L6, SR 1.08) | 1.07 | 1.17 / 0.96 | 0.90 | -19% | -0.16 |
| VT11_monthly_ewma (+ monthly EWMA risk model) | 1.30 | 1.39 / 1.15 | 1.07 | -13% | -0.06 |
| VT11_ewma (+ daily EWMA risk model) | 1.28 | 1.33 / 1.18 | 1.09 | -15% | -0.07 |
| CTA_v1 (+ soft risk caps) | 1.29 | 1.36 / 1.16 | 1.06 | -14% | -0.13 |
| **CTA_v2** (+ trend-continuity filter) | **1.34** | **1.44 / 1.17** | **1.08** | **-12%** | **-0.06** |

The risk-model ladder's out-of-sample SR is 0.96 (historical) -> 1.15 (monthly
EWMA) -> 1.18 (daily EWMA). Most of the gain is already achieved by monthly
exponential weighting. Daily data adds a small out-of-sample SR improvement and
brings realised out-of-sample volatility closer to the 11% target (monthly 9.79%,
daily 11.02%). This is not a uniform improvement: full-sample SR falls from 1.30
to 1.28, maximum drawdown worsens from -12.86% to -14.58%, and annual turnover
rises from 13.47 to 15.60. Full-sample arithmetic annualised returns rise from
12.47% to 14.76%, alongside realised volatility rising from 9.60% to 11.56%.
The continuity filter improves CTA_v1 in both periods with slightly lower turnover,
but CTA_v2's test SR remains slightly below the uncapped daily EWMA version.

Signal research (`research_signals.py`, `evaluate_signal_strength.py`): continuous t-stat
signals, multi-horizon signs (1/3/6/12m, raw and risk-adjusted) and the hybrid_v2 p_trend
filter (windows x1-x4, thresholds 0.3-0.5) all either lost to sign(12m) before 1999 or
improved before 1999 and lost after 2000; fast horizons add turnover and their edge
fades after 2000. Trend continuity was the only strength measure with a clear training
relation to the next-month TSMOM payoff (top vs bottom quintile t = 2.6).

Crisis alpha and SG Trend, 2000-2014 (`output/crisis_alpha_2000_2014.csv`,
`crisis_alpha_smile.png`): CTA_v2 SR 1.17 vs SG Trend 0.53 (SG is net of fees, ours is
before costs), correlation with SG Trend 0.63, market beta ~0, +1.9% per month in the
worst 10% of equity months, positive but insignificant Mkt^2 coefficient (SG Trend 1.43).
Vol targeting trades some convexity for a steadier risk level.

**Open issue (post-midterm):** CTA_v2 has less crisis convexity than SG Trend (Mkt^2
coefficient 0.38, not significant, vs 1.43; worst-equity-month return +1.9% vs +3.6%). The
goal is a strategy that is more trend-like in crises, not only a higher Sharpe. Candidate
levers: let the vol target react less (or not at all) when trend signals are strong in a
sell-off, a crisis-aware cap, and signals that move faster in large drawdowns. Not changed
for the midterm.

Universe over time: the book starts in 1976-02 with 11 instruments (first month with at
least 10 instruments that have 12 months of signal and 36 months of vol history) and grows
as markets list: about 19 in 1980, 32 in 1990, 48 in 2000, 58 from 2010.

## No look-ahead

At each month-end t the engine hands the strategy `returns.iloc[:i+1]`, a copy that ends
at t, and multiplies the resulting weights only with month t+1 returns. Tests scramble
every return after t and require the weights up to t to be bit-for-bit unchanged.
