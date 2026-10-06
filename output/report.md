# Futures Data Analysis Results

## Data and Validation
- Raw instruments: 62; reference reconstruction: 58; unique economic instruments: 57.
- Monthly calendar: 1969-01 to 2014-12.
- Quality events (including calendar-gap flags): 9; flagged daily returns: 582; flagged monthly returns: 105.
- Reconciliation status: {'match': 20987, 'both_missing': 10997, 'reference_missing': 32}. Numeric tolerance: 1e-06.
- All input SHA-256 checks passed; source files remain unchanged.

## Methodology
- Returns are adjacent month-end Close ratios minus one. Missing prices are not forward-filled; dates are aligned by calendar month.
- Starting in 2008-12, RL uses ER's own monthly returns. Price levels are not spliced directly.
- The primary exports MonthlyReturns.csv and data/MonthlyReturns_cleaned.csv retain all 58 reference columns, including both ZD and YM. The two files are byte-identical.
- The optional 57-column analysis excludes YM and retains ZD for Dow Jones. This is an additional modeling choice, not a requirement of the screenshot. SP and ND represent their respective ordinary/mini candidate groups.
- The primary exports preserve the reference date labels and column order, retaining 32 reconstructable reference-missing values without statistical imputation. The original data/MonthlyReturns.csv remains unchanged.
- Asset classes equally weight instruments with observed returns each month and rebalance monthly. Actual weights are saved in returns/asset_class_weights.csv.
- Quoted currencies are not converted to USD. Continuous-contract adjustment and roll methods require clarification from the course data provider.
- Full-history samples differ across instruments; the common observed sample contains 75 months.
- Cumulative returns stop at the first internal missing return; unknown returns are not set to zero.
- Autocorrelation uses monthly-return lags 1-24. Reference bands are not formal significance conclusions.

## Items Requiring Review
- AssetMap entries BC, BG, FF, and ZH have no corresponding raw files.
- Raw OHLC anomalies and large returns are retained; see quality/.
- reference_missing indicates a missing reference value that can be reconstructed from raw prices; rebuilt_missing indicates the reverse. See the reconciliation differences.
- Statistics are preliminary pending manual outlier review. Genuine extreme market moves are not removed automatically.

## Output Guide
- quality/: input inventory, coverage checks, and anomaly records.
- cleaned/: standardized daily observations, asset mapping, selection rules, and supplementary factor data.
- returns/: month-end prices, 62/58/57-instrument returns, class returns, weights, cumulative returns, and full reconciliation.
- statistics/: instrument and class summaries, common-sample statistics, and ACF results.
- figures/: all PNG and SVG charts.
