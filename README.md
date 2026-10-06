# ORIE 5260 Futures Analysis

## Project layout

```text
ORIE 5260/
  ORIE5260_Futures_Analysis.ipynb
  MonthlyReturns.csv            # Primary reconstructed returns (58 instruments)
  data/
    FuturesUnderlyingData/       # Original daily instrument CSV files
    AssetMapCsv.csv              # Original asset mapping
    MonthlyReturns.csv          # Original reference monthly returns
    MonthlyReturns_cleaned.csv  # Identical copy of the primary reconstructed CSV
    Lecture3_livedata.xlsx       # Supplementary monthly factors
  output/
    cleaned/                    # Standardized data and instrument rules
    quality/                    # Data-quality checks and input inventory
    returns/                    # Reconstructed returns and reconciliation
    statistics/                 # Summary statistics and autocorrelation
    figures/                    # PNG and SVG charts
    report.md
    manifest.csv
  execute_analysis_notebook.py
  build_analysis_notebook.py
```

Open `ORIE5260_Futures_Analysis.ipynb` and run all cells in order.
The notebook finds the project root from the current directory or its ancestors,
reads inputs from `data/`, and writes generated results to `output/`.
The source data remain unchanged. Repeated runs update the same output files.

The primary deliverable is the root `MonthlyReturns.csv`, with a byte-identical
copy at `data/MonthlyReturns_cleaned.csv`. It preserves the original reference's
58 columns, column order, and date labels. The original `data/MonthlyReturns.csv`
remains unchanged. Missing reference values are reconstructed where raw prices
permit; unavailable history is not imputed.

The 57-instrument dataset excludes YM while retaining ZD to avoid duplicate
Dow Jones exposure in the supplementary portfolio analysis. This is an optional
modeling choice, not a requirement of the assignment screenshot. Notebook
Section 5 documents the complete 62-to-58 and optional 58-to-57 selection rules.

Python dependencies: pandas, numpy, matplotlib, openpyxl, and IPython.
The optional execution helper also requires nbformat, nbclient, and ipykernel.
Local dependency and runtime directories remain at the project root.

`build_analysis_notebook.py` rebuilds the notebook source and resets saved outputs;
use the existing notebook for normal analysis.
