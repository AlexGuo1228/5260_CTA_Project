# ORIE 5260 Futures Analysis

## Project layout

```text
ORIE 5260/
  ORIE5260_Futures_Analysis.ipynb
  data/
    FuturesUnderlyingData/       # Original daily instrument CSV files
    AssetMapCsv.csv              # Original asset mapping
    MonthlyReturns.csv          # Original reference monthly returns
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

Python dependencies: pandas, numpy, matplotlib, openpyxl, and IPython.
The optional execution helper also requires nbformat, nbclient, and ipykernel.
Local dependency and runtime directories remain at the project root.

`build_analysis_notebook.py` rebuilds the notebook source and resets saved outputs;
use the existing notebook for normal analysis.
