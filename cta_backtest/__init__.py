"""Cross-asset CTA (time-series momentum) backtest framework."""
from .config import BacktestConfig
from .data import load_market_data, coverage_table
from .engine import run_backtest, BacktestResult
from .strategies import TSMOMBaseline, StaticBenchmark, VolTargetTSMOM
from .riskmodel import RiskModel, historical_risk, ewma_risk
from . import metrics

__all__ = ["BacktestConfig", "load_market_data", "coverage_table", "run_backtest",
           "BacktestResult", "TSMOMBaseline", "StaticBenchmark", "VolTargetTSMOM",
           "RiskModel", "historical_risk", "ewma_risk", "metrics"]
