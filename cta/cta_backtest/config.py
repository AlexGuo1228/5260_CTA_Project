"""Strategy and backtest configuration.

All tunable parameters live here; a run is fully described by one
`BacktestConfig` (it is also written next to the results).

Defaults reproduce the course baseline (Lecture 6, "Baseline Strategy:
Description" and "Risk Model: Code snippet").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BacktestConfig:
    # ---- Data -----------------------------------------------------------
    # Single source of truth: the cleaned 58-column panel produced by the
    # data notebook at the project root, and the course asset map.
    returns_path: str = "../MonthlyReturns.csv"
    asset_map_path: str = "../data/AssetMapCsv.csv"

    # Instruments removed from the universe. Empty = the course's 58-column
    # set (its own de-duplication: ES/SC -> SP, EN -> ND, ER -> RL).
    drop_instruments: List[str] = field(default_factory=list)

    # Asset-class corrections on top of the asset map. EC (Eurodollar) is a
    # 3-month USD interest-rate future; the map labels it FX while labelling
    # the equivalent UK contract (SS) FI. Affects class reporting only.
    asset_class_overrides: Dict[str, str] = field(default_factory=lambda: {"EC": "FI"})

    # ---- Sample ---------------------------------------------------------
    # First / last month whose returns are evaluated ("YYYY-MM"); None = all.
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # ---- Signal ---------------------------------------------------------
    # sign of the past return over months t-L+1 .. t (all L months observed).
    momentum_lookback: int = 12
    # "sum" (arithmetic sum of monthly returns) reproduces the course numbers;
    # "compound" uses prod(1 + r) - 1.
    momentum_aggregation: str = "sum"

    # CTA_v2 strength filter: trend continuity = -Information Discreteness over the
    # last `continuity_window_days` trading days; positions ranked below the
    # month's cross-sectional median are scaled by `continuity_low_weight`.
    continuity_window_days: int = 252
    continuity_low_weight: float = 0.5

    # ---- Risk model -----------------------------------------------------
    # Course baseline: expanding-window realised volatility of monthly
    # returns (annualised), available once an instrument has at least
    # `vol_min_months` returns.
    vol_min_months: int = 36

    # Daily EWMA volatility (risk-model upgrade, step 1a).
    daily_dir: str = "../data/FuturesUnderlyingData"
    # Panel column -> (source file, first month it is used). Mirrors the
    # monthly panel: RL uses ER returns from 2008-12 (Lecture 4 merge rule).
    daily_splices: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {"RL": ("ER", "2008-12")})
    # A daily return spanning more than this many calendar days is dropped
    # (data hole, not a holiday). Affects 8 SS returns.
    daily_max_gap_days: int = 7
    # Half-life in trading days of the EWMA on squared daily returns.
    vol_halflife_days: float = 60.0
    # Floor on the EWMA vol: at least this fraction of the instrument's own
    # historical to-date vol. Stops a vol collapse (e.g. Eurodollar at zero
    # rates, EWMA vol 0.07%) from turning into a 14x notional position.
    vol_floor_vs_history: float = 0.5
    # Minimum daily returns before an EWMA estimate is produced.
    vol_ewma_min_days: int = 126
    # Half-life candidates and the last month of the selection (training) sample.
    vol_halflife_grid: List[float] = field(default_factory=lambda: [10.0, 20.0, 40.0, 60.0, 90.0])
    risk_train_end: str = "1999-12"

    # Daily EWMA correlation (risk-model upgrade, step 1b). Covariance =
    # D C D with D the EWMA vols above and C an EWMA correlation matrix
    # estimated on overlapping `corr_overlap_days`-day returns (k > 1 absorbs
    # asynchronous closes across time zones).
    corr_halflife_days: float = 120.0
    corr_overlap_days: int = 3
    # A pair's correlation is used once its joint EWMA weight corresponds to
    # at least this many joint observations; otherwise it is set to 0.
    corr_min_days: int = 126
    corr_halflife_grid: List[float] = field(default_factory=lambda: [60.0, 120.0, 250.0, 500.0])
    corr_overlap_grid: List[int] = field(default_factory=lambda: [1, 3])
    # Random test portfolios per month for the covariance evaluation.
    n_random_portfolios: int = 50
    random_seed: int = 0

    # ---- Position sizing ------------------------------------------------
    # Each position is scaled to this annualised volatility: signal * target / vol.
    position_vol_target: float = 0.40
    # Portfolio = average of the positions (1 / number of positions).
    # The portfolio trades only when at least this many positions exist.
    min_assets: int = 10

    # ---- Vol targeting and risk caps (step 2) ---------------------------
    # Ex-ante annualised portfolio volatility every month (scales up and down).
    # 11% = course L6 target and the realised vol of the course baseline.
    portfolio_vol_target: float = 0.11
    # Soft caps on shares of standalone risk: share = cap * tanh(lam * raw / cap),
    # lam solved so shares sum to 1. Near-linear for small shares; each step
    # closer to the cap needs a larger raw signal; equal shares are unchanged.
    class_risk_cap: float = 0.40
    instrument_risk_cap: float = 0.10

    # ---- Costs ----------------------------------------------------------
    # One-way transaction cost in basis points of traded notional.
    cost_bps: float = 0.0

    name: str = "cta"

    def to_dict(self) -> dict:
        return asdict(self)
