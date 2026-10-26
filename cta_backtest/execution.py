"""Monthly execution experiments on frozen, point-in-time strategy targets.

Keep the original engine unchanged for midterm replication. This layer evolves
actual positions and accounts for costs in the NAV denominator used for drift.
Costs are booked at the end of the funded month, matching the original engine's
timing convention. This is a monthly approximation, not a fill-level simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .riskmodel import RiskModel


@dataclass(frozen=True)
class ExecutionConfig:
    # One-way bps per unit of traded notional, including both buys and sells.
    trade_bps: float = 2.0
    # Combined close-and-open cost per unit rolled, NOT a one-way quote.
    roll_roundtrip_bps: float = 2.0
    # Uniform monthly accrual is a scenario assumption, not a contract calendar.
    rolls_per_year: float = 4.0
    policy: str = "full"
    # Relative no-trade band around each target; no absolute weight floor.
    buffer: float = 0.10
    fraction: float = 0.50
    # Reduced-turnover policies fall back to the target if stale risk is excessive.
    max_risk_ratio: float = 1.25

    def __post_init__(self):
        values = (self.trade_bps, self.roll_roundtrip_bps, self.rolls_per_year,
                  self.buffer, self.fraction, self.max_risk_ratio)
        if not all(np.isfinite(values)):
            raise ValueError("Execution parameters must be finite")
        if min(self.trade_bps, self.roll_roundtrip_bps, self.rolls_per_year, self.buffer) < 0:
            raise ValueError("Costs, roll frequency and buffer must be nonnegative")
        if self.policy not in {"full", "buffer", "partial"}:
            raise ValueError("Unknown execution policy")
        if not 0 < self.fraction <= 1 or self.max_risk_ratio < 1:
            raise ValueError("Invalid fraction or risk ratio")


@dataclass
class ExecutionResult:
    # Positions and trades use signal dates; ledger rows use earned-return dates.
    positions: pd.DataFrame
    trades: pd.DataFrame
    ledger: pd.DataFrame


def execute(target: np.ndarray, current: np.ndarray, cfg: ExecutionConfig) -> np.ndarray:
    """Apply execution rules without looking at the next month's return.

Initial entries, complete exits and sign reversals execute immediately. This
prevents a turnover rule from retaining an obsolete direction or ineligible name.
"""
    delta = target - current
    if cfg.policy == "full":
        return target.copy()
    if cfg.policy == "buffer":
        out = np.where(np.abs(delta) <= cfg.buffer * np.abs(target), current, target)
    else:
        out = current + cfg.fraction * delta
    mandatory = (target == 0) | (current == 0) | (target * current < 0)
    return np.where(mandatory, target, out)


def replay_execution(monthly: pd.DataFrame, targets: pd.DataFrame,
                     cfg: ExecutionConfig, risk: RiskModel | None = None) -> ExecutionResult:
    """Replay precomputed causal targets with a one-row execution lag.

The caller must supply targets generated with information available on their
signal dates. Missing held returns retain the legacy zero-return convention and
are explicitly counted. No terminal liquidation is assumed.
"""
    if monthly.index.has_duplicates or not monthly.index.is_monotonic_increasing:
        raise ValueError("Monthly dates must be unique and increasing")
    if not targets.columns.equals(monthly.columns) or not np.isfinite(targets.values).all():
        raise ValueError("Targets must be finite and use the monthly column order")
    if len(targets) == 0 or targets.index.has_duplicates or not targets.index.is_monotonic_increasing:
        raise ValueError("Target dates must be nonempty, unique and increasing")
    locs = monthly.index.get_indexer(targets.index)
    if np.any(locs < 0) or np.any(locs >= len(monthly) - 1) or np.any(np.diff(locs) != 1):
        raise ValueError("Targets must cover consecutive signal dates with next-month returns")
    if risk is not None and list(monthly.columns) != risk.ids:
        raise ValueError("Risk model column order differs")
    current = np.zeros(len(monthly.columns))
    positions, trades, ledger = [], [], []
    for t, loc in zip(targets.index, locs):
        target = targets.loc[t].to_numpy(dtype=float)
        held = execute(target, current, cfg)
        forced = False
        forecast = np.nan
        target_forecast = np.nan
        if risk is not None:
            _, cov = risk.at(t.to_period("M"))
            # The existing risk model marks unavailable instruments with NaN.
            # Its eligible submatrix is finite; flat instruments add no risk.
            active = (held != 0) | (target != 0)
            sub = cov[np.ix_(active, active)]
            if not np.isfinite(sub).all():
                raise ValueError(f"Missing covariance for held assets at {t}")
            h, g = held[active], target[active]
            target_forecast = float(np.sqrt(max(g @ sub @ g, 0)))
            forecast = float(np.sqrt(max(h @ sub @ h, 0)))
            if cfg.policy != "full" and forecast > cfg.max_risk_ratio * target_forecast + 1e-12:
                held = target.copy()
                forecast = target_forecast
                forced = True
        delta = held - current
        turnover = float(np.abs(delta).sum())
        gross_exposure = float(np.abs(held).sum())
        trade_cost = turnover * cfg.trade_bps / 1e4
        roll_cost = gross_exposure * cfg.rolls_per_year / 12 * cfg.roll_roundtrip_bps / 1e4
        next_returns = monthly.iloc[loc + 1]
        r = next_returns.fillna(0).to_numpy()
        gross = float(held @ r)
        net = gross - trade_cost - roll_cost
        if 1 + net <= 0:
            raise ValueError("Portfolio NAV is nonpositive")
        positions.append(held.copy())
        trades.append(delta.copy())
        ledger.append(dict(date=monthly.index[loc + 1], gross_return=gross, net_return=net,
                           trade_cost=trade_cost, roll_cost=roll_cost, turnover=turnover,
                           gross_exposure=gross_exposure, forecast_vol=forecast,
                           target_forecast_vol=target_forecast, forced_rebalance=forced,
                           target_distance=float(np.abs(held - target).sum()),
                           missing_held=int(((held != 0) & next_returns.isna().values).sum())))
        # Costs reduce NAV, so subsequent positions drift relative to net NAV.
        current = held * (1 + r) / (1 + net)
    return ExecutionResult(pd.DataFrame(positions, index=targets.index, columns=targets.columns),
                           pd.DataFrame(trades, index=targets.index, columns=targets.columns),
                           pd.DataFrame(ledger).set_index("date"))


def performance(ledger: pd.DataFrame) -> dict:
    """Annualized monthly statistics, including initial deployment turnover.

Drawdown includes initial NAV=1, so a loss in the first month is not hidden.
Gross return uses the executed positions of this run, not another policy's book.
"""
    r = ledger.net_return
    nav = (1 + r).cumprod()
    peak = nav.cummax().clip(lower=1)
    vol = r.std(ddof=1) * np.sqrt(12)
    return dict(months=len(r), ann_return=r.mean() * 12,
                cagr=nav.iloc[-1] ** (12 / len(r)) - 1,
                ann_vol=vol, sharpe=r.mean() * 12 / vol if vol > 0 else np.nan,
                max_drawdown=(nav / peak - 1).min(),
                gross_ann_return=ledger.gross_return.mean() * 12,
                ann_trade_cost=ledger.trade_cost.mean() * 12,
                ann_roll_cost=ledger.roll_cost.mean() * 12,
                ann_turnover=ledger.turnover.mean() * 12,
                avg_gross=ledger.gross_exposure.mean(),
                avg_forecast_vol=ledger.forecast_vol.mean(),
                avg_target_distance=ledger.target_distance.mean(),
                forced_rebalances=int(ledger.forced_rebalance.sum()),
                missing_held=int(ledger.missing_held.sum()))
