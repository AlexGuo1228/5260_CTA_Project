"""Accounting, timing and execution-policy checks using hand-computable cases."""
import numpy as np
import pandas as pd
import pytest

from cta_backtest.engine import run_backtest
from cta_backtest.execution import ExecutionConfig, execute, performance, replay_execution
from cta_backtest.riskmodel import RiskModel


def panel(values):
    return pd.DataFrame(values, index=pd.date_range("2000-01-31", periods=len(values), freq="ME"),
                        columns=["A", "B"])


def test_zero_cost_replay_matches_original_engine():
    class Fixed:
        def compute_weights(self, history):
            return pd.Series([0.8, -0.3], index=history.columns)
    m = panel([[0, 0], [.1, -.1], [-.05, .02], [.03, .02]])
    old = run_backtest(m, Fixed())
    new = replay_execution(m, old.weights, ExecutionConfig(trade_bps=0, roll_roundtrip_bps=0))
    np.testing.assert_allclose(new.ledger.net_return, old.returns, atol=1e-15)
    np.testing.assert_allclose(new.ledger.turnover, old.turnover, atol=1e-15)
    np.testing.assert_allclose(new.positions, old.weights)


def test_cost_units_roll_roundtrip_and_net_nav_drift():
    m = panel([[0, 0], [.1, 0], [0, 0]])
    target = panel([[1, 0], [1, 0]])
    res = replay_execution(m, target, ExecutionConfig(trade_bps=10, roll_roundtrip_bps=6))
    first = res.ledger.iloc[0]
    assert first.trade_cost == pytest.approx(.001)
    assert first.roll_cost == pytest.approx(.0002)
    assert first.net_return == pytest.approx(.0988)
    assert res.ledger.iloc[1].turnover == pytest.approx(abs(1 - 1.1 / 1.0988))
    np.testing.assert_allclose(res.ledger.net_return,
                               res.ledger.gross_return - res.ledger.trade_cost - res.ledger.roll_cost)


@pytest.mark.parametrize("policy", ["buffer", "partial"])
def test_exits_entries_and_reversals_are_immediate(policy):
    cfg = ExecutionConfig(policy=policy)
    np.testing.assert_allclose(execute(np.array([0., -1., 2.]), np.array([1., 1., 0.]), cfg),
                               [0., -1., 2.])


def test_reduced_trading_changes_actual_next_month_pnl():
    m = panel([[0, 0], [0, 0], [.1, 0]])
    targets = panel([[1, 0], [2, 0]])
    res = replay_execution(m, targets, ExecutionConfig(policy="partial", fraction=.5,
                                                        trade_bps=0, roll_roundtrip_bps=0))
    assert res.positions.iloc[1, 0] == pytest.approx(1.5)
    assert res.ledger.iloc[1].net_return == pytest.approx(.15)
    assert res.ledger.iloc[1].turnover == pytest.approx(.5)


def test_buffer_keeps_small_deviation_and_future_shock_cannot_change_trade():
    m = panel([[0, 0], [0, 0], [.1, 0]])
    targets = panel([[1, 0], [1.05, 0]])
    cfg = ExecutionConfig(policy="buffer", trade_bps=0, roll_roundtrip_bps=0)
    a = replay_execution(m, targets, cfg)
    m.iloc[-1, 0] = .5
    b = replay_execution(m, targets, cfg)
    pd.testing.assert_frame_equal(a.positions, b.positions)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    assert a.positions.iloc[1, 0] == 1
    assert b.ledger.iloc[-1].net_return == pytest.approx(.5)


def test_risk_guard_forces_target_after_large_risk_reduction():
    m = panel([[0, 0], [0, 0], [0, 0]])
    targets = panel([[1, 0], [.1, 0]])
    risk = RiskModel("test", list(m.columns), m.index.to_period("M"),
                     np.ones((3, 2)) * .2, np.tile(np.eye(2) * .04, (3, 1, 1)))
    res = replay_execution(m, targets, ExecutionConfig(policy="partial", trade_bps=0,
                                                        roll_roundtrip_bps=0), risk)
    assert res.ledger.iloc[1].forced_rebalance
    assert res.positions.iloc[1, 0] == pytest.approx(.1)


def test_first_month_loss_is_in_drawdown_and_missing_is_disclosed():
    m = panel([[0, 0], [-.1, np.nan]])
    res = replay_execution(m, panel([[1, 1]]), ExecutionConfig(trade_bps=0, roll_roundtrip_bps=0))
    assert performance(res.ledger)["max_drawdown"] == pytest.approx(-.1)
    assert res.ledger.iloc[0].missing_held == 1


@pytest.mark.parametrize("kwargs", [{"trade_bps": -1}, {"fraction": 0},
                                    {"policy": "other"}, {"buffer": np.nan}])
def test_invalid_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        ExecutionConfig(**kwargs)


def test_gap_in_target_dates_fails():
    m = panel([[0, 0]] * 5)
    with pytest.raises(ValueError, match="consecutive"):
        replay_execution(m, m.iloc[[0, 2]], ExecutionConfig())
