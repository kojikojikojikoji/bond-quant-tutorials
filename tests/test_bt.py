"""bondlab.bt の回帰テスト（S9）。"""
import numpy as np
import pandas as pd

from bondlab import bt


def test_backtest_lag_prevents_lookahead():
    idx = pd.date_range("2026-01-01", periods=40, freq="B")
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.standard_normal((40, 5)), index=idx)
    ret = pd.DataFrame(rng.standard_normal((40, 5)) * 1e-3, index=idx)
    res = bt.backtest(sig, ret, cost_bps=0.0, lag=1)
    # lag=1 は先頭のポジションが0
    assert (res["positions"].iloc[0] == 0).all()


def test_backtest_matches_manual():
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    rng = np.random.default_rng(1)
    sig = pd.DataFrame(rng.standard_normal((30, 3)), index=idx)
    ret = pd.DataFrame(rng.standard_normal((30, 3)) * 1e-3, index=idx)
    res = bt.backtest(sig, ret, cost_bps=0.5, lag=1)
    pos = sig.shift(1).fillna(0.0)
    gross = (pos * ret).sum(axis=1)
    turn = pos.diff().abs().sum(axis=1).fillna(pos.abs().sum(axis=1))
    manual = gross - turn * (0.5 * 1e-4)
    assert np.allclose(manual.values, res["pnl"].values, atol=1e-12)


def test_lookahead_cheat_overstates_sharpe():
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    ret = pd.DataFrame(np.random.default_rng(2).standard_normal((60, 4)) * 1e-3, index=idx)
    cheat = np.sign(ret)  # 当期リターンを覗き見た反則シグナル
    s0 = bt.performance(bt.backtest(cheat, ret, lag=0)["pnl"])["sharpe"]
    s1 = bt.performance(bt.backtest(cheat, ret, lag=1)["pnl"])["sharpe"]
    assert s0 > s1 + 5.0


def test_performance_keys_and_cost():
    pnl = pd.Series(np.random.default_rng(3).standard_normal(252) * 1e-3)
    perf = bt.performance(pnl)
    assert set(perf) == {"ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate", "n"}
    assert perf["max_drawdown"] <= 0


def test_conversion_factor():
    assert abs(bt.conversion_factor(0.06, 10) - 1.0) < 1e-9
    assert bt.conversion_factor(0.02, 10) < 1.0
    assert bt.conversion_factor(0.10, 10) > 1.0
