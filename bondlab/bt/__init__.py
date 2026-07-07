"""戦略バックテスト基盤。

シグナルとリターンから日次のポジション・損益を計算する軽量バックテスター。
ルックアヘッド（先読み）を防ぐため、シグナルは1期ずらして翌営業日に約定する
のが既定。取引コストは回転率に比例させる。S9 で理論を扱う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(signals, returns, cost_bps: float = 0.0, lag: int = 1):
    """シグナルとリターンから日次損益を計算する。

    Parameters
    ----------
    signals : pandas.DataFrame or Series
        各資産のターゲットポジション（例 -1〜+1）。行=日付、列=資産。
    returns : 同形状
        各資産の当日リターン（価格変化率、または DV01 調整済み損益）。
    cost_bps : float
        片道の取引コスト（bp）。回転（ポジション変化の絶対値）に掛ける。
    lag : int
        シグナルを何営業日ずらして約定するか。既定 1（翌日約定＝先読み防止）。

    Returns
    -------
    dict(pnl, positions, gross_pnl, cost, turnover)
    """
    sig = _as_frame(signals)
    ret = _as_frame(returns)
    ret = ret.reindex_like(sig)
    # lag 日ずらして約定。最初の lag 日はポジション0。
    positions = sig.shift(lag).fillna(0.0)
    gross = (positions * ret).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    cost = turnover * (cost_bps * 1e-4)
    pnl = gross - cost
    return dict(pnl=pnl, positions=positions, gross_pnl=gross, cost=cost, turnover=turnover)


def performance(pnl, periods_per_year: int = 252) -> dict:
    """損益系列のパフォーマンス指標。"""
    pnl = pd.Series(pnl).dropna()
    mean = pnl.mean()
    std = pnl.std(ddof=1)
    sharpe = np.sqrt(periods_per_year) * mean / std if std > 0 else np.nan
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    return dict(
        ann_return=float(mean * periods_per_year),
        ann_vol=float(std * np.sqrt(periods_per_year)),
        sharpe=float(sharpe),
        max_drawdown=float(drawdown.min()),
        hit_rate=float((pnl > 0).mean()),
        n=int(pnl.size),
    )


def _as_frame(x):
    if isinstance(x, pd.Series):
        return x.to_frame()
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(np.atleast_2d(np.asarray(x, dtype=float)))


# ---- 国債先物ヘルパ（S9-4） -------------------------------------------------

def conversion_factor(coupon: float, years_to_maturity: float, notional_coupon: float = 0.06,
                      freq: int = 2) -> float:
    """国債先物のコンバージョンファクター（簡易版）。

    受渡適格銘柄を、想定クーポン（日本は 6%）を利回りとして額面1で割り引いた
    価格を返す（額面1基準なのでクーポン=想定クーポンなら CF=1）。厳密な取引所式は
    端数期間の丸めを含むが、ここでは半年グリッドに丸めた素朴版を使う（notebook で
    端数期間を含む厳密版に拡張する）。
    """
    n = int(round(years_to_maturity * freq))
    c = coupon / freq
    y = notional_coupon / freq
    # クーポン債の想定利回りでの価格 / 100
    disc = (1 + y) ** (-np.arange(1, n + 1))
    price = c * disc.sum() + (1 + y) ** (-n)
    return float(price)
