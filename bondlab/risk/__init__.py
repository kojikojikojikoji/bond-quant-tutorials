"""VaR / Expected Shortfall とバックテスト検定。

損益系列（正が利益）に対する Value at Risk と Expected Shortfall を、
ヒストリカル・パラメトリック・モンテカルロの3方式で提供する（S3-5）。
VaR は「損失側の分位点」を正の数で返す慣行に合わせる。
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def historical_var(pnl, alpha: float = 0.99) -> float:
    """ヒストリカル VaR。損失分布の (1-alpha) 分位点を正の損失で返す。"""
    pnl = np.asarray(pnl, dtype=float)
    return -np.quantile(pnl, 1.0 - alpha)


def historical_es(pnl, alpha: float = 0.99) -> float:
    """ヒストリカル Expected Shortfall（最悪 (1-alpha) 質量の平均損失）。

    VaR 超過の閾値判定ではなく、最悪側から $k=\\lceil (1-\\alpha)n \\rceil$ 個の
    観測を平均する。閾値方式は離散分布や境界の同値点（atom）でVaRが利益側に
    来ると全標本を拾って壊れるため、順序統計量ベースにする。
    """
    pnl = np.asarray(pnl, dtype=float)
    n = pnl.size
    if n == 0:
        return float("nan")
    # 1e-9 は (1-alpha)*n が整数のとき浮動小数の丸め上げで k が1つ増えるのを防ぐ。
    k = max(int(np.ceil((1.0 - alpha) * n - 1e-9)), 1)
    worst = np.sort(pnl)[:k]  # 最も小さい（最悪の）k 個
    return -worst.mean()


def parametric_var(pnl, alpha: float = 0.99) -> float:
    """正規分布を仮定したパラメトリック VaR。"""
    pnl = np.asarray(pnl, dtype=float)
    mu, sigma = pnl.mean(), pnl.std(ddof=1)
    z = stats.norm.ppf(1.0 - alpha)
    return -(mu + sigma * z)


def parametric_es(pnl, alpha: float = 0.99) -> float:
    """正規分布を仮定した Expected Shortfall。"""
    pnl = np.asarray(pnl, dtype=float)
    mu, sigma = pnl.mean(), pnl.std(ddof=1)
    z = stats.norm.ppf(1.0 - alpha)
    return -(mu - sigma * stats.norm.pdf(z) / (1.0 - alpha))


def kupiec_pof(exceptions: int, n: int, alpha: float = 0.99):
    """Kupiec の POF（proportion of failures）検定。

    VaR 例外数が想定被覆率と整合するかを尤度比検定で評価する。

    Returns
    -------
    dict(lr, p_value, expected, observed)
        lr は検定統計量（自由度1のカイ二乗）、p_value はその上側確率。
    """
    p = 1.0 - alpha
    x = exceptions
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        lr = -2.0 * n * np.log(p)
    else:
        pi = x / n
        lr = -2.0 * (
            (n - x) * np.log(1.0 - p) + x * np.log(p)
            - (n - x) * np.log(1.0 - pi) - x * np.log(pi)
        )
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return dict(lr=float(lr), p_value=float(p_value), expected=p * n, observed=x)
