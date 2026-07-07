"""金利・割引の基本ユーティリティ。

複利規約（compounding convention）の相互変換と、割引係数とゼロレートの
往復変換を提供する。S1-1 で理論を扱い、以降のカーブ・モデル層が土台として
利用する。S0-1 では、bondlab の最初の実モジュールとして「自作関数には
説明とテストを添える」開発フローの実演に使う。

規約は文字列で表す:
    "annual"      年1回複利
    "semiannual"  年2回複利
    "quarterly"   年4回複利
    "monthly"     年12回複利
    "continuous"  連続複利

いずれの関数も numpy 配列を受け付ける（ベクトル化）。
"""
from __future__ import annotations

from typing import Union

import numpy as np

Number = Union[float, np.ndarray]

# 規約名 -> 年間の複利回数。連続複利は None で表す。
_FREQ = {
    "annual": 1,
    "semiannual": 2,
    "quarterly": 4,
    "monthly": 12,
    "continuous": None,
}


def _frequency(convention: str) -> int | None:
    if convention not in _FREQ:
        raise ValueError(
            f"未知の複利規約: {convention!r}。利用可能: {sorted(_FREQ)}"
        )
    return _FREQ[convention]


def discount_factor(rate: Number, t: Number, convention: str = "continuous") -> Number:
    """レートと年数 t から割引係数を求める。

    連続複利なら DF = exp(-r t)、m 回複利なら DF = (1 + r/m)^(-m t)。

    Parameters
    ----------
    rate : float or ndarray
        年率（小数。5% は 0.05）。
    t : float or ndarray
        年数。
    convention : str
        複利規約。

    Returns
    -------
    float or ndarray
        割引係数。
    """
    m = _frequency(convention)
    if m is None:
        return np.exp(-np.asarray(rate, dtype=float) * np.asarray(t, dtype=float))
    r = np.asarray(rate, dtype=float)
    return (1.0 + r / m) ** (-m * np.asarray(t, dtype=float))


def rate_from_discount(df: Number, t: Number, convention: str = "continuous") -> Number:
    """割引係数と年数 t からレートを逆算する。

    discount_factor の逆関数。t=0 では未定義のため呼び出し側で除外する。
    """
    df = np.asarray(df, dtype=float)
    t = np.asarray(t, dtype=float)
    m = _frequency(convention)
    if m is None:
        return -np.log(df) / t
    return m * (df ** (-1.0 / (m * t)) - 1.0)


def convert_rate(rate: Number, t: Number, src: str, dst: str) -> Number:
    """同一の割引係数を保ったままレートを規約変換する。

    src 規約のレートを一旦割引係数に落とし、dst 規約のレートへ戻す。
    割引係数を経由することで全規約を一様に扱える。
    """
    df = discount_factor(rate, t, src)
    return rate_from_discount(df, t, dst)


def to_continuous(rate: Number, convention: str) -> Number:
    """m 回複利レートを連続複利レートへ変換する（t 非依存）。

    r_c = m * ln(1 + r_m / m)。連続複利どうしは恒等。
    """
    m = _frequency(convention)
    if m is None:
        return np.asarray(rate, dtype=float)
    r = np.asarray(rate, dtype=float)
    return m * np.log1p(r / m)


def from_continuous(rate_c: Number, convention: str) -> Number:
    """連続複利レートを m 回複利レートへ変換する（t 非依存）。

    r_m = m * (exp(r_c / m) - 1)。
    """
    m = _frequency(convention)
    if m is None:
        return np.asarray(rate_c, dtype=float)
    rc = np.asarray(rate_c, dtype=float)
    return m * (np.exp(rc / m) - 1.0)
