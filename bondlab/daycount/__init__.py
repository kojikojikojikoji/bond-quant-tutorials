"""Day count convention（日数計算規約）のスクラッチ実装。

債券のクーポン・経過利子・割引年数はすべて「2日付間の年数」に依存する。
その数え方の取り決めが day count convention で、市場ごとに異なる。S1-2 で
理論を扱い、ここに再利用可能な実装を置く。QuantLib と突合して検証する。

対応規約:
    "ACT/360"    実日数 / 360     マネーマーケット
    "ACT/365F"   実日数 / 365     JGB 等（Fixed）
    "30/360"     30日/月・360日/年 米国社債の慣行
    "ACT/ACT"    実日数 / その年の実日数（ISDA）
"""
from __future__ import annotations

import datetime as _dt
from typing import Union

Date = Union[_dt.date, _dt.datetime]


def _d(x: Date) -> _dt.date:
    return x.date() if isinstance(x, _dt.datetime) else x


def year_fraction(start: Date, end: Date, convention: str = "ACT/365F") -> float:
    """start から end までの年数を規約に従って求める。"""
    s, e = _d(start), _d(end)
    if convention == "ACT/360":
        return (e - s).days / 360.0
    if convention == "ACT/365F":
        return (e - s).days / 365.0
    if convention == "30/360":
        return _thirty_360(s, e)
    if convention == "ACT/ACT":
        return _act_act_isda(s, e)
    raise ValueError(f"未知の day count 規約: {convention!r}")


def _thirty_360(s: _dt.date, e: _dt.date) -> float:
    """30/360 (US, Bond Basis)。月末調整つき。"""
    d1, d2 = s.day, e.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (
        360 * (e.year - s.year)
        + 30 * (e.month - s.month)
        + (d2 - d1)
    ) / 360.0


def _is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _act_act_isda(s: _dt.date, e: _dt.date) -> float:
    """ACT/ACT (ISDA)。年をまたぐ日数を、各年の実日数で按分する。"""
    if s == e:
        return 0.0
    if s.year == e.year:
        denom = 366.0 if _is_leap(s.year) else 365.0
        return (e - s).days / denom
    # 開始年の残り
    start_year_end = _dt.date(s.year + 1, 1, 1)
    denom_s = 366.0 if _is_leap(s.year) else 365.0
    frac = (start_year_end - s).days / denom_s
    # 中間の満年
    frac += e.year - s.year - 1
    # 終了年の頭から
    end_year_start = _dt.date(e.year, 1, 1)
    denom_e = 366.0 if _is_leap(e.year) else 365.0
    frac += (e - end_year_start).days / denom_e
    return frac
