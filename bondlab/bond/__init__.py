"""固定利付債の価格・YTM・経過利子。

S1-3 / S1-4 で理論を扱い、ここに再利用可能な実装を置く。キャッシュフローを
明示的に生成し、割引の合計として価格を出す素朴な実装にとどめ、可読性を優先
する。QuantLib の FixedRateBond と突合して検証する。
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import List

from scipy.optimize import brentq

from bondlab import daycount


@dataclass
class FixedRateBond:
    """固定利付債。額面 100 を基準にする。

    Attributes
    ----------
    issue : date
        発行日（最初のクーポン計算の起点）。
    maturity : date
        満期日。
    coupon : float
        年クーポン率（小数。2% は 0.02）。
    frequency : int
        年間の利払回数（2 なら半年）。
    convention : str
        経過利子計算の day count 規約。
    face : float
        額面。
    """

    issue: _dt.date
    maturity: _dt.date
    coupon: float
    frequency: int = 2
    convention: str = "ACT/ACT"
    face: float = 100.0
    _schedule: List[_dt.date] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._schedule = self._build_schedule()

    def _build_schedule(self) -> List[_dt.date]:
        """満期から逆算してクーポン日を刻む（月数ベースの素朴な生成）。"""
        months = 12 // self.frequency
        dates = [self.maturity]
        d = self.maturity
        while d > self.issue:
            d = _add_months(d, -months)
            dates.append(d)
        dates = sorted(set(dates))
        # issue より前は落とす（逆算で1つ手前まで入るため）。
        return [x for x in dates if x > self.issue]

    def cashflows(self):
        """(日付, キャッシュフロー) のリスト。最後に額面償還を足す。"""
        cpn = self.face * self.coupon / self.frequency
        flows = [(d, cpn) for d in self._schedule]
        # 満期に額面を上乗せ。
        d_last, c_last = flows[-1]
        flows[-1] = (d_last, c_last + self.face)
        return flows

    def dirty_price(self, ytm: float, settlement: _dt.date) -> float:
        """YTM から dirty price（経過利子込み）を求める。

        市場慣行の periodic exponent で割り引く。次クーポンまでの残り期間割合
        を w とすると、j 番目（j=0 が次クーポン）の割引指数は w + j。
        満期までの完全な期間は指数 1 として数え、期中の端数だけ day count で
        測る。これが street convention で、QuantLib の複利利回りと一致する。

            w = yf(settlement, next_cpn) / yf(prev_cpn, next_cpn)
            price = Σ_j  c_j / (1 + y/f)^(w + j)
        """
        f = self.frequency
        prev, nxt = self._surrounding(settlement)
        if nxt is None:
            return 0.0
        w = 1.0 - self._elapsed_fraction(prev, settlement, nxt)
        future = [(d, c) for d, c in self.cashflows() if d > settlement]
        pv = 0.0
        for j, (_d, c) in enumerate(future):
            pv += c * (1.0 + ytm / f) ** (-(w + j))
        return pv

    def accrued(self, settlement: _dt.date) -> float:
        """経過利子。直前クーポン日から決済日までの期間割合でクーポンを按分する。"""
        prev, nxt = self._surrounding(settlement)
        if prev is None or nxt is None:
            return 0.0
        cpn = self.face * self.coupon / self.frequency
        return cpn * self._elapsed_fraction(prev, settlement, nxt)

    def _elapsed_fraction(self, prev: _dt.date, settlement: _dt.date, nxt: _dt.date) -> float:
        """クーポン期間のうち経過した割合を返す（0〜1）。

        ACT/ACT は債券標準の ISMA/Bond 変種、すなわち実日数の比で測る。
        ISDA の暦年按分を期間比に使ううるう年境界のずれ（最大 5e-4 程度）を避ける。
        他の規約は年数が区間について加法的なので、規約年数の比で一致する。
        """
        if nxt <= prev:
            return 0.0
        if self.convention == "ACT/ACT":
            full = (nxt - prev).days
            part = (settlement - prev).days
        else:
            full = daycount.year_fraction(prev, nxt, self.convention)
            part = daycount.year_fraction(prev, settlement, self.convention)
        return part / full if full > 0 else 0.0

    def clean_price(self, ytm: float, settlement: _dt.date) -> float:
        """clean price = dirty price − 経過利子。"""
        return self.dirty_price(ytm, settlement) - self.accrued(settlement)

    def yield_from_price(self, clean: float, settlement: _dt.date) -> float:
        """clean price から YTM を数値解法（Brent 法）で逆算する。"""
        target = clean + self.accrued(settlement)

        def f(y: float) -> float:
            return self.dirty_price(y, settlement) - target

        return brentq(f, -0.5, 2.0, xtol=1e-12, maxiter=200)

    def _surrounding(self, settlement: _dt.date):
        """決済日を挟む (直前クーポン日, 次クーポン日) を返す。"""
        all_dates = [self.issue] + self._schedule
        prev = None
        nxt = None
        for d in all_dates:
            if d <= settlement:
                prev = d
            elif nxt is None:
                nxt = d
        return prev, nxt


def _add_months(d: _dt.date, months: int) -> _dt.date:
    """月加算（月末は各月の末日に丸める）。"""
    m0 = d.month - 1 + months
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    # その月の末日
    if m == 12:
        last = 31
    else:
        last = (_dt.date(y, m + 1, 1) - _dt.timedelta(days=1)).day
    return _dt.date(y, m, min(d.day, last))
