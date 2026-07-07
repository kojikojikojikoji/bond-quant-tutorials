"""bondlab.bond の回帰テスト（S1-3 / S1-4）。QuantLib と突合する。

経過利子・クリーン価格は ACT/ACT の ISMA/Bond 変種で一致する。うるう年境界を
またぐクーポン期間でも machine precision で一致することを含める（過去の
ISDA 比によるバグの回帰防止）。
"""
import datetime as dt

import QuantLib as ql

from bondlab.bond import FixedRateBond


def _schedule(issue, maturity):
    return ql.Schedule(
        ql.Date(issue.day, issue.month, issue.year),
        ql.Date(maturity.day, maturity.month, maturity.year),
        ql.Period(ql.Semiannual), ql.NullCalendar(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False,
    )


def test_yield_price_roundtrip():
    b = FixedRateBond(dt.date(2024, 6, 15), dt.date(2029, 6, 15), 0.02, 2, "ACT/ACT")
    s = dt.date(2026, 9, 10)
    clean = b.clean_price(0.025, s)
    assert abs(b.yield_from_price(clean, s) - 0.025) < 1e-10


def test_clean_and_accrued_match_quantlib_isma():
    issue, mat, cpn = dt.date(2024, 6, 15), dt.date(2029, 6, 15), 0.02
    b = FixedRateBond(issue, mat, cpn, 2, "ACT/ACT")
    sched = _schedule(issue, mat)
    qlb = ql.FixedRateBond(0, 100.0, sched, [cpn], ql.ActualActual(ql.ActualActual.Bond, sched))
    for s in [dt.date(2026, 9, 10), dt.date(2027, 3, 1)]:
        d = ql.Date(s.day, s.month, s.year)
        ql.Settings.instance().evaluationDate = d
        ir = ql.InterestRate(0.025, ql.ActualActual(ql.ActualActual.Bond, sched), ql.Compounded, ql.Semiannual)
        assert abs(b.accrued(s) - qlb.accruedAmount(d)) < 1e-10
        assert abs(b.clean_price(0.025, s) - ql.BondFunctions.cleanPrice(qlb, ir, d)) < 1e-8


def test_accrued_matches_isma_across_leap_boundary():
    # クーポン期間がうるう年境界（2024/2025）をまたぐ債券。
    issue, mat, cpn = dt.date(2023, 10, 15), dt.date(2029, 4, 15), 0.02
    b = FixedRateBond(issue, mat, cpn, 2, "ACT/ACT")
    sched = _schedule(issue, mat)
    qlb = ql.FixedRateBond(0, 100.0, sched, [cpn], ql.ActualActual(ql.ActualActual.Bond, sched))
    for s in [dt.date(2025, 1, 20), dt.date(2028, 3, 1)]:
        d = ql.Date(s.day, s.month, s.year)
        assert abs(b.accrued(s) - qlb.accruedAmount(d)) < 1e-10
