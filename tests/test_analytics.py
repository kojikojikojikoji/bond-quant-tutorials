"""bondlab.analytics の回帰テスト（S3）。QuantLib と突合する。"""
import datetime as dt

import numpy as np
import QuantLib as ql

from bondlab.bond import FixedRateBond
from bondlab.analytics import duration_convexity, effective_duration, bump_curve, pca
from bondlab.curve import bootstrap_par


def test_duration_convexity_matches_quantlib():
    b = FixedRateBond(dt.date(2024, 6, 15), dt.date(2034, 6, 15), 0.03, 2, "ACT/ACT")
    settle, y = dt.date(2026, 6, 15), 0.035
    dc = duration_convexity(b, y, settle)
    sched = ql.Schedule(ql.Date(15, 6, 2024), ql.Date(15, 6, 2034), ql.Period(ql.Semiannual),
                        ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    aa = ql.ActualActual(ql.ActualActual.Bond, sched)
    qlb = ql.FixedRateBond(0, 100.0, sched, [0.03], aa)
    d = ql.Date(15, 6, 2026)
    ql.Settings.instance().evaluationDate = d
    ir = ql.InterestRate(y, aa, ql.Compounded, ql.Semiannual)
    assert abs(dc["modified"] - ql.BondFunctions.duration(qlb, ir, ql.Duration.Modified, d)) < 1e-10
    assert abs(dc["macaulay"] - ql.BondFunctions.duration(qlb, ir, ql.Duration.Macaulay, d)) < 1e-10
    assert abs(dc["convexity"] - ql.BondFunctions.convexity(qlb, ir, d)) < 1e-8


def test_effective_equals_modified_for_fixed_cf():
    b = FixedRateBond(dt.date(2024, 6, 15), dt.date(2034, 6, 15), 0.03, 2, "ACT/ACT")
    settle, y = dt.date(2026, 6, 15), 0.035
    eff = effective_duration(lambda yy: b.dirty_price(yy, settle), y, bump=1e-5)
    assert abs(eff - duration_convexity(b, y, settle)["modified"]) < 1e-5


def test_krd_sum_equals_parallel_dv01():
    c = bootstrap_par(np.arange(1.0, 21.0), np.full(20, 0.03), frequency=1)
    cf = np.array([[k, 4.0] for k in range(1, 21)] + [[20.0, 100.0]])

    def pv(curve):
        return float(np.sum(cf[:, 1] * curve.discount(cf[:, 0])))

    nodes = c.times[c.times > 0]
    v0 = pv(c)
    krd_sum = sum(-(pv(bump_curve(c, float(n), 1e-4)) - v0) for n in nodes)
    zb = np.array([c.zero_rate(x) if x > 0 else 0.0 for x in c.times])
    m = c.times > 0
    from bondlab.curve import DiscountCurve
    par = DiscountCurve(c.times[m], np.exp(-(zb[m] + 1e-4) * c.times[m]))
    par_dv01 = -(pv(par) - v0)
    assert abs(krd_sum - par_dv01) / abs(par_dv01) < 1e-3


def test_pca_level_dominates():
    rng = np.random.default_rng(0)
    tenors = np.array([2, 5, 10, 20, 30.0])
    lvl = rng.standard_normal((400, 1))
    changes = lvl * np.ones_like(tenors) * 0.8 + 0.02 * rng.standard_normal((400, 5))
    p = pca(changes)
    assert p["explained_ratio"][0] > 0.8
    assert abs(p["explained_ratio"].sum() - 1.0) < 1e-12
