"""bondlab.credit の回帰テスト（S7）。QuantLib と突合する。"""
import numpy as np
import QuantLib as ql

from bondlab.credit import (HazardCurve, cds_par_spread, bootstrap_hazard,
                            merton_equity, distance_to_default, merton_pd, solve_asset)
from bondlab.curve import bootstrap_par


def test_survival_constant_hazard():
    hc = HazardCurve([5.0, 10.0], [0.03, 0.03])
    t = np.linspace(0, 10, 41)
    assert np.allclose(hc.survival(t), np.exp(-0.03 * t), atol=1e-12)


def test_cds_par_spread_matches_quantlib():
    disc = bootstrap_par(np.arange(1, 11.0), np.full(10, 0.02), frequency=1)
    today = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dq = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.02, ql.Actual365Fixed()))
    for lam in [0.01, 0.03, 0.06]:
        end = today + ql.Period(5, ql.Years)
        hz = ql.HazardRateCurve([today, end], [lam, lam], ql.Actual365Fixed())
        hz.enableExtrapolation()
        sched = ql.MakeSchedule(today, end, ql.Period(ql.Quarterly))
        cds = ql.CreditDefaultSwap(ql.Protection.Buyer, 1e6, 0.01, sched, ql.Following, ql.Actual365Fixed())
        cds.setPricingEngine(ql.MidPointCdsEngine(ql.DefaultProbabilityTermStructureHandle(hz), 0.4, dq))
        mine = cds_par_spread(HazardCurve([5.0], [lam]), disc, 5.0, 0.4, 4)
        assert abs(mine - cds.fairSpread()) * 1e4 < 0.05  # within 0.05 bp


def test_bootstrap_hazard_roundtrip():
    disc = bootstrap_par(np.arange(1, 11.0), np.full(10, 0.02), frequency=1)
    tenors = np.array([1, 3, 5, 7, 10.0])
    spreads = np.array([80, 100, 120, 130, 140.0]) / 1e4
    hc = bootstrap_hazard(disc, tenors, spreads, 0.4, 4)
    recon = np.array([cds_par_spread(hc, disc, T, 0.4, 4) for T in tenors])
    assert np.allclose(recon, spreads, atol=1e-10)


def test_merton_roundtrip():
    from scipy import stats
    V, sV, D, T, r = 120.0, 0.25, 80.0, 1.0, 0.02
    d1 = (np.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
    E = merton_equity(V, sV, D, T, r)
    assert abs(d1 - sV * np.sqrt(T) - distance_to_default(V, sV, D, T, r)) < 1e-12
    assert abs(merton_pd(V, sV, D, T, r) - stats.norm.cdf(-distance_to_default(V, sV, D, T, r))) < 1e-12
    sE = stats.norm.cdf(d1) * V / E * sV
    sol = solve_asset(E, sE, D, T, r)
    assert abs(sol["asset"] - V) < 1e-5 and abs(sol["asset_vol"] - sV) < 1e-5
