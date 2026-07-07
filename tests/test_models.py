"""bondlab.models の回帰テスト（S5）。QuantLib と突合する。"""
import numpy as np
import QuantLib as ql

from bondlab.models import Vasicek, CIR, HullWhite
from bondlab.curve import bootstrap_par


def test_vasicek_zcb_matches_quantlib():
    a, b, s, r0 = 0.30, 0.05, 0.02, 0.03
    v = Vasicek(a, b, s, r0)
    qv = ql.Vasicek(r0, a, b, s)
    for tau in [0.5, 1, 5, 10, 30]:
        assert abs(v.zcb(tau) - qv.discountBond(0.0, tau, r0)) < 1e-10


def test_cir_zcb_matches_quantlib():
    a, b, s, r0 = 0.20, 0.05, 0.08, 0.03
    c = CIR(a, b, s, r0)
    qc = ql.CoxIngersollRoss(r0, b, a, s)  # (r0, theta=level, k=speed, sigma)
    for tau in [0.5, 1, 5, 10, 20]:
        assert abs(c.zcb(tau) - qc.discountBond(0.0, tau, r0)) < 1e-10


def test_cir_feller_and_exact_mc():
    c = CIR(0.20, 0.05, 0.08, 0.03)
    assert c.feller()
    t, paths = c.simulate_exact(3.0, 150, 40000, seed=1)
    dt = t[1] - t[0]
    integ = (paths[:, 0] * 0.5 + paths[:, 1:-1].sum(1) + paths[:, -1] * 0.5) * dt
    disc = np.exp(-integ)
    se = disc.std(ddof=1) / np.sqrt(disc.size)
    assert abs(disc.mean() - c.zcb(3.0)) < 4 * se


def test_vasicek_sim_stationary_moments():
    a, b, s, r0 = 0.1, 0.05, 0.01, 0.03
    v = Vasicek(a, b, s, r0)
    _, paths = v.simulate(5.0, 60, 40000, seed=1)
    mean_th = b + (r0 - b) * np.exp(-a * 5)
    var_th = s ** 2 / (2 * a) * (1 - np.exp(-2 * a * 5))
    assert abs(paths[:, -1].mean() - mean_th) < 5e-4
    assert abs(paths[:, -1].var() - var_th) < 5e-5


def test_hull_white_reproduces_initial_curve():
    curve = bootstrap_par(np.arange(1.0, 11.0), np.linspace(0.02, 0.035, 10), frequency=1)
    hw = HullWhite(0.05, 0.01, curve)
    for T in [1.0, 5.0, 10.0]:
        assert abs(hw.zcb(0.0, T) - curve.discount(T)) < 1e-10
