"""bondlab.curve の回帰テスト（S2）。"""
import numpy as np

from bondlab.curve import bootstrap_par, DiscountCurve, nss, fit_nss


def test_bootstrap_reprices_par_bonds():
    grid = np.arange(1.0, 11.0)
    par = np.linspace(0.03, 0.045, grid.size)
    c = bootstrap_par(grid, par, frequency=1)
    dfs = c.dfs[1:]  # skip the t=0, DF=1 node
    for i, r in enumerate(par):
        price = r * dfs[: i + 1].sum() + dfs[i]
        assert abs(price - 1.0) < 1e-12


def test_discount_scalar_returns_float():
    c = bootstrap_par(np.arange(1.0, 6.0), np.full(5, 0.03), frequency=1)
    assert isinstance(c.discount(2.5), float)
    assert isinstance(c.zero_rate(2.5), float)
    assert isinstance(c.forward_rate(2.0, 3.0), float)
    # 配列入力は配列を返す
    assert c.discount(np.array([1.0, 2.0])).shape == (2,)


def test_discount_curve_monotone_and_positive():
    c = bootstrap_par(np.arange(1.0, 11.0), np.linspace(0.03, 0.045, 10), frequency=1)
    ts = np.linspace(0.1, 10.0, 50)
    dfs = c.discount(ts)
    assert np.all(dfs > 0)
    assert np.all(np.diff(dfs) < 0)  # 割引係数は単調減少


def test_nss_zero_limit_is_finite():
    # tau=0 の極限 z(0)=beta0+beta1
    z0 = nss(0.0, 0.03, -0.02, 0.02, 0.01, 1.5, 5.0)
    assert abs(float(z0) - 0.01) < 1e-12


def test_fit_nss_roundtrip():
    ten = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])
    p = dict(beta0=0.03, beta1=-0.02, beta2=0.02, beta3=0.01, lam1=1.5, lam2=5.0)
    y = nss(ten, **p)
    fit = fit_nss(ten, y)
    assert np.max(np.abs(nss(ten, **fit) - y)) < 1e-8
