"""bondlab.pricing の回帰テスト（S6）。QuantLib と突合する。"""
import numpy as np
import QuantLib as ql

from bondlab import pricing
from bondlab.curve import bootstrap_par


def test_black76_matches_quantlib():
    F, K, T, vol, df = 0.03, 0.032, 2.0, 0.25, 0.95
    for opt, qopt in [("call", ql.Option.Call), ("put", ql.Option.Put)]:
        q = df * ql.blackFormula(qopt, K, F, vol * np.sqrt(T))
        assert abs(pricing.black76(F, K, T, vol, df, opt) - q) < 1e-14


def test_bachelier_matches_quantlib():
    F, K, T, nvol, df = 0.03, 0.032, 2.0, 0.008, 0.95
    for opt, qopt in [("call", ql.Option.Call), ("put", ql.Option.Put)]:
        q = df * ql.bachelierBlackFormula(qopt, K, F, nvol * np.sqrt(T))
        assert abs(pricing.bachelier(F, K, T, nvol, df, opt) - q) < 1e-14


def test_implied_vol_roundtrip():
    F, K, T, vol = 0.03, 0.035, 2.0, 0.30
    px = pricing.black76(F, K, T, vol)
    assert abs(pricing.implied_vol_black(px, F, K, T) - vol) < 1e-8


def test_sabr_matches_quantlib():
    rng = np.random.default_rng(1)
    for _ in range(100):
        F, K, T = rng.uniform(0.01, 0.06), rng.uniform(0.005, 0.08), rng.uniform(0.25, 10)
        a, b, r, n = (rng.uniform(0.005, 0.05), rng.uniform(0.1, 0.9),
                      rng.uniform(-0.7, 0.7), rng.uniform(0.05, 0.8))
        assert abs(pricing.sabr_vol(F, K, T, a, b, r, n) - ql.sabrVolatility(K, F, T, a, b, n, r)) < 1e-12


def test_par_swap_flat_and_forward():
    c = bootstrap_par(np.arange(1, 31.0), np.full(30, 0.03), frequency=1)
    assert abs(pricing.par_swap_rate(c, c, np.arange(1, 6.0)) - 0.03) < 1e-12
    # フォワードスタート 5y5y: (P(5)-P(10))/annuity(start=5)
    fwd = pricing.par_swap_rate(c, c, np.arange(6, 11.0), start=5.0)
    ann = pricing.swap_annuity(c, np.arange(6, 11.0), start=5.0)
    assert abs(fwd - (c.discount(5.0) - c.discount(10.0)) / ann) < 1e-12


def test_swaption_equals_annuity_times_black():
    F, K, T, vol, A = 0.03, 0.03, 5.0, 0.25, 4.0
    assert abs(pricing.swaption_black(F, K, T, vol, A, "payer")
               - A * pricing.black76(F, K, T, vol, 1.0, "call")) < 1e-12
