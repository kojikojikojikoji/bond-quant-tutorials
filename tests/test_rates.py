"""bondlab.rates の回帰テスト（S0-1 / S1-1）。"""
import numpy as np
import pytest

from bondlab import rates


@pytest.mark.parametrize("conv", ["annual", "semiannual", "quarterly", "monthly", "continuous"])
def test_discount_roundtrip(conv):
    r, t = 0.037, 4.5
    df = rates.discount_factor(r, t, conv)
    r_back = rates.rate_from_discount(df, t, conv)
    assert abs(r_back - r) < 1e-12


def test_convert_rate_preserves_discount():
    # 規約変換しても同じ t での割引係数は不変。
    r, t = 0.05, 3.0
    df0 = rates.discount_factor(r, t, "semiannual")
    r2 = rates.convert_rate(r, t, "semiannual", "continuous")
    df1 = rates.discount_factor(r2, t, "continuous")
    assert abs(df0 - df1) < 1e-12


def test_continuous_conversion_is_time_independent():
    r = 0.05
    rc = rates.to_continuous(r, "semiannual")
    assert abs(rc - 2 * np.log1p(r / 2)) < 1e-14
    assert abs(rates.from_continuous(rc, "semiannual") - r) < 1e-14


def test_vectorized():
    rs = np.array([0.01, 0.02, 0.05])
    ts = np.array([1.0, 5.0, 10.0])
    df = rates.discount_factor(rs, ts, "annual")
    expected = (1 + rs) ** (-ts)
    assert np.allclose(df, expected)


def test_unknown_convention_raises():
    with pytest.raises(ValueError):
        rates.discount_factor(0.05, 1.0, "weekly")
