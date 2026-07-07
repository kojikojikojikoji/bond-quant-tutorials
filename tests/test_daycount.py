"""bondlab.daycount の回帰テスト（S1-2）。QuantLib と突合する。"""
import datetime as dt

import QuantLib as ql

from bondlab.daycount import year_fraction


def _qld(d: dt.date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


def test_known_values():
    s, e = dt.date(2025, 1, 31), dt.date(2025, 7, 31)
    assert abs(year_fraction(s, e, "30/360") - 0.5) < 1e-12
    assert abs(year_fraction(s, e, "ACT/360") - 181 / 360) < 1e-12
    assert abs(year_fraction(s, e, "ACT/365F") - 181 / 365) < 1e-12
    assert abs(year_fraction(dt.date(2024, 1, 1), dt.date(2025, 1, 1), "ACT/ACT") - 1.0) < 1e-12


def test_matches_quantlib():
    pairs = [
        (dt.date(2024, 2, 29), dt.date(2025, 2, 28)),
        (dt.date(2023, 10, 15), dt.date(2024, 4, 15)),
        (dt.date(2025, 1, 31), dt.date(2025, 3, 31)),
    ]
    mapping = {
        "ACT/360": ql.Actual360(),
        "ACT/365F": ql.Actual365Fixed(),
        "30/360": ql.Thirty360(ql.Thirty360.BondBasis),
        "ACT/ACT": ql.ActualActual(ql.ActualActual.ISDA),
    }
    for s, e in pairs:
        for conv, qdc in mapping.items():
            mine = year_fraction(s, e, conv)
            q = qdc.yearFraction(_qld(s), _qld(e))
            assert abs(mine - q) < 1e-12, (conv, s, e, mine, q)
