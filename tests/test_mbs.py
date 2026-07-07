"""bondlab.mbs の回帰テスト（S8）。PSA 定義・CF 保存則で検証する。"""
import numpy as np

from bondlab.mbs import psa_cpr, cpr_to_smm, smm_to_cpr, mbs_cashflows, weighted_average_life


def test_psa_ramp():
    # PSA100: age15 -> 3%, age30以降 -> 6%
    assert abs(psa_cpr(15) - 0.03) < 1e-12
    assert abs(psa_cpr(30) - 0.06) < 1e-12
    assert abs(psa_cpr(60) - 0.06) < 1e-12
    # PSA200 は倍
    assert abs(psa_cpr(15, 200) - 0.06) < 1e-12


def test_cpr_smm_roundtrip():
    for cpr in [0.02, 0.06, 0.30]:
        assert abs(smm_to_cpr(cpr_to_smm(cpr)) - cpr) < 1e-12


def test_cashflows_conserve_principal():
    age = np.arange(1, 361)
    smm = cpr_to_smm(psa_cpr(age, 100))
    cf = mbs_cashflows(100.0, 0.05, 360, smm)
    assert abs(cf["total_principal"].sum() - 100.0) < 1e-6
    assert abs(cf["balance"][-1]) < 1e-6


def test_higher_psa_shortens_wal():
    age = np.arange(1, 361)
    wal_100 = weighted_average_life(mbs_cashflows(100.0, 0.05, 360, cpr_to_smm(psa_cpr(age, 100))))
    wal_300 = weighted_average_life(mbs_cashflows(100.0, 0.05, 360, cpr_to_smm(psa_cpr(age, 300))))
    wal_0 = weighted_average_life(mbs_cashflows(100.0, 0.05, 360, np.zeros(360)))
    assert wal_300 < wal_100 < wal_0
