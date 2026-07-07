"""bondlab.risk の回帰テスト（S3-5）。"""
import numpy as np

from bondlab import risk


def test_historical_var_matches_quantile():
    x = np.random.default_rng(0).normal(0, 1, 5000)
    for a in (0.95, 0.99):
        assert np.isclose(-np.quantile(x, 1 - a), risk.historical_var(x, a))


def test_es_ge_var():
    x = np.random.default_rng(1).normal(0, 1, 100000)
    assert risk.historical_es(x, 0.99) >= risk.historical_var(x, 0.99)


def test_es_worst_mass_discrete():
    # 100サンプル・99%: 最悪1個の平均。境界の丸め上げに影響されない。
    x = np.array([-100.0] + [1.0] * 99)
    assert abs(risk.historical_es(x, 0.99) - 100.0) < 1e-9
    # 200サンプル・99%: 最悪2個の平均。
    x2 = np.array([-100.0, -50.0] + [1.0] * 198)
    assert abs(risk.historical_es(x2, 0.99) - 75.0) < 1e-9


def test_parametric_var_normal():
    from scipy import stats
    x = np.random.default_rng(2).normal(0.5, 2.0, 200000)
    z = stats.norm.ppf(0.01)
    assert abs(risk.parametric_var(x, 0.99) - (-(x.mean() + x.std(ddof=1) * z))) < 1e-9


def test_kupiec_expected_and_keys():
    k = risk.kupiec_pof(6, 600, 0.99)
    assert set(k) == {"lr", "p_value", "expected", "observed"}
    assert np.isclose(k["expected"], 6.0)
    # 例外数が期待どおりなら棄却しない
    assert k["p_value"] > 0.05
