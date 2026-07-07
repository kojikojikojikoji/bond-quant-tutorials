"""bondlab.sim の回帰テスト（S4）。解析解・収束次数と突合する。"""
import numpy as np

from bondlab import sim


def test_brownian_moments_and_quadratic_variation():
    _, W = sim.brownian_paths(20000, 100, 1.0, seed=42)
    assert W.shape == (20000, 101)
    assert np.allclose(W[:, 0], 0.0)
    assert abs(W[:, -1].mean()) < 0.03
    assert abs(W[:, -1].var(ddof=1) - 1.0) < 0.05
    qv = (np.diff(W, axis=1) ** 2).sum(axis=1)
    assert abs(qv.mean() - 1.0) < 0.02


def test_antithetic_symmetry():
    _, W = sim.brownian_paths(2, 50, 1.0, seed=0, antithetic=True)
    assert np.allclose(W[0], -W[1])


def _make_dW(n_paths, n_steps, dt, seed):
    return np.sqrt(dt) * np.random.default_rng(seed).standard_normal((n_paths, n_steps))


def _solve(drift, diff, x0, T, dW, scheme="euler", diff_x=None):
    n, N = dW.shape
    dt = T / N
    X = np.empty((n, N + 1))
    X[:, 0] = x0
    t = np.linspace(0, T, N + 1)
    for i in range(N):
        a, b = drift(t[i], X[:, i]), diff(t[i], X[:, i])
        step = a * dt + b * dW[:, i]
        if scheme == "milstein":
            step += 0.5 * b * diff_x(t[i], X[:, i]) * (dW[:, i] ** 2 - dt)
        X[:, i + 1] = X[:, i] + step
    return X


def test_strong_orders_gbm():
    mu, sig, x0, T, M = 0.05, 0.2, 1.0, 1.0, 20000
    d, b, bx = (lambda t, x: mu * x), (lambda t, x: sig * x), (lambda t, x: sig * np.ones_like(x))
    dts, ee, em = [], [], []
    for N in [8, 16, 32, 64, 128, 256]:
        dW = _make_dW(M, N, T / N, 2024 + N)
        WT = dW.sum(1)
        xT = x0 * np.exp((mu - 0.5 * sig ** 2) * T + sig * WT)
        ee.append(np.mean(np.abs(_solve(d, b, x0, T, dW, "euler")[:, -1] - xT)))
        em.append(np.mean(np.abs(_solve(d, b, x0, T, dW, "milstein", bx)[:, -1] - xT)))
        dts.append(T / N)
    assert 0.4 < np.polyfit(np.log(dts), np.log(ee), 1)[0] < 0.65
    assert 0.85 < np.polyfit(np.log(dts), np.log(em), 1)[0] < 1.15


def test_control_variate_reduces_stderr():
    rng = np.random.default_rng(0)
    S0, K, r, s, T = 100.0, 100.0, 0.03, 0.2, 1.0
    Z = rng.standard_normal(200000)
    ST = S0 * np.exp((r - 0.5 * s ** 2) * T + s * np.sqrt(T) * Z)
    target = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    control = np.exp(-r * T) * ST
    crude = sim.mc_stats(target)
    cv = sim.control_variate(target, control, S0)
    assert cv["stderr"] < crude["stderr"]
