"""確率過程シミュレーション。

ブラウン運動の生成、SDE の数値解法（Euler-Maruyama / Milstein）、モンテカルロ
推定と分散削減（対照変量・制御変量）を提供する。S4 で理論を扱い、S5 の金利
モデルや S8 の MC-OAS が土台に使う。
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def brownian_paths(n_paths: int, n_steps: int, T: float, seed: int | None = None,
                   antithetic: bool = False):
    """標準ブラウン運動のパスを生成する。

    Parameters
    ----------
    n_paths, n_steps : int
        パス数と時間ステップ数。
    T : float
        満期（年）。刻み dt = T / n_steps。
    seed : int, optional
        乱数シード。
    antithetic : bool
        True なら増分の符号反転パスを後半に足して対称化する（分散削減）。

    Returns
    -------
    times : ndarray shape (n_steps+1,)
    W : ndarray shape (n_paths, n_steps+1)   W[:,0]=0
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))
    incr = np.sqrt(dt) * z
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    times = np.linspace(0.0, T, n_steps + 1)
    return times, W


def simulate_sde(drift: Callable, diffusion: Callable, x0: float, T: float,
                 n_steps: int, n_paths: int, scheme: str = "euler",
                 seed: int | None = None, diffusion_x: Callable | None = None,
                 antithetic: bool = False):
    """一般の1次元 SDE dX = a(t,X)dt + b(t,X)dW を数値積分する。

    Parameters
    ----------
    drift, diffusion : callable
        a(t, x), b(t, x)。ベクトル化された x を受ける。
    scheme : str
        "euler"（Euler-Maruyama）または "milstein"。
    diffusion_x : callable, optional
        Milstein に必要な b の x 偏微分 ∂b/∂x(t, x)。
    antithetic : bool
        対称化した増分を使う。

    Returns
    -------
    times : ndarray (n_steps+1,)
    X : ndarray (n_paths, n_steps+1)
    """
    if scheme == "milstein" and diffusion_x is None:
        raise ValueError("milstein には diffusion_x（∂b/∂x）が必要")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))
    dW = np.sqrt(dt) * z
    times = np.linspace(0.0, T, n_steps + 1)
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for i in range(n_steps):
        t = times[i]
        x = X[:, i]
        a = drift(t, x)
        b = diffusion(t, x)
        step = a * dt + b * dW[:, i]
        if scheme == "milstein":
            bx = diffusion_x(t, x)
            step = step + 0.5 * b * bx * (dW[:, i] ** 2 - dt)
        X[:, i + 1] = x + step
    return times, X


def mc_stats(samples, confidence: float = 0.95) -> dict:
    """モンテカルロ推定量の平均・標準誤差・信頼区間を返す。"""
    from scipy import stats

    s = np.asarray(samples, dtype=float)
    n = s.size
    mean = s.mean()
    stderr = s.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    return dict(mean=float(mean), stderr=float(stderr),
                ci_low=float(mean - z * stderr), ci_high=float(mean + z * stderr), n=n)


def control_variate(target, control, control_mean: float) -> dict:
    """制御変量法。target と相関する control（既知の平均）で分散を削減する。

    最適係数 c* = Cov(target, control)/Var(control) を推定して補正する。
    """
    target = np.asarray(target, dtype=float)
    control = np.asarray(control, dtype=float)
    cov = np.cov(target, control)
    c = cov[0, 1] / cov[1, 1]
    adjusted = target - c * (control - control_mean)
    return dict(estimate=float(adjusted.mean()),
                stderr=float(adjusted.std(ddof=1) / np.sqrt(target.size)),
                beta=float(c))
