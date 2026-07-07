"""短期金利モデル。

Vasicek・CIR（アフィン期間構造モデル。解析的なゼロクーポン債価格つき）と、
市場カーブに整合する Hull-White（拡張 Vasicek）を提供する。S5 で理論を扱い、
S6 のデリバティブ評価・S8 の MC-OAS が土台に使う。

いずれも QuantLib の対応クラスと突合して検証する。
"""
from __future__ import annotations

import numpy as np

from bondlab.curve import DiscountCurve


class Vasicek:
    """Vasicek モデル dr = a(b - r)dt + σ dW。

    a=平均回帰速度、b=長期水準、σ=ボラティリティ。アフィンモデルなので
    ゼロクーポン債価格が解析的に書ける。金利は負になりうる。
    """

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        self.a, self.b, self.sigma, self.r0 = a, b, sigma, r0

    def _B(self, tau):
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def _A(self, tau):
        a, b, s = self.a, self.b, self.sigma
        B = self._B(tau)
        return np.exp((B - tau) * (a ** 2 * b - 0.5 * s ** 2) / a ** 2 - s ** 2 * B ** 2 / (4 * a))

    def zcb(self, tau, r=None):
        """満期までの年数 tau のゼロクーポン債価格 P = A·exp(-B·r)。"""
        r = self.r0 if r is None else r
        return self._A(tau) * np.exp(-self._B(tau) * r)

    def zero_rate(self, tau, r=None):
        tau = np.asarray(tau, dtype=float)
        return -np.log(self.zcb(tau, r)) / tau

    def simulate(self, T, n_steps, n_paths, seed=None):
        """短期金利パスを厳密サンプリング（OU 遷移分布）で生成する。"""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        a, b, s = self.a, self.b, self.sigma
        r = np.full(n_paths, self.r0)
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = r
        mean_rev = np.exp(-a * dt)
        var = s ** 2 / (2 * a) * (1 - mean_rev ** 2)
        for i in range(n_steps):
            mean = r * mean_rev + b * (1 - mean_rev)
            r = mean + np.sqrt(var) * rng.standard_normal(n_paths)
            out[:, i + 1] = r
        return np.linspace(0, T, n_steps + 1), out


class CIR:
    """CIR モデル dr = a(b - r)dt + σ√r dW。

    平方根拡散により金利は非負（Feller 条件 2ab ≥ σ² で厳密に正）。ZCB は
    解析的。厳密サンプリングは非中心カイ二乗分布による。
    """

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        self.a, self.b, self.sigma, self.r0 = a, b, sigma, r0

    def feller(self) -> bool:
        return 2 * self.a * self.b >= self.sigma ** 2

    def _gamma(self):
        return np.sqrt(self.a ** 2 + 2 * self.sigma ** 2)

    def zcb(self, tau, r=None):
        r = self.r0 if r is None else r
        a, b, s = self.a, self.b, self.sigma
        g = self._gamma()
        eg = np.exp(g * tau) - 1.0
        denom = 2 * g + (a + g) * eg
        A = (2 * g * np.exp((a + g) * tau / 2.0) / denom) ** (2 * a * b / s ** 2)
        B = 2 * eg / denom
        return A * np.exp(-B * r)

    def zero_rate(self, tau, r=None):
        tau = np.asarray(tau, dtype=float)
        return -np.log(self.zcb(tau, r)) / tau

    def simulate_exact(self, T, n_steps, n_paths, seed=None):
        """非中心カイ二乗分布による厳密サンプリング。"""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        a, b, s = self.a, self.b, self.sigma
        c = s ** 2 * (1 - np.exp(-a * dt)) / (4 * a)
        df = 4 * a * b / s ** 2
        r = np.full(n_paths, self.r0)
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = r
        for i in range(n_steps):
            nc = r * np.exp(-a * dt) / c
            r = c * rng.noncentral_chisquare(df, nc, n_paths)
            out[:, i + 1] = r
        return np.linspace(0, T, n_steps + 1), out


class HullWhite:
    """Hull-White（拡張 Vasicek）dr = (θ(t) - a r)dt + σ dW。

    θ(t) を初期割引カーブに合わせることで、市場のゼロクーポン債価格を厳密に
    再現する no-arbitrage モデル。ZCB は Brigo-Mercurio の式で与える。
    """

    def __init__(self, a: float, sigma: float, curve: DiscountCurve):
        self.a, self.sigma, self.curve = a, sigma, curve

    def _pm(self, t):
        """市場割引係数 P^M(0,t)。"""
        return self.curve.discount(t)

    def _fm(self, t, eps: float = 1e-4):
        """市場の瞬間フォワード f^M(0,t) = -d/dt ln P^M(0,t)。"""
        t = max(t, eps)
        return -(np.log(self._pm(t + eps)) - np.log(self._pm(t - eps))) / (2 * eps)

    def _B(self, t, T):
        return (1.0 - np.exp(-self.a * (T - t))) / self.a

    def zcb(self, t, T, r_t=None):
        """時点 t・状態 r_t のゼロクーポン債価格 P(t,T)。

        r_t を省略すると t=0（r_0 = f^M(0,0)）とみなし、初期カーブを再現する。
        """
        a, s = self.a, self.sigma
        B = self._B(t, T)
        if t == 0.0:
            return self._pm(T)
        r_t = self._fm(0.0) if r_t is None else r_t
        lnA = (np.log(self._pm(T) / self._pm(t))
               + B * self._fm(t)
               - s ** 2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B ** 2)
        return np.exp(lnA - B * r_t)
