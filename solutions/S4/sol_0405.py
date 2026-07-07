# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # S4-5 演習 解答例
#
# リスク中立評価とニュメレール変換の演習 2 問の解答例です。

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from bondlab.sim import brownian_paths


def crr_params(sigma, dt):
    """CRR の上昇率 u・下落率 d（u*d=1）。"""
    u = np.exp(sigma * np.sqrt(dt))
    return u, 1.0 / u


def binomial_price(S0, K, r_cont, sigma, T, n, kind="call"):
    """CRR n ステップ二項モデルのヨーロピアン価格（連続複利金利）。"""
    dt = T / n
    u, d = crr_params(sigma, dt)
    disc = np.exp(-r_cont * dt)
    q = (np.exp(r_cont * dt) - d) / (u - d)
    j = np.arange(n + 1)
    ST = S0 * u ** j * d ** (n - j)
    values = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    for _ in range(n):
        values = disc * (q * values[1:] + (1 - q) * values[:-1])
    return float(values[0])


def bs_price(S0, K, r_cont, sigma, T, kind="call"):
    """Black–Scholes 解析解。"""
    d1 = (np.log(S0 / K) + (r_cont + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return float(S0 * norm.cdf(d1) - K * np.exp(-r_cont * T) * norm.cdf(d2))
    return float(K * np.exp(-r_cont * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


# %% [markdown]
# ## 演習1：プットの二項価格の BS 極限への収束
#
# ステップ数を $2$ のべき乗で増やし、Black–Scholes 解析解との絶対誤差を両対数で見ます。

# %%
S0, K, r_cont, sigma, T = 100.0, 100.0, 0.03, 0.25, 1.0
bs_put = bs_price(S0, K, r_cont, sigma, T, kind="put")
print(f"Black–Scholes プット解析解 = {bs_put:.6f}\n")

ns = [2 ** k for k in range(0, 12)]   # 1 .. 2048
prices, errs = [], []
print(f"{'n':>6} {'二項プット':>12} {'BSとの差(符号付)':>16}")
for n in ns:
    bp = binomial_price(S0, K, r_cont, sigma, T, n, kind="put")
    prices.append(bp)
    errs.append(abs(bp - bs_put))
    print(f"{n:6d} {bp:12.6f} {bp - bs_put:16.6e}")

assert errs[-1] < 1e-2

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.loglog(ns, errs, "o-", color="steelblue", label="|二項価格 - BS解|")
ax.loglog(ns, errs[0] * ns[0] / np.array(ns), "k--", alpha=0.6, label="傾き -1 の参照線")
ax.set_xlabel("二項ステップ数 n")
ax.set_ylabel("絶対誤差")
ax.set_title("プット二項価格の BS 極限への収束")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# 誤差はおおむね $O(1/n)$ の傾きで減ります。ただし CRR 格子では、行使価格をまたぐノードの
# 位置が $n$ の偶奇で変わるため、誤差の絶対値は単調に減らず**振動**します。奇数・偶数で
# 別々に見ると、それぞれの系列は滑らかに $0$ へ近づきます。

# %%
odd = [(n, e) for n, e in zip(ns, errs) if n % 2 == 1]
even = [(n, e) for n, e in zip(ns, errs) if n % 2 == 0]
print("奇数ステップの誤差:", [f"{n}:{e:.2e}" for n, e in odd])
print("偶数ステップの誤差:", [f"{n}:{e:.2e}" for n, e in even])
print("→ 偶奇で誤差水準が異なり、系列が振動するのが分かる")

# %% [markdown]
# ## 演習2：Girsanov による測度変換とマルチンゲール性
#
# 実測度 $\mathbb{P}$（ドリフト $\mu$）で株価を生成し、ラドン・ニコディム微分
# $L=\exp(-\theta W_T-\tfrac12\theta^2 T)$、$\theta=(\mu-r)/\sigma$ で重み付けすると
# $\mathbb{Q}$ 期待値になります。$\mathbb{Q}$ の下で割引株価 $e^{-rT}S_T$ の期待値が $S_0$
# （マルチンゲール）になることを確認します。

# %%
S0, mu, r_cont, sigma, T = 100.0, 0.15, 0.03, 0.20, 1.0
theta = (mu - r_cont) / sigma
print(f"リスクの市場価格 θ = (μ-r)/σ = {theta:.4f}\n")

n_mc = 300_000
_, W = brownian_paths(n_mc, 1, T, seed=0)
WT = W[:, -1]

ST_P = S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * WT)   # 実測度 P で生成
L = np.exp(-theta * WT - 0.5 * theta ** 2 * T)                # dQ/dP

disc_ST = np.exp(-r_cont * T) * ST_P

# 実測度のまま（重みなし）→ S0*exp((mu-r)T) にずれる
mean_P = disc_ST.mean()
# Q に変換（重み L）→ S0（マルチンゲール）
mean_Q = (L * disc_ST).mean()

print(f"重みなし E_P[e^-rT S_T] = {mean_P:.4f}  （理論 S0*exp((μ-r)T) = "
      f"{S0*np.exp((mu-r_cont)*T):.4f}）")
print(f"重みあり E_Q[e^-rT S_T] = {mean_Q:.4f}  （理論 S0 = {S0:.4f}）")

assert abs(mean_Q - S0) / S0 < 0.01
assert abs(mean_P - S0 * np.exp((mu - r_cont) * T)) / S0 < 0.01
print("\n→ Q では割引株価の期待値が S0（マルチンゲール）。P ではドリフト分だけずれる")

# %% [markdown]
# ### ラドン・ニコディム微分の健全性チェック
#
# $L=d\mathbb{Q}/d\mathbb{P}$ は期待値 $\mathbb{E}^{\mathbb{P}}[L]=1$ を満たす確率密度です。数値でも確認します。

# %%
print(f"E_P[L] = {L.mean():.6f}（理論 1.0）")
assert abs(L.mean() - 1.0) < 0.01
print("→ 重みの総和が 1 に正規化されている（正しい測度変換）")
