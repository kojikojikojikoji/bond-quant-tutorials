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
# # S5-2 演習 解答例
#
# CIR モデルのフェラー条件と、厳密サンプリング・離散化の差を扱う2問の解答例。

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

from bondlab.models import CIR


def cir_exact_ncx2(a, b, sigma, r0, T, n_steps, n_paths, rng):
    """scipy.stats.ncx2 による厳密サンプリング（本編と同じ実装）。"""
    from scipy.stats import ncx2
    dt = T / n_steps
    c = sigma ** 2 * (1.0 - np.exp(-a * dt)) / (4.0 * a)
    d = 4.0 * a * b / sigma ** 2
    decay = np.exp(-a * dt)
    r = np.full(n_paths, float(r0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = r
    for i in range(n_steps):
        lam = r * decay / c
        r = c * ncx2.rvs(df=d, nc=lam, size=n_paths, random_state=rng)
        out[:, i + 1] = r
    return np.linspace(0.0, T, n_steps + 1), out


def cir_euler_ft(a, b, sigma, r0, T, n_steps, n_paths, rng):
    """Euler full-truncation 離散化（本編と同じ実装）。"""
    dt = T / n_steps
    sqdt = np.sqrt(dt)
    r = np.full(n_paths, float(r0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = r
    n_neg = 0
    for i in range(n_steps):
        rp = np.maximum(r, 0.0)
        dW = rng.standard_normal(n_paths) * sqdt
        r = r + a * (b - rp) * dt + sigma * np.sqrt(rp) * dW
        n_neg += int((r < 0).sum())
        out[:, i + 1] = np.maximum(r, 0.0)
    return np.linspace(0.0, T, n_steps + 1), out, n_neg


def mc_zcb(paths, times, tau):
    """割引債価格 E[exp(-∫_0^τ r ds)] を台形則 + MC 推定（推定値, 標準誤差）。"""
    idx = int(np.round(tau / (times[1] - times[0])))
    dt = times[1] - times[0]
    seg = paths[:, : idx + 1]
    integral = (seg[:, 0] * 0.5 + seg[:, 1:-1].sum(axis=1) + seg[:, -1] * 0.5) * dt
    disc = np.exp(-integral)
    return disc.mean(), disc.std(ddof=1) / np.sqrt(len(disc))


def cir_zcb(a, b, sigma, r0, tau):
    """解析的 ZCB（解答の基準値に使う）。"""
    tau = np.asarray(tau, dtype=float)
    g = np.sqrt(a ** 2 + 2 * sigma ** 2)
    eg = np.exp(g * tau) - 1.0
    denom = 2 * g + (a + g) * eg
    A = (2 * g * np.exp((a + g) * tau / 2.0) / denom) ** (2 * a * b / sigma ** 2)
    B = 2 * eg / denom
    return A * np.exp(-B * r0)


# %% [markdown]
# ## 演習1：フェラー条件と原点近傍の分布形状
#
# $a=0.5,\ b=0.04,\ r_0=0.04$ を固定し、$\sigma$ を条件成立側・違反側で変えて
# 満期 $T=10$ の $r_T$ 分布を厳密サンプリングで描く。

# %%
a, b, r0 = 0.5, 0.04, 0.04
T, n_steps, n_paths = 10.0, 400, 60000

cases = {
    "条件成立 σ=0.10": 0.10,
    "条件違反 σ=0.30": 0.30,
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for sigma in cases.values():
    rng = np.random.default_rng(2026)
    _, paths = cir_exact_ncx2(a, b, sigma, r0, T, n_steps, n_paths, rng)
    rT = paths[:, -1]
    shape = 2 * a * b / sigma ** 2
    scale = sigma ** 2 / (2 * a)
    lab = f"σ={sigma:.2f}, 2ab/σ²={shape:.2f}"
    axes[0].hist(rT, bins=80, density=True, alpha=0.5, label=lab)
    xx = np.linspace(1e-6, rT.max(), 400)
    axes[0].plot(xx, gamma.pdf(xx, a=shape, scale=scale), lw=1)

axes[0].set_title("満期分布 r_T（全体）")
axes[0].set_xlabel("r_T")
axes[0].set_ylabel("密度")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 原点近傍を拡大。
for name, sigma in cases.items():
    rng = np.random.default_rng(2026)
    _, paths = cir_exact_ncx2(a, b, sigma, r0, T, n_steps, n_paths, rng)
    rT = paths[:, -1]
    shape = 2 * a * b / sigma ** 2
    scale = sigma ** 2 / (2 * a)
    bins = np.linspace(0, 0.01, 40)
    axes[1].hist(rT, bins=bins, density=True, alpha=0.5, label=name)
    xx = np.linspace(1e-6, 0.01, 300)
    axes[1].plot(xx, gamma.pdf(xx, a=shape, scale=scale), lw=1)

axes[1].set_xlim(0, 0.01)
axes[1].set_title("原点近傍（r_T < 0.01）の拡大")
axes[1].set_xlabel("r_T")
axes[1].set_ylabel("密度")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

for name, sigma in cases.items():
    shape = 2 * a * b / sigma ** 2
    rng = np.random.default_rng(2026)
    _, paths = cir_exact_ncx2(a, b, sigma, r0, T, n_steps, n_paths, rng)
    rT = paths[:, -1]
    near0 = np.mean(rT < 0.005)
    print(f"{name}: shape=2ab/σ²={shape:.3f}, "
          f"feller={CIR(a, b, sigma, r0).feller()}, "
          f"P(r_T<0.005)={100*near0:.2f}%")

# %% [markdown]
# **考察**：定常ガンマ分布の shape パラメータは $2ab/\sigma^2$。条件成立側
# （$\sigma=0.10$）では shape $>1$ で密度は原点で $0$ から立ち上がり、金利が $0$
# 近傍に来にくい。条件違反側（$\sigma=0.30$）では shape $<1$ となり密度が原点で
# 発散する形になり、$r_T$ が $0$ 付近に張り付く確率が跳ね上がる。フェラー条件
# $2ab\ge\sigma^2$ は shape $\ge1$（＝非中心カイ二乗の自由度 $4ab/\sigma^2\ge2$）と
# 同値であり、原点近傍の形状変化の境目そのものである。

# %% [markdown]
# ## 演習2：離散化の負値発生と厳密サンプリングの差
#
# 演習1の条件違反パラメータ（$\sigma=0.30$）で、時間刻み $dt$ を細かくしながら
# 離散化の負値発生率と割引債価格の推定バイアスを表にまとめる。厳密サンプリングは
# $dt$ に依存せず正しい値を返すことを確認する。

# %%
sigma = 0.30
tau = 5.0
n_paths = 60000
p_analytic = float(cir_zcb(a, b, sigma, r0, tau))
print(f"解析的割引債価格 P({tau:.0f}) = {p_analytic:.6f}\n")

dt_list = [0.1, 0.02, 0.004]
print(f"{'dt':>7} {'n_steps':>8} {'負値率(%)':>10} "
      f"{'離散化 P':>12} {'離散化 bias(σ)':>14} {'厳密 P':>12} {'厳密 bias(σ)':>14}")
rows = []
for dt in dt_list:
    ns = int(round(tau / dt))
    rng_e = np.random.default_rng(101)
    times_e, eu_paths, n_neg = cir_euler_ft(a, b, sigma, r0, tau, ns, n_paths, rng_e)
    neg_rate = 100 * n_neg / (ns * n_paths)
    e_eu, s_eu = mc_zcb(eu_paths, times_e, tau)

    rng_x = np.random.default_rng(202)
    times_x, ex_paths = cir_exact_ncx2(a, b, sigma, r0, tau, ns, n_paths, rng_x)
    e_ex, s_ex = mc_zcb(ex_paths, times_x, tau)

    print(f"{dt:7.3f} {ns:8d} {neg_rate:10.3f} "
          f"{e_eu:12.6f} {(e_eu-p_analytic)/s_eu:14.2f} "
          f"{e_ex:12.6f} {(e_ex-p_analytic)/s_ex:14.2f}")
    rows.append((dt, neg_rate, e_eu, e_ex))

# %% [markdown]
# **考察**：条件違反下では、粗い刻み $dt=0.1$ で離散化が頻繁に負値を踏み
# （full-truncation で $0$ に切り詰められる）、割引債価格に無視できないバイアスが
# 出る。$dt$ を細かくすると負値発生率もバイアスも縮むが、収束は遅い。一方、厳密
# サンプリングは非中心カイ二乗の遷移分布から直接引くため、どの $dt$ でも解析解と
# 標準誤差の範囲で一致し、離散化誤差を持たない。条件違反領域では厳密サンプリングを
# 使うのが定石である。

# %%
# 負値発生率が dt とともに単調に減ることを可視化。
dts = [r[0] for r in rows]
negs = [r[1] for r in rows]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dts, negs, "o-")
ax.set_xscale("log")
ax.set_xlabel("時間刻み dt（年, 対数軸）")
ax.set_ylabel("負値（切り詰め）発生率 (%)")
ax.set_title("離散化の負値発生率と時間刻みの関係（条件違反）")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()
