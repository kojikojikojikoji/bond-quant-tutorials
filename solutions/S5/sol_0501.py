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
# # S5-1 演習 解答例
#
# Vasicek モデルの演習 2 問の解答例です。

# %%
import numpy as np
import matplotlib.pyplot as plt


def vasicek_B(a, tau):
    """B(tau) = (1 - exp(-a*tau)) / a。"""
    tau = np.asarray(tau, dtype=float)
    return (1.0 - np.exp(-a * tau)) / a


def vasicek_A(a, b, sigma, tau):
    """A(tau)（P = exp(A - B r) の指数の A 部分）。"""
    tau = np.asarray(tau, dtype=float)
    B = vasicek_B(a, tau)
    return (B - tau) * (a ** 2 * b - 0.5 * sigma ** 2) / a ** 2 - sigma ** 2 * B ** 2 / (4 * a)


def vasicek_zcb(a, b, sigma, r, tau):
    """ZCB 価格 P = exp(A(tau) - B(tau) r)。"""
    return np.exp(vasicek_A(a, b, sigma, tau) - vasicek_B(a, tau) * r)


def vasicek_zero_rate(a, b, sigma, r, tau):
    """ゼロレート y = -ln P / tau。"""
    tau = np.asarray(tau, dtype=float)
    return -np.log(vasicek_zcb(a, b, sigma, r, tau)) / tau


def simulate_short_rate(a, b, sigma, r0, T, n_steps, n_paths, seed=None):
    """OU 厳密サンプリングで短期金利パスを生成する。"""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    mean_rev = np.exp(-a * dt)
    var = sigma ** 2 / (2 * a) * (1 - mean_rev ** 2)
    r = np.full(n_paths, float(r0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = r
    for i in range(n_steps):
        mean = r * mean_rev + b * (1 - mean_rev)
        r = mean + np.sqrt(var) * rng.standard_normal(n_paths)
        out[:, i + 1] = r
    return np.linspace(0.0, T, n_steps + 1), out


def vasicek_mc_zcb(a, b, sigma, r0, tau, n_steps, n_paths, seed=None):
    """∫r ds の割引を MC で平均して ZCB 価格を推定する。"""
    times, paths = simulate_short_rate(a, b, sigma, r0, tau, n_steps, n_paths, seed)
    dt = tau / n_steps
    integral = dt * (paths[:, 1:-1].sum(axis=1) + 0.5 * (paths[:, 0] + paths[:, -1]))
    discounts = np.exp(-integral)
    return dict(price=float(discounts.mean()),
                stderr=float(discounts.std(ddof=1) / np.sqrt(n_paths)))


# %% [markdown]
# ## 演習1：カーブ形状のレポート
#
# $a\in\{0.1, 0.5, 1.5\}$、$\sigma\in\{0.005, 0.02\}$ を組み合わせ、$b=3\%$・$r_0=1\%$ 固定で
# ゼロレートカーブを描きます。

# %%
b, r0 = 0.03, 0.01
a_list = [0.1, 0.5, 1.5]
sigma_list = [0.005, 0.02]
tt = np.linspace(0.1, 30, 200)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, sigma in zip(axes, sigma_list):
    for a in a_list:
        y = vasicek_zero_rate(a, b, sigma, r0, tt)
        ax.plot(tt, y * 100, lw=2, label=f"a={a}")
    ax.axhline(b * 100, color="black", ls="--", lw=1, label=f"b={b}")
    ax.set_title(f"σ={sigma}")
    ax.set_xlabel("年限 τ（年）")
    ax.legend(fontsize=9)
axes[0].set_ylabel("ゼロレート（%）")
plt.tight_layout()
plt.show()

# %% [markdown]
# **観察と説明**
#
# - $r_0=1\% < b=3\%$ なので、いずれのカーブも右上がり（順イールド）です。金利が長期水準 $b$ へ
#   上がっていく期待を反映します。
# - $a$ を大きくすると、短期金利が $b$ へ**速く**回帰するため、カーブは短い年限で $b$ に到達し
#   早々に平坦化します。$a$ が小さいと回帰が遅く、長い年限までかけてゆるやかに上昇します。
# - $\sigma$ を大きくすると、右パネルのように長期側が押し下げられます。ゼロレートに含まれる
#   $-A(\tau)/\tau$ の項に $-\sigma^2 B^2/(4a)$ という**凸性（convexity）**由来の負の寄与があり、
#   $\tau$ が大きいほど効いてカーブを沈めるためです。

# %%
# 平坦化の速さを数値で: 各 a について「y が b の 1bp 以内に入る最小年限」
b_pct = b
for a in a_list:
    y = vasicek_zero_rate(a, b, 0.005, r0, tt)
    reached = tt[np.abs(y - b_pct) < 1e-4]
    first = reached[0] if reached.size else np.nan
    print(f"a={a}: ゼロレートが b の 1bp 以内に入る最小年限 ≈ {first:.1f} 年")

# %% [markdown]
# ## 演習2：MC 標準誤差の $1/\sqrt{n}$ 収束
#
# $\tau=5$ 年の ZCB を、パス数を増やしながら MC 推定し、標準誤差が $\propto 1/\sqrt{n}$ で
# 縮むことを両対数で確認します。

# %%
a, b, sigma, r0 = 0.30, 0.05, 0.02, 0.03
tau = 5.0
analytic = float(vasicek_zcb(a, b, sigma, r0, tau))
print(f"解析解 ZCB(τ={tau}) = {analytic:.6f}\n")

n_list = [2_000, 8_000, 32_000, 128_000, 512_000]
stderrs, prices = [], []
print(f"{'n':>9} {'MC価格':>12} {'標準誤差':>12} {'解析解との差':>14}")
for n in n_list:
    mc = vasicek_mc_zcb(a, b, sigma, r0, tau, n_steps=500, n_paths=n, seed=100 + n)
    stderrs.append(mc["stderr"])
    prices.append(mc["price"])
    print(f"{n:9d} {mc['price']:12.6f} {mc['stderr']:12.2e} {mc['price']-analytic:+14.6f}")
    # 解析解との差が標準誤差の範囲（4σ）に収まる
    assert abs(mc["price"] - analytic) < 4 * mc["stderr"]

# %%
stderrs = np.array(stderrs)
n_arr = np.array(n_list, dtype=float)
# 傾き -1/2 の参照線（最初の点を通す）
ref = stderrs[0] * np.sqrt(n_arr[0] / n_arr)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(n_arr, stderrs, "o-", color="steelblue", label="MC 標準誤差")
ax.loglog(n_arr, ref, "k--", alpha=0.7, label="傾き -1/2 の参照線 (∝ 1/√n)")
ax.set_xlabel("パス数 n")
ax.set_ylabel("標準誤差")
ax.set_title("MC 標準誤差の 1/√n 収束（Vasicek ZCB, τ=5年）")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# 標準誤差は参照線（傾き $-1/2$）にほぼ重なり、$n$ を 4 倍にするごとに約半分になります。
# 対数傾きを数値でも確認します。

# %%
log_slope = np.polyfit(np.log(n_arr), np.log(stderrs), 1)[0]
print(f"log-log 回帰の傾き = {log_slope:.4f}（理論 -0.5）")
assert abs(log_slope - (-0.5)) < 0.05
print("→ 標準誤差は 1/√n で縮み、MC 価格は解析解と標準誤差の範囲で一致する")
