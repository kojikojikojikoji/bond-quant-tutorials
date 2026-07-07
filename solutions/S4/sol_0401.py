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
# # S4-1 演習 解答例

# %% [markdown]
# ## 準備
#
# 本編の自作関数を最小限だけ再掲する。ベクトル化したパス生成器と、累積2次変分・
# 累積1次変分を手元に置く。

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import bondlab

print("bondlab version:", bondlab.__version__)


def bm_paths(n_paths, n_steps, T, seed=None):
    """標準ブラウン運動のパスをベクトル化生成する（本編と同一）。"""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    incr = np.sqrt(dt) * z
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    times = np.linspace(0.0, T, n_steps + 1)
    return times, W


def quadratic_variation(W):
    """各時点までの累積2次変分 Σ(ΔW)^2 を返す（本編と同一）。"""
    incr = np.diff(W, axis=1)
    qv = np.cumsum(incr ** 2, axis=1)
    return np.concatenate([np.zeros((W.shape[0], 1)), qv], axis=1)


def total_variation(W):
    """各時点までの累積1次変分 Σ|ΔW| を返す（本編と同一）。"""
    incr = np.diff(W, axis=1)
    tv = np.cumsum(np.abs(incr), axis=1)
    return np.concatenate([np.zeros((W.shape[0], 1)), tv], axis=1)


# %% [markdown]
# ## 演習1：2次変分と1次変分の対比
#
# 満期 $T=1$ を固定し、ステップ数 $m$ を変えて多数パスを生成する。各パスの終端
# 2次変分 $[W]_T$ と1次変分 $V^{(1)}_T$ を測り、パス平均をとる。
#
# 理論の予想は次の通り。
#
# - 2次変分：$\mathbb{E}[[W]_T] = T$（$m$ によらず一定）
# - 1次変分：$\mathbb{E}[V^{(1)}_T] = m \cdot \mathbb{E}\lvert \Delta W \rvert
#   = m \sqrt{2\Delta t/\pi} = \sqrt{2/\pi}\,\sqrt{m}\,\sqrt{T}$
#   （$m^{1/2}$ で発散）

# %%
T = 1.0
n_paths = 3000
ms = [25, 100, 400, 1600, 6400]

rows = []
for m in ms:
    _, W = bm_paths(n_paths, m, T, seed=2024)
    qv_T = quadratic_variation(W)[:, -1].mean()
    tv_T = total_variation(W)[:, -1].mean()
    tv_theory = np.sqrt(2.0 / np.pi) * np.sqrt(m) * np.sqrt(T)
    rows.append((m, qv_T, tv_T, tv_theory))

print(f"{'m':>7s} {'平均[W]_T':>12s} {'平均V(1)_T':>12s} {'V(1)理論線':>12s}")
for m, qv, tv, tvt in rows:
    print(f"{m:>7d} {qv:>12.4f} {tv:>12.3f} {tvt:>12.3f}")

# 2次変分は m によらず T、1次変分は理論線に一致し m とともに増える。
for m, qv, tv, tvt in rows:
    assert abs(qv - T) < 0.02
    assert abs(tv - tvt) < 0.03 * tvt
assert rows[-1][2] > rows[0][2] * 10  # 1次変分は明確に増大する
print("\n2次変分は一定（→T）、1次変分は sqrt(m) で発散")

# %%
ms_arr = np.array([r[0] for r in rows], dtype=float)
qv_arr = np.array([r[1] for r in rows])
tv_arr = np.array([r[2] for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].axhline(T, color="r", ls="--", lw=1.2, label=r"理論 $T$")
axes[0].plot(ms_arr, qv_arr, "o-", color="steelblue", label="2次変分")
axes[0].set_xscale("log")
axes[0].set_xlabel("ステップ数 m")
axes[0].set_ylabel(r"平均 $[W]_T$")
axes[0].set_title("2次変分は m によらず T")
axes[0].set_ylim(0.0, 2.0 * T)
axes[0].legend()

axes[1].plot(ms_arr, np.sqrt(2.0 / np.pi) * np.sqrt(ms_arr) * np.sqrt(T),
             "r--", lw=1.2, label=r"理論 $\sqrt{2/\pi}\sqrt{mT}$")
axes[1].plot(ms_arr, tv_arr, "o-", color="darkorange", label="1次変分")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("ステップ数 m")
axes[1].set_ylabel(r"平均 $V^{(1)}_T$")
axes[1].set_title("1次変分は発散（両対数）")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：2次変分は $m$ を上げても $T=1$ に張り付いたまま動かない。細分すると
# 各項 $(\Delta W)^2 \approx \Delta t$ が確定的に積み上がり、揺らぎ（分散
# $2T^2/m$）だけが消えるためである。対して1次変分は各項 $\lvert \Delta W \rvert
# \sim \sqrt{\Delta t}$ が $\Delta t$ より遅く縮むので、$m$ 個の和が
# $\sqrt{m}$ で増え続ける。この「2乗和は残り、絶対値和は発散する」非対称性が、
# ブラウン運動を通常の微積分から切り離し、伊藤解析を必要にする核心である。

# %% [markdown]
# ## 演習2：独立増分・定常性の確認
#
# ステップ数の大きいパスを1本作り、増分 $\Delta W_i$ の分布・定常性・自己相関を
# 調べる。

# %%
T = 1.0
m = 40000
dt = T / m
_, W = bm_paths(n_paths=1, n_steps=m, T=T, seed=99)
incr = np.diff(W[0])  # 増分列 ΔW_i（長さ m）

print(f"増分の本数 = {incr.size},  刻み dt = {dt:.3e}")
print(f"増分の平均 = {incr.mean():.3e} (理論 0)")
print(f"増分の分散 = {incr.var(ddof=1):.3e} (理論 dt = {dt:.3e})")

# %% [markdown]
# ### (a) 増分のヒストグラムと N(0, dt) の密度

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sd = np.sqrt(dt)
axes[0].hist(incr, bins=60, density=True, alpha=0.6, color="steelblue")
xs = np.linspace(-4 * sd, 4 * sd, 200)
axes[0].plot(xs, stats.norm.pdf(xs, 0.0, sd), "r-", lw=1.5,
             label=r"$\mathcal{N}(0,\,dt)$")
axes[0].set_title("増分 ΔW の分布")
axes[0].set_xlabel("ΔW")
axes[0].legend()

# ### (b) 前半・後半の増分分布（定常性）
half = m // 2
first, second = incr[:half], incr[half:]
axes[1].hist(first, bins=50, density=True, alpha=0.5, label="前半", color="steelblue")
axes[1].hist(second, bins=50, density=True, alpha=0.5, label="後半", color="darkorange")
axes[1].set_title("前半 vs 後半の増分分布（定常性）")
axes[1].set_xlabel("ΔW")
axes[1].legend()

# ### (c) 増分列の標本自己相関（独立性）
x = incr - incr.mean()
denom = np.sum(x ** 2)
max_lag = 20
acf = np.array([1.0] + [np.sum(x[:-k] * x[k:]) / denom for k in range(1, max_lag + 1)])
lags = np.arange(max_lag + 1)
conf = 1.96 / np.sqrt(incr.size)  # 95% 信頼帯（白色雑音の目安）
axes[2].bar(lags, acf, width=0.3, color="steelblue")
axes[2].axhline(conf, color="r", ls="--", lw=0.8)
axes[2].axhline(-conf, color="r", ls="--", lw=0.8)
axes[2].set_title("増分の自己相関（独立性）")
axes[2].set_xlabel("ラグ")
axes[2].set_ylabel("ACF")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 定常性・独立性の数値検定

# %%
# 定常性：前半・後半の分布が等しいか（2標本コルモゴロフ–スミルノフ検定）。
ks2 = stats.ks_2samp(first, second)
print(f"前半 vs 後半の KS 検定: 統計量 = {ks2.statistic:.4f}, p 値 = {ks2.pvalue:.3f}")
assert ks2.pvalue > 0.01  # 分布は同じ（棄却されない）＝定常

# 独立性：ラグ 1 以上の自己相関が信頼帯にほぼ収まる。
n_outside = int(np.sum(np.abs(acf[1:]) > conf))
print(f"ラグ1〜{max_lag} で信頼帯を超えた本数 = {n_outside} / {max_lag}")
assert n_outside <= 2  # 20 本中 2 本程度までは偶然の範囲

# 増分の分散が dt に一致（定常増分の分散が刻み幅で決まる）。
assert abs(incr.var(ddof=1) - dt) < 0.05 * dt
print("増分は N(0, dt) に一致・前後半で不変・無相関 → 定常かつ独立増分")

# %% [markdown]
# **解釈**：増分は $\mathcal{N}(0, \Delta t)$ に重なり、前半・後半で分布が変わらず
# （定常増分）、自己相関はラグ $\ge 1$ で信頼帯に収まる（独立増分）。これらは
# ブラウン運動の定義そのものであり、シミュレーションが定義を正しく満たすことの
# 数値的な裏付けになる。定常性は分散が絶対時刻ではなく刻み幅 $\Delta t$ だけで
# 決まる点に、独立性は無相関（正規性のもとでは独立と同値）に現れる。
