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
# # S5-3 演習 解答例

# %% [markdown]
# ## 準備
#
# 本編の `HullWhiteTree` を最小限だけ再掲する。第2段階の前向き帰納で $\alpha_i$ と
# Arrow-Debreu 価格 $Q_{i,j}$ を作り、後ろ向き帰納でゼロクーポン債を評価する。

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bondlab
from bondlab.curve import bootstrap_par
from bondlab.models import HullWhite

print("bondlab version:", bondlab.__version__)


class HullWhiteTree:
    """Hull-White 三項ツリー（本編と同一）。"""

    def __init__(self, a, sigma, curve, T, n_steps):
        self.a, self.sigma, self.curve = a, sigma, curve
        self.T, self.N = T, n_steps
        self.dt = T / n_steps
        self.dx = sigma * np.sqrt(3.0 * self.dt)
        self.jmax = int(np.ceil(0.184 / (a * self.dt)))
        self.build()

    def _branch(self, j):
        m = j * (1.0 - self.a * self.dt)
        if j >= self.jmax:
            k = j - 1
        elif j <= -self.jmax:
            k = j + 1
        else:
            k = int(np.round(m))
        eta = m - k
        pu = 1.0 / 6.0 + (eta ** 2 + eta) / 2.0
        pm = 2.0 / 3.0 - eta ** 2
        pd = 1.0 / 6.0 + (eta ** 2 - eta) / 2.0
        return k, pu, pm, pd

    def build(self):
        Q = {0: 1.0}
        self.alpha = np.empty(self.N)
        self.Q_levels = [dict(Q)]
        for i in range(self.N):
            s = sum(q * np.exp(-j * self.dx * self.dt) for j, q in Q.items())
            pm_disc = self.curve.discount((i + 1) * self.dt)
            self.alpha[i] = (np.log(s) - np.log(pm_disc)) / self.dt
            Q_next = {}
            for j, q in Q.items():
                k, pu, pm, pd = self._branch(j)
                disc = np.exp(-(self.alpha[i] + j * self.dx) * self.dt)
                for jj, p in ((k + 1, pu), (k, pm), (k - 1, pd)):
                    Q_next[jj] = Q_next.get(jj, 0.0) + q * p * disc
            Q = Q_next
            self.Q_levels.append(dict(Q))

    def short_rate(self, i, j):
        # α_i は割引段 i=0..N-1 で定義。最終段は作図用に線形外挿する。
        a_i = self.alpha[i] if i < self.N else 2 * self.alpha[-1] - self.alpha[-2]
        return a_i + j * self.dx

    def reproduction_error(self):
        errs = []
        for k in range(1, self.N + 1):
            t = k * self.dt
            p_tree = sum(self.Q_levels[k].values())
            errs.append((-np.log(p_tree) / t - self.curve.zero_rate(t)) * 1e4)
        return np.array(errs)

    def zcb(self, mat_step, from_step=0):
        V = {j: 1.0 for j in self.Q_levels[mat_step].keys()}
        for i in range(mat_step - 1, from_step - 1, -1):
            V_next = {}
            for j in self.Q_levels[i].keys():
                k, pu, pm, pd = self._branch(j)
                disc = np.exp(-self.short_rate(i, j) * self.dt)
                V_next[j] = disc * (pu * V.get(k + 1, 0.0)
                                    + pm * V.get(k, 0.0)
                                    + pd * V.get(k - 1, 0.0))
            V = V_next
        return V

    def reachable(self):
        return [(min(d), max(d)) for d in self.Q_levels]


# 本編と同じ実カーブ（合成パー利回りを年次補間 → ブートストラップ）。
par_df = pd.read_csv("data/samples/synthetic_ust_par_curve.csv")
annual = np.arange(1, 31)
par_annual = np.interp(annual, par_df["tenor"].values, par_df["par_yield"].values)
curve = bootstrap_par(annual, par_annual, frequency=1, interp="linear_zero")

a, sigma = 0.10, 0.01
hw = HullWhite(a, sigma, curve)

# %% [markdown]
# ## 演習1：ツリーステップ数と価格収束
#
# 内部時点 $t_0=2.5$（整数年ノードを避ける）・満期 $T=10$ の中央節点の条件付き ZCB
# 価格を段数 $N$ に対して求め、`bondlab` 解析値 `hw.zcb(t0, T, r0)`（$r_0$ はその節点の
# 短期金利）からの誤差を両対数でプロットする。あわせてグリッド満期 $T=N\Delta t$ の
# ゼロクーポン債では誤差がほぼ出ないことを確認する。整数年ノードでは補間フォワード
# $f^M$ が折れ、解析値側の $f^M$ 評価に段差が入って収束が乱れるため、$t_0$ はノード間に取る。

# %%
t0, T_mat = 2.5, 10.0
Ns = np.array([20, 40, 80, 160])

err_interior = []   # 内部条件付き価格の誤差
err_grid = []       # グリッド満期 P(0,T) の誤差（無裁定構成でほぼ0）
for N in Ns:
    tr = HullWhiteTree(a, sigma, curve, T=T_mat, n_steps=N)
    i0 = int(round(t0 / tr.dt))
    v_int = tr.zcb(mat_step=N, from_step=i0)[0]
    r0 = tr.short_rate(i0, 0)
    err_interior.append(abs(v_int - hw.zcb(t0, T_mat, r0)))
    v_grid = tr.zcb(mat_step=N, from_step=0)[0]     # 根の値 = P(0,T)
    err_grid.append(abs(v_grid - hw.zcb(0.0, T_mat)))

err_interior = np.array(err_interior)
err_grid = np.array(err_grid)
dts = T_mat / Ns

print(f"{'N':>5}{'Δt':>9}{'内部誤差':>12}{'グリッド満期誤差':>18}")
for N, dt, ei, eg in zip(Ns, dts, err_interior, err_grid):
    print(f"{N:>5}{dt:>9.4f}{ei:>12.2e}{eg:>18.2e}")

# 内部誤差は O(Δt)、グリッド満期誤差は機械精度近くで一定。
slope = np.polyfit(np.log(dts), np.log(err_interior), 1)[0]
print(f"\n内部誤差の両対数の傾き = {slope:.2f}（1 に近ければ O(Δt)）")
print(f"グリッド満期誤差の最大 = {np.max(err_grid):.2e}（無裁定構成でほぼ0）")
assert 0.7 < slope < 1.3
assert np.max(err_grid) < 1e-6

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(dts, err_interior, "o-", color="steelblue", label=r"内部条件付き $P(t_0,T;r_0)$")
ax.loglog(dts, err_interior[0] * (dts / dts[0]), "r--", lw=1.0,
          label=r"傾き1の基準線 $O(\Delta t)$")
ax.loglog(dts, np.maximum(err_grid, 1e-16), "s--", color="green",
          label=r"グリッド満期 $P(0,T)$")
ax.set_xlabel(r"時間刻み $\Delta t$")
ax.set_ylabel("解析値からの誤差")
ax.set_title(r"ステップ数と価格収束（$t_0=3$, $T=10$）")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：内部の条件付き価格 $P(t_0,T;r_0)$ は、$t_0$ から $T$ まで多数の刻みで
# 割引を積み上げるため離散化誤差を持ち、$\Delta t$ に比例して縮む（傾き $\approx 1$）。
# 一方、グリッド満期 $T=N\Delta t$ のゼロクーポン債は、前向き帰納で $\alpha_i$ を
# $P^M(0,k\Delta t)$ に合わせて構成した結果として機械精度で再現される。これは無裁定
# モデルが「初期カーブを厳密に通す」性質そのものであり、収束を論じる対象ですらない。
# 離散化誤差が意味を持つのは、初期カーブに直接固定されていない内部の状態依存量の方だ、
# という切り分けが要点である。

# %% [markdown]
# ## 演習2：$a$ を変えてツリー形状と初期カーブ再現精度を評価
#
# $\sigma=0.01$ 固定、満期10年・段数80のツリーを複数の $a$ で組み、打ち切り幅
# $j_{\max}$・格子幅・初期カーブ再現誤差の最大値(bp) を表にまとめる。

# %%
a_list = [0.02, 0.05, 0.10, 0.30]
N = 80
T = 10.0
dt = T / N

table = []
trees = {}
for a_i in a_list:
    tr = HullWhiteTree(a_i, sigma, curve, T=T, n_steps=N)
    trees[a_i] = tr
    max_err = np.max(np.abs(tr.reproduction_error()))
    table.append((a_i, tr.jmax, 2 * tr.jmax + 1, max_err))

print(f"{'a':>6}{'j_max':>7}{'格子幅':>8}{'再現誤差最大(bp)':>18}")
for a_i, jm, width, err in table:
    print(f"{a_i:>6.2f}{jm:>7d}{width:>8d}{err:>18.4f}")

# どの a でも初期カーブは <1bp で再現される。
for a_i, jm, width, err in table:
    assert err < 1.0
# a が小さいほど j_max（格子幅）は大きい。
jmaxes = [row[1] for row in table]
assert jmaxes[0] > jmaxes[-1]
print("\n→ どの a でも再現誤差 <1bp。a が小さいほど格子は広い。")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

# (左) 到達領域の上下端を a ごとに重ねる。
colors = plt.cm.viridis(np.linspace(0, 0.85, len(a_list)))
for a_i, col in zip(a_list, colors):
    tr = trees[a_i]
    times = np.arange(tr.N + 1) * tr.dt
    reach = tr.reachable()
    hi = np.array([tr.short_rate(i, reach[i][1]) for i in range(tr.N + 1)])
    lo = np.array([tr.short_rate(i, reach[i][0]) for i in range(tr.N + 1)])
    axes[0].plot(times, hi, "-", color=col, lw=1.3, label=f"a={a_i}")
    axes[0].plot(times, lo, "-", color=col, lw=1.3)
axes[0].set_xlabel("時間 (年)")
axes[0].set_ylabel(r"短期金利 $r$（到達領域の上下端）")
axes[0].set_title("a が大きいほど格子は細い")
axes[0].legend(fontsize=8)

# (右) j_max（格子幅の半分）と再現誤差を a に対して。
ax_r = axes[1]
ax_r.bar([str(a_i) for a_i in a_list], [row[1] for row in table], color=colors)
ax_r.set_xlabel("平均回帰速度 a")
ax_r.set_ylabel(r"打ち切り幅 $j_{\max}$")
ax_r.set_title("格子幅と再現誤差")
ax_t = ax_r.twinx()
ax_t.plot([str(a_i) for a_i in a_list], [row[3] for row in table],
          "r.-", ms=10, label="再現誤差最大(bp)")
ax_t.set_ylabel("再現誤差最大 (bp)", color="r")
ax_t.set_ylim(0, 1.0)
ax_t.tick_params(axis="y", labelcolor="r")
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：$a$ が小さいほど平均回帰が弱く、短期金利が中心から離れた状態にも到達し
# やすい。打ち切り幅 $j_{\max}=\lceil 0.184/(a\Delta t)\rceil$ が大きくなり、格子は広く
# （背が高く）なる。逆に $a$ が大きいと格子は細く、上下端での分枝の切替が早い段から
# 頻繁に起こる。いずれの $a$ でも初期カーブ再現は <1bp に保たれる。これは再現精度が
# 第1段階の格子形状ではなく、第2段階のシフト $\alpha_i$ の決め方（前向き帰納で
# $P^M(0,k\Delta t)$ に一致させる）だけで担保されるためである。$a$ の選択は初期
# フィットの良し悪しではなく、金利の分布の広がり方、ひいては金利オプションの
# ボラティリティ構造に効いてくる。
