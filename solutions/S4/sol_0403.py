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
# # S4-3 演習 解答例

# %% [markdown]
# ## 準備
#
# 本編の自作ソルバーを最小限だけ再掲する。ブラウン増分を外部から受け取る
# `solve_sde` と、強収束の誤差指標を手元に置く。

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bondlab

np.random.seed(0)
print("bondlab version:", bondlab.__version__)


def make_increments(n_paths, n_steps, dt, seed):
    """ブラウン増分 dW を生成する（本編と同一）。"""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    return np.sqrt(dt) * z


def solve_sde(drift, diffusion, x0, T, dW, scheme="euler", diffusion_x=None):
    """与えた増分 dW で 1次元 SDE を数値積分する（本編と同一）。"""
    if scheme == "milstein" and diffusion_x is None:
        raise ValueError("milstein には diffusion_x（∂b/∂x）が必要です")
    n_paths, n_steps = dW.shape
    dt = T / n_steps
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


def strong_error(x_num, x_exact):
    """終端でのパス単位平均絶対誤差 E[|X_N - X_T|]（本編と同一）。"""
    return float(np.mean(np.abs(x_num - x_exact)))


# %% [markdown]
# ## 演習1：OU 過程（加法ノイズ）での強収束次数
#
# Ornstein-Uhlenbeck 過程 $dX = \kappa(\theta - X)\,dt + \sigma\,dW$ を考える。拡散
# $b(t,x)=\sigma$ は状態に依存しないので $\partial_x b = 0$、したがって Milstein の
# 補正項が消え、Euler-Maruyama と Milstein は**同一の更新式**になる。
#
# OU は線形 SDE なので厳密解を持つ。区間 $[t_i, t_{i+1}]$ の厳密な遷移は
#
# $$ X_{i+1} = \theta + (X_i - \theta)e^{-\kappa \Delta t}
#    + \sigma\sqrt{\tfrac{1 - e^{-2\kappa\Delta t}}{2\kappa}}\,Z_i $$
#
# だが、強収束の比較では**同一のブラウン運動**を使う必要があるため、ここでは各刻みで
# 生成した増分 $\Delta W_i$ をそのまま使い、非常に細かい参照解（reference solution）を
# 厳密解の代理として用いる。参照解と粗い解が同じブラウン運動を共有するよう、細かい
# 増分を束ねて粗い増分を作る。

# %%
kappa, theta, sigma, x0 = 1.0, 0.03, 0.02, 0.05
T = 1.0

# 参照解のための最細グリッド。粗いグリッドは細かい増分を束ねて作る。
N_ref = 4096
M = 20000
seed = 4242
dt_ref = T / N_ref
dW_ref = make_increments(M, N_ref, dt_ref, seed=seed)

drift = lambda t, x: kappa * (theta - x)
diffusion = lambda t, x: sigma * np.ones_like(x)
diffusion_x = lambda t, x: np.zeros_like(x)

# 最細グリッドでの参照解（厳密解の代理）
_, X_ref = solve_sde(drift, diffusion, x0, T, dW_ref, scheme="euler")
x_ref_T = X_ref[:, -1]

# 粗いグリッド：N_ref を割り切る N で、連続する増分を束ねる（同一ブラウン運動を共有）
step_list = [8, 16, 32, 64, 128, 256]
rows = []
for N in step_list:
    factor = N_ref // N
    # 束ね：(M, N, factor) に整形して factor 方向を合計 → 粗い増分（分散が dt にそろう）
    dW_coarse = dW_ref.reshape(M, N, factor).sum(axis=2)
    _, Xe = solve_sde(drift, diffusion, x0, T, dW_coarse, scheme="euler")
    _, Xm = solve_sde(drift, diffusion, x0, T, dW_coarse, scheme="milstein",
                      diffusion_x=diffusion_x)
    rows.append((N, T / N, strong_error(Xe[:, -1], x_ref_T),
                 strong_error(Xm[:, -1], x_ref_T),
                 float(np.max(np.abs(Xe[:, -1] - Xm[:, -1])))))

dt_arr = np.array([r[1] for r in rows])
err_e = np.array([r[2] for r in rows])
err_m = np.array([r[3] for r in rows])
em_diff = np.array([r[4] for r in rows])

slope_e = np.polyfit(np.log(dt_arr), np.log(err_e), 1)[0]
slope_m = np.polyfit(np.log(dt_arr), np.log(err_m), 1)[0]

tbl = pd.DataFrame({
    "N": [r[0] for r in rows],
    "dt": dt_arr.round(5),
    "EM強誤差": err_e,
    "Milstein強誤差": err_m,
    "EM-Milstein最大差": em_diff,
})
print(tbl.to_string(index=False))
print(f"\n強収束次数（log-log 回帰の傾き）")
print(f"  Euler-Maruyama : {slope_e:.3f}")
print(f"  Milstein       : {slope_m:.3f}")

# 加法ノイズなので EM と Milstein は完全一致（更新式が同じ）。
assert em_diff.max() < 1e-12
# この設定では強収束次数が 0.5 を大きく上回り 1.0 付近に達する。
assert slope_e > 0.85

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(dt_arr, err_e, "o-", color="#1f77b4", label=f"Euler-Maruyama（傾き {slope_e:.2f}）")
ax.loglog(dt_arr, err_m, "s--", color="#d62728", label=f"Milstein（傾き {slope_m:.2f}）")
ax.loglog(dt_arr, err_e[0] * (dt_arr / dt_arr[0]) ** 1.0, ":",
          color="gray", alpha=0.6, label="傾き 1.0 参照")
ax.loglog(dt_arr, err_e[0] * (dt_arr / dt_arr[0]) ** 0.5, "-.",
          color="gray", alpha=0.4, label="傾き 0.5 参照")
ax.set_xlabel("時間刻み Δt")
ax.set_ylabel("強収束誤差  E[|X_N − X_ref|]")
ax.set_title("OU 過程（加法ノイズ）の強収束次数")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：加法ノイズ（$b$ が定数）では $\partial_x b = 0$ なので Milstein の補正項
# $\tfrac12 b\,\partial_x b(\Delta W^2 - \Delta t)$ が消え、Euler-Maruyama と Milstein は
# 更新式そのものが一致する（両者の差は機械精度でゼロ）。さらに強収束次数は 0.5 では
# なく 1.0 付近に達する。GBM のような乗法ノイズで Euler-Maruyama が 0.5 に留まったのは、
# 拡散係数 $b(t,X)=\sigma X$ が区間内で揺らぐのにそれを無視したためだった。加法ノイズでは
# $b$ が揺らがないので、その誤差源が最初から存在せず、精度が落ちない。「Euler-Maruyama の
# 強収束は 0.5」は乗法ノイズでの上限であって、拡散が状態依存しない問題には当てはまらない、
# というのが要点である。

# %% [markdown]
# ## 演習2：CIR の負値到達率と破綻率を刻みとパラメータで評価
#
# 時間刻み（ステップ数 $N$）とボラティリティ $\sigma$ を変え、(1) 負値到達率、
# (2) 素朴な $\sqrt{r}$ の破綻（NaN）率、(3) full truncation の破綻率を比べる。
# 「負値到達率」は「満期までに1回でも $r<0$ になったパスの割合」と定義する。素朴版と
# full truncation は初めて負に落ちるまで同一の軌道なので、負値到達率は両者で一致する。
# 差が出るのはその後で、素朴版は破綻（NaN）、full truncation は完走する。

# %%
def simulate_cir(kappa, theta, sigma, r0, T, dW, scheme="full_trunc"):
    """CIR を Euler-Maruyama で離散化（本編と同一）。

    (パス, 負値到達フラグ, NaN 破綻フラグ) を返す。
    """
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    went_negative = np.zeros(n_paths, dtype=bool)
    for i in range(n_steps):
        ri = r[:, i]
        with np.errstate(invalid="ignore"):
            root = np.sqrt(np.maximum(ri, 0.0)) if scheme == "full_trunc" else np.sqrt(ri)
        r_next = ri + kappa * (theta - ri) * dt + sigma * root * dW[:, i]
        went_negative |= r_next < 0.0
        r[:, i + 1] = r_next
    has_nan = np.isnan(r).any(axis=1)
    return r, went_negative, has_nan


kappa, theta, r0, T = 0.5, 0.02, 0.02, 1.0
M = 8000

N_list = [50, 100, 250, 500]
sigma_list = [0.15, 0.25, 0.35]

records = []
for sig in sigma_list:
    for N in N_list:
        dt = T / N
        dW = make_increments(M, N, dt, seed=1000 + N)  # 両スキームで同一増分
        _, neg_p, nan_p = simulate_cir(kappa, theta, sig, r0, T, dW, scheme="plain")
        _, neg_ft, nan_ft = simulate_cir(kappa, theta, sig, r0, T, dW, scheme="full_trunc")
        feller = "満たす" if 2 * kappa * theta >= sig ** 2 else "破る"
        records.append(dict(
            σ=sig, N=N, dt=round(dt, 4), フェラー=feller,
            負値到達率=round(neg_ft.mean() * 100, 2),
            素朴破綻率=round(nan_p.mean() * 100, 2),
            FT破綻率=round(nan_ft.mean() * 100, 2),
        ))
        # 負値到達率は両スキームで一致、full truncation は決して破綻しない。
        assert neg_p.mean() == neg_ft.mean()
        assert nan_ft.sum() == 0

res = pd.DataFrame(records)
print(res.to_string(index=False))

# フェラー条件を破る σ ほど負値到達率が高い（同一 N での比較）。
r_hi = res[(res["N"] == 250) & (res["σ"] == 0.35)]["負値到達率"].iloc[0]
r_lo = res[(res["N"] == 250) & (res["σ"] == 0.15)]["負値到達率"].iloc[0]
print(f"\nN=250: σ=0.35 の負値到達率 {r_hi}% > σ=0.15 の負値到達率 {r_lo}%")
assert r_hi > r_lo

# %% [markdown]
# 刻み依存を1本のパラメータ（σ=0.35）で可視化する。負値到達率（＝素朴破綻率）と、
# full truncation の破綻率（常に 0）を並べる。

# %%
sig = 0.35
Ns = np.array([25, 50, 100, 200, 400, 800])
neg_rate, plain_nan, ft_nan = [], [], []
for N in Ns:
    dt = T / N
    dW = make_increments(M, int(N), dt, seed=333 + int(N))
    _, ng, npn = simulate_cir(kappa, theta, sig, r0, T, dW, scheme="plain")
    _, _, ftn = simulate_cir(kappa, theta, sig, r0, T, dW, scheme="full_trunc")
    neg_rate.append(ng.mean() * 100)
    plain_nan.append(npn.mean() * 100)
    ft_nan.append(ftn.mean() * 100)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(Ns, neg_rate, "o-", color="#ff7f0e", label="負値到達率（両スキーム共通）")
ax.plot(Ns, plain_nan, "^--", color="#d62728", label="素朴 √r の破綻(NaN)率")
ax.plot(Ns, ft_nan, "s-", color="#1f77b4", label="full truncation の破綻率")
ax.set_xscale("log", base=2)
ax.set_xlabel("ステップ数 N（刻み Δt=T/N は右ほど細かい）")
ax.set_ylabel("割合 (%)")
ax.set_title(f"CIR の負値到達率と破綻率（σ={sig}, フェラー破り）")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：
#
# - **σ の効果**：ボラティリティが上がりフェラー条件 $2\kappa\theta \ge \sigma^2$ を
#   破るほど、拡散項 $\sigma\sqrt{r}\,\Delta W$ が原点近傍で大きく振れ、負値到達率が上がる。
# - **刻みの効果**：$\Delta t$ を細かくする（$N$ を増やす）と、1ステップの増分
#   $\sqrt{\Delta t}$ が小さくなり負値到達率は下がるが、フェラー条件を破る限りゼロには
#   ならない。刻みを細かくするだけでは非負性を保証できないのが要点である。
# - **full truncation の効き所**：負値到達率そのものは素朴版と同じで、full truncation が
#   下げるのは負値ではなく**破綻（NaN）率**である。素朴な $\sqrt{r}$ は負値に落ちた瞬間に
#   NaN で破綻するのに対し、full truncation は破綻をゼロに抑え、負に落ちてもドリフトの
#   平均回帰で速やかに正へ戻す。フェラー条件を破る高ボラ・粗い刻みの領域ほど破綻率が
#   高く、full truncation の価値が大きい。厳密な非負性が要るなら、対数変換や厳密
#   サンプリング（非心カイ二乗分布）へ進む必要がある。
