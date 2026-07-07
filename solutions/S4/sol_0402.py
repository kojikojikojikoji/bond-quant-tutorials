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
# # S4-2 演習 解答例

# %% [markdown]
# ## 準備
#
# 本編の自作関数のうち、演習で使うものを最小限だけ再掲します。確率積分（左端点）
# と OU の厳密サンプリングを手元に置きます。

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bondlab
from bondlab.sim import brownian_paths, simulate_sde

print("bondlab version:", bondlab.__version__)


def ito_integral(W):
    """左端点評価で各パスの ∫_0^T W dW を近似する（本編と同一）。"""
    dW = np.diff(W, axis=1)
    return np.sum(W[:, :-1] * dW, axis=1)


def ou_exact_paths(x0, kappa, theta, sigma, times, n_paths, seed):
    """OU 過程を厳密な遷移分布でサンプリングする（本編と同一）。"""
    rng = np.random.default_rng(seed)
    n_steps = times.size - 1
    X = np.empty((n_paths, times.size))
    X[:, 0] = x0
    for i in range(n_steps):
        dt = times[i + 1] - times[i]
        e = np.exp(-kappa * dt)
        var = sigma ** 2 / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * dt))
        mean = X[:, i] * e + theta * (1.0 - e)
        X[:, i + 1] = mean + np.sqrt(var) * rng.standard_normal(n_paths)
    return X

# %% [markdown]
# ## 演習 1：$f(W)=W^2$ の伊藤展開の数値検証（別ルート）
#
# 伊藤の公式より $W_T^2 = 2\int_0^T W\,dW + T$ です。分割数 $n$ を変えて、
# $2\int W\,dW + T$ の平均と $E[W_T^2]=T$ の差（バイアス）、および伊藤積分の
# 平均を表にします。

# %%
T = 1.0
n_paths = 40000
rows = []
for n in [10, 50, 200, 1000]:
    times, W = brownian_paths(n_paths, n, T, seed=2026)
    it = ito_integral(W)
    reconstructed = 2.0 * it + T          # 伊藤展開から組み立てた W_T²
    wt2 = W[:, -1] ** 2                    # 直接の W_T²
    rows.append({
        "n_steps": n,
        "E[2∫WdW+T]": reconstructed.mean(),
        "E[W_T²]=T との差": reconstructed.mean() - T,
        "E[∫W dW]": it.mean(),
        "パス最大誤差": float(np.max(np.abs(reconstructed - wt2))),
    })

df1 = pd.DataFrame(rows)
print(df1.to_string(index=False,
                    formatters={c: (lambda v: f"{v:.6f}") for c in df1.columns[1:]}))

# %% [markdown]
# **考察**：$W_T^2 = 2\int W\,dW + T$ は恒等式なので、パス最大誤差は分割によらず
# 機械精度（丸め）です。一方、伊藤積分の平均 $E[\int W\,dW]$ はサンプル誤差で
# 0 のまわりに揺れ、パス数を増やせば 0 へ収束します。再構成した $E[W_T^2]$ は
# 理論値 $T$ に近く、分割 $n$ を細かくしても（恒等式ゆえ）バイアスは増えません。
# 「伊藤の公式による書き換え」は近似ではなく厳密な関係だと確認できます。

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0.0, color="gray", lw=0.8)
ax.plot(df1["n_steps"], df1["E[W_T²]=T との差"], "o-", label="E[2∫WdW+T] - T")
ax.plot(df1["n_steps"], df1["E[∫W dW]"], "s--", label="E[∫W dW]（理論 0）")
ax.set_xscale("log")
ax.set_xlabel("n_steps（分割数）")
ax.set_ylabel("平均の残差")
ax.set_title("演習1：伊藤展開の残差（サンプル誤差のみ）")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 演習 2：OU の平均回帰を初期値を変えて可視化
#
# $\kappa=1.0,\ \theta=0,\ \sigma=0.5$ を固定し、初期値を変えて平均パスと
# 分散の立ち上がりを 1 枚にまとめます。

# %%
kappa, theta, sigma = 1.0, 0.0, 0.5
T_ou, n_steps, n_paths_ou = 6.0, 1200, 20000
times = np.linspace(0.0, T_ou, n_steps + 1)
stat_var = sigma ** 2 / (2.0 * kappa)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for x0 in [-3.0, -1.0, 0.0, 2.0, 4.0]:
    X = ou_exact_paths(x0, kappa, theta, sigma, times, n_paths_ou, seed=int(100 + x0 * 10))
    ax[0].plot(times, X.mean(axis=0), label=f"X0={x0:+.0f}")
    if x0 in (4.0, -3.0):
        ax[1].plot(times, X.var(axis=0), label=f"分散 (X0={x0:+.0f})")

ax[0].axhline(theta, color="gray", lw=0.8)
ax[0].set_title("平均パス：どの初期値からも θ へ回帰")
ax[0].set_xlabel("t")
ax[0].set_ylabel("E[X_t]")
ax[0].legend(fontsize=8)

var_theory_t = stat_var * (1.0 - np.exp(-2.0 * kappa * times))
ax[1].plot(times, var_theory_t, "k--", label="理論 σ²/(2κ)(1-e^{-2κt})")
ax[1].axhline(stat_var, color="gray", lw=0.8, label="定常分散 σ²/(2κ)")
ax[1].set_title("分散：初期値によらず定常分散へ")
ax[1].set_xlabel("t")
ax[1].set_ylabel("Var[X_t]")
ax[1].legend(fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# **考察**：平均は $E[X_t]=X_0 e^{-\kappa t}$ に従って、初期値の符号・大きさに
# よらず水準 $\theta=0$ へ指数的に回帰します。回帰の速さは $\kappa$ が決め、
# 特性時間はおよそ $1/\kappa$（値が半分に戻るまで約 $\ln 2/\kappa$）です。
# 分散は初期値によらず $\sigma^2/(2\kappa)$ へ立ち上がります。半減期を確認します。

# %%
half_life = np.log(2.0) / kappa
print(f"平均回帰の半減期 ln2/κ = {half_life:.4f} 年")
print(f"定常分散 σ²/(2κ) = {stat_var:.6f}")

# 数値でも半減期を確認：X0=4 の平均が 2 を切る最初の時刻。
X = ou_exact_paths(4.0, kappa, theta, sigma, times, n_paths_ou, seed=999)
m = X.mean(axis=0)
t_half = times[np.argmax(m < 2.0)]
print(f"数値で観測した半減時刻（X0=4→2）≈ {t_half:.4f} 年")
assert abs(t_half - half_life) < 0.05
print("半減期の理論と数値が整合（assert 通過）")
