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
# # S4-4 演習 解答例

# %% [markdown]
# ## 準備
#
# 本編の自作関数を最小限だけ再掲する。GBM 満期株価・割引ペイオフ・Black-Scholes の
# 解析解（価格とデルタ）、および各分散削減の推定量を手元に置く。

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, qmc

import bondlab
from bondlab import sim

print("bondlab version:", bondlab.__version__)

S0, K, r, SIGMA, T = 100.0, 100.0, 0.03, 0.20, 1.0
SEED = 20260707


def bs_call_price(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def gbm_terminal(S0, r, sigma, T, Z):
    return S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)


def discounted_call_payoff(ST, K, r, T):
    return np.exp(-r * T) * np.maximum(ST - K, 0.0)


def mc_call_crude(n, seed):
    rng = np.random.default_rng(seed)
    ST = gbm_terminal(S0, r, SIGMA, T, rng.standard_normal(n))
    return sim.mc_stats(discounted_call_payoff(ST, K, r, T))


def mc_call_antithetic(n_pairs, seed):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_pairs)
    pay_p = discounted_call_payoff(gbm_terminal(S0, r, SIGMA, T, Z), K, r, T)
    pay_m = discounted_call_payoff(gbm_terminal(S0, r, SIGMA, T, -Z), K, r, T)
    return sim.mc_stats(0.5 * (pay_p + pay_m))


def mc_call_control(n, seed):
    rng = np.random.default_rng(seed)
    ST = gbm_terminal(S0, r, SIGMA, T, rng.standard_normal(n))
    target = discounted_call_payoff(ST, K, r, T)
    control = np.exp(-r * T) * ST
    return sim.control_variate(target, control, S0)  # E[control] = S0


def sobol_normals(n_log2, seed):
    u = qmc.Sobol(d=1, scramble=True, seed=np.random.default_rng(seed)).random_base2(m=n_log2).ravel()
    return norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))


def mc_call_sobol_mean(n_log2, seed):
    ST = gbm_terminal(S0, r, SIGMA, T, sobol_normals(n_log2, seed))
    return float(discounted_call_payoff(ST, K, r, T).mean())


def sobol_stderr(n_log2, n_batches, seed):
    seeds = np.random.SeedSequence(seed).spawn(n_batches)
    means = np.array([mc_call_sobol_mean(n_log2, s) for s in seeds])
    return float(means.mean()), float(means.std(ddof=1) / np.sqrt(n_batches))


BS_TRUE = bs_call_price(S0, K, r, SIGMA, T)
DELTA_TRUE = bs_call_delta(S0, K, r, SIGMA, T)
print(f"BS 真値 価格 = {BS_TRUE:.6f}, デルタ = {DELTA_TRUE:.6f}")

# %% [markdown]
# ## 演習1：分散削減率の予算依存
#
# パス予算 $N \in \{2^{12}, 2^{14}, 2^{16}\}$ を振り、対照変量・制御変量・Sobol の分散
# 削減率（対 素朴MC）を1表にまとめる。分散削減率は同じ予算での分散比
# $\operatorname{Var}_{\text{crude}} / \operatorname{Var}_{\text{method}}
# = (\operatorname{SE}_{\text{crude}}/\operatorname{SE}_{\text{method}})^2$ で測る。
# Sobol の標準誤差はスクランブルを変えた 32 バッチのばらつきから見積もる。

# %%
budgets_log2 = [12, 14, 16]
records = []
for m in budgets_log2:
    N = 2 ** m
    se_crude = mc_call_crude(N, seed=SEED + m)["stderr"]
    se_anti = mc_call_antithetic(N // 2, seed=SEED + m)["stderr"]
    se_ctrl = mc_call_control(N, seed=SEED + m)["stderr"]
    _, se_sobol = sobol_stderr(m, n_batches=32, seed=SEED + m)
    records.append(dict(
        予算N=N,
        対照変量=(se_crude / se_anti) ** 2,
        制御変量=(se_crude / se_ctrl) ** 2,
        Sobol=(se_crude / se_sobol) ** 2,
    ))

ex1 = pd.DataFrame(records)
print("分散削減率（素朴MC=1、大きいほど効率的）")
print(ex1.assign(
    対照変量=ex1["対照変量"].round(1),
    制御変量=ex1["制御変量"].round(1),
    Sobol=ex1["Sobol"].round(1),
).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for col, color, marker in [("対照変量", "#1f77b4", "o"),
                           ("制御変量", "#2ca02c", "s"),
                           ("Sobol", "#d62728", "^")]:
    ax.plot(ex1["予算N"], ex1[col], marker + "-", color=color, label=col)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("パス予算 N（対数軸）")
ax.set_ylabel("分散削減率（対 素朴MC）")
ax.set_title("分散削減率のパス予算依存")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# 対照変量・制御変量は予算にほぼ依存しない定数倍、Sobol は予算とともに優位が拡大する。
sobol_growth = ex1["Sobol"].iloc[-1] / ex1["Sobol"].iloc[0]
ctrl_growth = ex1["制御変量"].iloc[-1] / ex1["制御変量"].iloc[0]
print(f"\nSobol の削減率の伸び（2^16 / 2^12）  = {sobol_growth:.2f} 倍")
print(f"制御変量の削減率の伸び（同上）        = {ctrl_growth:.2f} 倍")
assert sobol_growth > ctrl_growth  # Sobol の優位は予算とともに拡大

# %% [markdown]
# **解釈**：対照変量と制御変量の分散削減率は、分散の比 $1/(1-\rho^2)$ 等が予算に
# 依存しない定数なので、$N$ を増やしてもほぼ一定倍にとどまる。一方 Sobol は誤差自体が
# $1/\sqrt{n}$ より速く（実質 $(\log n)^d/n$ に近く）縮むため、素朴MC との比＝分散削減率が
# $N$ とともに拡大する。予算が大きいほど準乱数が有利になるのはこのためである。

# %% [markdown]
# ## 演習2：pathwise vs バンプの Greeks
#
# デルタについて pathwise 推定量とバンプ法を比較する。バンプは同一乱数（CRN）と
# 非同一乱数の両方を実装し、バンプ幅 $h \in \{0.01, 0.1, 1, 5\}$ で推定値・バイアス
# （真値 $N(d_1)$ との差）・標準誤差を表にする。

# %%
def pathwise_delta(n, seed):
    rng = np.random.default_rng(seed)
    ST = gbm_terminal(S0, r, SIGMA, T, rng.standard_normal(n))
    est = np.exp(-r * T) * (ST > K).astype(float) * ST / S0
    return sim.mc_stats(est)


def bump_delta(n, h, seed, common_rn=True):
    rng = np.random.default_rng(seed)
    Z_up = rng.standard_normal(n)
    Z_dn = Z_up if common_rn else rng.standard_normal(n)
    pay_up = discounted_call_payoff(gbm_terminal(S0 + h, r, SIGMA, T, Z_up), K, r, T)
    pay_dn = discounted_call_payoff(gbm_terminal(S0 - h, r, SIGMA, T, Z_dn), K, r, T)
    return sim.mc_stats((pay_up - pay_dn) / (2.0 * h))


N_PATHS = 200_000
pw = pathwise_delta(N_PATHS, seed=SEED)

rows = [dict(手法="pathwise", h=np.nan,
             推定値=pw["mean"], バイアス=pw["mean"] - DELTA_TRUE, 標準誤差=pw["stderr"])]
for h in [0.01, 0.1, 1.0, 5.0]:
    for common, tag in [(True, "バンプ(同一乱数)"), (False, "バンプ(非同一乱数)")]:
        bd = bump_delta(N_PATHS, h=h, seed=SEED, common_rn=common)
        rows.append(dict(手法=tag, h=h, 推定値=bd["mean"],
                         バイアス=bd["mean"] - DELTA_TRUE, 標準誤差=bd["stderr"]))

ex2 = pd.DataFrame(rows)
print(f"デルタ真値 N(d1) = {DELTA_TRUE:.6f}  （パス数 {N_PATHS}）")
print(ex2.assign(
    推定値=ex2["推定値"].round(5),
    バイアス=ex2["バイアス"].round(5),
    標準誤差=ex2["標準誤差"].round(6),
).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for tag, color, marker in [("バンプ(同一乱数)", "#1f77b4", "o"),
                           ("バンプ(非同一乱数)", "#d62728", "s")]:
    sub = ex2[ex2["手法"] == tag]
    ax.loglog(sub["h"], sub["標準誤差"], marker + "-", color=color, label=tag)
ax.axhline(pw["stderr"], color="#2ca02c", ls="--", label="pathwise（h非依存）")
# 非同一乱数バンプの理論オーダー O(1/h) 参照線。
sub_ind = ex2[ex2["手法"] == "バンプ(非同一乱数)"].sort_values("h")
ax.loglog(sub_ind["h"], sub_ind["標準誤差"].iloc[-1] * (sub_ind["h"].iloc[-1] / sub_ind["h"]),
          ":", color="#d62728", alpha=0.5, label="傾き -1 参照 (1/h)")
ax.set_xlabel("バンプ幅 h（対数軸）")
ax.set_ylabel("デルタ推定量の標準誤差")
ax.set_title("Greeks 推定：pathwise vs バンプ（分散）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# 検証：pathwise は不偏（バイアス小）かつ低分散。
assert abs(pw["mean"] - DELTA_TRUE) < 3.0 * pw["stderr"]
# 非同一乱数バンプは h を小さくすると標準誤差が増大する（分散爆発）。
se_ind = ex2[ex2["手法"] == "バンプ(非同一乱数)"].sort_values("h")["標準誤差"].to_numpy()
assert se_ind[0] > se_ind[-1]  # h=0.01 の SE > h=5 の SE
# 同一乱数バンプは pathwise に匹敵する低分散。
se_crn_small = ex2[(ex2["手法"] == "バンプ(同一乱数)") & (ex2["h"] == 0.01)]["標準誤差"].iloc[0]
print(f"\npathwise SE            = {pw['stderr']:.6f}")
print(f"同一乱数バンプ(h=0.01) SE = {se_crn_small:.6f}")
print(f"非同一乱数バンプ(h=0.01) SE = {se_ind[0]:.4f}（桁違いに大きい）")

# %% [markdown]
# **解釈**：pathwise 推定量はバイアスがゼロ（微分と期待値の交換が厳密に成り立つ）で、
# 標準誤差もバンプ幅に依存しない。バンプ法は中心差分の打ち切りで $O(h^2)$ のバイアスを
# 持つが、より効くのは分散である。非同一乱数だと分子 $\text{pay}_{up}-\text{pay}_{dn}$ の
# 分散が $h$ に依存せず一定なのに $1/(2h)$ で割るため、標準誤差が $O(1/h)$ で発散する。
# 同一乱数（CRN）を使うと分子の差が $O(h)$ になり、$1/(2h)$ で割っても標準誤差が抑えられ、
# pathwise に匹敵する。実務で Greeks を MC で出すなら、まず pathwise、それが使えない
# （ペイオフが不連続な）場合は同一乱数バンプ、という順に検討する。
