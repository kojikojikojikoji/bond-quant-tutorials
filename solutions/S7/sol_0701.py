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
# # S7-1 演習 解答例
#
# ハザードレートとサバイバル確率の演習2問の解答例です。本文と同じ
# `bondlab.credit` / `bondlab.curve` を使います。

# %%
import numpy as np
import matplotlib.pyplot as plt

from bondlab.credit import HazardCurve

np.random.seed(0)

# %% [markdown]
# ## 演習 1：近似 $s \approx \lambda(1-R)$ の誤差マップ
#
# 本文と同じリスキーゼロ債スプレッド（回収を満期に固定した閉形式）を真値とし、
# $\lambda(1-R)$ の相対誤差（%）を $\lambda \times R$ の格子で求めます。

# %%
def risky_zero_spread(lam, R, t):
    """リスキー割引ゼロ債（回収を満期に固定）の信用スプレッドを閉形式で返す。"""
    S = np.exp(-lam * t)
    return -np.log(S + R * (1 - S)) / t


t_ref = 5.0
lam_grid = np.array([0.005, 0.010, 0.030, 0.080])   # 50,100,300,800 bp
R_grid = np.array([0.2, 0.4, 0.6, 0.8])

err = np.zeros((len(lam_grid), len(R_grid)))
for i, lam in enumerate(lam_grid):
    for j, R in enumerate(R_grid):
        s_true = risky_zero_spread(lam, R, t_ref)
        s_approx = lam * (1 - R)
        err[i, j] = (s_approx - s_true) / s_true * 100

# 表形式で出力
header = "λ\\R    " + "".join(f"{R:>9.1f}" for R in R_grid)
print(header)
for i, lam in enumerate(lam_grid):
    row = f"{lam*1e4:5.0f}bp " + "".join(f"{err[i, j]:>8.2f}%" for j in range(len(R_grid)))
    print(row)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(err, aspect="auto", cmap="viridis", origin="lower")
ax.set_xticks(range(len(R_grid)))
ax.set_xticklabels([f"{R:.1f}" for R in R_grid])
ax.set_yticks(range(len(lam_grid)))
ax.set_yticklabels([f"{lam*1e4:.0f}bp" for lam in lam_grid])
ax.set_xlabel("回収率 R")
ax.set_ylabel("ハザード λ")
ax.set_title("λ(1-R) 近似の相対誤差（%）")
for i in range(len(lam_grid)):
    for j in range(len(R_grid)):
        ax.text(j, i, f"{err[i, j]:.1f}", ha="center", va="center",
                color="white", fontsize=9)
fig.colorbar(im, ax=ax, label="相対誤差（%）")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察
#
# 誤差は $\lambda$・$R$ の双方に対して単調に増えますが、効き方が大きいのは $\lambda$ です。
# $\lambda$ を50bpから800bpへ上げると各 $R$ 列で誤差が約0.3%→十数%へ跳ね、$R$ を
# 0.2から0.8へ上げても各 $\lambda$ 行で数倍にとどまります。$\lambda(1-R)$ は
# $s=-\ln[S+R(1-S)]/t$ の $\lambda t$ に関する一次項なので、$\lambda t$ が大きいほど
# 生存確率の指数減衰の非線形性（二次以降）が効いて誤差が拡大します。$R$ を上げると
# 一次項 $\lambda(1-R)$ 自体が縮み、相対的に高次項の寄与が増えるため誤差が増えます。
# 小さい $\lambda$（投資適格レンジ）では誤差1%前後で実務的に十分、大きい $\lambda$
# （ハイイールド）ではフル評価が要る、というのが結論です。

# %%
assert np.all(err > 0)                    # 近似は常に過大評価（正の誤差）
assert np.all(np.diff(err, axis=0) > 0)   # λ が大きいほど誤差拡大（列方向）
assert np.all(np.diff(err, axis=1) > 0)   # R が大きいほど誤差拡大（行方向）
assert err[0, 0] < 1.0                     # 最小 λ・最小 R では1%未満
print("演習1 チェックを通過しました")

# %% [markdown]
# ## 演習 2：折れ曲がる生存確率
#
# 区分定数ハザード $\lambda=[0.01, 0.05, 0.01]$（区間右端 $[2, 4, 8]$ 年）で、
# $S(t)$ と $\ln S(t)$ をプロットし、$\ln S(t)$ の傾きが区間ごとに $-\lambda$ に
# なることを数値で確認します。

# %%
times = np.array([2.0, 4.0, 8.0])
lams = np.array([0.01, 0.05, 0.01])
haz = HazardCurve(times, lams)

t = np.linspace(0.0, 8.0, 401)
S = haz.survival(t)
logS = np.log(S)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(t, S, color="#2ca02c")
for xb in times[:-1]:
    ax1.axvline(xb, ls="--", color="gray", alpha=0.6)
ax1.set_xlabel("年数 t")
ax1.set_ylabel("生存確率 S(t)")
ax1.set_title("区分定数ハザードの生存確率（折れ曲がる）")
ax1.set_ylim(0, 1.02)
ax1.grid(alpha=0.3)

ax2.plot(t, logS, color="#1f77b4")
for xb in times[:-1]:
    ax2.axvline(xb, ls="--", color="gray", alpha=0.6)
ax2.set_xlabel("年数 t")
ax2.set_ylabel("ln S(t)")
ax2.set_title("ln S(t) は区間ごとに直線（傾き = -λ）")
ax2.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# 各区間の中央で傾き d(lnS)/dt を数値微分し、-λ と突合する。
segments = [(0.5, 1.5, lams[0]), (2.5, 3.5, lams[1]), (5.0, 7.0, lams[2])]
print(f"{'区間':>12} {'数値傾き':>12} {'-λ':>10}")
for a, b, lam in segments:
    slope = (np.log(haz.survival(b)) - np.log(haz.survival(a))) / (b - a)
    print(f"[{a:.1f},{b:.1f}]   {slope:>12.6f} {-lam:>10.6f}")
    assert abs(slope - (-lam)) < 1e-9

# %% [markdown]
# ### 考察
#
# $\ln S(t) = -\int_0^t \lambda(u)\,du$ なので、$\lambda$ が区間内で一定なら
# $\ln S(t)$ はその区間で傾き $-\lambda$ の直線になります。$\lambda$ が
# $0.01 \to 0.05 \to 0.01$ と切り替わる $t=2, 4$ で傾きが折れ、生存確率 $S(t)$ 自体は
# $[2,4]$ 年で急に減り、前後の区間では緩やかに減ります。数値微分した傾きが各区間で
# $-\lambda$ に一致することが、区分定数ハザードと折れ線累積ハザードの整合を裏づけます。

# %%
print("演習2 チェックを通過しました")
