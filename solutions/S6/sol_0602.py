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
# # S6-2 演習 解答例

# %% [markdown]
# ## 準備
#
# 本編で使ったキャップレット評価・剥ぎ取りの部品を最小限だけ再掲する。合成カーブ、
# 単利フォワード、キャップ価格（Black／Bachelier）、スポットボラ剥ぎ取りを手元に
# 置く。

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import bondlab
from bondlab import pricing
from bondlab.curve import bootstrap_par

print("bondlab version:", bondlab.__version__)

SEED = 20260707
rng = np.random.default_rng(SEED)
plt.rcParams["axes.unicode_minus"] = False

par_tenors = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
par_rates = np.array([0.030, 0.032, 0.034, 0.0355, 0.0365, 0.038, 0.0395])
curve = bootstrap_par(par_tenors, par_rates, frequency=1)

tau = 0.25
reset_times = np.arange(0.5, 2.5, tau)
pay_times = reset_times + tau


def forward_rates(curve, reset_times, tau):
    """単利フォワード L_i = (P(T_{i-1})/P(T_i) - 1)/tau。"""
    p_start = curve.discount(reset_times)
    p_end = curve.discount(reset_times + tau)
    return (p_start / p_end - 1.0) / tau


fwds = forward_rates(curve, reset_times, tau)
dfs_pay = curve.discount(pay_times)
K_cap = 0.035


def cap_price_black(fwds, K, expiries, vol, tau, dfs_pay):
    return sum(pricing.caplet(f, K, T, vol, tau, dp, "black")
               for f, T, dp in zip(fwds, expiries, dfs_pay))


def strip_spot_vols(fwds, expiries, pay_dfs, tau, strike, flat_vols):
    """フラットボラ列からスポットボラ列を剥ぎ取る（キャップレット単位）。"""
    n = len(fwds)
    spot = np.empty(n)
    cum_price = 0.0
    for i in range(n):
        cap_i = sum(
            pricing.caplet(fwds[j], strike, expiries[j], flat_vols[i], tau, pay_dfs[j], "black")
            for j in range(i + 1)
        )
        target_caplet = cap_i - cum_price
        f, T_exp, dp = fwds[i], expiries[i], pay_dfs[i]
        spot[i] = brentq(
            lambda v: pricing.caplet(f, strike, T_exp, v, tau, dp, "black") - target_caplet,
            1e-6, 5.0, xtol=1e-10,
        )
        cum_price += pricing.caplet(f, strike, T_exp, spot[i], tau, dp, "black")
    return spot

# %% [markdown]
# ## 演習1：フラットボラからスポットボラを剥ぎ取る
#
# ### (a) 右上がりフラットボラ → スポットボラは上側へ急に伸びる

# %%
flat_up = 0.24 + 0.05 * (reset_times - reset_times.min())   # 短期低め・長期高め
spot_up = strip_spot_vols(fwds, reset_times, dfs_pay, tau, K_cap, flat_up)

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(reset_times, flat_up * 100, "o-", label="フラットボラ（入力・右上がり）")
ax.plot(reset_times, spot_up * 100, "s--", label="スポットボラ（剥ぎ取り）")
ax.set_xlabel("キャップレット満期（年）")
ax.set_ylabel("ボラ（%）")
ax.set_title("(a) 右上がりフラットボラではスポットボラが上側へ急伸")
ax.legend()
fig.tight_layout()
plt.show()

print("長期端: フラット =", round(flat_up[-1] * 100, 2), "% / スポット =", round(spot_up[-1] * 100, 2), "%")
assert spot_up[-1] > flat_up[-1]   # 右上がりでは長期スポットがフラットの上側
print("右上がりのとき、長期スポットボラはフラットボラの上側に来ることを確認しました")

# %% [markdown]
# ### (b) 剥ぎ取ったスポットボラでキャップを再現できるか

# %%
for i in range(len(fwds)):
    cap_flat = sum(pricing.caplet(fwds[j], K_cap, reset_times[j], flat_up[i], tau, dfs_pay[j], "black")
                   for j in range(i + 1))
    cap_spot = sum(pricing.caplet(fwds[j], K_cap, reset_times[j], spot_up[j], tau, dfs_pay[j], "black")
                   for j in range(i + 1))
    assert abs(cap_flat - cap_spot) < 1e-10
print("全満期でフラット価格とスポット再構成価格が機械精度一致することを確認しました")

# %% [markdown]
# ### (c) フラットボラが平坦ならスポットボラも平坦か
#
# フラットボラが全満期で同一なら、剥ぎ取っても同じ値が戻る。理由：フラットボラが
# 一定 $\sigma$ のとき、どのキャップも全キャップレットを同じ $\sigma$ で評価した和
# であり、各キャップレットの限界価格も $\sigma$ で作った値そのもの。よって逐次求解の
# 解は常に $\sigma$ になる（スポット＝フラット）。

# %%
flat_const = np.full_like(reset_times, 0.30)
spot_const = strip_spot_vols(fwds, reset_times, dfs_pay, tau, K_cap, flat_const)
print("スポットボラ（平坦入力）:", np.round(spot_const, 6))
assert np.allclose(spot_const, 0.30, atol=1e-8)
print("フラットボラが平坦なら、スポットボラも同一値に戻ることを確認しました")

# %% [markdown]
# ## 演習2：低金利・負金利で Black が壊れ Bachelier が機能する
#
# ATM キャップレット1本を取り出す。真の価格は正規ボラ 60bp で作り、そこから
# 対数正規ボラ・正規ボラの両方を逆算する。

# %%
T_exp = 1.0                      # 満期1年のキャップレット
tau_1 = 0.5                      # 半年
df_pay = float(curve.discount(T_exp + tau_1))
true_nvol = 0.0060               # 真の正規ボラ 60bp

L_grid = np.array([0.030, 0.020, 0.010, 0.005, 0.002, 0.000, -0.005, -0.010])

rows = []
for L in L_grid:
    K_atm = L                    # ATM：行使をフォワードに一致させる
    price = tau_1 * pricing.bachelier(L, K_atm, T_exp, true_nvol, df_pay, "call")
    # 対数正規ボラの逆算（F>0 のときのみ試みる）
    ln_vol = np.nan
    if L > 0:
        try:
            ln_vol = brentq(
                lambda v: tau_1 * pricing.black76(L, K_atm, T_exp, v, df_pay, "call") - price,
                1e-6, 5.0, xtol=1e-10,
            )
        except (ValueError, ZeroDivisionError):
            ln_vol = np.nan
    # 正規ボラの逆算（負金利でも定義できる）
    try:
        n_vol = brentq(
            lambda v: tau_1 * pricing.bachelier(L, K_atm, T_exp, v, df_pay, "call") - price,
            1e-8, 0.10, xtol=1e-12,
        )
    except (ValueError, ZeroDivisionError):
        n_vol = np.nan
    rows.append((L, price, ln_vol, n_vol))

print(f"{'F=K(ATM)':>10} {'価格':>14} {'対数正規ボラ':>14} {'正規ボラ(bp)':>14}")
for L, price, ln_vol, n_vol in rows:
    ln_disp = "破綻(NaN)" if np.isnan(ln_vol) else f"{ln_vol:14.4f}"
    print(f"{L:10.3f} {price:14.8f} {ln_disp:>14} {n_vol * 1e4:14.2f}")

# %% [markdown]
# ### (a)(b) 図示：対数正規ボラは低金利で発散・負金利で破綻、正規ボラは滑らか

# %%
L_arr = np.array([r[0] for r in rows])
ln_arr = np.array([r[2] for r in rows])
n_arr = np.array([r[3] for r in rows]) * 1e4

fig, ax1 = plt.subplots(figsize=(7.4, 3.6))
ax1.plot(L_arr, ln_arr, "o-", color="C3", label="対数正規ボラ（Black）")
ax1.set_xlabel("ATM フォワード L0 = K")
ax1.set_ylabel("対数正規ボラ", color="C3")
ax1.axvline(0.0, color="gray", ls=":", lw=1)
ax2 = ax1.twinx()
ax2.plot(L_arr, n_arr, "s--", color="C0", label="正規ボラ（bp）")
ax2.set_ylabel("正規ボラ（bp）", color="C0")
ax1.set_title("(a)(b) 低金利で対数正規ボラは発散、負金利で破綻／正規ボラは一定")
fig.tight_layout()
plt.show()

# 正規ボラは全水準で真値 60bp を復元する（負金利含む）
n_valid = n_arr[~np.isnan(n_arr)]
assert np.allclose(n_valid, 60.0, atol=1e-2)
# 対数正規ボラは L<=0 で NaN（破綻）
assert np.all(np.isnan(ln_arr[L_arr <= 0]))
print("正規ボラは負金利まで真値 60bp を復元、対数正規ボラは L<=0 で破綻することを確認しました")

# %% [markdown]
# ### (c) なぜ正規モデルは負金利で壊れないか
#
# Black'76 は $d_{1,2}$ に $\ln(L_0/K)$ を含む。$L_0 \le 0$ または $K \le 0$ では
# 対数が定義できず、$L_0 \to 0^{+}$ では $\ln(L_0/K) \to -\infty$ となってボラが
# 発散する。対して Bachelier の価格式は
#
# $$
# \tau P\,\bigl[(L_0 - K)\Phi(d) + \sigma_N\sqrt{T}\,\phi(d)\bigr],\quad d = \frac{L_0-K}{\sigma_N\sqrt{T}}
# $$
#
# のように、$L_0$ と $K$ を常に**差** $(L_0 - K)$ の形でしか使わない。差は $L_0$ や
# $K$ が負でも普通の実数なので、価格もボラ逆算も負金利側へ滑らかに延長できる。
# これが低金利・負金利局面で正規ボラが標準になる数理的な理由である。

# %%
print("演習1・2の検証がすべて通過しました")
