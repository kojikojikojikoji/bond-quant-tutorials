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
# # S6-3 演習 解答例
#
# ## 準備
#
# 本編の合成カーブ・合成ボラ・フォワードスワップ計算・スクラッチ評価を最小限だけ再掲します。

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import bondlab
from bondlab.pricing import swap_annuity, swaption_black, black76
from bondlab.curve import bootstrap_par

print("bondlab version:", bondlab.__version__)

SEED = 20260707
rng = np.random.default_rng(SEED)
plt.rcParams["axes.unicode_minus"] = False

par_tenors = np.arange(1, 31, dtype=float)
par_rates = 0.03 + 0.01 * (1.0 - np.exp(-par_tenors / 8.0))
curve = bootstrap_par(par_tenors, par_rates, frequency=1)


def forward_swap(curve, expiry, tenor, freq=1):
    """フォワードスワップレート S0 と前方アニュイティ A を返す（本編と同一）。"""
    tau = 1.0 / freq
    n_pre = int(round(expiry * freq))
    n_all = int(round((expiry + tenor) * freq))
    times_all = tau * np.arange(1, n_all + 1)
    times_pre = tau * np.arange(1, n_pre + 1)
    ann = swap_annuity(curve, times_all) - swap_annuity(curve, times_pre)
    s0 = (curve.discount(expiry) - curve.discount(expiry + tenor)) / ann
    return float(s0), float(ann)


def synthetic_vol(expiry, tenor):
    """満期・テナー依存の合成インプライドボラ（本編と同一）。"""
    base = 0.22
    hump = 0.03 * np.exp(-((np.log(expiry) - np.log(2.0)) ** 2) / 0.8)
    tenor_decay = -0.008 * np.log(tenor)
    return base + hump + tenor_decay


def straddle_price(F, K, T, vol, annuity):
    """ATM ストラドル価格 = payer + receiver。"""
    return (swaption_black(F, K, T, vol, annuity, option="payer")
            + swaption_black(F, K, T, vol, annuity, option="receiver"))


# %% [markdown]
# ## 演習1：ATM ストラドルのベガ・ガンマとボラ取引
#
# 満期 3y・テナー 5y の ATM ストラドルについて、ベガ（ボラの中心差分）とガンマ
# （フォワードの 2 階差分）を数値計算し、解析式と突き合わせます。

# %%
e, te = 3, 5
s0, ann = forward_swap(curve, e, te, freq=1)
vol = synthetic_vol(e, te)
T = float(e)
K = s0  # ATM

# 数値ベガ・ガンマ
h_v, h_f = 1e-4, 1e-4
vega_num = (straddle_price(s0, K, T, vol + h_v, ann)
            - straddle_price(s0, K, T, vol - h_v, ann)) / (2 * h_v)
gamma_num = (straddle_price(s0 + h_f, K, T, vol, ann)
             - 2 * straddle_price(s0, K, T, vol, ann)
             + straddle_price(s0 - h_f, K, T, vol, ann)) / h_f ** 2

# 解析式（単一オプションを 2 倍）
d1 = (np.log(s0 / K) + 0.5 * vol ** 2 * T) / (vol * np.sqrt(T))
vega_ana = 2 * ann * s0 * norm.pdf(d1) * np.sqrt(T)
gamma_ana = 2 * ann * norm.pdf(d1) / (s0 * vol * np.sqrt(T))

print(f"対象           : {e}y×{te}y, S0={s0:.5f}, vol={vol:.4f}, annuity={ann:.4f}")
print(f"ストラドル価格 : {straddle_price(s0, K, T, vol, ann):.8f}")
print(f"ベガ  : 数値={vega_num:.6f}  解析={vega_ana:.6f}  差={abs(vega_num - vega_ana):.2e}")
print(f"ガンマ: 数値={gamma_num:.6f}  解析={gamma_ana:.6f}  差={abs(gamma_num - gamma_ana):.2e}")
assert abs(vega_num - vega_ana) < 1e-3
assert abs(gamma_num - gamma_ana) < 1e-2
print("→ 数値微分と解析式が一致")

# %% [markdown]
# ### ボラ取引としての説明
#
# ストラドル買いの損益は、満期での実現ボラとインプライドボラの綱引きで決まります。図で二つの
# 効果を分けて見ます。左：満期にスワップレートがどこへ動いたかに対するペイオフ（ガンマの効果、
# 大きく動くほど得）。右：ボラ水準に対する現在価値（ベガの効果、ボラが上がると価値が増える）。

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 左：満期ペイオフ（ガンマ＝動くほど得）
s_realized = np.linspace(s0 - 0.03, s0 + 0.03, 121)
payoff = ann * (np.abs(s_realized - K))  # payer/receiver のどちらかが行使される
axes[0].plot(s_realized * 100, payoff, color="crimson", lw=2, label="straddle payoff at expiry")
axes[0].axhline(straddle_price(s0, K, T, vol, ann), color="gray", ls="--", lw=1.2,
                label="premium paid")
axes[0].axvline(s0 * 100, color="gray", ls=":", lw=1)
axes[0].set_xlabel("realized swap rate at expiry (%)")
axes[0].set_ylabel("payoff")
axes[0].set_title("gamma: larger moves pay more")
axes[0].legend(fontsize=9)

# 右：ボラに対する価値（ベガ）
vols = np.linspace(0.10, 0.40, 61)
prices = np.array([straddle_price(s0, K, T, v, ann) for v in vols])
axes[1].plot(vols * 100, prices, color="steelblue", lw=2, label="straddle price")
axes[1].plot(vols * 100,
             straddle_price(s0, K, T, vol, ann) + vega_ana * (vols - vol),
             color="crimson", ls="--", lw=1.5, label="vega tangent")
axes[1].axvline(vol * 100, color="gray", ls=":", lw=1)
axes[1].set_xlabel("implied vol (%)")
axes[1].set_ylabel("straddle price")
axes[1].set_title("vega: higher vol, higher value")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()

# 損益分岐：実現ボラが払ったプレミアムを賄うだけ動けばよい。
premium = straddle_price(s0, K, T, vol, ann)
breakeven_move = premium / ann  # A*|dS| = premium
print(f"支払プレミアム             = {premium:.6f}")
print(f"損益分岐のレート変動 |ΔS|  = {breakeven_move*1e4:.1f} bp")
print("解釈: 満期までに実現ボラがインプライドを上回り、スワップレートが損益分岐を超えて")
print("      動けば買い手の勝ち。動かなければベガ分の時間価値を失う（ボラの売買）。")

# %% [markdown]
# ## 演習2：ペイヤー・レシーバー・パリティ（フォワードスワップ）
#
# 満期 5y・テナー 5y で行使価格 $K$ を振り、$V^{\text{pay}}-V^{\text{rec}}=A(0)(S_0-K)$ を確認します。

# %%
e, te = 5, 5
s0, ann = forward_swap(curve, e, te, freq=1)
vol = synthetic_vol(e, te)
T = float(e)

print(f"対象: {e}y×{te}y, S0={s0:.5f}, annuity={ann:.5f}, vol={vol:.4f}\n")
print(f"{'K':>10s}{'V_payer':>12s}{'V_receiver':>12s}{'pay-rec':>12s}{'A*(S0-K)':>12s}")
for k in (s0 - 0.01, s0, s0 + 0.01):
    vpay = swaption_black(s0, k, T, vol, ann, option="payer")
    vrec = swaption_black(s0, k, T, vol, ann, option="receiver")
    parity = ann * (s0 - k)
    print(f"{k:10.5f}{vpay:12.6f}{vrec:12.6f}{vpay - vrec:12.6f}{parity:12.6f}")
    assert abs((vpay - vrec) - parity) < 1e-12
print("\n→ payer − receiver = A(0)·(S0 − K) を厳密確認")

# %% [markdown]
# ### 符号反転の理由
#
# 連続的に $K$ を振ってパリティを図示します。$V^{\text{pay}}-V^{\text{rec}}=A(0)(S_0-K)$ は $K$ の
# 一次関数で、$K=S_0$（ATM）で 0 になり、そこを境に符号が反転します。

# %%
strikes = np.linspace(s0 - 0.02, s0 + 0.02, 81)
diff = np.array([swaption_black(s0, k, T, vol, ann, option="payer")
                 - swaption_black(s0, k, T, vol, ann, option="receiver")
                 for k in strikes])
parity = ann * (s0 - strikes)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(strikes * 100, diff, color="crimson", lw=2, label="payer − receiver")
ax.plot(strikes * 100, parity, color="black", ls="--", lw=1.2, label="A(0)·(S0 − K)")
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(s0 * 100, color="gray", ls=":", lw=1, label=f"ATM = {s0*100:.2f}%")
ax.set_xlabel("strike (%)")
ax.set_ylabel("payer − receiver")
ax.set_title("payer−receiver parity is linear in strike, sign flips at ATM")
ax.legend()
plt.tight_layout()
plt.show()

print("解釈: payer − receiver は行使価格 K のフォワードペイヤースワップと同じ持ち高。")
print("      K < S0 では固定を安く払える in-the-money のスワップなので価値は正、")
print("      K > S0 では固定を高く払う out-of-the-money なので価値は負。ATM で 0 に交差。")
assert abs(np.max(np.abs(diff - parity))) < 1e-12
print("→ 全ストライクでパリティが厳密成立")
