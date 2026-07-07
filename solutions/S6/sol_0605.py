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
# # S6-5 演習 解答例
#
# SABR スマイルの演習2問の解答例です。本編（`nb_0605_sabr.py`）で組んだ Hagan 近似の
# スクラッチ実装とフィット関数を流用します。

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from bondlab.pricing import sabr_vol

np.random.seed(0)


def my_sabr_vol(F, K, T, alpha, beta, rho, nu):
    """本編と同じ Hagan(2002) 対数正規 SABR ボラのスクラッチ実装。"""
    if abs(F - K) < 1e-12:
        term_atm = (((1 - beta) ** 2 / 24) * alpha ** 2 / F ** (2 - 2 * beta)
                    + 0.25 * rho * beta * nu * alpha / F ** (1 - beta)
                    + (2 - 3 * rho ** 2) / 24 * nu ** 2)
        return alpha / F ** (1 - beta) * (1 + term_atm * T)
    log_fk = np.log(F / K)
    fk_beta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk
    x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    denom = fk_beta * (1
                       + (1 - beta) ** 2 / 24 * log_fk ** 2
                       + (1 - beta) ** 4 / 1920 * log_fk ** 4)
    prefactor = alpha / denom * (z / x)
    correction = (((1 - beta) ** 2 / 24) * alpha ** 2 / fk_beta ** 2
                  + 0.25 * rho * beta * nu * alpha / fk_beta
                  + (2 - 3 * rho ** 2) / 24 * nu ** 2)
    return prefactor * (1 + correction * T)


def sabr_smile(F, Ks, T, alpha, beta, rho, nu):
    """行使配列に対する my_sabr_vol のベクトル化。"""
    return np.array([my_sabr_vol(F, float(K), T, alpha, beta, rho, nu) for K in Ks])


def fit_sabr(F, Ks, vols, T, beta):
    """beta 固定で (alpha, rho, nu) を最小二乗フィットし、RMSE(bp) も返す。"""
    Ks = np.asarray(Ks, dtype=float)
    vols = np.asarray(vols, dtype=float)
    atm_idx = int(np.argmin(np.abs(Ks - F)))
    alpha0 = vols[atm_idx] * F ** (1 - beta)

    def resid(p):
        a, r, n = p
        model = np.array([my_sabr_vol(F, float(K), T, a, beta, r, n) for K in Ks])
        return (model - vols) * 1e4

    fit = least_squares(resid, x0=[alpha0, -0.2, 0.3],
                        bounds=([1e-6, -0.999, 1e-6], [1.0, 0.999, 3.0]))
    return tuple(fit.x), float(np.sqrt(np.mean(fit.fun ** 2)))


# 自作実装が bondlab と一致していることを念のため確認
assert abs(my_sabr_vol(0.03, 0.035, 5.0, 0.02, 0.5, -0.3, 0.4)
           - sabr_vol(0.03, 0.035, 5.0, 0.02, 0.5, -0.3, 0.4)) < 1e-13

# %% [markdown]
# ## 演習1：4パラメータの感応度とバックボーン
#
# 各パラメータを単独で3水準動かした4枚のスマイルを描き、さらに $\beta\in\{0,0.5,1\}$ に
# ついて **ATM ボラのバックボーン**（$F$ を動かしたときの ATM ボラの軌跡）を確認します。

# %%
F0, T0 = 0.030, 5.0
base = dict(alpha=0.020, beta=0.5, rho=-0.30, nu=0.40)
Ks = np.linspace(0.012, 0.055, 80)

grids = {"alpha": [0.014, 0.020, 0.026], "beta": [0.0, 0.5, 1.0],
         "rho": [-0.6, 0.0, 0.6], "nu": [0.15, 0.40, 0.70]}
labels = {"alpha": r"$\alpha$（高さ）", "beta": r"$\beta$（バックボーン）",
          "rho": r"$\rho$（スキュー）", "nu": r"$\nu$（曲率）"}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (pname, values) in zip(axes.ravel(), grids.items()):
    for v in values:
        p = dict(base)
        p[pname] = v
        ax.plot(Ks * 100, sabr_smile(F0, Ks, T0, **p) * 100, lw=1.8,
                label=f"{pname}={v}")
    ax.axvline(F0 * 100, color="gray", ls=":", lw=1)
    ax.set_title(labels[pname])
    ax.set_xlabel("行使レート K(%)"); ax.set_ylabel("対数正規ボラ(%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("演習1：SABR 4パラメータの単独感応度", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### バックボーン：$F$ を動かしたときの ATM ボラの軌跡
#
# $\alpha,\rho,\nu$ を固定し $F$ を $0.01\to0.05$ に動かして、各 $\beta$ での ATM ボラを
# 追います。$\beta=1$ なら ATM ボラはほぼ横ばい（対数正規的）、$\beta=0$ なら $F$ の
# 低下とともに ATM ボラが上がる（正規的）ことを確認します。

# %%
F_grid = np.linspace(0.010, 0.050, 60)
fig, ax = plt.subplots(figsize=(9, 5.5))
backbone_rows = []
for beta in [0.0, 0.5, 1.0]:
    atm = np.array([my_sabr_vol(F, F, T0, base["alpha"], beta, base["rho"], base["nu"])
                    for F in F_grid])
    ax.plot(F_grid * 100, atm * 100, lw=2.0, label=f"β={beta}")
    backbone_rows.append({"beta": beta,
                          "ATMボラ@F=1%": atm[0] * 100,
                          "ATMボラ@F=5%": atm[-1] * 100,
                          "傾き(高F-低F)": (atm[-1] - atm[0]) * 100})
ax.set_title("演習1：ATM ボラのバックボーン（F を動かす）")
ax.set_xlabel("フォワード F(%)"); ax.set_ylabel("ATM 対数正規ボラ(%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

backbone_df = pd.DataFrame(backbone_rows)
print(backbone_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# β=1 は F を上げても ATM ボラがほぼ横ばい、β=0 は右下がり（低 F で高い）
slopes = backbone_df.set_index("beta")["傾き(高F-低F)"]
assert slopes[1.0] > slopes[0.0]  # β=1 のほうが傾きが大きい（＝横ばい〜微増）
assert slopes[0.0] < 0            # β=0 は F 上昇で ATM ボラ低下
print("\n→ β はスマイル断面ではなく『F の変化に対する ATM ボラの動き方』を決める。"
      "β=0 は正規的（右下がり）、β=1 は対数正規的（横ばい）")

# %% [markdown]
# ## 演習2：β=0 vs β=1 のフィット・外挿・識別問題
#
# 真値 $\beta=0.5$ のスマイルを狭い行使域（ATM±50bp）で生成し、$\beta=0$ と $\beta=1$ で
# フィットします。(a) フィット域 RMSE がほぼ等しいこと、(b) 外挿域で乖離すること、
# (c) フィット済み $\rho$ が $\beta$ で系統的にずれることを示します。狭い域では曲率の
# 情報が乏しく、水準とスキューは3自由度 $(\alpha,\rho,\nu)$ でどの $\beta$ でも合わせ
# られるため、$\beta$ が識別できません。

# %%
F_c, T_c = 0.030, 5.0
alpha_true, beta_true, rho_true, nu_true = 0.020, 0.5, -0.30, 0.45

Ks_calib = F_c + np.linspace(-0.005, 0.005, 5)
vols_calib = sabr_smile(F_c, Ks_calib, T_c, alpha_true, beta_true, rho_true, nu_true)
Ks_wide = np.linspace(F_c - 0.028, F_c + 0.030, 140)
vols_true_wide = sabr_smile(F_c, Ks_wide, T_c, alpha_true, beta_true, rho_true, nu_true)

fits = {}
for beta in [0.0, 0.5, 1.0]:
    (a, r, n), rmse = fit_sabr(F_c, Ks_calib, vols_calib, T_c, beta=beta)
    fits[beta] = dict(alpha=a, rho=r, nu=n, rmse=rmse,
                      wide=sabr_smile(F_c, Ks_wide, T_c, a, beta, r, n))

# (a) フィット域 RMSE の比較
fit_summary = pd.DataFrame(
    {beta: {"alpha": f["alpha"], "rho": f["rho"], "nu": f["nu"],
            "フィット域RMSE(bp)": f["rmse"]} for beta, f in fits.items()}
).T
fit_summary.index.name = "beta"
print("フィット結果（真値 β=0.5, ρ=-0.30）:")
print(fit_summary.to_string(float_format=lambda x: f"{x:.5f}"))

# フィット域では β を変えてもほぼ同じ品質（＝β を識別できない）
assert fits[0.0]["rmse"] < 2.0 and fits[1.0]["rmse"] < 2.0
assert abs(fits[0.0]["rmse"] - fits[1.0]["rmse"]) < 0.3  # RMSE がほぼ等しい

# %% [markdown]
# ### (b) 外挿域の乖離を図示

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot((Ks_wide - F_c) * 1e4, vols_true_wide * 100, "k-", lw=2.4,
        label=f"真値（β={beta_true}）")
colors = {0.0: "C0", 0.5: "C2", 1.0: "C3"}
for beta in [0.0, 1.0]:
    ax.plot((Ks_wide - F_c) * 1e4, fits[beta]["wide"] * 100, "--", lw=1.8,
            color=colors[beta], label=f"β={beta} フィット")
ax.plot((Ks_calib - F_c) * 1e4, vols_calib * 100, "o", ms=7, color="gray",
        label="キャリブレーション点")
ax.axvspan(-50, 50, color="green", alpha=0.06, label="フィット域")
ax.set_title("演習2：β=0 と β=1 のフィットと外挿")
ax.set_xlabel("ATM からの行使乖離(bp)"); ax.set_ylabel("対数正規ボラ(%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 外挿域（フィット域の外）では両者が乖離する
outside = np.abs(Ks_wide - F_c) > 0.005
gap_outside = np.max(np.abs(fits[0.0]["wide"][outside] - fits[1.0]["wide"][outside])) * 1e4
gap_inside = np.max(np.abs(fits[0.0]["wide"][~outside] - fits[1.0]["wide"][~outside])) * 1e4
print(f"β=0 と β=1 の最大乖離: フィット域内={gap_inside:.2f}bp, 外挿域={gap_outside:.2f}bp")
assert gap_outside > gap_inside

# %% [markdown]
# ### (c) $\rho$ が $\beta$ で系統的にずれる（識別問題）
#
# スキューは $\beta$ と $\rho$ が奪い合います。$\beta$ を下げると（低ストライクを持ち上げる
# 方向）、同じ観測スキューを保つためフィット済み $\rho$ は逆に緩みます。真値 $\beta=0.5$ で
# $\rho$ が復元される一方、$\beta$ を誤って固定すると $\rho$ に系統的なバイアスが乗ります。

# %%
rho_by_beta = pd.Series({beta: fits[beta]["rho"] for beta in [0.0, 0.5, 1.0]},
                        name="フィット済みρ")
rho_by_beta.index.name = "beta"
print(rho_by_beta.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\n真値 β=0.5 での復元 ρ = {fits[0.5]['rho']:.4f}（真値 {rho_true}）")

# 真の β=0.5 では rho を正しく復元
assert abs(fits[0.5]["rho"] - rho_true) < 1e-3
# β を上げると（対数正規寄り）フィット済み ρ はより負に振れてスキューを補う
assert fits[1.0]["rho"] < fits[0.0]["rho"]
print("\n結論：フィット域だけでは β と ρ を分離できない（識別問題）。β を履歴由来の"
      "バックボーンから固定し、(α,ρ,ν) を日々キャリブレーションするのが実務の型。")
