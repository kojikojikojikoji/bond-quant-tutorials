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
# # S2-3 演習 解答例

# %% [markdown]
# ## 準備：因子負荷・NSS・2段階推定
#
# 本編の自作関数を最小限だけ再掲する。

# %%
import numpy as np
import pandas as pd

import bondlab
from bondlab import curve as blcurve

np.random.seed(0)


def ns_loadings(tau, lam):
    tau = np.asarray(tau, dtype=float)
    x = tau / lam
    L0 = np.ones_like(tau)
    L1 = (1.0 - np.exp(-x)) / x
    L2 = L1 - np.exp(-x)
    return L0, L1, L2


def nss_zero(tau, b0, b1, b2, b3, lam1, lam2):
    _, L1, L2 = ns_loadings(tau, lam1)
    _, _, L2b = ns_loadings(tau, lam2)
    return b0 + b1 * L1 + b2 * L2 + b3 * L2b


def design_matrix(tenors, lam1, lam2):
    _, L1, L2 = ns_loadings(tenors, lam1)
    _, _, L2b = ns_loadings(tenors, lam2)
    L0 = np.ones_like(np.asarray(tenors, dtype=float))
    return np.column_stack([L0, L1, L2, L2b])


def fit_beta_ols(tenors, yields, lam1, lam2):
    X = design_matrix(tenors, lam1, lam2)
    beta, *_ = np.linalg.lstsq(X, np.asarray(yields, dtype=float), rcond=None)
    resid = X @ beta - yields
    return beta, float(resid @ resid)


def fit_nss_2stage(tenors, yields, lam_grid=None):
    tenors = np.asarray(tenors, dtype=float)
    yields = np.asarray(yields, dtype=float)
    if lam_grid is None:
        lam_grid = np.linspace(0.5, 10.0, 20)
    best = None
    for lam1 in lam_grid:
        for lam2 in lam_grid:
            if lam2 <= lam1:
                continue
            beta, ssr = fit_beta_ols(tenors, yields, lam1, lam2)
            if best is None or ssr < best[0]:
                best = (ssr, beta, lam1, lam2)
    _, beta, lam1, lam2 = best
    return dict(beta0=beta[0], beta1=beta[1], beta2=beta[2], beta3=beta[3],
                lam1=lam1, lam2=lam2)


tenors = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

# %% [markdown]
# ## 演習1：格子の粗さと復元精度
#
# 既知パラメータ `p` から曲線を生成し、λ 格子の点数を変えてフィットする。
# 各格子で「係数の最大復元誤差」と「曲線の最大復元誤差」を測る。

# %%
p = dict(b0=0.045, b1=-0.020, b2=0.030, b3=-0.015, lam1=1.5, lam2=5.0)
y_true = nss_zero(tenors, **p)
truth = [p["b0"], p["b1"], p["b2"], p["b3"], p["lam1"], p["lam2"]]

rows = []
for n in (5, 8, 12, 20, 40):
    grid = np.linspace(0.5, 10.0, n)
    f = fit_nss_2stage(tenors, y_true, lam_grid=grid)
    est = [f["beta0"], f["beta1"], f["beta2"], f["beta3"], f["lam1"], f["lam2"]]
    coef_err = max(abs(a - b) for a, b in zip(est, truth))
    z = nss_zero(tenors, f["beta0"], f["beta1"], f["beta2"], f["beta3"],
                 f["lam1"], f["lam2"])
    curve_err = float(np.max(np.abs(z - y_true)))
    rows.append({
        "格子点数": n,
        "λ刻み幅": round((10.0 - 0.5) / (n - 1), 3),
        "係数最大誤差": coef_err,
        "曲線最大誤差": curve_err,
    })

ex1 = pd.DataFrame(rows)
print(ex1.to_string(index=False,
                    formatters={"係数最大誤差": lambda v: f"{v:.2e}",
                                "曲線最大誤差": lambda v: f"{v:.2e}"}))

# %% [markdown]
# **解釈**：曲線の最大誤差は格子を細かくするとほぼ単調に小さくなり、比較的
# 少ない点数でも実用水準（数 bp 未満）に達する。これは「係数が多少ずれても
# 曲線は復元される」多重共線性の裏返しである。一方、係数の最大誤差は滑らかには
# 減らず、**真の λ（1.5, 5.0）が格子点に載るかどうかで階段状**に動く。点数 20
# の既定格子は刻み 0.5 で 1.5・5.0 をちょうど含むため係数まで機械精度で戻るが、
# 5 点・12 点のように真値を外す格子では、曲線が合っていても係数（特に β2, β3, λ）
# は残差なりの丸めを受ける。要するに「曲線精度は格子密度に素直、係数精度は真値が
# 格子に載るかに依存」する。

# %% [markdown]
# ## 演習2：日次因子の解釈（スティープ化／フラット化）
#
# 実データパネルからレベル・スロープ・曲率の日次時系列を推定し、最も
# スティープ化した局面とフラット化した局面を特定する。

# %%
panel = pd.read_csv("data/samples/synthetic_ust_par_panel.csv")
wide = panel.pivot(index="date", columns="tenor", values="par_yield").sort_index()
panel_tenors = wide.columns.values.astype(float)
coarse_grid = np.linspace(0.8, 8.0, 8)

records = []
for date, row in wide.iterrows():
    f = fit_nss_2stage(panel_tenors, row.values, lam_grid=coarse_grid)
    records.append({
        "date": date,
        "レベル": f["beta0"],
        "スロープ": -f["beta1"],
        "曲率": f["beta2"],
    })

betas = pd.DataFrame(records).set_index("date")
betas.index = pd.to_datetime(betas.index)

# 日次のスロープ変化（bp）で、最スティープ化日・最フラット化日を特定
slope_chg = betas["スロープ"].diff() * 1e4  # bp
steepen_day = slope_chg.idxmax()
flatten_day = slope_chg.idxmin()
print(f"最スティープ化: {steepen_day.date()}  Δスロープ {slope_chg.max():+.2f} bp")
print(f"最フラット化  : {flatten_day.date()}  Δスロープ {slope_chg.min():+.2f} bp")

# %% [markdown]
# スロープ変化が短期側・長期側どちらの利回り変化によるものかを、観測パネルの
# 端点（0.5年・30年）の日次変化と照合する。

# %%
short_chg = wide[0.5].astype(float).diff() * 1e4   # bp
long_chg = wide[30.0].astype(float).diff() * 1e4   # bp
short_chg.index = betas.index
long_chg.index = betas.index

for label, day in [("スティープ化", steepen_day), ("フラット化", flatten_day)]:
    print(f"[{label}] {day.date()}: "
          f"0.5年 {short_chg.loc[day]:+.2f} bp / "
          f"30年 {long_chg.loc[day]:+.2f} bp")

# 期間全体で、スロープ変化が短期・長期どちらとより連動するか
print("\nΔスロープ と Δ短期(0.5y) の相関:",
      round(np.corrcoef(slope_chg.dropna(), (-short_chg).dropna())[0, 1], 3))
print("Δスロープ と Δ長期(30y) の相関 :",
      round(np.corrcoef(slope_chg.dropna(), long_chg.dropna())[0, 1], 3))

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(betas.index, betas["レベル"] * 100, color="C0")
axes[0].set_ylabel("レベル (%)")
axes[0].set_title("NSS 因子の日次時系列（レベル・スロープ・曲率）")
axes[1].plot(betas.index, betas["スロープ"] * 100, color="C1")
axes[1].axvline(steepen_day, color="g", ls="--", alpha=0.7, label="最スティープ化")
axes[1].axvline(flatten_day, color="r", ls="--", alpha=0.7, label="最フラット化")
axes[1].set_ylabel("スロープ (%)")
axes[1].legend()
axes[2].plot(betas.index, betas["曲率"] * 100, color="C2")
axes[2].set_ylabel("曲率 (%)")
axes[2].set_xlabel("日付")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：スロープ（$=-\beta_1$、長期$-$短期）は、短期利回りが下がるか長期が
# 上がるとスティープ化し、逆でフラット化する。上の照合では、スロープの日次変化は
# 短期端の変化（符号反転）と強く連動し、長期端との連動は相対的に弱い。つまり
# この合成パネルのスロープの動きは、主に**短期側の利回り変化が駆動**している。
# レベルは緩やかなトレンドで動き、曲率は中期の相対的な突出を表して短期・スロープ
# とは別のリズムを持つ。3因子がそれぞれ独立した情報を担うことが、パラメトリック
# 表現の利点である。
