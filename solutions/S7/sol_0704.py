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
# # S7-4 演習 解答例
#
# クレジットスプレッド分析の演習2問の解答例です。本文の合成スプレッド時系列と
# ゼロカーブを同じシードで再構築して使います。

# %%
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from bondlab.bond import FixedRateBond
from bondlab.curve import bootstrap_par

np.random.seed(7)

# %% [markdown]
# ## 演習1：IG/HY時系列からの局面判定
#
# 本文と同じ合成スプレッドを生成し、各月を「平常」「警戒」「ストレス」に分類します。
# しきい値は **IG・HY 双方の過去パーセンタイル** で決めます。
# 根拠：単一系列だと個別要因で誤検知しうるため、IG（質の高い層）と HY（脆弱な層）が
# **同時に**ワイド化した月ほど、市場全体のストレスとみなせるという考え方です。

# %%
rng = np.random.default_rng(7)
months = pd.date_range("2005-01-01", "2025-12-01", freq="MS")
n = len(months)

stress_windows = [
    ("2007-08-01", "2009-12-01", 1.00),
    ("2011-07-01", "2012-06-01", 0.35),
    ("2015-08-01", "2016-02-01", 0.30),
    ("2020-02-01", "2020-08-01", 0.65),
    ("2022-04-01", "2022-10-01", 0.35),
]
stress = np.zeros(n)
t_idx = np.arange(n)
for start, end, peak in stress_windows:
    s = months.get_indexer([pd.Timestamp(start)])[0]
    e = months.get_indexer([pd.Timestamp(end)])[0]
    mid = (s + e) / 2.0
    half = max((e - s) / 2.0, 1.0)
    bump = peak * np.clip(1 - ((t_idx - mid) / half) ** 2, 0, None)
    stress = np.maximum(stress, bump)


def ar1_noise(sd, rho=0.7):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, sd)
    return e


ig_oas = np.clip(110 + 500 * stress + ar1_noise(12.0), 70, None)
hy_oas = np.clip(350 + 1650 * stress + ar1_noise(45.0), 260, None)
spreads = pd.DataFrame({"IG": ig_oas, "HY": hy_oas}, index=months)


# %%
def classify_regime(df, warn=0.60, stress_q=0.80):
    """各月を局面に分類する。

    IG・HY をそれぞれ過去全体でのパーセンタイル順位に変換し、
    両系列のパーセンタイルの小さい方（＝両方が高くて初めて高い）で判定する。
      - 両方が stress_q 以上 → 「ストレス」
      - 両方が warn 以上     → 「警戒」
      - それ以外            → 「平常」
    """
    ig_rank = df["IG"].rank(pct=True)
    hy_rank = df["HY"].rank(pct=True)
    both = np.minimum(ig_rank, hy_rank)
    regime = np.where(both >= stress_q, "ストレス",
                      np.where(both >= warn, "警戒", "平常"))
    return pd.Series(regime, index=df.index, name="局面")


regime = classify_regime(spreads)
counts = regime.value_counts().reindex(["平常", "警戒", "ストレス"]).fillna(0).astype(int)
print("局面別の月数:")
print(counts.to_string())

print("\n直近12か月の局面推移:")
recent = pd.concat([spreads.tail(12).round(0), regime.tail(12)], axis=1)
print(recent.to_string())

# %%
fig, ax = plt.subplots(figsize=(9, 4))
color_map = {"平常": "C2", "警戒": "C1", "ストレス": "C3"}
ax.plot(spreads.index, spreads["HY"], color="0.6", lw=0.8, label="HY OAS")
for lab, col in color_map.items():
    mask = regime == lab
    ax.scatter(spreads.index[mask], spreads["HY"][mask], s=12, color=col, label=lab)
ax.set_ylabel("HY OAS (bp)")
ax.set_title("局面判定（HY系列上に色分け）")
ax.legend(ncol=4, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
print("金融危機・コロナ期が『ストレス』として抽出されています。")

# %% [markdown]
# ## 演習2：同一発行体の複数銘柄で rich/cheap を見る
#
# 1発行体の年限違い5銘柄からG-spreadを計算し、年限に対するスプレッドカーブ（2次）を
# フィットします。各銘柄の残差（実測−フィット）が正なら割安（cheap、利回りが高い＝安い）、
# 負なら割高（rich）と判定します。

# %%
SETTLE = dt.date(2026, 6, 15)
TENORS = np.arange(1, 11, dtype=float)
GOVT_PAR = np.array([0.015, 0.018, 0.020, 0.022, 0.024, 0.025, 0.026, 0.0265, 0.027, 0.0275])
govt_zc = bootstrap_par(TENORS, GOVT_PAR, frequency=1)


def future_cashflows(bond, settle):
    flows = [(d, c) for d, c in bond.cashflows() if d > settle]
    times = np.array([(d - settle).days / 365.25 for d, _ in flows], dtype=float)
    cfs = np.array([c for _, c in flows], dtype=float)
    return times, cfs


def dirty_from_curve(bond, settle, zc, s):
    times, cfs = future_cashflows(bond, settle)
    return float(np.sum(cfs * zc.discount(times) * np.exp(-s * times)))


def bond_gspread(bond, clean, settle):
    y = bond.yield_from_price(clean, settle)
    T = (bond.maturity - settle).days / 365.25
    return T, y - float(np.interp(T, TENORS, GOVT_PAR))


# 同一発行体（BBB）の5銘柄。基準カーブ + 年限依存の真スプレッドで価格を作り、
# 一部銘柄に銘柄固有のズレ（発行額・流動性差など）を+/-で与えて rich/cheap を作る。
def make_bond(mat_years, coupon):
    mat = dt.date(SETTLE.year + mat_years, SETTLE.month, SETTLE.day)
    iss = dt.date(SETTLE.year - 3, SETTLE.month, SETTLE.day)
    return FixedRateBond(iss, mat, coupon=coupon, frequency=2, convention="ACT/ACT")


# (年限, 真の年限依存スプレッド, 銘柄固有ズレbp)
issues = [
    (2, 0.0150, +8.0),
    (4, 0.0180, -6.0),
    (5, 0.0195, +0.0),
    (7, 0.0220, +12.0),
    (9, 0.0245, -10.0),
]

rows = []
for my, base_sp, idio_bp in issues:
    b = make_bond(my, coupon=0.03 + base_sp)
    dirty = dirty_from_curve(b, SETTLE, govt_zc, base_sp + idio_bp / 1e4)
    clean = dirty - b.accrued(SETTLE)
    T, g = bond_gspread(b, clean, SETTLE)
    rows.append({"年限": round(T, 2), "G_spread_bp": g * 1e4})

df = pd.DataFrame(rows)

# 年限に対する2次フィットと残差。
coef = np.polyfit(df["年限"], df["G_spread_bp"], 2)
df["fit_bp"] = np.polyval(coef, df["年限"])
df["残差_bp"] = df["G_spread_bp"] - df["fit_bp"]
df["判定"] = np.where(df["残差_bp"] > 1.0, "cheap（割安）",
                     np.where(df["残差_bp"] < -1.0, "rich（割高）", "fair"))

print("同一発行体の rich/cheap 判定:")
print(df.round(2).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
grid = np.linspace(df["年限"].min(), df["年限"].max(), 50)
ax.plot(grid, np.polyval(coef, grid), color="C0", label="フィット曲線")
for _, r in df.iterrows():
    col = "C3" if r["残差_bp"] < -1 else ("C2" if r["残差_bp"] > 1 else "C7")
    ax.scatter(r["年限"], r["G_spread_bp"], s=55, color=col)
    ax.annotate(f"{r['残差_bp']:+.0f}", (r["年限"], r["G_spread_bp"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("残存年限（年）")
ax.set_ylabel("G-spread (bp)")
ax.set_title("同一発行体スプレッドカーブと rich/cheap（残差 bp）")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
print("フィット曲線より上（正の残差）が cheap、下（負の残差）が rich です。")
