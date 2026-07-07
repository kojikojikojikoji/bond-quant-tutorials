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
# # S2-4 演習 解答例

# %% [markdown]
# ## 準備：本編の自作関数を再掲
#
# デュアルブートストラップと評価・凸性調整の関数を最小限だけ再掲します。
# 割引係数の器は `bondlab.curve.DiscountCurve`、シングルカーブ比較用に
# `bootstrap_par` を借ります。

# %%
import numpy as np
import pandas as pd

from bondlab.curve import DiscountCurve, bootstrap_par

np.random.seed(0)


def df(curve, t):
    return float(np.ravel(curve.discount(t))[0])


def bootstrap_ois(tenors, rates):
    tenors = np.asarray(tenors, float)
    rates = np.asarray(rates, float)
    dfs = np.empty_like(tenors)
    annuity = 0.0
    for i, (_T, K) in enumerate(zip(tenors, rates)):
        dfs[i] = (1.0 - K * annuity) / (1.0 + K)
        annuity += dfs[i]
    return DiscountCurve(tenors, dfs)


def bootstrap_dual(tenors, swap_rates, disc):
    tenors = np.asarray(tenors, float)
    swap_rates = np.asarray(swap_rates, float)
    proj_nodes = [1.0]
    dfs = np.empty_like(tenors)
    annuity = 0.0
    prev_float = 0.0
    for i, (T, S) in enumerate(zip(tenors, swap_rates)):
        dfo = df(disc, T)
        annuity += dfo
        rhs = S * annuity - prev_float
        dfp_n = proj_nodes[-1] / (1.0 + rhs / dfo)
        proj_nodes.append(dfp_n)
        dfs[i] = dfp_n
        prev_float += dfo * (proj_nodes[-2] / dfp_n - 1.0)
    return DiscountCurve(tenors, dfs)


def swap_npv(fixed, tenor, disc, proj):
    years = range(1, tenor + 1)
    annuity = sum(df(disc, t) for t in years)
    floatpv = sum(df(disc, t) * (df(proj, t - 1) / df(proj, t) - 1.0) for t in years)
    return floatpv - fixed * annuity


def swap_npv_single(fixed, tenor, curve):
    years = range(1, tenor + 1)
    annuity = sum(df(curve, t) for t in years)
    floatpv = sum(df(curve, t) * (df(curve, t - 1) / df(curve, t) - 1.0) for t in years)
    return floatpv - fixed * annuity


def hw_convexity(t1, t2, a, sigma):
    def bfun(u, v):
        return (1.0 - np.exp(-a * (v - u))) / a
    b = bfun(t1, t2)
    return b / (t2 - t1) * (b * (1.0 - np.exp(-2.0 * a * t1))
                            + 2.0 * a * bfun(0.0, t1) ** 2) * sigma ** 2 / (4.0 * a)


def holee_convexity(t1, t2, sigma):
    return 0.5 * sigma ** 2 * t1 * t2


tenors = list(range(1, 11))
ois_rates = np.array([0.0300, 0.0310, 0.0320, 0.0326, 0.0330, 0.0333, 0.0335, 0.0337, 0.0339, 0.0340])
swap_rates = np.array([0.0320, 0.0330, 0.0338, 0.0344, 0.0349, 0.0353, 0.0356, 0.0359, 0.0361, 0.0363])
fixed_off = 0.030

# %% [markdown]
# ## 演習1：NPV差はベーシスか水準か
#
# 5年・固定3.0%スワップについて、(a) テナーベーシス（スワップだけを持ち上げる）
# と (b) 全体水準（OISもスワップも同じ幅で平行シフト）の2軸で、マルチと
# シングルのNPV差を表にします。

# %%
# (a) ベーシスを振る：OISは固定、スワップだけ持ち上げる。
rows_basis = []
for db_bp in [0, 20, 40, 60, 80]:
    db = db_bp / 1e4
    oc = bootstrap_ois(tenors, ois_rates)
    pc = bootstrap_dual(tenors, swap_rates + db, oc)
    sc = bootstrap_par(tenors, swap_rates + db, frequency=1)
    diff = (swap_npv(fixed_off, 5, oc, pc) - swap_npv_single(fixed_off, 5, sc)) * 1e4
    rows_basis.append({"追加ベーシス(bp)": db_bp,
                       "5yスワップ-OIS(bp)": round((swap_rates[4] - ois_rates[4]) * 1e4 + db_bp, 1),
                       "NPV差(bp)": round(diff, 3)})

ex1a = pd.DataFrame(rows_basis)
print("(a) テナーベーシスを振る")
print(ex1a.to_string(index=False))

# %%
# (b) 全体水準を振る：OISもスワップも同じ幅で平行シフト（ベーシスは一定）。
rows_level = []
for sh_bp in [-100, -50, 0, 50, 100]:
    sh = sh_bp / 1e4
    oc = bootstrap_ois(tenors, ois_rates + sh)
    pc = bootstrap_dual(tenors, swap_rates + sh, oc)
    sc = bootstrap_par(tenors, swap_rates + sh, frequency=1)
    diff = (swap_npv(fixed_off, 5, oc, pc) - swap_npv_single(fixed_off, 5, sc)) * 1e4
    rows_level.append({"水準シフト(bp)": sh_bp,
                       "5yスワップ-OIS(bp)": round((swap_rates[4] - ois_rates[4]) * 1e4, 1),
                       "NPV差(bp)": round(diff, 3)})

ex1b = pd.DataFrame(rows_level)
print("\n(b) 全体水準を平行シフト（ベーシス一定）")
print(ex1b.to_string(index=False))

# %% [markdown]
# **解釈**：NPV差を主に動かすのはベーシスです。(a) ではベーシスを広げるほど
# 差が単調に拡大します。マルチとシングルの違いは「割引をOIS（低め）で行うか
# スワップ（高め）で行うか」だけなので、OISとスワップの水準差＝ベーシスが
# ドライバーになります。一方 (b) のようにOISとスワップを同じ幅で動かすと
# ベーシスが変わらないため、割引カーブの乖離幅もほぼ一定で、NPV差は水準に
# 対してほとんど動きません（オフマーケット度合いによる二次的な変化のみ）。
# 要するに、割引カーブ分離の効き目はベーシスに支配され、金利水準そのものには
# 一次では依存しません。

# %% [markdown]
# ## 演習2：SOFR先物の凸性調整の大きさ
#
# 3か月物SOFR先物（$t_2 - t_1 = 0.25$）の凸性調整を、先物満期とボラの関数として
# Hull-White（$a=0.03$）とHo-Lee（$a\to0$）で計算し、bpで表にします。

# %%
tau = 0.25
maturities = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
vols = [0.005, 0.010, 0.015]

rows = []
for t1 in maturities:
    row = {"先物満期t1(年)": t1}
    for sigma in vols:
        hl = holee_convexity(t1, t1 + tau, sigma) * 1e4
        hw = hw_convexity(t1, t1 + tau, 0.03, sigma) * 1e4
        row[f"HL σ={sigma * 100:.1f}%(bp)"] = round(hl, 3)
        row[f"HW σ={sigma * 100:.1f}%(bp)"] = round(hw, 3)
    rows.append(row)

ex2 = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(ex2.to_string(index=False))

# %% [markdown]
# 次数の確認：Ho-Lee式 $\tfrac{1}{2}\sigma^2 t_1 t_2$ は満期にほぼ二乗
# （$t_1 t_2 = t_1(t_1+\tau) \approx t_1^2$）、ボラに二乗で効きます。数値で
# 傾きを確かめます。

# %%
sigma = 0.010
# 満期を2倍にすると調整は約4倍（二乗）になるはず。
r1 = holee_convexity(1.0, 1.0 + tau, sigma)
r2 = holee_convexity(2.0, 2.0 + tau, sigma)
print(f"満期 1→2年 の比 (≈4を期待): {r2 / r1:.3f}")
# ボラを2倍にすると調整は4倍（二乗）になるはず。
rv1 = holee_convexity(5.0, 5.0 + tau, 0.010)
rv2 = holee_convexity(5.0, 5.0 + tau, 0.020)
print(f"ボラ 1.0→2.0% の比 (=4を期待): {rv2 / rv1:.3f}")

# 平均回帰があるとHWはHo-Leeより小さい（遠い満期ほど差が開く）。
for t1 in [1.0, 5.0, 10.0]:
    hl = holee_convexity(t1, t1 + tau, sigma)
    hw = hw_convexity(t1, t1 + tau, 0.03, sigma)
    print(f"t1={t1:>4}年  HL={hl * 1e4:7.3f}bp  HW={hw * 1e4:7.3f}bp  HL-HW={ (hl - hw) * 1e4:6.3f}bp")

# %% [markdown]
# **解釈**：凸性調整は満期にほぼ二乗、ボラに二乗で増えます（比がいずれも約4）。
# 短期ゾーン（$t_1 \lesssim 1$ 年）ではサブbpで無視できますが、満期が延びると
# 急速に効き、遠い先物をカーブ入力に使うほど補正が必要です。平均回帰 $a>0$ は
# 遠い将来の金利分散の伸びを抑えるため、Hull-Whiteの調整はHo-Leeより小さく
# 出ます。差 $HL-HW$ は満期が延びるほど開き、平均回帰の効果が長期ほど強く
# 効くことを示します。実務では、SOFR先物は短期の数四半期分だけカーブに入れ、
# それより先はスワップに委ねることで、凸性調整の不確実性を小さく抑えます。
