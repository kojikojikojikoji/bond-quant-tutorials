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
# # S2-1 演習 解答例

# %% [markdown]
# ## 準備：ブートストラップとパー逆算
#
# 本編の自作関数を最小限だけ再掲する。剥ぎ取り本体は `bondlab.curve.bootstrap_par` を使い、
# パーレートの逆算だけ手元に置く。

# %%
import numpy as np
import pandas as pd

import bondlab
from bondlab import curve as blcurve

np.random.seed(0)
print("bondlab version:", bondlab.__version__)


def par_from_curve(dfs, freq=1):
    """割引係数から各テナーのパーレート c_n = f(1 - DF_n)/Σ_{k≤n} DF_k を逆算する。"""
    dfs = np.asarray(dfs, dtype=float)
    return freq * (1.0 - dfs) / np.cumsum(dfs)


# 合成パーカーブを読み、年1回払いグリッドへ線形補間する。
sample = pd.read_csv("data/samples/synthetic_ust_par_curve.csv")
grid = np.arange(1.0, 31.0)
par_base = np.interp(grid, sample["tenor"].to_numpy(), sample["par_yield"].to_numpy())

# %% [markdown]
# ## 演習1：ゼロ・パー・フォワードの描画と QuantLib 突合
#
# 合成カーブを剥ぎ取り、3本のレートを1枚に描く。あわせて `bootstrap_par` と、時間軸を
# 揃えた QuantLib `DiscountCurve` の割引係数一致を確認する。

# %%
curve = blcurve.bootstrap_par(grid, par_base, frequency=1, interp="log_linear")

zero = curve.zero_rate(grid)
par = par_from_curve(curve.dfs[1:], freq=1)
fwd_1y = np.array([float(curve.forward_rate(t, t + 1.0)) for t in grid])

summary = pd.DataFrame({
    "テナー(年)": grid.astype(int),
    "パー%": np.round(par * 100, 4),
    "ゼロ%": np.round(zero * 100, 4),
    "1年ﾌｫﾜｰﾄﾞ%": np.round(fwd_1y * 100, 4),
})
print(summary.to_string(index=False))

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(grid, par * 100, "o-", label="パーレート", color="#1f77b4")
ax.plot(grid, zero * 100, "s-", label="ゼロレート", color="#d62728")
ax.plot(grid, fwd_1y * 100, "^-", label="1年フォワード", color="#2ca02c")
ax.set_xlabel("テナー (年)")
ax.set_ylabel("レート (%)")
ax.set_title("合成パーカーブから剥ぎ取った3本のレート")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# QuantLib との突合：ノードを $t\times365$ 日に置き、year fraction を整数年へ揃える。

# %%
import QuantLib as ql

day_count = ql.Actual365Fixed()
today = ql.Date(15, 5, 2026)
ql.Settings.instance().evaluationDate = today

node_dates = [today + ql.Period(int(round(t * 365)), ql.Days) for t in curve.times]
ql_curve = ql.DiscountCurve(node_dates, list(curve.dfs), day_count)

node_diff = max(abs(curve.discount(t) - ql_curve.discount(float(t))) for t in curve.times[1:])
interior = np.array([1.4, 6.5, 13.3, 24.7])
interior_diff = max(abs(curve.discount(t) - ql_curve.discount(float(t))) for t in interior)
print(f"ノード割引係数の最大差 : {node_diff:.2e}")
print(f"内部点割引係数の最大差 : {interior_diff:.2e}")
assert node_diff < 1e-12 and interior_diff < 1e-12
print("時間軸を揃えれば QuantLib DiscountCurve と一致します（< 1e-12）")

# %% [markdown]
# **解釈**：右上がりカーブでは 1年フォワード > ゼロ > パー の順序が全域で保たれる。フォワードは
# 限界金利なので長期の傾きを最も敏感に映し、パーはクーポン全体の平均利回りなので最も鈍い。
# 時間軸（year fraction）を揃えれば log-linear 補間は QuantLib と機械精度で一致し、日付ベースで
# 出る $\sim 10^{-4}$ の差は補間ロジックではなく閏日由来だと確かめられる。

# %% [markdown]
# ## 演習2：パー利回りショックの長期ゾーンへの伝播
#
# 2年テナーのパー利回りに $+10\,\mathrm{bp}$ を与え、ベースラインとのゼロレート差（bp）を
# テナー横断でプロットする。比較のため 10 年テナーへのショックも重ねる。

# %%
def shock_zero_diff(base_par, grid, shock_tenor, shock_bp=10.0):
    """指定テナーのパー利回りに +shock_bp のショックを与え、ゼロレート差(bp)を返す。"""
    bumped = base_par.copy()
    idx = int(np.where(np.isclose(grid, shock_tenor))[0][0])
    bumped[idx] += shock_bp / 1e4
    c0 = blcurve.bootstrap_par(grid, base_par, frequency=1)
    c1 = blcurve.bootstrap_par(grid, bumped, frequency=1)
    return (c1.zero_rate(grid) - c0.zero_rate(grid)) * 1e4  # bp


diff_2y = shock_zero_diff(par_base, grid, 2.0)
diff_10y = shock_zero_diff(par_base, grid, 10.0)

shock_tbl = pd.DataFrame({
    "テナー(年)": grid.astype(int),
    "2年ショック→ゼロ差(bp)": np.round(diff_2y, 3),
    "10年ショック→ゼロ差(bp)": np.round(diff_10y, 3),
})
print(shock_tbl[(shock_tbl["テナー(年)"] <= 12)].to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(grid, diff_2y, "o-", label="2年パーに +10bp", color="#1f77b4")
ax.plot(grid, diff_10y, "s-", label="10年パーに +10bp", color="#d62728")
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel("テナー (年)")
ax.set_ylabel("ゼロレート差 (bp)")
ax.set_title("パー利回り +10bp ショックのゼロカーブへの伝播")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：ショックを与えたテナーより手前のゼロレートは変わらない（前進代入はまだショック点の
# パー利回りに触れていないため）。ショック点で差が立ち上がり、そこを起点に長期ゾーンへ繰り越される。
# 前進代入 $DF(t_i) = (1 - (c_i/f)\sum_{k<i} DF(t_k))/(1 + c_i/f)$ では、ショック点で汚れた
# $DF$ が running sum に混ざり、以降の全テナーの分子を通じて伝播し続ける。伝播した差は減衰せず
# 残るが、長期ではゼロレートが年数 $t$ で割られるため、割引係数の絶対ずれが同じでも bp 表示の
# ゼロ差は薄まって見える。2年ショックが 10年ショックより広い範囲に効くのは、起点が手前にある
# ぶん繰り越される区間が長いからである。

# %% [markdown]
# ショック点より手前のゼロレートが不変であることを数値で確認する。

# %%
before = grid < 2.0
assert np.max(np.abs(diff_2y[before])) < 1e-9
print("2年ショックは 1年ゾーンのゼロを動かしません（前進代入の因果性）")
