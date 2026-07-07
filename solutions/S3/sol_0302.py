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
# # S3-2 演習 解答例

# %% [markdown]
# ## 準備：本編の部品を再掲
#
# 本編で自作したDV01・ヘッジの部品を最小限だけ再掲します。解析DV01は
# `bondlab.analytics.duration_convexity` を部品に使います。

# %%
import datetime as dt

import numpy as np
import pandas as pd

import bondlab
from bondlab import bond as blbond
from bondlab.analytics import duration_convexity

np.random.seed(0)
print("bondlab version:", bondlab.__version__)


def dv01_bump(bond, ytm, settle, bump=1e-4, method="central"):
    """利回りをバンプして再評価した数値DV01（額面100あたり）。中心/片側差分。"""
    p0 = bond.dirty_price(ytm, settle)
    p_up = bond.dirty_price(ytm + bump, settle)
    if method == "central":
        p_dn = bond.dirty_price(ytm - bump, settle)
        deriv = (p_up - p_dn) / (2.0 * bump)
    elif method == "forward":
        deriv = (p_up - p0) / bump
    else:
        raise ValueError(f"未知の差分方式: {method!r}")
    return -deriv * 1e-4


def portfolio_dv01(positions, settle):
    """(ラベル, 債券, 利回り, 面額) の列から合計DV01と明細を返す。"""
    rows, total = [], 0.0
    for label, bond, ytm, face in positions:
        dc = duration_convexity(bond, ytm, settle)
        pos = (face / 100.0) * dc["dv01"]
        total += pos
        rows.append({"銘柄": label, "面額": face, "DV01/100": round(dc["dv01"], 6),
                     "ポジションDV01": round(pos, 4)})
    return total, pd.DataFrame(rows)


def solve_two_bond_hedge(target, hedge_a, hedge_b, settle, t_ref):
    """水準・傾き中立の2本連立を解き、ヘッジ2本の面額を返す。"""
    def dv01_of(item):
        T, bond, ytm, face = item
        return T, (face / 100.0) * duration_convexity(bond, ytm, settle)["dv01"]

    Tt, dt_dv01 = dv01_of(target)
    Ta, da = dv01_of(hedge_a)
    Tb, db = dv01_of(hedge_b)
    A = np.array([[da, db], [da * (Ta - t_ref), db * (Tb - t_ref)]])
    rhs = np.array([-dt_dv01, -dt_dv01 * (Tt - t_ref)])
    n_a, n_b = np.linalg.solve(A, rhs)
    return n_a * 100.0, n_b * 100.0


def pnl_reprice(bond, y0, y1, settle):
    """決済日固定で利回りだけ y0→y1 に動かした dirty price 差（額面100あたり）。"""
    return bond.dirty_price(y1, settle) - bond.dirty_price(y0, settle)


# %% [markdown]
# ## 演習1：5年ロングを2年+30年でヘッジし残存PnLを分解
#
# バタフライ変化が2番目に大きい日を選び、5年債ロングを2年+30年で水準・傾き中立に
# ヘッジして、非平行シフトの残存PnLを分解します。

# %%
panel = pd.read_csv("data/samples/synthetic_ust_par_panel.csv")
wide = panel.pivot(index="date", columns="tenor", values="par_yield").sort_index()

chg = wide.diff().dropna()
butterfly = (chg[2.0] - 2.0 * chg[10.0] + chg[30.0]).abs()
# 大きい順に並べ、2番目の日を取る。
day1 = butterfly.sort_values(ascending=False).index[1]
dates = list(wide.index)
day0 = dates[dates.index(day1) - 1]

shift_bp = (wide.loc[day1] - wide.loc[day0]) * 1e4
print(f"基準日: {day0}   シフト日: {day1}（バタフライ2番手）")
print("年限別シフト (bp):")
print(shift_bp.round(3).to_string())

# %% [markdown]
# 前日水準でパー発行の実在風銘柄（5年・2年・30年）を作ります。

# %%
issue_d = dt.date(*map(int, day0.split("-")))


def make_par_bond(tenor_years):
    y0 = float(wide.loc[day0, tenor_years])
    mat = dt.date(issue_d.year + tenor_years, issue_d.month, issue_d.day)
    return blbond.FixedRateBond(issue_d, mat, coupon=y0, frequency=2), y0


bond5, y5_0 = make_par_bond(5)
bond2, y2_0 = make_par_bond(2)
bond30, y30_0 = make_par_bond(30)
settle_d = issue_d

y5_1 = y5_0 + float(shift_bp[5.0]) * 1e-4
y2_1 = y2_0 + float(shift_bp[2.0]) * 1e-4
y30_1 = y30_0 + float(shift_bp[30.0]) * 1e-4

FACE5 = 10_000_000.0

# 基準年限を対象年限（5年）に取る。対象の傾き寄与 (T_t - t_ref) = 0 となり、
# 第2式（傾き中立）の右辺 -DV01_target*(T_t - t_ref) = 0 になる。
face2, face30 = solve_two_bond_hedge(
    target=(5.0, bond5, y5_0, FACE5),
    hedge_a=(2.0, bond2, y2_0, 100.0),
    hedge_b=(30.0, bond30, y30_0, 100.0),
    settle=settle_d, t_ref=5.0,
)
print(f"ヘッジ面額: 2年 = {face2:,.0f} / 30年 = {face30:,.0f}")

positions = [("5年ロング", bond5, y5_0, FACE5),
             ("2年ヘッジ", bond2, y2_0, face2),
             ("30年ヘッジ", bond30, y30_0, face30)]
total, tbl = portfolio_dv01(positions, settle_d)
print(tbl.to_string(index=False))
print(f"合計DV01: {total:,.6f}")
assert abs(total) < 1e-6

# %% [markdown]
# 決済日を固定して再評価し、残存PnLを水準・傾き・曲率に分解します。基準 $T_0=5$ の
# まわりで年限 $\{2,5,30\}$ の利回り変化を $a+b(T-5)+c(T-5)^2$ に当てはめます。

# %%
pnl5 = FACE5 / 100.0 * pnl_reprice(bond5, y5_0, y5_1, settle_d)
pnl2 = face2 / 100.0 * pnl_reprice(bond2, y2_0, y2_1, settle_d)
pnl30 = face30 / 100.0 * pnl_reprice(bond30, y30_0, y30_1, settle_d)
pnl_unhedged, pnl_hedged = pnl5, pnl5 + pnl2 + pnl30
print(f"ヘッジ前PnL: {pnl_unhedged:,.2f} / ヘッジ後PnL: {pnl_hedged:,.2f}")

Tn = np.array([2.0, 5.0, 30.0])
dy = np.array([shift_bp[2.0], shift_bp[5.0], shift_bp[30.0]]) * 1e-4
V = np.column_stack([np.ones_like(Tn), (Tn - 5.0), (Tn - 5.0) ** 2])
a, b, c = np.linalg.solve(V, dy)

dv01_pos = {T: face / 100.0 * duration_convexity(bond, y, settle_d)["dv01"]
            for T, (bond, y, face) in {
                2.0: (bond2, y2_0, face2),
                5.0: (bond5, y5_0, FACE5),
                30.0: (bond30, y30_0, face30)}.items()}
level = -sum(dv01_pos[T] * a for T in Tn) / 1e-4
slope = -sum(dv01_pos[T] * b * (T - 5.0) for T in Tn) / 1e-4
curv = -sum(dv01_pos[T] * c * (T - 5.0) ** 2 for T in Tn) / 1e-4

decomp = pd.DataFrame({
    "要因": ["水準", "傾き", "曲率", "一次合計", "実測ヘッジ後", "残差"],
    "PnL": [level, slope, curv, level + slope + curv, pnl_hedged,
            pnl_hedged - (level + slope + curv)],
})
print(decomp.to_string(index=False, formatters={"PnL": "{:,.2f}".format}))

# 水準・傾きはヘッジで消えるので、残存PnLの主因は曲率。
assert abs(level) < abs(curv) + 1e-9
assert abs(slope) < abs(curv) + 1e-9
print("\n基準年限を対象年限に取ると傾き中立式の右辺がゼロになり、"
      "残存PnLは曲率が主因になります。")

# %% [markdown]
# **解釈**：基準年限 $T_0$ を対象年限（5年）に取ると、対象自身の傾き寄与 $(T_t-T_0)=0$
# なので、傾き中立の連立式は「2本のヘッジだけで傾きゼロ」を課す形になります。水準と
# 傾きの2成分を消した結果、非平行シフトのうち曲率成分だけが残存PnLの主因として残ります。

# %% [markdown]
# ## 演習2：バンプ幅・差分方式の定量化
#
# 満期30年・クーポン4.458%の債券で、バンプ幅×差分方式の4通りの相対誤差を表にし、
# バンプ幅を振った両対数プロットの傾きから精度の次数を確認します。

# %%
issue = dt.date(2026, 1, 2)
mat30 = dt.date(2056, 1, 2)
bond = blbond.FixedRateBond(issue, mat30, coupon=0.04458, frequency=2)
y0 = 0.04458
settle = issue
dv01_exact = duration_convexity(bond, y0, settle)["dv01"]

rows = []
for bump in (1e-4, 10e-4):
    for method in ("central", "forward"):
        approx = dv01_bump(bond, y0, settle, bump=bump, method=method)
        rows.append({
            "バンプ幅": f"{int(round(bump * 1e4))}bp",
            "差分方式": "中心" if method == "central" else "片側",
            "数値DV01": round(approx, 8),
            "相対誤差": abs(approx - dv01_exact) / dv01_exact,
        })
err_tbl = pd.DataFrame(rows)
print(f"解析DV01 = {dv01_exact:.8f}")
print(err_tbl.to_string(index=False, formatters={"相対誤差": "{:.3e}".format}))

# 片側10bp が最悪、中心1bp が最良になることを確認。
worst = err_tbl.loc[err_tbl["相対誤差"].idxmax()]
best = err_tbl.loc[err_tbl["相対誤差"].idxmin()]
assert worst["差分方式"] == "片側" and worst["バンプ幅"] == "10bp"
assert best["差分方式"] == "中心" and best["バンプ幅"] == "1bp"

# %% [markdown]
# バンプ幅を $0.5$〜$50\,\mathrm{bp}$ で振り、両対数プロットの傾きを最小二乗で推定します。
# 中心差分は傾き $\approx 2$（$O(h^2)$）、片側差分は傾き $\approx 1$（$O(h)$）になります。

# %%
import matplotlib.pyplot as plt

hs = np.array([0.5, 1, 2, 5, 10, 20, 50]) * 1e-4
fig, ax = plt.subplots(figsize=(8, 5))
slopes = {}
for method, marker, color in [("central", "o", "#1f77b4"), ("forward", "s", "#d62728")]:
    errs = np.array([abs(dv01_bump(bond, y0, settle, bump=h, method=method) - dv01_exact)
                     for h in hs])
    ax.loglog(hs * 1e4, errs, marker + "-", color=color,
              label="中心差分" if method == "central" else "片側差分")
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    slopes[method] = slope
ax.set_xlabel("バンプ幅 (bp)")
ax.set_ylabel("解析DV01との絶対誤差")
ax.set_title("差分方式・バンプ幅とDV01誤差の次数（30年債）")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

print(f"両対数の傾き: 中心差分 = {slopes['central']:.2f} / 片側差分 = {slopes['forward']:.2f}")
assert abs(slopes["central"] - 2.0) < 0.3
assert abs(slopes["forward"] - 1.0) < 0.3

# %% [markdown]
# **解釈**：中心差分は打ち切り誤差が $O(h^2)$ なので、両対数プロットの傾きが約2になり、
# バンプ幅を小さくすると誤差が急速に減ります。片側差分は $O(h)$ で傾きが約1です。
# ただしバンプを小さくしすぎると丸め誤差が支配的になるため、実務では中心差分で
# $1$〜$5\,\mathrm{bp}$ 程度が安定します。
