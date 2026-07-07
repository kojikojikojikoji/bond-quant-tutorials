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
# # S3-1 演習 解答例

# %% [markdown]
# ## 準備
#
# 本編の自作関数を最小限だけ再掲する。デュレーション・コンベクシティの解析計算と
# テイラー近似を手元に置き、2銘柄を定義する。

# %%
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bondlab
from bondlab.bond import FixedRateBond

np.random.seed(0)
print("bondlab version:", bondlab.__version__)


def analytic_risk(bond, ytm, settle):
    """解析式で modified/convexity/dirty_price を計算する（本編と同一）。"""
    f = bond.frequency
    n, cf = bond.period_cashflows(settle)
    disc = (1.0 + ytm / f) ** (-n)
    pv = cf * disc
    price = pv.sum()
    t = n / f
    macaulay = (t * pv).sum() / price
    modified = macaulay / (1.0 + ytm / f)
    convexity = (cf * n * (n + 1.0) * (1.0 + ytm / f) ** (-n - 2.0)).sum() / (f ** 2) / price
    return dict(macaulay=float(macaulay), modified=float(modified),
                convexity=float(convexity), dirty_price=float(price))


def taylor_approx(risk, price0, dy):
    """1次・2次のテイラー近似価格を返す（本編と同一）。"""
    lin = price0 * (1.0 - risk["modified"] * dy)
    quad = price0 * (1.0 - risk["modified"] * dy + 0.5 * risk["convexity"] * dy ** 2)
    return dict(linear=float(lin), quadratic=float(quad))


settle = dt.date(2026, 6, 15)
instruments = {
    "10年JGB相当 (0.8%)": FixedRateBond(dt.date(2024, 6, 20), dt.date(2034, 6, 20), 0.008, 2, "ACT/ACT", 100.0),
    "10年UST相当 (4.25%)": FixedRateBond(dt.date(2024, 6, 15), dt.date(2034, 6, 15), 0.0425, 2, "ACT/ACT", 100.0),
}
y0s = {"10年JGB相当 (0.8%)": 0.009, "10年UST相当 (4.25%)": 0.043}

# %% [markdown]
# ## 演習1：近似誤差のシフト幅依存（オーダー確認）
#
# シフト幅 $|\Delta y|$ を細かく振り、1次近似・2次近似の絶対誤差（価格比、bp）を
# 両対数でプロットする。理論では1次誤差は $O(\Delta y^2)$、2次誤差は $O(\Delta y^3)$
# なので、両対数上でそれぞれ傾き2・傾き3の直線に乗るはず。

# %%
shift_bp = np.array([5, 10, 25, 50, 100, 150, 200], dtype=float)

records = []
for name, b in instruments.items():
    y0 = y0s[name]
    risk = analytic_risk(b, y0, settle)
    p0 = risk["dirty_price"]
    for s in shift_bp:
        for sign in (+1.0, -1.0):
            dy = sign * s / 1e4
            p_true = b.dirty_price(y0 + dy, settle)
            ap = taylor_approx(risk, p0, dy)
            records.append(dict(
                銘柄=name, シフトbp=sign * s,
                絶対1次誤差bp=abs(ap["linear"] - p_true) / p0 * 1e4,
                絶対2次誤差bp=abs(ap["quadratic"] - p_true) / p0 * 1e4,
            ))

err = pd.DataFrame(records)
tbl = err[err["シフトbp"] == 200.0].copy()
tbl["絶対1次誤差bp"] = tbl["絶対1次誤差bp"].round(2)
tbl["絶対2次誤差bp"] = tbl["絶対2次誤差bp"].round(3)
print("＋200bp での近似誤差（価格比 bp）")
print(tbl[["銘柄", "絶対1次誤差bp", "絶対2次誤差bp"]].to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, b) in zip(axes, instruments.items()):
    sub = err[(err["銘柄"] == name) & (err["シフトbp"] > 0)].sort_values("シフトbp")
    x = sub["シフトbp"].to_numpy()
    ax.loglog(x, sub["絶対1次誤差bp"], "o-", color="#1f77b4", label="1次近似誤差")
    ax.loglog(x, sub["絶対2次誤差bp"], "s-", color="#d62728", label="2次近似誤差")
    # 傾き2・3の参照線
    ax.loglog(x, sub["絶対1次誤差bp"].iloc[0] * (x / x[0]) ** 2, "--", color="#1f77b4", alpha=0.4, label="傾き2 参照")
    ax.loglog(x, sub["絶対2次誤差bp"].iloc[0] * (x / x[0]) ** 3, "--", color="#d62728", alpha=0.4, label="傾き3 参照")
    ax.set_title(name)
    ax.set_xlabel("利回りシフト |Δy| (bp)")
    ax.set_ylabel("絶対誤差 (価格比 bp)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# 傾きを数値で確認：log-log 回帰の傾きが 1次≒2, 2次≒3 に近い。
for name, b in instruments.items():
    sub = err[(err["銘柄"] == name) & (err["シフトbp"] > 0)].sort_values("シフトbp")
    lx = np.log(sub["シフトbp"].to_numpy())
    s1 = np.polyfit(lx, np.log(sub["絶対1次誤差bp"].to_numpy()), 1)[0]
    s2 = np.polyfit(lx, np.log(sub["絶対2次誤差bp"].to_numpy()), 1)[0]
    print(f"{name}: 1次誤差の傾き={s1:.2f}（理論2）, 2次誤差の傾き={s2:.2f}（理論3）")
    assert 1.8 < s1 < 2.2
    assert 2.7 < s2 < 3.3

# %% [markdown]
# **解釈**：両対数上で1次誤差は傾き2、2次誤差は傾き3の直線に乗る。テイラー展開の
# 打ち切り誤差が、1次近似では2次項（$\propto\Delta y^2$）、2次近似では3次項
# （$\propto\Delta y^3$）に支配されるという理論とちょうど一致する。同じ $\Delta y$ でも
# 2次近似の誤差が桁違いに小さいのは、$|\Delta y|<1$ の領域で $\Delta y^3\ll\Delta y^2$
# だからである。

# %% [markdown]
# ## 演習2：クーポン・満期に対するデュレーションとコンベクシティの依存
#
# クーポンを 0〜6%、満期を 2〜30 年で振り、修正デュレーションとコンベクシティを
# ヒートマップで可視化する。利回りは 3% 固定、決済日は発行と同時（フルターム）とする。

# %%
coupons = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
mats = np.array([2, 3, 5, 7, 10, 15, 20, 30])
y_fix = 0.03
issue = dt.date(2026, 6, 15)

dur_grid = np.zeros((len(coupons), len(mats)))
cvx_grid = np.zeros((len(coupons), len(mats)))
for i, c in enumerate(coupons):
    for j, m in enumerate(mats):
        b = FixedRateBond(issue, dt.date(2026 + int(m), 6, 15), float(c), 2, "ACT/ACT", 100.0)
        r = analytic_risk(b, y_fix, issue)
        dur_grid[i, j] = r["modified"]
        cvx_grid[i, j] = r["convexity"]

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, grid, title in zip(axes, (dur_grid, cvx_grid), ("修正デュレーション (年)", "コンベクシティ")):
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(mats)))
    ax.set_xticklabels(mats)
    ax.set_yticks(range(len(coupons)))
    ax.set_yticklabels([f"{c*100:.0f}%" for c in coupons])
    ax.set_xlabel("満期 (年)")
    ax.set_ylabel("クーポン")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
plt.tight_layout()
plt.show()

# %% [markdown]
# 満期方向の断面（クーポン 2% 固定）で、満期が延びるとデュレーションはほぼ線形、
# コンベクシティは加速度的に増えることを数値で確認する。

# %%
c_slice = 0.02
line = []
for m in mats:
    b = FixedRateBond(issue, dt.date(2026 + int(m), 6, 15), c_slice, 2, "ACT/ACT", 100.0)
    r = analytic_risk(b, y_fix, issue)
    line.append((int(m), r["modified"], r["convexity"]))
line_df = pd.DataFrame(line, columns=["満期(年)", "修正Dur", "コンベクシティ"])
line_df["修正Dur"] = line_df["修正Dur"].round(3)
line_df["コンベクシティ"] = line_df["コンベクシティ"].round(2)
print(f"クーポン {c_slice*100:.0f}% 固定・利回り {y_fix*100:.0f}% の満期依存")
print(line_df.to_string(index=False))

# ゼロクーポン（0%）のマコーレーデュレーション ≒ 残存年数を確認。
bz = FixedRateBond(issue, dt.date(2026 + 10, 6, 15), 0.0, 2, "ACT/ACT", 100.0)
rz = analytic_risk(bz, y_fix, issue)
print(f"\n10年ゼロクーポンのマコーレーDur = {rz['macaulay']:.4f} 年（残存10年に一致）")
assert abs(rz["macaulay"] - 10.0) < 1e-6

# 同一満期ではクーポンが低いほどデュレーションが長い。
d_low = analytic_risk(FixedRateBond(issue, dt.date(2036, 6, 15), 0.01, 2, "ACT/ACT", 100.0), y_fix, issue)["modified"]
d_high = analytic_risk(FixedRateBond(issue, dt.date(2036, 6, 15), 0.06, 2, "ACT/ACT", 100.0), y_fix, issue)["modified"]
print(f"\n10年債: クーポン1% Dur={d_low:.3f} > クーポン6% Dur={d_high:.3f}")
assert d_low > d_high

# %% [markdown]
# **解釈**：低クーポン・長満期でデュレーションもコンベクシティも大きくなる。理由は
# キャッシュフローの重心が後ろへ寄るからである。クーポンが低いほど回収が満期の
# 元本償還に集中し（極限のゼロクーポンでは重心＝満期）、満期が延びればその重心自体が
# 遠くなる。デュレーションは重心までの距離、コンベクシティはその距離の二乗的な広がりを
# 測るので、後ろ重心の債券ほど両者がそろって大きくなる。
