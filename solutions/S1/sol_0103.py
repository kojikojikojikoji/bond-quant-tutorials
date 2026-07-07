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
# # S1-3 演習 解答例

# %% [markdown]
# ## 準備：価格関数とソルバー
#
# 本編の自作関数を最小限だけ再掲し、演習で使う。day count は
# `bondlab.daycount.year_fraction` を借りる。

# %%
import datetime as dt

import numpy as np
import pandas as pd

import bondlab
from bondlab import bond as blbond
from bondlab import daycount

np.random.seed(0)


def add_months(d, months):
    m0 = d.month - 1 + months
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    last = 31 if m == 12 else (dt.date(y, m + 1, 1) - dt.timedelta(days=1)).day
    return dt.date(y, m, min(d.day, last))


def coupon_schedule(issue, maturity, freq):
    step = 12 // freq
    dates, d = [maturity], maturity
    while d > issue:
        d = add_months(d, -step)
        dates.append(d)
    return [x for x in sorted(set(dates)) if x > issue]


def cashflows(schedule, coupon, freq, face=100.0):
    cpn = face * coupon / freq
    flows = [(d, cpn) for d in schedule]
    dl, cl = flows[-1]
    flows[-1] = (dl, cl + face)
    return flows


def surrounding(issue, schedule, settlement):
    prev = nxt = None
    for d in [issue] + list(schedule):
        if d <= settlement:
            prev = d
        elif nxt is None:
            nxt = d
    return prev, nxt


def dirty_price(flows, ytm, freq, prev, nxt, settlement, conv):
    full = daycount.year_fraction(prev, nxt, conv)
    rem = daycount.year_fraction(settlement, nxt, conv)
    w = rem / full if full > 0 else 0.0
    future = [(d, c) for d, c in flows if d > settlement]
    return sum(c * (1.0 + ytm / freq) ** (-(w + j)) for j, (_d, c) in enumerate(future))


def accrued(coupon, freq, face, prev, nxt, settlement, conv):
    full = daycount.year_fraction(prev, nxt, conv)
    part = daycount.year_fraction(prev, settlement, conv)
    cpn = face * coupon / freq
    return cpn * (part / full) if full > 0 else 0.0


def price_and_deriv(flows, ytm, freq, prev, nxt, settlement, conv):
    full = daycount.year_fraction(prev, nxt, conv)
    rem = daycount.year_fraction(settlement, nxt, conv)
    w = rem / full if full > 0 else 0.0
    future = [(d, c) for d, c in flows if d > settlement]
    base = 1.0 + ytm / freq
    price = deriv = 0.0
    for j, (_d, c) in enumerate(future):
        expo = w + j
        price += c * base ** (-expo)
        deriv += c * (-expo / freq) * base ** (-(expo + 1.0))
    return price, deriv


def ytm_newton(target, freq, flows, prev, nxt, settlement, conv, guess=0.03,
               tol=1e-12, maxiter=100):
    y = guess
    for _ in range(maxiter):
        price, deriv = price_and_deriv(flows, y, freq, prev, nxt, settlement, conv)
        diff = price - target
        if abs(diff) < tol:
            return y
        y = y - diff / deriv
    return y


# %% [markdown]
# ## 演習1：複利YTMと単利最終利回りの差
#
# 3銘柄の市場 clean price から、複利 YTM と単利最終利回りを求め、その差（bp）を
# 区分と結びつける。

# %%
settle = dt.date(2026, 9, 10)
universe = [
    ("JGB 5年 (低クーポン)", dt.date(2024, 6, 20), dt.date(2029, 6, 20), 0.004, 99.10),
    ("JGB 10年 (中期)", dt.date(2023, 9, 20), dt.date(2033, 9, 20), 0.008, 97.50),
    ("JGB 20年 (高クーポン)", dt.date(2020, 6, 20), dt.date(2040, 6, 20), 0.015, 101.20),
]

rows = []
for name, iss, mat, cpn, mkt in universe:
    sc = coupon_schedule(iss, mat, 2)
    fl = cashflows(sc, cpn, 2)
    pv, nx = surrounding(iss, sc, settle)
    accr = accrued(cpn, 2, 100.0, pv, nx, settle, "ACT/ACT")
    ytm = ytm_newton(mkt + accr, 2, fl, pv, nx, settle, "ACT/ACT", guess=0.01)
    T = daycount.year_fraction(settle, mat, "ACT/365F")
    simple = (cpn * 100.0 + (100.0 - mkt) / T) / mkt
    kind = "プレミアム" if cpn > ytm else ("ディスカウント" if cpn < ytm else "パー")
    rows.append({
        "銘柄": name,
        "複利YTM%": round(ytm * 100, 4),
        "単利利回り%": round(simple * 100, 4),
        "差(bp)": round((simple - ytm) * 1e4, 2),
        "区分": kind,
    })

ex1 = pd.DataFrame(rows)
print(ex1.to_string(index=False))

# %% [markdown]
# **解釈**：ディスカウント債（市場価格 < 100）では、満期償還益を単純に年割りする
# 単利利回りが複利 YTM より高めに出る（差が正）。複利 YTM は償還益にも割引・
# 再投資を効かせるため、単利より控えめになるからである。価格が額面から離れる
# ほど（残存が長く乖離が大きいほど）差は開きやすい。プレミアム債では逆に単利が
# 低めに出て差の符号が反転する。要するに単利と複利の差は「価格が額面からどちら側に
# どれだけ離れているか」で決まる。

# %% [markdown]
# ## 演習2：価格変化の実測と一次近似のずれ

# %%
name, iss, mat, cpn, _ = universe[1]
sc = coupon_schedule(iss, mat, 2)
fl = cashflows(sc, cpn, 2)
pv, nx = surrounding(iss, sc, settle)
accr10 = accrued(cpn, 2, 100.0, pv, nx, settle, "ACT/ACT")


def clean_of(yv):
    return dirty_price(fl, yv, 2, pv, nx, settle, "ACT/ACT") - accr10


y0 = 0.012
p0, dp = price_and_deriv(fl, y0, 2, pv, nx, settle, "ACT/ACT")
p0_clean = p0 - accr10

rows = []
for bp in (-200, -50, 50, 200):
    yv = y0 + bp / 1e4
    actual = clean_of(yv)
    approx = p0_clean + dp * (yv - y0)  # 接線（一次近似）
    rows.append({
        "Δy(bp)": bp,
        "実際の価格": round(actual, 4),
        "一次近似": round(approx, 4),
        "実際ΔP": round(actual - p0_clean, 4),
        "近似ΔP": round(approx - p0_clean, 4),
        "近似誤差": round(approx - actual, 4),
    })

ex2 = pd.DataFrame(rows)
print(ex2.to_string(index=False))

# %%
import matplotlib.pyplot as plt

ys = np.linspace(-0.008, 0.032, 200)
prices = np.array([clean_of(v) for v in ys])
tangent = p0_clean + dp * (ys - y0)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ys * 100, prices, label="実際の価格（凸）")
ax.plot(ys * 100, tangent, "--", label="一次近似（接線）")
ax.scatter([y0 * 100], [p0_clean], color="k", zorder=5, label="基準点 1.2%")
ax.set_xlabel("利回り (%)")
ax.set_ylabel("clean price")
ax.set_title("実際の価格変化 vs 一次近似")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：価格-利回り曲線は下に凸なので、接線は常に曲線より下側を通る。よって
# 一次近似は「近似誤差 = 近似 − 実際 $\le 0$」、すなわち実際の価格を過小評価する。
# 利回り低下側（Δy<0, 値上がり）では実際の値上がりを小さく見積もり、利回り上昇側
# （Δy>0, 値下がり）では値下がりを大きく見積もる。どちらも投資家に不利な方向へ
# 誤るわけではなく、凸性は実際の価格が近似より常に有利（高い）側にあることを意味する。
# ずれの幅は $|\Delta y|$ の二乗にほぼ比例して増え、$\pm200$bp では $\pm50$bp より
# 大きい。この二次の項がコンベクシティ（S1-4）である。

# %% [markdown]
# bondlab との一致も確認しておく。

# %%
for name, iss, mat, cpn, mkt in universe:
    bb = blbond.FixedRateBond(iss, mat, cpn, 2, "ACT/ACT")
    sc = coupon_schedule(iss, mat, 2)
    fl = cashflows(sc, cpn, 2)
    pv, nx = surrounding(iss, sc, settle)
    accr = accrued(cpn, 2, 100.0, pv, nx, settle, "ACT/ACT")
    ytm_s = ytm_newton(mkt + accr, 2, fl, pv, nx, settle, "ACT/ACT", guess=0.01)
    assert abs(ytm_s - bb.yield_from_price(mkt, settle)) < 1e-9
print("bondlab との YTM 一致を確認しました")
