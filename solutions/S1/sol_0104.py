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
# # S1-4 演習 解答例
#
# クリーン/ダーティ価格と市場慣行の国際比較（`nb_0104_clean_dirty`）の演習解答。

# %%
import datetime as dt

import numpy as np

from bondlab import daycount
from bondlab.bond import FixedRateBond

np.random.seed(0)


def _add_months(d: dt.date, months: int) -> dt.date:
    """月加算（月末は各月の末日に丸める）。"""
    m0 = d.month - 1 + months
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    last = 31 if m == 12 else (dt.date(y, m + 1, 1) - dt.timedelta(days=1)).day
    return dt.date(y, m, min(d.day, last))


def coupon_dates(issue: dt.date, maturity: dt.date, freq: int):
    """満期から逆算したクーポン日の列。"""
    step = 12 // freq
    dates, d = [maturity], maturity
    while d > issue:
        d = _add_months(d, -step)
        dates.append(d)
    return [x for x in sorted(set(dates)) if x > issue]


def surrounding(dates, issue, settle):
    """受渡日を挟む (直前利払日, 次利払日)。"""
    prev, nxt = None, None
    for d in [issue] + dates:
        if d <= settle:
            prev = d
        elif nxt is None:
            nxt = d
    return prev, nxt


def accrued_isma(prev, nxt, settle, coupon, freq, face=100.0):
    """ISMA（Bond）変種の経過利子。"""
    d_full = (nxt - prev).days
    d_part = (settle - prev).days
    return (face * coupon / freq) * d_part / d_full if d_full > 0 else 0.0


def accrued_isda(prev, settle, coupon, face=100.0):
    """ISDA 変種の経過利子。"""
    return face * coupon * daycount.year_fraction(prev, settle, "ACT/ACT")


def simple_yield(price, annual_coupon, years):
    """JGB 単利最終利回り。"""
    return (annual_coupon + (100.0 - price) / years) / price


# %% [markdown]
# ## 演習1：同一 CF の利回りを UST 慣行 / JGB 慣行で比較
#
# クーポン 3%・残存 7 年・半年利払い、クリーン価格 96.0、受渡日は利払日ちょうど
# （経過利子ゼロ）。UST 慣行（半年複利 YTM）と JGB 慣行（単利）の利回り差を出す。

# %%
def compare_conventions(coupon, price, years, settle_on_coupon):
    """半年複利 YTM と単利利回りを返す。settle_on_coupon は利払日ちょうど。"""
    issue = _add_months(settle_on_coupon, -12)          # 発行日は適当に過去へ
    maturity = _add_months(settle_on_coupon, int(years * 12))
    bond = FixedRateBond(issue, maturity, coupon, frequency=2, convention="ACT/ACT")
    y_compound = bond.yield_from_price(price, settle_on_coupon)  # UST 慣行
    y_simple = simple_yield(price, 100.0 * coupon, years)        # JGB 慣行
    return y_compound, y_simple


settle = dt.date(2026, 6, 15)
for cpn in [0.03, 0.01]:
    y_c, y_s = compare_conventions(cpn, 96.0, 7.0, settle)
    diff_bp = (y_c - y_s) * 1e4
    print(f"クーポン {cpn * 100:.1f}%: 半年複利YTM = {y_c * 100:.5f}%  "
          f"単利 = {y_s * 100:.5f}%  差 = {diff_bp:+.3f} bp")

print()
print("クーポンを 3% → 1% に下げると、複利効果が小さくなり両慣行の差は縮む。")

# %% [markdown]
# ## 演習2：ISDA vs ISMA の経過利子差を複数受渡日で比較
#
# 本文の 2% 半年債について、直前利払日から次利払日まで 10 日刻みで受渡日を動かし、
# ISMA と ISDA の経過利子差を集める。差が最大の受渡日と、期首でのゼロ一致を確認する。

# %%
issue = dt.date(2024, 6, 15)
maturity = dt.date(2029, 6, 15)
coupon, freq = 0.02, 2
dates = coupon_dates(issue, maturity, freq)
prev, nxt = surrounding(dates, issue, dt.date(2026, 9, 10))

offsets = list(range(0, (nxt - prev).days + 1, 10))
settlements = [prev + dt.timedelta(days=k) for k in offsets]
diffs = np.array([
    accrued_isda(prev, s, coupon) - accrued_isma(prev, nxt, s, coupon, freq)
    for s in settlements
])

k_max = int(np.argmax(np.abs(diffs)))
print(f"クーポン期間: {prev} → {nxt}（{(nxt - prev).days} 日）")
print(f"差が最大の受渡日: {settlements[k_max]}  差 = {diffs[k_max]:.8f}")
print(f"期首の差 = {diffs[0]:.2e}   期末の差 = {diffs[-1]:.2e}")

# 期首（受渡日=直前利払日）で差はゼロ。期が進むほど差は単調に開く。
assert abs(diffs[0]) < 1e-12
assert abs(diffs[k_max]) == np.max(np.abs(diffs))
ai_isma_full = accrued_isma(prev, nxt, nxt, coupon, freq)
ai_isda_full = accrued_isda(prev, nxt, coupon)
print(f"次利払日での ISMA = {ai_isma_full:.8f}（= 半期クーポン 1.0）")
print(f"次利払日での ISDA = {ai_isda_full:.8f}（暦年 365 基準のため 1.0 を僅かに超える）")
