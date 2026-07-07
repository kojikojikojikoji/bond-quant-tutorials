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
# # S1-1 演習 解答例

# %%
import numpy as np

from bondlab import rates

conventions = ["annual", "semiannual", "quarterly", "monthly", "continuous"]

# 本文のスクラッチ関数を再掲（解答を自己完結させるため）。
_FREQ = {"annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12, "continuous": None}


def freq(convention):
    return _FREQ[convention]


def df_scratch(rate, t, convention):
    m = freq(convention)
    r = np.asarray(rate, dtype=float)
    tt = np.asarray(t, dtype=float)
    if m is None:
        return np.exp(-r * tt)
    return (1.0 + r / m) ** (-m * tt)


def rate_scratch(df, t, convention):
    m = freq(convention)
    d = np.asarray(df, dtype=float)
    tt = np.asarray(t, dtype=float)
    if m is None:
        return -np.log(d) / tt
    return m * (d ** (-1.0 / (m * tt)) - 1.0)


def to_cont_scratch(rate, convention):
    m = freq(convention)
    r = np.asarray(rate, dtype=float)
    if m is None:
        return r
    return m * np.log1p(r / m)


def from_cont_scratch(rate_c, convention):
    m = freq(convention)
    rc = np.asarray(rate_c, dtype=float)
    if m is None:
        return rc
    return m * (np.exp(rc / m) - 1.0)


# %% [markdown]
# ## 演習1：往復誤差ゼロのテスト
#
# 半年複利 5% を連続・月複利へ変換し、半年複利へ戻す。規約変換は $t$ 非依存なので、
# `to_continuous`／`from_continuous` を経由すれば任意の規約対を往復できる。

# %%
r_semi = 0.05

# スクラッチ: 半年 -> 連続 -> 半年、半年 -> 月 -> 半年。
rc = to_cont_scratch(r_semi, "semiannual")
back_via_cont = from_cont_scratch(rc, "semiannual")

r_month = from_cont_scratch(to_cont_scratch(r_semi, "semiannual"), "monthly")
back_via_month = from_cont_scratch(to_cont_scratch(r_month, "monthly"), "semiannual")

print(f"半年複利         : {r_semi:.10%}")
print(f"連続複利へ       : {float(rc):.10%}")
print(f"月複利へ         : {float(r_month):.10%}")
print(f"連続経由で復元   : {float(back_via_cont):.10%}")
print(f"月経由で復元     : {float(back_via_month):.10%}")

assert abs(float(back_via_cont) - r_semi) < 1e-12
assert abs(float(back_via_month) - r_semi) < 1e-12

# bondlab.rates でも同じ往復を行い、結果がスクラッチと一致することを確認。
b_cont = rates.convert_rate(r_semi, 1.0, "semiannual", "continuous")
b_month = rates.convert_rate(r_semi, 1.0, "semiannual", "monthly")
b_back_cont = rates.convert_rate(b_cont, 1.0, "continuous", "semiannual")
b_back_month = rates.convert_rate(b_month, 1.0, "monthly", "semiannual")

assert abs(float(b_back_cont) - r_semi) < 1e-12
assert abs(float(b_back_month) - r_semi) < 1e-12
assert abs(float(b_cont) - float(rc)) < 1e-12
assert abs(float(b_month) - float(r_month)) < 1e-12
print("往復誤差ゼロ・スクラッチと bondlab の一致を確認")

# %% [markdown]
# ## 演習2：同一 DF からの逆算
#
# $t=4$、$DF=0.85$ を固定し、5規約のゼロレートを逆算する。逆算レートから DF を
# 再計算すると 0.85 に戻り、レートは複利回数が多いほど小さくなる。

# %%
t = 4.0
df_fixed = 0.85

recovered = {}
for conv in conventions:
    r = float(rates.rate_from_discount(df_fixed, t, conv))
    recovered[conv] = r
    # 再計算した DF が元に戻る。
    df_again = float(rates.discount_factor(r, t, conv))
    assert abs(df_again - df_fixed) < 1e-12
    # スクラッチと一致。
    assert abs(float(rate_scratch(df_fixed, t, conv)) - r) < 1e-12

print(f"{'規約':12s} {'ゼロレート(%)':>14s}")
for conv in conventions:
    print(f"{conv:12s} {recovered[conv]*100:14.6f}")

# 大小関係 annual >= semiannual >= quarterly >= monthly >= continuous。
order = [recovered[c] for c in conventions]
assert all(order[i] >= order[i + 1] - 1e-12 for i in range(len(order) - 1))
print("DF 再計算での復元・規約の大小関係を確認")
