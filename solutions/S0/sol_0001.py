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
# # S0-1 演習 解答例

# %% [markdown]
# ## 演習1：市場別・月別の営業日数

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

calendars = {
    "Tokyo": ql.Japan(),
    "New York": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    "London": ql.UnitedKingdom(),
}
rows = []
for m in range(1, 13):
    start = ql.Date(1, m, 2026)
    end = ql.Date(ql.Date.endOfMonth(start).dayOfMonth(), m, 2026)
    row = {}
    for label, cal in calendars.items():
        row[label] = sum(
            1 for n in range(start.serialNumber(), end.serialNumber() + 1)
            if cal.isBusinessDay(ql.Date(n))
        )
    rows.append(row)

bd = pd.DataFrame(rows, index=[f"2026-{m:02d}" for m in range(1, 13)])
print(bd)

ax = bd.plot.bar(figsize=(11, 4))
ax.set_ylabel("営業日数")
ax.set_title("2026年 月別営業日数")
plt.tight_layout()
plt.show()

# %% [markdown]
# 差が大きい月は、各国固有の連休が入る月。日本は1月（正月）・5月（GW）、
# 米国は休日が分散、英国は Bank Holiday が5月・8月に入る。年末年始は3市場とも
# 休むため差が出にくい。

# %% [markdown]
# ## 演習2：規約変換の年数依存

# %%
from bondlab import rates

t = np.arange(1, 31)
r_annual = 0.04
r_semi = rates.convert_rate(r_annual, t, "annual", "semiannual")
r_cont = rates.convert_rate(r_annual, t, "annual", "continuous")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, (r_annual - r_semi) * 1e4, label="年1回 − 半年 (bp)")
ax.plot(t, (r_annual - r_cont) * 1e4, label="年1回 − 連続 (bp)")
ax.set_xlabel("年数 t")
ax.set_ylabel("レート差 (bp)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# 規約間のレート差は年数にほぼ依存しない。割引係数を保つ変換のうち、
# 連続 ⇔ 離散の関係 $r_c = m\ln(1+r_m/m)$ は $t$ を含まないため。数値上わずかな
# ぶれが出るのは丸めの範囲。年1回 4% は半年複利で約 3.96%、連続で約 3.92%。
