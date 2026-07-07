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
# # S0-2 演習 解答例

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bondlab import data

us10 = data.load_sample("synthetic_us10y")
jp10 = data.load_sample("synthetic_jp10y")
raw = pd.concat({"US10Y": us10, "JP10Y": jp10}, axis=1)

# %% [markdown]
# ## 演習1：欠損処理の方針比較（除外 vs 前値埋め）

# %%
drop = raw.dropna(how="any")
spread_drop = (drop["US10Y"] - drop["JP10Y"]) * 100

ffilled = raw.ffill().dropna(how="any")
spread_ffill = (ffilled["US10Y"] - ffilled["JP10Y"]) * 100

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(spread_drop, bins=60, alpha=0.6, label="除外", density=True)
ax.hist(spread_ffill, bins=60, alpha=0.6, label="前値埋め", density=True)
ax.set_xlabel("スプレッド (bp)")
ax.legend()
plt.tight_layout()
plt.show()

print("除外    件数", len(spread_drop), "std", round(spread_drop.std(), 1))
print("前値埋め件数", len(spread_ffill), "std", round(spread_ffill.std(), 1))

# %% [markdown]
# 前値埋めは、片方だけ休場の日に古い値を「観測」として足すため、同じ値が
# 連続して分布の山を高くし、日次変化の見かけのボラティリティを下げる。
# 分析用途では除外方式の方が裾を歪めない。

# %% [markdown]
# ## 演習2：cache_only の挙動

# %%
try:
    data.fred_series("NONEXISTENT", cache_only=True)
except RuntimeError as e:
    print("想定どおり例外:", str(e)[:60], "...")

# 鍵のある環境では以下でキャッシュが作られ、次回から cache_only で同じ結果が返る:
#   s1 = data.fred_series("DGS10")                 # 実取得＋キャッシュ
#   s2 = data.fred_series("DGS10", cache_only=True) # キャッシュのみ
#   assert s1.equals(s2)

# %% [markdown]
# ## 演習3：スプレッド変動が大きい月・月次ボラ

# %%
d = drop.copy()
d["spread_bp"] = spread_drop
monthly_vol = d["spread_bp"].diff().groupby(d.index.to_period("M")).std()
top5 = monthly_vol.sort_values(ascending=False).head(5)
print("日次スプレッド変化の月次ボラ 上位5:")
print(top5.round(2))
