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
# # S6-1 演習 解答例
#
# FRA と金利スワップの演習 2 問の解答例です。本文の合成カーブを再構築して使います。

# %%
import numpy as np
import matplotlib.pyplot as plt

from bondlab.curve import bootstrap_par
from bondlab.pricing import par_swap_rate, swap_annuity

np.random.seed(0)

years = [1, 2, 3, 4, 5, 6, 7]
ois_rates = [0.015, 0.019, 0.022, 0.024, 0.025, 0.026, 0.0265]
proj_rates = [0.018, 0.023, 0.026, 0.028, 0.029, 0.0295, 0.030]
disc_full = bootstrap_par(years, ois_rates, frequency=1)
proj_full = bootstrap_par(years, proj_rates, frequency=1)


def swap_schedule(maturity, frequency=1):
    """満期 maturity（年）まで、年 frequency 回払いの支払時点を等間隔で生成する。"""
    n = int(round(maturity * frequency))
    return np.arange(1, n + 1, dtype=float) / frequency


# %% [markdown]
# ## 演習 1：シングルカーブ vs マルチカーブのパーレート差
#
# 各年限について、シングルカーブ（割引・推計とも推計カーブ）とマルチカーブ
# （割引＝OIS、推計＝推計カーブ）のパースワップレートを求め、差をベーシスで測ります。

# %%
tenors = list(range(1, 8))
single_rates = []
multi_rates = []
for m in tenors:
    tt = swap_schedule(float(m))
    single_rates.append(par_swap_rate(proj_full, proj_full, tt))  # 割引=推計=proj
    multi_rates.append(par_swap_rate(disc_full, proj_full, tt))   # 割引=OIS, 推計=proj

single_rates = np.array(single_rates)
multi_rates = np.array(multi_rates)
basis_bp = (multi_rates - single_rates) * 1e4

for m, s, mc, b in zip(tenors, single_rates, multi_rates, basis_bp):
    print(f"{m}年  シングル={s:.5%}  マルチ={mc:.5%}  差={b:+.3f}bp")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tenors, basis_bp, "o-", color="#1f77b4")
ax.axhline(0.0, ls="--", color="gray")
ax.set_xlabel("年限（年）")
ax.set_ylabel("マルチ − シングル（bp）")
ax.set_title("割引カーブ差し替えによるパーレートのベーシス")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察
#
# 変動レッグの PV は $\sum \tau_i P^{d}(t_i) F_i^{p}$ で、割引カーブ $P^{d}$ を OIS に
# 差し替えると各キャッシュフローの現在価値ウェイトが変わります。OIS レートが推計
# カーブより低いこの設定では割引係数が大きく、後半（金利の高い区間の）フォワードを
# 相対的に重く評価するため、マルチカーブのパーレートはシングルカーブよりわずかに
# 高く出ます。1年スワップは支払いが1回だけで、パーレートがその区間フォワードに一致し
# 割引カーブに依存しないため差はゼロです。2年以降は複数キャッシュフローの割引ウェイト
# が効いて差が開き、数分の1〜数bp の範囲に収まります。OIS と推計カーブの上下関係が
# 逆なら符号も反転します。

# %%
assert abs(basis_bp[0]) < 1e-9      # 1年スワップは割引カーブに依存せず差はゼロ
assert np.all(basis_bp[1:] > 0)     # OIS < 推計なので2年以降はマルチが上回る
print("演習1 チェックを通過しました")

# %% [markdown]
# ## 演習 2：年限別 DV01 の構造
#
# `proj_full` をシングルカーブとして、年限 1〜10 年のパースワップの DV01 を
# 解析式 $\text{DV01} = N \cdot A \cdot 10^{-4}$ で計算します。アニュイティ
# $A = \sum \tau_i P(t_i)$ には bondlab の `swap_annuity` を使います。

# %%
notional = 100_000_000.0  # 1億円
tenors10 = list(range(1, 11))
dv01_list = []
annuity_list = []
for m in tenors10:
    tt = swap_schedule(float(m))
    A = swap_annuity(proj_full, tt)
    annuity_list.append(A)
    dv01_list.append(notional * A * 1e-4)

for m, A, d in zip(tenors10, annuity_list, dv01_list):
    print(f"{m:2d}年  A={A:7.4f}  DV01={d:>12,.0f} 円/bp")

increments = np.diff(dv01_list)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(tenors10, np.array(dv01_list) / 1e3, "o-", color="#d62728")
ax1.set_xlabel("年限（年）")
ax1.set_ylabel("DV01（千円/bp）")
ax1.set_title("年限別 DV01")
ax1.grid(alpha=0.3)

ax2.bar(tenors10[1:], increments / 1e3, color="#2ca02c", alpha=0.7)
ax2.set_xlabel("年限（年）")
ax2.set_ylabel("DV01 の増分（千円/bp）")
ax2.set_title("年限を1年延ばしたときの DV01 増分")
ax2.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察
#
# DV01 はアニュイティ $A = \sum_i \tau_i P(t_i)$ に比例します。年限を1年延ばすと
# 項 $\tau_{n+1} P(t_{n+1})$ が1つ増えるので DV01 は必ず増えます。ただし追加される
# 割引係数 $P(t_{n+1})$ は年限が伸びるほど小さくなる（遠い将来ほど現在価値が薄い）
# ため、増分は逓減します。右図のとおり DV01 は単調増加しつつ、1年あたりの伸びは
# 年限とともに縮みます。長期スワップほど金利感応度は大きいが、限界的な寄与は
# 割引で目減りする、という構造です。

# %%
assert np.all(np.array(dv01_list) > 0)
assert np.all(increments > 0)          # 単調増加
assert np.all(np.diff(increments) < 0)  # 増分は逓減
print("演習2 チェックを通過しました")
