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
# # S7-3 演習 解答例
#
# Merton構造モデルの演習2問の解答例です。本文と同じ `bondlab.credit` を使い、
# 合成企業のデータで DD/PD を計算します。

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bondlab import credit

np.random.seed(0)

r, T = 0.01, 1.0

# %% [markdown]
# ## 演習1：2社のDD/PD計算と格付整合
#
# A・B・C の3社について、観測量（株式時価総額・株式ボラ・負債＝default point）から
# 企業価値・資産ボラを逆算し、DD と PD を求めます。想定格付順序は A > C > B です。

# %%
firms = {
    "A社（高格付）": dict(equity=3200.0, equity_vol=0.24, debt=2600.0),
    "C社（中格付）": dict(equity=1200.0, equity_vol=0.35, debt=2000.0),
    "B社（低格付）": dict(equity=500.0, equity_vol=0.50, debt=1800.0),
}

results = {}
for name, f in firms.items():
    sol = credit.solve_asset(f["equity"], f["equity_vol"], f["debt"], T, r)
    V, sV = sol["asset"], sol["asset_vol"]
    dd = credit.distance_to_default(V, sV, f["debt"], T, r)
    pd_val = credit.merton_pd(V, sV, f["debt"], T, r)
    results[name] = dict(V=V, sV=sV, dd=dd, pd=pd_val)
    print(f"{name}: V={V:8.1f}  σV={sV:.4f}  DD={dd:6.3f}  PD={pd_val * 100:.4f}%")

dds = [results[n]["dd"] for n in firms]
# 想定順序 A > C > B と整合するか
assert dds[0] > dds[1] > dds[2]
print("\nDD の順序 A > C > B が想定格付順序と整合しました")

# %% [markdown]
# **考察**：DD の順序（A 約6.0 > C 約3.5 > B 約2.2）は想定格付順序と一致します。
# 序列を作っている主因はレバレッジと株式ボラの組み合わせです。もし C社の株式ボラ
# だけを大きく引き上げると、逆算後の資産ボラが上がって DD が縮み、C社が B社を
# 下回って順序が崩れます。DD はレバレッジとボラを1つの尺度に束ねているため、
# 片方の入力だけでも順序は容易に逆転します。

# %% [markdown]
# ## 演習2：資産ボラ・負債水準を振ってDDの感応度
#
# A社の逆算後の企業価値 V を固定し、(a) 資産ボラ、(b) 負債水準 を振って DD の
# 動きを見ます。

# %%
VA = results["A社（高格付）"]["V"]   # 固定する企業価値

# (a) 資産ボラを振る（負債は A社の値に固定）
DA = firms["A社（高格付）"]["debt"]
vol_grid = np.linspace(0.10, 0.40, 60)
dd_vol = [credit.distance_to_default(VA, sv, DA, T, r) for sv in vol_grid]
pd_vol = [credit.merton_pd(VA, sv, DA, T, r) for sv in vol_grid]

# (b) 負債を振る（資産ボラは A社の逆算値に固定）
sVA = results["A社（高格付）"]["sV"]
debt_grid = np.linspace(0.5 * VA, 0.95 * VA, 60)
dd_debt = [credit.distance_to_default(VA, sVA, d, T, r) for d in debt_grid]
pd_debt = [credit.merton_pd(VA, sVA, d, T, r) for d in debt_grid]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(vol_grid, dd_vol, color="#a13d3d")
axes[0].set_xlabel("資産ボラ σV")
axes[0].set_ylabel("DD")
axes[0].set_title("(a) 資産ボラに対する DD")
axes[0].grid(alpha=0.3)

axes[1].plot(debt_grid / VA, dd_debt, color="#2f6f4f")
axes[1].set_xlabel("負債 / 企業価値（D/V）")
axes[1].set_ylabel("DD")
axes[1].set_title("(b) 負債水準に対する DD")
axes[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()

print(f"σV 0.10→0.40 で DD: {dd_vol[0]:.2f} → {dd_vol[-1]:.2f}")
print(f"D/V 0.50→0.95 で DD: {dd_debt[0]:.2f} → {dd_debt[-1]:.2f}")

# %% [markdown]
# **考察**：DD は資産ボラの上昇に対してほぼ反比例で急落します（分母が $\sigma_V$
# のため）。負債の増加に対しては対数レバレッジ $\ln(V/D)$ を通じて低下しますが、
# 対数のぶん反応はなだらかで、D が V に近づく領域で初めて DD が急速に潰れます。
# PD は $N(-\mathrm{DD})$ なので、DD が数標準偏差ある領域では入力を動かしても
# ほぼ 0 のままで、DD が 2〜3 を割ってから初めて目に見えて立ち上がります。
# したがって高格付社では、ボラの急変が負債の小幅増より PD を効かせます。
