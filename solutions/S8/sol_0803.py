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
# # S8-3 演習 解答例
#
# 日本の証券化市場（RMBS・機構債）の演習2問の解答例です。本文と同じシード・
# 同じ自作関数を再掲し、日本型プリペイメントの低感応度と、団信・超過担保の
# キャッシュフロー影響を数値で確認します。

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bondlab
from bondlab.mbs import cpr_to_smm, mbs_cashflows, weighted_average_life

print("bondlab version:", bondlab.__version__)

SEED = 20260707
rng = np.random.default_rng(SEED)


def seasoning_ramp(age, ramp=30.0, plateau=1.0):
    age = np.asarray(age, dtype=float)
    return plateau * np.minimum(1.0, age / ramp)


def refi_s_curve(incentive, k, center):
    incentive = np.asarray(incentive, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (incentive - center)))


def cpr_japan(age, incentive, center=0.006, dansin_floor=0.006):
    """本文の日本型CPR。演習用に center と団信の床を引数化する。"""
    season = seasoning_ramp(age, ramp=30.0, plateau=1.0)
    turnover = 0.045
    refi = 0.030 * refi_s_curve(incentive, k=150.0, center=center)
    return np.clip(dansin_floor + season * (turnover + refi), 0.0, 0.40)


POOL_BAL = 100.0
POOL_WAC = 0.020
POOL_WAM = 360
ages_life = np.arange(1, POOL_WAM + 1)

# %% [markdown]
# ## 演習1：低い金利感応度の理由 — 仮説3つと1つの検証
#
# **仮説（制度要因）**
#
# 1. **借換コストが高い**：抵当権抹消・再設定の登録免許税、司法書士報酬、
#    保証料等により、小さな金利差では借換が採算に乗らない。→ S字の変曲点
#    `center` が右（高インセンティブ側）にずれる。
# 2. **金利環境が平坦**：長期低金利でインセンティブの分布が0近傍に集中し、
#    S字の急峻部に届かない。→ 実効的な反応が小さい。
# 3. **商慣行・情報摩擦**：借換が生活イベント主導で、金利最適化の主体性が弱い。
#    → 非自発的成分（団信・住み替え）の比重が上がり、金利反応が薄まる。
#
# **検証するもの：仮説1（借換コストの高さ = 変曲点 center の右シフト）**
#
# `center` を大きくすると、低〜中インセンティブ域でCPRが立ち上がりにくく
# なります。同じインセンティブ経路に対して `center` を振り、平均CPRと
# WALがどう動くかを定量化します。

# %%
# 本文と同じインセンティブ経路（120ヶ月の観測に相当する将来シナリオを流用）
inc_axis = np.linspace(-0.005, 0.015, 200)
age_fixed = 60.0

fig, ax = plt.subplots(figsize=(8, 4.5))
centers = [0.002, 0.006, 0.010]
for c in centers:
    ax.plot(inc_axis * 100, cpr_japan(age_fixed, inc_axis, center=c) * 100,
            label=f"center={c*1e4:.0f}bp（借換コスト大ほど右）")
ax.set_xlabel("借換インセンティブ (bp)")
ax.set_ylabel("CPR (%)  ※月齢60ヶ月")
ax.set_title("借換コスト（変曲点 center）と金利感応度")
ax.legend()
fig.tight_layout()
plt.show()

# 変曲点を右シフトすると、持続的 +60bp シナリオでの平均CPR・WALはどう動くか
scn = np.full(POOL_WAM, 0.006)  # +60bp の持続的インセンティブ
rows = []
for c in [0.002, 0.006, 0.010, 0.014]:
    cpr = cpr_japan(ages_life, scn, center=c)
    cf = mbs_cashflows(POOL_BAL, POOL_WAC, POOL_WAM, cpr_to_smm(cpr))
    rows.append({"center(bp)": c * 1e4, "平均CPR": float(cpr.mean()),
                 "WAL(年)": weighted_average_life(cf)})
tab1 = pd.DataFrame(rows)
print(tab1.to_string(index=False))
print("\n結論: 借換コストが高い（center が右）ほど、同じ +60bp のインセンティブでも")
print("平均CPRは下がりWALは伸びる。すなわち金利感応度の低さは借換コストで説明できる。")

# %% [markdown]
# ## 演習2：団信・超過担保が投資家CFに与える影響
#
# ### (2-1) 団信 — 非自発的償還の床
#
# 団信の床 `dansin_floor` を $0$（団信なし）と $0.006$（団信あり）で切り替え、
# WALの差を見ます。団信は金利に鈍感な非自発的償還を上乗せするため、CPRの
# 下限を押し上げ、WALをわずかに短くしつつ、**金利感応度をさらに薄めます**。

# %%
scn_low = np.full(POOL_WAM, -0.002)   # インセンティブがほぼ無い（借換が起きにくい）局面
for label, floor in [("団信なし(floor=0)", 0.0), ("団信あり(floor=0.006)", 0.006)]:
    cpr = cpr_japan(ages_life, scn_low, dansin_floor=floor)
    cf = mbs_cashflows(POOL_BAL, POOL_WAC, POOL_WAM, cpr_to_smm(cpr))
    print(f"{label:24s}  平均CPR={cpr.mean():.4f}  WAL={weighted_average_life(cf):5.2f}年")

print("\n→ 低インセンティブ局面でも団信の床があるとCPRが0に落ちず、WALが短くなる。")
print("  この成分は金利にほぼ無関係なので、日本型の低感応度をさらに強める。")

# %% [markdown]
# ### (2-2) 超過担保 — 信用損失の吸収
#
# 超過担保（OC）は、裏付けプール残高を発行債券残高より多く置く構造です。
# 信用損失が発生しても、まずOC部分が吸収するため、投資家（債券側）の元本が
# 毀損しにくくなります。簡単な損失シナリオで、OCの有無による投資家元本の
# 毀損の違いを示します。
#
# 設定：発行債券残高 $100$、超過担保 $3\%$（プール残高 $103$）。プール全体で
# 累計 $2\%$ / $4\%$ の信用損失が起きたとき、投資家元本が守られるかを見ます。

# %%
bond_balance = 100.0
oc_ratio = 0.03
pool_balance = bond_balance * (1.0 + oc_ratio)   # 103

for loss_rate in [0.00, 0.02, 0.04]:
    pool_loss = pool_balance * loss_rate
    # OC（超過担保 3）が先に損失を吸収し、超えた分だけ投資家元本を毀損する
    oc_amount = pool_balance - bond_balance       # = 3
    investor_loss = max(0.0, pool_loss - oc_amount)
    investor_principal = bond_balance - investor_loss
    print(f"累計損失 {loss_rate*100:4.1f}%  プール損失={pool_loss:5.2f}  "
          f"OC吸収={min(pool_loss, oc_amount):4.2f}  "
          f"投資家元本={investor_principal:6.2f}  "
          f"{'毀損なし' if investor_loss == 0 else f'毀損 {investor_loss:.2f}'}")

print("\n→ 累計損失がOC(=3)以内なら投資家元本は額面100を維持。OCを超えて初めて毀損する。")
print("  超過担保は投資家CFの元本部分に対する信用バッファとして機能する。")
