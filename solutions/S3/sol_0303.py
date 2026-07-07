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
# # S3-3 演習 解答例

# %% [markdown]
# ## 準備：本編の KRD 関数を再掲
#
# 本編と同一の実装を、演習で使う分だけ再掲します。

# %%
import numpy as np
import pandas as pd

from bondlab.curve import DiscountCurve, bootstrap_par
from bondlab.analytics import bump_curve

np.random.seed(20260507)


def price_portfolio(curve, cashflows):
    cf = np.asarray(cashflows, dtype=float)
    return float(np.sum(cf[:, 1] * curve.discount(cf[:, 0])))


def key_rate_dv01(curve, cashflows, tenors, size=1e-4, width=None):
    v0 = price_portfolio(curve, cashflows)
    out = []
    for tau in tenors:
        bumped = bump_curve(curve, float(tau), size, width=width)
        out.append(-(price_portfolio(bumped, cashflows) - v0))
    return np.array(out)


def key_rate_duration(curve, cashflows, tenors, size=1e-4, width=None):
    v0 = price_portfolio(curve, cashflows)
    return key_rate_dv01(curve, cashflows, tenors, size=size, width=width) / (v0 * size)


# 本編と同じ実データカーブ。
par_df = pd.read_csv("data/samples/synthetic_ust_par_curve.csv")
grid = np.arange(1.0, 31.0)
par = np.interp(grid, par_df["tenor"].to_numpy(), par_df["par_yield"].to_numpy())
curve = bootstrap_par(grid, par, frequency=1)

key_tenors = np.array([2.0, 5.0, 10.0, 20.0, 30.0])


def zero_bond(t, mv):
    """時価 mv の t 年ゼロクーポン債を (年数, CF) で返す。"""
    return np.array([[t, mv / curve.discount(t)]])


# %% [markdown]
# ## 演習1：5年+20年バーベル vs 2年+30年バーベル
#
# ### 1. 時価配分
#
# 同一時価 100・同一デュレーション 10 の条件 $A+B=100,\ 5A+20B=1000$ を解きます。

# %%
Amat = np.array([[1.0, 1.0], [5.0, 20.0]])
bvec = np.array([100.0, 10.0 * 100.0])
mv5, mv20 = np.linalg.solve(Amat, bvec)
print(f"5年+20年バーベル 時価配分: 5年 = {mv5:.4f}, 20年 = {mv20:.4f}")

# 比較対象：本編の 2年+30年バーベル。
mv2, mv30 = np.linalg.solve(np.array([[1.0, 1.0], [2.0, 30.0]]), bvec)
print(f"2年+30年バーベル 時価配分: 2年 = {mv2:.4f}, 30年 = {mv30:.4f}")

barbell_5_20 = np.vstack([zero_bond(5.0, mv5), zero_bond(20.0, mv20)])
barbell_2_30 = np.vstack([zero_bond(2.0, mv2), zero_bond(30.0, mv30)])

# %% [markdown]
# ### 2. KRD プロファイルの比較

# %%
krd_5_20 = key_rate_duration(curve, barbell_5_20, key_tenors, width=None)
krd_2_30 = key_rate_duration(curve, barbell_2_30, key_tenors, width=None)

tbl = pd.DataFrame({
    "キーテナー(年)": key_tenors,
    "5年+20年KRD": np.round(krd_5_20, 4),
    "2年+30年KRD": np.round(krd_2_30, 4),
})
print(tbl.to_string(index=False))
print(f"\n5年+20年 KRD合計 = {krd_5_20.sum():.4f}")
print(f"2年+30年 KRD合計 = {krd_2_30.sum():.4f}")

# KRD の広がり（デュレーション加重の分散）で分散度を測る。
def krd_dispersion(krd, tenors):
    w = krd / krd.sum()
    center = np.sum(w * tenors)
    return float(np.sqrt(np.sum(w * (tenors - center) ** 2)))


print(f"5年+20年 分散指標 = {krd_dispersion(krd_5_20, key_tenors):.3f}")
print(f"2年+30年 分散指標 = {krd_dispersion(krd_2_30, key_tenors):.3f}")

# %% [markdown]
# 両バーベルとも KRD 合計は約 10 年で一致します。KRD の広がり（分散指標）は
# **2年+30年バーベルの方が大きい**——端に振った分だけ、テナー方向の分散が大きく
# なります。5年+20年バーベルは中央寄りで、ブレットに近い形です。

# %% [markdown]
# ### 3. スティープナーでの損益

# %%
steepener_bp = key_tenors - 10.0  # 10年中心の回転
pdv01_5_20 = key_rate_dv01(curve, barbell_5_20, key_tenors, width=None)
pdv01_2_30 = key_rate_dv01(curve, barbell_2_30, key_tenors, width=None)

pnl_5_20 = -np.sum(pdv01_5_20 * steepener_bp)
pnl_2_30 = -np.sum(pdv01_2_30 * steepener_bp)
print(f"スティープナー損益  5年+20年バーベル = {pnl_5_20:+.5f}")
print(f"スティープナー損益  2年+30年バーベル = {pnl_2_30:+.5f}")

# %% [markdown]
# **結論**：どちらのバーベルもスティープナーで損失ですが、損失が小さいのは
# **5年+20年バーベル**です。端に振った 2年+30年バーベルは 30 年側のパーシャルDV01
# が大きく、そこに最大のレート上昇（+20bp）が当たるため損失が膨らみます。
# 5年+20年は年限が中央寄りで、回転中心 10 年からの距離（＝当たる $\Delta r$ の
# 大きさ）が小さいため、ツイストへの感応度が低いのです。バーベルを端へ振るほど
# スティープナーに弱く、フラットナーに強くなります。

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(key_tenors))
w = 0.38
ax.bar(x - w / 2, krd_5_20, w, label="5年+20年バーベル", color="#3b6ea5")
ax.bar(x + w / 2, krd_2_30, w, label="2年+30年バーベル", color="#c0654a")
ax.set_xticks(x)
ax.set_xticklabels([f"{t:.0f}年" for t in key_tenors])
ax.set_xlabel("キーテナー")
ax.set_ylabel("KRD（年）")
ax.set_title("端に振るほど KRD の分散が大きい")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 演習2：バケット粒度と KRD の安定性
#
# ### 1〜2. 粒度別の KRD 合計（ノード単体バンプ）

# %%
grids = {
    "粗い [2,10,30]": np.array([2.0, 10.0, 30.0]),
    "標準 [2,5,10,20,30]": np.array([2.0, 5.0, 10.0, 20.0, 30.0]),
    "細かい [2,3,5,7,10,15,20,25,30]": np.array([2, 3, 5, 7, 10, 15, 20, 25, 30], dtype=float),
}

barbell = barbell_2_30  # 本編の 2年+30年バーベルを対象にする
rows = []
for label, g in grids.items():
    krd = key_rate_duration(curve, barbell, g, width=None)
    rows.append({"グリッド": label, "本数": len(g), "KRD合計(単体)": round(krd.sum(), 4)})

print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# 2年+30年バーベルは KRD が 2 年と 30 年に集中します。どのグリッドも 2 年・30 年を
# キーテナーに含むため、ノード単体バンプでも合計はほぼ 10 に届きます。取りこぼしが
# 起きるのは、**リスクの所在をキーテナーが覆っていないとき**です。それを次で
# 作ります。

# %%
# 8年+25年バーベル（同一時価・同一デュレーション 10）で、粗いグリッドが
# リスク年限を覆えないケースを作る。
mv8, mv25 = np.linalg.solve(np.array([[1.0, 1.0], [8.0, 25.0]]), bvec)
barbell_8_25 = np.vstack([zero_bond(8.0, mv8), zero_bond(25.0, mv25)])

coarse = np.array([2.0, 10.0, 30.0])  # 8年も25年も覆っていない
krd_node = key_rate_duration(curve, barbell_8_25, coarse, width=None)
print(f"8年+25年バーベル / 粗いグリッド[2,10,30]")
print(f"  ノード単体 KRD合計 = {krd_node.sum():.4f}（10 に不足）")

# %% [markdown]
# 8 年も 25 年もキーテナーに無いため、単体バンプでは 8 年・25 年ノードの動きを
# 拾えず、KRD 合計が 10 に届きません（**バケットの取りこぼし**）。

# %% [markdown]
# ### 3. 三角バンプによる改善

# %%
for w in [3.0, 8.0, 12.0]:
    krd_tri = key_rate_duration(curve, barbell_8_25, coarse, width=w)
    print(f"  三角バンプ w={w:4.1f}  KRD合計 = {krd_tri.sum():.4f}")

# %% [markdown]
# 三角バンプは、キーテナー 2・10・30 のテントが 8 年・25 年へ**染み出す**ことで、
# 覆えていなかった年限のゼロレート変化を拾います。幅 $w$ を広げると隣接テントの
# 重なりが増え、8 年・25 年が近傍のキーからより多く配分され、KRD 合計が 10 へ
# 近づきます。ただし幅を広げすぎると各テナーの局所性が失われ、隣接パーシャルDV01
# の相関が高まって解釈が難しくなります。粒度と幅は、リスクの所在を覆いつつ
# 局所性を保てる範囲で選ぶのが実務的な設計です。

# %% [markdown]
# ### 補足：KRD と PCA の橋渡し（本編理論の数値確認）
#
# カーブ変動データから主成分（水準・スロープ・曲率）を取り出し、バーベルとブレットの
# KRD ベクトルを主成分へ射影して「主成分感応度」を比べます。ブレットは水準（第1主成分）
# に、バーベルはスロープ（第2主成分）に相対的に強く反応することを確認します。

# %%
from bondlab.analytics import pca

# 合成のカーブ変動（水準・スロープ・曲率の3因子＋ノイズ）を生成する。
tenors_full = key_tenors
n_days = 500
level = np.random.normal(0, 1.0, n_days)
slope = np.random.normal(0, 0.6, n_days)
curv = np.random.normal(0, 0.3, n_days)
load_level = np.ones_like(tenors_full)
load_slope = (tenors_full - tenors_full.mean()) / tenors_full.std()
load_curv = ((tenors_full - tenors_full.mean()) ** 2)
load_curv = (load_curv - load_curv.mean()) / load_curv.std()
changes = (np.outer(level, load_level) + np.outer(slope, load_slope)
           + np.outer(curv, load_curv) + np.random.normal(0, 0.05, (n_days, len(tenors_full))))

res = pca(changes)
pcs = res["eigenvectors"]  # 列が各主成分
print("主成分の寄与率:", np.round(res["explained_ratio"][:3], 3))

krd_bullet = key_rate_duration(curve, zero_bond(10.0, 100.0), key_tenors, width=None)
krd_barbell = key_rate_duration(curve, barbell_2_30, key_tenors, width=None)

for name, k in [("ブレット", krd_bullet), ("バーベル", krd_barbell)]:
    proj = k @ pcs[:, :3]
    print(f"{name} の主成分感応度  第1(水準)={proj[0]:+.3f}  第2(スロープ)={proj[1]:+.3f}  第3(曲率)={proj[2]:+.3f}")

# %% [markdown]
# KRD ベクトルを主成分基底へ射影すると、テナー別の感応度が「水準・スロープ・曲率」
# への感応度へ変換されます。ブレットとバーベルは水準（デュレーション）への感応度は
# そろえてありますが、スロープ・曲率への感応度が異なります。これが、並行シフトでは
# 同じでもスティープナー（スロープ変化）で損益が分かれる理由を、主成分の言葉で
# 言い換えたものです。KRD（テナー基底）と PCA 感応度（主成分基底）は、負荷行列で
# 相互に変換できる同じリスクの2つの見方だと分かります。
