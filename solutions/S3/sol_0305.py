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
# # S3-5 演習 解答例
#
# 本編 `notebooks/S03_リスク指標/nb_0305_var_es.py` の演習2問の解答です。VaR / ES / Kupiec 検定は
# `bondlab.risk` を用い、本編の慣行（PnL は正が利益、VaR / ES は正の損失）に合わせます。

# %%
import datetime as _dt

import numpy as np
import pandas as pd

import bondlab
from bondlab import risk as blrisk
from bondlab.bond import FixedRateBond

SEED = 20260707
print("bondlab version:", bondlab.__version__)

# %% [markdown]
# ## 演習1：VaR が劣加法性を破る反例と、ES では破れないこと
#
# 独立な2資産 $A, B$ を、確率 $p=0.03$ で損失 $L=100$、確率 $1-p$ で利益 $c=1$ とします
# （PnL は正が利益）。$\alpha=0.95$ なので $1-\alpha=0.05 > p$ です。単独の下側 5% 分位点は
# デフォルトを捉えないため単独 VaR は損失にならず、合算すると「少なくとも一方がデフォルト」の
# 確率が $1-(1-p)^2 = 0.0591 > 0.05$ に上がって分位点が損失域に落ちる、という筋書きです。
#
# この反例は**離散分布に確率の塊（atom）を置く**構成です。ヒストリカル推定量
# `bondlab.risk.historical_es` は「$\le -\mathrm{VaR}$ の標本平均」で ES を近似するため、
# 分位点がちょうど atom に重なると裾の質量を取りすぎ、離散分布では正しい ES を返しません
# （連続分布・大標本では問題になりません）。そこで**厳密な離散分位点・裾平均**を小さな補助関数で
# 実装し、VaR はライブラリと突合、ES は厳密値で評価します。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `disc_var(values, probs, alpha)` | 値, 確率, 信頼水準 | float | 離散分布の厳密 VaR（下側 (1-α) 分位点を正の損失で） |
# | `disc_es(values, probs, alpha)` | 値, 確率, 信頼水準 | float | 最悪 (1-α) の質量を平均した厳密 ES |

# %%
def disc_var(values, probs, alpha=0.95):
    """離散分布の厳密 VaR。累積確率が (1-alpha) に達する最小値を正の損失で返す。"""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(probs)[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, (1 - alpha) - 1e-12))
    return -float(v[idx])


def disc_es(values, probs, alpha=0.95):
    """離散分布の厳密 ES。最悪 (1-alpha) の確率質量にわたる平均損失を返す。"""
    order = np.argsort(values)          # 値の昇順＝損失の大きい順
    v, w = np.asarray(values)[order], np.asarray(probs)[order]
    remaining, tail_loss = 1 - alpha, 0.0
    for vi, wi in zip(v, w):
        take = min(wi, remaining)       # 境界の atom は必要な分だけ切り出す
        tail_loss += (-vi) * take
        remaining -= take
        if remaining <= 1e-15:
            break
    return tail_loss / (1 - alpha)


p, L, c = 0.03, 100.0, 1.0
alpha = 0.95

# 単独資産 A の全事象（PnL, 確率）。
single = np.array([c, -L])
single_w = np.array([1 - p, p])

# 合算 A+B の全事象（独立なので確率は積）。
combo = np.array([2 * c, c - L, -2 * L])                 # 両生存 / 片方デフォルト / 両デフォルト
combo_w = np.array([(1 - p) ** 2, 2 * p * (1 - p), p ** 2])

var_A = disc_var(single, single_w, alpha)
var_AB = disc_var(combo, combo_w, alpha)
es_A = disc_es(single, single_w, alpha)
es_AB = disc_es(combo, combo_w, alpha)

# VaR は塊があってもライブラリの推定量と一致する（大標本で突合）。
rng = np.random.default_rng(SEED)
sA = rng.choice(single, size=2_000_000, p=single_w)
sAB = rng.choice(combo, size=2_000_000, p=combo_w)
assert np.isclose(var_A, blrisk.historical_var(sA, alpha), atol=1e-9)
assert np.isclose(var_AB, blrisk.historical_var(sAB, alpha), atol=1e-9)

print(f"P(少なくとも一方デフォルト) = {1 - (1 - p) ** 2:.4f}  （> 1-alpha = {1 - alpha:.2f}）")
print("（VaR は bondlab.risk.historical_var と一致することを確認済み）\n")

tbl = pd.DataFrame({
    "指標": ["95%VaR", "95%ES"],
    "単独A": [var_A, es_A],
    "単独B": [var_A, es_A],           # B は A と同分布
    "A+B": [var_AB, es_AB],
    "単独の和 A+B": [2 * var_A, 2 * es_A],
})
print(tbl.to_string(index=False,
      formatters={col: "{:.3f}".format for col in ["単独A", "単独B", "A+B", "単独の和 A+B"]}))

print("\n--- 判定 ---")
print(f"VaR(A+B) = {var_AB:.3f}  vs  VaR(A)+VaR(B) = {2 * var_A:.3f}"
      f"  → {'劣加法性を破る（VaRが大きい）' if var_AB > 2 * var_A else '破らない'}")
print(f"ES(A+B)  = {es_AB:.3f}  vs  ES(A)+ES(B)  = {2 * es_A:.3f}"
      f"  → {'劣加法性を保つ' if es_AB <= 2 * es_A + 1e-9 else '破る'}")

assert var_AB > 2 * var_A          # VaR は劣加法性を破る
assert es_AB <= 2 * es_A + 1e-9    # ES は劣加法性を保つ

# %% [markdown]
# 単独 VaR は $-c=-1$（5% 分位点が利益 $+c$ に当たり、損失ではない）で、その和は $-2$。
# 一方 A+B の 95% VaR は片方デフォルトの損失 $L-c=99$ に跳ね上がり、
# $\mathrm{VaR}(A+B) \gg \mathrm{VaR}(A)+\mathrm{VaR}(B)$ となって劣加法性が破れます。
# 同じ設定でも ES は裾全体を平均するため $\mathrm{ES}(A+B)\le\mathrm{ES}(A)+\mathrm{ES}(B)$ を保ち、
# コヒーレントなリスク尺度であることが確認できます。

# %% [markdown]
# ## 演習2：日次損益のバックテスト（パラメトリック vs ヒストリカル）
#
# 本編と同じ手順で日次損益 `pnl` を再構成し、パラメトリック 99% VaR をしきい値に取り直して
# Kupiec 検定にかけ、ヒストリカル VaR の場合と比較します。

# %%
panel = pd.read_csv("data/samples/synthetic_ust_par_panel.csv", parse_dates=["date"])
wide = panel.pivot(index="date", columns="tenor", values="par_yield").sort_index()
tenor_grid = wide.columns.to_numpy(dtype=float)
dates = [d.date() for d in wide.index]

holdings = [
    ("2年", 2.0, FixedRateBond(_dt.date(2024, 1, 2), _dt.date(2028, 1, 2), 0.030, 2)),
    ("5年", 5.0, FixedRateBond(_dt.date(2024, 1, 2), _dt.date(2031, 1, 2), 0.035, 2)),
    ("10年", 10.0, FixedRateBond(_dt.date(2024, 1, 2), _dt.date(2036, 1, 2), 0.040, 2)),
    ("30年", 30.0, FixedRateBond(_dt.date(2024, 1, 2), _dt.date(2056, 1, 2), 0.045, 2)),
]
weights = np.full(len(holdings), 1.0 / len(holdings))

values = np.zeros(len(dates))
for i, d in enumerate(dates):
    row = wide.iloc[i].to_numpy(dtype=float)
    pv = 0.0
    for j, (_lbl, ten, bond) in enumerate(holdings):
        y = float(np.interp(ten, tenor_grid, row))
        pv += weights[j] * bond.clean_price(y, d)
    values[i] = pv

pnl = np.diff(values)
n = pnl.size
print(f"日次損益: {n}日,  平均 {pnl.mean():+.4f},  標準偏差 {pnl.std(ddof=1):.4f}")

# %%
rows = []
for name, var in [("ヒストリカル", blrisk.historical_var(pnl, 0.99)),
                  ("パラメトリック", blrisk.parametric_var(pnl, 0.99))]:
    exc = int(np.sum(pnl < -var))
    kp = blrisk.kupiec_pof(exc, n, 0.99)
    rows.append((name, var, exc, kp["expected"], kp["lr"], kp["p_value"]))

bt = pd.DataFrame(rows, columns=["方式", "99%VaR", "例外数", "期待例外数", "Kupiec LR", "p値"])
print(bt.to_string(index=False, formatters={
    "99%VaR": "{:.4f}".format, "期待例外数": "{:.2f}".format,
    "Kupiec LR": "{:.4f}".format, "p値": "{:.4f}".format}))

print("\n--- 考察 ---")
print("パラメトリック VaR は正規当てはめで滑らかなしきい値を与え、ヒストリカル VaR は")
print("経験分位点なので最悪日付近に張り付く。しきい値の高低で例外数がずれ、標本が約60日と")
print("短いため p 値はどちらも大きく（=棄却できず）、方式差を統計的に判別する検出力は乏しい。")
print("実務で250日以上を要求するのは、この検出力不足を補うため。")

# %% [markdown]
# どちらの方式でも例外数は期待例外数（$0.01\times n \approx 0.6$ 件）の近傍にとどまり、
# Kupiec 検定は帰無仮説を棄却しません。しきい値の作り方（正規当てはめか経験分位点か）で
# 例外数は前後しますが、短い標本では p 値の差に統計的な意味を持たせにくい、というのが結論です。
