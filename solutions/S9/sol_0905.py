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
# # S9-5 演習 解答例
#
# バックテスト基盤と戦略評価の演習2問の解答例です。本文と同じ
# `bondlab.bt` と rich/cheap 残差シグナルを使います。

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bondlab import bt

np.random.seed(0)

DATA = Path("data/samples/synthetic_jgb_yield_panel.csv")
if not DATA.exists():
    for up in (Path(".."), Path("../.."), Path("../../..")):
        if (up / DATA).exists():
            DATA = up / DATA
            break


# %%
def build_signals(path):
    """本文と同じ手順で rich/cheap とバタフライのシグナル・リターンを作る。"""
    raw = pd.read_csv(path, parse_dates=["date"])
    Y = raw.pivot(index="date", columns="bond_id", values="yield").sort_index()
    mat = raw.groupby("bond_id")["maturity_years"].mean().reindex(Y.columns)

    # rich/cheap 残差
    x = mat.values.astype(float)
    resid = pd.DataFrame(index=Y.index, columns=Y.columns, dtype=float)
    for d, row in Y.iterrows():
        coef = np.polyfit(x, row.values.astype(float), 3)
        resid.loc[d] = row.values - np.polyval(coef, x)
    mu, sd = resid.mean(axis=1), resid.std(axis=1, ddof=0).replace(0.0, np.nan)
    sig_rv = resid.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
    ret_rv = (-resid.diff()).mul(mat, axis=1)

    # バタフライ 2-5-10
    nearest = lambda t: (mat - t).abs().idxmin()
    s, b, l = nearest(2.0), nearest(5.0), nearest(10.0)
    fly = Y[b] - 0.5 * (Y[s] + Y[l])
    z = (fly - fly.mean()) / fly.std(ddof=0)
    sig_fly = (-z).rename("FLY").to_frame()
    ret_fly = (-fly.diff() * mat[b]).rename("FLY").to_frame()
    return sig_rv, ret_rv, sig_fly, ret_fly


sig_rv, ret_rv, sig_fly, ret_fly = build_signals(DATA)
sig_all = pd.concat([sig_rv, sig_fly], axis=1)
ret_all = pd.concat([ret_rv, ret_fly], axis=1).reindex_like(sig_all)
print("シグナル形状:", sig_all.shape)

# %% [markdown]
# ## 演習1：取引コスト感応度
#
# rich/cheap 単独・バタフライ単独・統合の3戦略について、片道コストを 0〜10 bp
# で振り、各戦略のシャープと平均回転率を求めます。シャープが 1.0 を割る水準と
# 0 を割る損益分岐水準を読み取り、最もコストに脆い戦略を回転率で説明します。

# %%
cost_grid = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
strategies = {
    "rich/cheap": (sig_rv, ret_rv),
    "butterfly": (sig_fly, ret_fly),
    "統合": (sig_all, ret_all),
}

sharpe_tbl = pd.DataFrame(index=cost_grid, columns=strategies.keys(), dtype=float)
turnover_avg = {}
for name, (s, r) in strategies.items():
    turnover_avg[name] = bt.backtest(s, r, cost_bps=0.0, lag=1)["turnover"].mean()
    for c in cost_grid:
        res = bt.backtest(s, r, cost_bps=c, lag=1)
        sharpe_tbl_val = bt.performance(res["pnl"])["sharpe"]
        sharpe_tbl.loc[c, name] = sharpe_tbl_val

print("=== 片道コスト別シャープレシオ ===")
print(sharpe_tbl.to_string(float_format=lambda v: f"{v:.3f}"))
print("\n平均回転率:")
for name, tv in turnover_avg.items():
    print(f"  {name:12s}: {tv:.3f}")


# %%
def thresholds(series):
    """シャープが1.0を割る/0を割る最初のコスト水準を返す。"""
    below1 = series.index[series <= 1.0]
    below0 = series.index[series <= 0.0]
    b1 = float(below1[0]) if len(below1) else np.nan
    b0 = float(below0[0]) if len(below0) else np.nan
    return b1, b0


print("戦略ごとの閾値コスト（bp）:")
for name in strategies:
    b1, b0 = thresholds(sharpe_tbl[name])
    print(f"  {name:12s}: シャープ<1.0 → {b1} bp,  損益分岐(<0) → {b0} bp")

most_fragile = max(turnover_avg, key=turnover_avg.get)
print(f"\n回転率が最大＝最もコストに脆い戦略: {most_fragile}")
print("コスト = 回転率 × 片道bp なので、回転率が高いほど同じコスト増でシャープが速く劣化する。")

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
for name in strategies:
    ax.plot(cost_grid, sharpe_tbl[name].values, marker="o", label=name)
ax.axhline(1.0, color="gray", ls="--", lw=0.8)
ax.axhline(0.0, color="black", lw=0.8)
ax.set_title("演習1：取引コスト感応度（戦略別シャープ）")
ax.set_xlabel("片道コスト (bp)")
ax.set_ylabel("シャープレシオ")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 演習2：ルックアヘッドの成績差
#
# rich/cheap 戦略について、lag=0（同時点シグナル）・lag=1（翌日約定）・lag=2
# （2日遅延）でシャープ・年率リターン・最大ドローダウンを比較します。平均回帰は
# 「直前に割安化した銘柄」を買うため、lag=0（同時点）は直前の逆行を損益に取り込み、
# 正しい lag=1 と符号すら食い違う別物になります。lag=1→lag=2 では鮮度が落ちて
# 成績が下がることを確認します。最後に、当日リターンを覗き見た反則シグナルで
# 「同時点評価が過大評価を生む」典型例も見ます。

# %%
rows = []
for lag in (0, 1, 2):
    res = bt.backtest(sig_rv, ret_rv, cost_bps=0.0, lag=lag)
    perf = bt.performance(res["pnl"])
    rows.append({
        "lag": lag,
        "約定": {0: "当日（ルックアヘッド）", 1: "翌日（正しい）", 2: "2日遅延"}[lag],
        "sharpe": perf["sharpe"],
        "ann_return": perf["ann_return"],
        "max_drawdown": perf["max_drawdown"],
        "hit_rate": perf["hit_rate"],
    })
la = pd.DataFrame(rows).set_index("lag")
print(la.to_string(float_format=lambda v: f"{v:.4f}"))

gap01 = la.loc[1, "sharpe"] - la.loc[0, "sharpe"]
print(f"\n正しい lag=1 と誤り lag=0 のシャープ差: {gap01:+.2f}（同時点評価は真値と別物）")
print("lag=1 → lag=2 でシャープ低下：約定が遅れるほどシグナルの鮮度が落ちる。")
assert abs(gap01) > 1.0, "同時点評価は真値から大きく乖離するはず"
assert la.loc[1, "sharpe"] > la.loc[2, "sharpe"], "遅延が増えると鮮度が落ちて成績は下がる"

# %% [markdown]
# ### 覗き見シグナルによる過大評価
#
# 当日リターンの符号 $\operatorname{sign}(r_t)$ を覗き見た反則シグナルは、lag=0
# （同時点）では $\sum|r_t|$ を稼ぐ非現実的な成績になり、lag=1（翌日）では前日符号が
# 当日を予測せず、過大なシャープは消えます。

# %%
sig_cheat = np.sign(ret_rv)
cheat_rows = []
for lag in (0, 1):
    res = bt.backtest(sig_cheat, ret_rv, cost_bps=0.0, lag=lag)
    perf = bt.performance(res["pnl"])
    cheat_rows.append({"lag": lag, "sharpe": perf["sharpe"], "ann_return": perf["ann_return"]})
cheat = pd.DataFrame(cheat_rows).set_index("lag")
print(cheat.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"覗き見(lag=0)の過大化: {cheat.loc[0, 'sharpe'] - cheat.loc[1, 'sharpe']:+.1f}")
assert cheat.loc[0, "sharpe"] > cheat.loc[1, "sharpe"] + 5.0

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
for lag, label in [(0, "lag=0（ルックアヘッド）"), (1, "lag=1（正しい）"), (2, "lag=2（2日遅延）")]:
    res = bt.backtest(sig_rv, ret_rv, cost_bps=0.0, lag=lag)
    ax.plot(res["pnl"].cumsum().values, label=label)
ax.set_title("演習2：約定ラグ別の累積損益")
ax.set_xlabel("営業日")
ax.set_ylabel("累積PnL")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
