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
# # S11-2 演習 解答例
#
# RFQ プライサーの演習2問の解答例です。本文と同じ `bondlab.curve` の NSS
# フィットでミッドを作り、Avellaneda-Stoikov 風の在庫スキューを再構築します。

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bondlab.curve import fit_nss, nss

np.random.seed(0)

LAM_GRID = np.linspace(0.8, 8.0, 8)
BP = 1e4
FREQ = 2


def price_from_yield(coupon, years, ytm, freq=FREQ, face=100.0):
    n = max(int(round(years * freq)), 1)
    c = coupon / freq * face
    y = ytm / freq
    disc = (1.0 + y) ** (-np.arange(1, n + 1))
    return float(c * disc.sum() + face * disc[-1])


def modified_duration(coupon, years, ytm, freq=FREQ, face=100.0, h=1e-4):
    p_up = price_from_yield(coupon, years, ytm + h, freq, face)
    p_dn = price_from_yield(coupon, years, ytm - h, freq, face)
    return -((p_up - p_dn) / (2.0 * h)) / price_from_yield(coupon, years, ytm, freq, face)


def build_residual_panel(panel, lam_grid=LAM_GRID):
    rows = {}
    for date, grp in panel.groupby("date"):
        grp = grp.sort_values("maturity_years")
        fit = fit_nss(grp["maturity_years"].values, grp["yield"].values, lam_grid=lam_grid)
        resid = (grp["yield"].values - nss(grp["maturity_years"].values, **fit)) * BP
        rows[date] = pd.Series(resid, index=grp["bond_id"].values)
    wide = pd.DataFrame(rows).T
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def estimate_daily_vol(panel):
    wide = panel.pivot(index="date", columns="bond_id", values="yield").sort_index()
    return wide.diff().std(axis=0, ddof=1)


@dataclass
class RFQParams:
    gamma: float = 0.15
    kappa: float = 1.3
    horizon: float = 5.0
    base_ticks: float = 0.02


class RFQPricer:
    def __init__(self, universe, snap_yields, resid_mean_bp, daily_vol, params=None):
        self.params = params or RFQParams()
        self.univ = universe.set_index("bond_id")
        mat = self.univ["maturity_years"].reindex(snap_yields.index).values
        fit = fit_nss(mat, snap_yields.values, lam_grid=LAM_GRID)
        model_y = pd.Series(nss(mat, **fit), index=snap_yields.index)
        resid_y = resid_mean_bp.reindex(snap_yields.index) / BP
        self.mid_yield = model_y + resid_y
        recs = {}
        for b in snap_yields.index:
            cpn = self.univ.loc[b, "coupon"]
            yrs = self.univ.loc[b, "maturity_years"]
            my = self.mid_yield[b]
            m = price_from_yield(cpn, yrs, my)
            dur = modified_duration(cpn, yrs, my)
            recs[b] = dict(mid=m, dur=dur, sigma_p=dur * m * daily_vol.get(b, np.nan))
        self.book = pd.DataFrame(recs).T

    def half_spread(self, bond_id, size):
        p = self.params
        sig_p = self.book.loc[bond_id, "sigma_p"]
        return p.base_ticks + 0.5 * p.gamma * sig_p ** 2 * p.horizon * size

    def reservation(self, bond_id, inventory, time_left=None):
        p = self.params
        tl = p.horizon if time_left is None else time_left
        m = self.book.loc[bond_id, "mid"]
        sig_p = self.book.loc[bond_id, "sigma_p"]
        return m - inventory * p.gamma * sig_p ** 2 * tl

    def quote(self, bond_id, size=1.0, inventory=0.0, time_left=None):
        r = self.reservation(bond_id, inventory, time_left)
        d = self.half_spread(bond_id, size)
        return dict(mid=self.book.loc[bond_id, "mid"], reservation=r,
                    bid=r - d, ask=r + d, half_spread=d)


universe = pd.read_csv("data/samples/synthetic_jgb_universe.csv")
panel = pd.read_csv("data/samples/synthetic_jgb_yield_panel.csv")
resid_panel = build_residual_panel(panel)
resid_mean_bp = resid_panel.mean(axis=0)
daily_vol = estimate_daily_vol(panel)
val_date = panel["date"].max()
snap_yields = panel[panel["date"] == val_date].set_index("bond_id")["yield"]
pricer = RFQPricer(universe, snap_yields, resid_mean_bp, daily_vol)

# %% [markdown]
# ## 演習 1：サイズ・在庫を変えて bid/ask を可視化
#
# 在庫 $q\in\{-5,0,+5\}$、サイズ 1〜15 で bid/ask を描きます。在庫はクォートの
# 中心（リザベーション価格）を上下させ、サイズはミッドからの幅を広げます。

# %%
target = snap_yields.index[10]
sizes = np.linspace(1, 15, 29)
inventories = [-5.0, 0.0, 5.0]
colors = {-5.0: "seagreen", 0.0: "black", 5.0: "firebrick"}

fig, ax = plt.subplots(figsize=(9, 5))
for q in inventories:
    bids = [pricer.quote(target, size=s, inventory=q)["bid"] for s in sizes]
    asks = [pricer.quote(target, size=s, inventory=q)["ask"] for s in sizes]
    ax.plot(sizes, asks, color=colors[q], lw=1.4, label=f"ask (在庫 {q:+.0f})")
    ax.plot(sizes, bids, color=colors[q], lw=1.4, ls="--", label=f"bid (在庫 {q:+.0f})")
ax.axhline(pricer.book.loc[target, "mid"], color="gray", lw=0.8, label="ミッド")
ax.set_xlabel("RFQ サイズ Q")
ax.set_ylabel("価格（価格点）")
ax.set_title(f"サイズ×在庫による bid/ask（銘柄 {target}）")
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察
#
# - **在庫**は bid/ask の対（中心）を平行移動させます。ロング在庫（$q=+5$）では
#   帯全体が下がり、ショート在庫（$q=-5$）では上がります。これは在庫を平常水準へ
#   戻す方向のスキューです。
# - **サイズ**は bid と ask の開き（帯の幅）を広げます。中心は動かさず、幅だけが
#   サイズに線形で増えます。役割分担は「在庫＝中心、サイズ＝幅」と整理できます。

# %%
# 数値でも役割分担を確認する
q_mid_0 = pricer.quote(target, size=1.0, inventory=0.0)
q_mid_long = pricer.quote(target, size=1.0, inventory=5.0)
q_wide = pricer.quote(target, size=15.0, inventory=0.0)
print(f"在庫0・サイズ1   : 中心 {q_mid_0['reservation']:.4f}  幅 {q_mid_0['ask']-q_mid_0['bid']:.4f}")
print(f"在庫+5・サイズ1  : 中心 {q_mid_long['reservation']:.4f}  幅 {q_mid_long['ask']-q_mid_long['bid']:.4f}")
print(f"在庫0・サイズ15  : 中心 {q_wide['reservation']:.4f}  幅 {q_wide['ask']-q_wide['bid']:.4f}")

# %% [markdown]
# ## 演習 2：Avellaneda-Stoikov 風のリザベーション価格シフト
#
# 在庫解消の残り時間 $(T-t)$ を、$t=0$（残り $T$）から $t=T$（残り 0）へ縮めます。
# リザベーション価格のスキュー幅は $(T-t)$ に比例するため、期限に近づくほど
# ゼロへ収束します。ロング在庫 $q=+5$ で、残り時間に対するスキューを描きます。

# %%
T = pricer.params.horizon
t_grid = np.linspace(0.0, T, 41)
time_left_grid = T - t_grid
q_inv = 5.0
mid = pricer.book.loc[target, "mid"]
res_over_time = [pricer.reservation(target, inventory=q_inv, time_left=tl) for tl in time_left_grid]
skew = np.array(res_over_time) - mid  # ミッドからのスキュー幅（負）

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t_grid, skew, color="firebrick", lw=1.6, label="リザベーションのスキュー幅")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("経過時間 t（在庫解消期限 T まで）")
ax.set_ylabel("リザベーション − ミッド（価格点）")
ax.set_title(f"残り時間 (T-t) に伴うスキューの収束（在庫 +{q_inv:.0f}）")
ax.legend()
fig.tight_layout()
plt.show()

# 期限に近づくとスキューは0へ収束するはず
assert abs(skew[-1]) < 1e-9, "t=T でスキューが0に収束していない"
assert abs(skew[0]) > abs(skew[-1]), "残り時間が長いほどスキューが大きいはず"
print(f"t=0 のスキュー幅 = {skew[0]:+.4f} 点,  t=T のスキュー幅 = {skew[-1]:+.4f} 点")

# %% [markdown]
# ### 考察
#
# - スキュー幅は $(T-t)$ に比例して線形に縮み、期限 $t=T$ でゼロになります。
# - 理由は在庫リスクの積分にあります。残り時間が短いほど、在庫を持ち続ける間に
#   価格が動く余地が小さく、在庫が生む将来の含み損益の分散 $\sigma^2(T-t)$ が
#   減ります。リスクが小さければ在庫を急いで捌く動機も弱まり、クォートを歪めて
#   まで在庫を減らす必要が薄れるため、スキューが緩みます。
# - 実務では、この $(T-t)$ を「日中の残り時間」や「次のヘッジ機会までの時間」と
#   解釈し、引け間際に在庫スキューを強める/緩める運用に対応します。
