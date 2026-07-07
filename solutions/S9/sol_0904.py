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
# # S9-4 演習 解答例
#
# 国債先物ベーシスとCTDの演習2問の解答例です。本文 `nb_0904_futures_basis` と同じ
# `bondlab.bt` / `bondlab.bond` / `bondlab.curve` を使います。

# %%
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bondlab.bond import FixedRateBond
from bondlab.curve import bootstrap_par

np.random.seed(0)

ISSUE = dt.date(2005, 1, 1)
today = dt.date(2026, 6, 22)
delivery = dt.date(2026, 9, 20)
horizon = (delivery - today).days / 365.0
REPO = 0.001

universe = pd.read_csv("data/samples/synthetic_jgb_universe.csv")
tenors = np.arange(1, 21).astype(float)
par_rates = np.clip(0.002 + 0.0016 * tenors, None, 0.020)
curve = bootstrap_par(tenors, par_rates, frequency=1)


def build_bond(coupon, maturity, freq=2):
    return FixedRateBond(issue=ISSUE, maturity=maturity, coupon=coupon, frequency=freq)


def exact_conversion_factor(coupon, maturity, delivery, notional_coupon=0.06):
    return build_bond(coupon, maturity).clean_price(notional_coupon, delivery) / 100.0


def basis_table(basket, shift_bp, repo=REPO, futures=None):
    """バスケットにシフトを与え、CF・ベーシス・インプライドレポを計算する。"""
    rows = []
    for _, r in basket.iterrows():
        mat = today + dt.timedelta(days=round(r.maturity_years * 365.25))
        bond = build_bond(r.coupon, mat)
        fair = np.interp(r.maturity_years, tenors, par_rates)
        y = fair + shift_bp * 1e-4 + r.rich_cheap_bp * 1e-4
        clean = bond.clean_price(y, today)
        ai_s = bond.accrued(today)
        ai_d = bond.accrued(delivery)
        coupons = sum(c for d, c in bond.cashflows() if today < d <= delivery)
        cf = exact_conversion_factor(r.coupon, mat, delivery)
        dirty = clean + ai_s
        fwd = dirty * (1.0 + repo * horizon) - coupons - ai_d
        rows.append(dict(bond_id=r.bond_id, coupon=r.coupon, maturity_years=r.maturity_years,
                         cf=cf, clean=clean, ai_s=ai_s, ai_d=ai_d, coupons=coupons, fwd=fwd))
    df = pd.DataFrame(rows)
    if futures is None:
        futures = float((df.fwd / df.cf).min())
    df["net_basis"] = df.fwd - futures * df.cf
    df["gross_basis"] = df.clean - futures * df.cf
    invoice = futures * df.cf + df.ai_d + df.coupons
    cost = df.clean + df.ai_s
    df["implied_repo"] = (invoice - cost) / (cost * horizon)
    return df, futures


# %% [markdown]
# ## 演習 1：CTD特定と金利シフトでのスイッチ分析
#
# 受渡適格帯を残存6〜11年に広げてバスケットを組み直し、現状の CTD を特定します。次に
# 平行シフトを $-100$〜$+400\text{bp}$ で動かし、CTD が最初に入れ替わるシフト幅を求めます。

# %%
basket = universe[(universe.maturity_years >= 6.0) & (universe.maturity_years <= 11.0)].reset_index(drop=True)
print("受渡適格バスケット（残存6〜11年）:")
print(basket.to_string(index=False))

base, F = basis_table(basket, 0.0)
ctd0 = base.loc[base.net_basis.idxmin(), "bond_id"]
ctd0_mat = base.loc[base.net_basis.idxmin(), "maturity_years"]
print(f"\n無裁定先物価格 F = {F:.4f}")
print(f"現状のCTD = {ctd0}（残存 {ctd0_mat:.2f} 年）")
print(base[["bond_id", "coupon", "maturity_years", "cf", "net_basis", "implied_repo"]].round(5).to_string(index=False))

# %%
shifts = np.arange(-100, 401, 10)
ctd_path, ctd_mat_path = [], []
for bp in shifts:
    df_bp, _ = basis_table(basket, float(bp), futures=F)
    idx = df_bp.net_basis.idxmin()
    ctd_path.append(df_bp.loc[idx, "bond_id"])
    ctd_mat_path.append(df_bp.loc[idx, "maturity_years"])

switches = [(int(shifts[i]), ctd_path[i - 1], ctd_path[i], ctd_mat_path[i - 1], ctd_mat_path[i])
            for i in range(1, len(shifts)) if ctd_path[i] != ctd_path[i - 1]]
if switches:
    bp, before, after, m0, m1 = switches[0]
    print(f"最初のCTDスイッチ: +{bp}bp で {before}（残存{m0:.2f}年）→ {after}（残存{m1:.2f}年）")
    print(f"CTDのデュレーション代理（残存）は {m0:.2f} → {m1:.2f} 年へ長期化しました。")
else:
    print("−100〜+400bp の範囲で CTD スイッチは発生しませんでした。")

# %% [markdown]
# 金利が上がって標準物クーポン6%へ近づくほど、CTD は低デュレーション（短い）銘柄から
# 高デュレーション（長い）銘柄へ移ります。デュレーションが長い銘柄ほど、利回り上昇で
# 価格が大きく下がり、$F\cdot\mathrm{CF}$ に対して割安になるためです。

# %%
fig, ax = plt.subplots(figsize=(9, 3.6))
ids = sorted(set(ctd_path), key=lambda b: base.set_index("bond_id").maturity_years.get(b, 0)
             if b in base.bond_id.values else 0)
# maturity で並べ替えた縦軸
mat_lookup = {r.bond_id: r.maturity_years for _, r in basket.iterrows()}
ids = sorted(set(ctd_path), key=lambda b: mat_lookup[b])
ymap = {b: k for k, b in enumerate(ids)}
ax.step(shifts, [ymap[b] for b in ctd_path], where="post", color="#1f77b4")
ax.set_yticks(range(len(ids)))
ax.set_yticklabels([f"{b} ({mat_lookup[b]:.1f}y)" for b in ids])
ax.set_xlabel("平行シフト（bp）")
ax.set_title("CTD銘柄と金利シフト（残存6〜11年バスケット）")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 演習 2：インプライドレポとネットベーシスの関係
#
# 各銘柄のインプライドレポとネットベーシスを散布図にし、逆相関（順位反転）を確認します。
# さらにレポ金利を 0.10% → 0.50% に上げたとき、ネットベーシスと CTD がどう変わるかを見ます。

# %%
base2, F2 = basis_table(basket, 0.0)
corr = np.corrcoef(base2.net_basis, base2.implied_repo)[0, 1]
order_net = base2.sort_values("net_basis").bond_id.tolist()
order_irr = base2.sort_values("implied_repo", ascending=False).bond_id.tolist()
print(f"ネットベーシスとインプライドレポの相関係数 = {corr:.4f}")
print(f"ネットベーシス昇順の並び   : {order_net}")
print(f"インプライドレポ降順の並び : {order_irr}")
print("順位は完全に一致:", order_net == order_irr)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(base2.net_basis, base2.implied_repo * 100, color="#1f77b4")
for _, r in base2.iterrows():
    ax.annotate(r.bond_id, (r.net_basis, r.implied_repo * 100),
                textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set_xlabel("ネットベーシス（円/額面100）")
ax.set_ylabel("インプライドレポ（%）")
ax.set_title("インプライドレポ vs ネットベーシス（逆相関）")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ネットベーシスが小さい銘柄ほどインプライドレポが高く、CTD（ネットベーシス最小＝
# インプライドレポ最大）が両指標で一致します。片方が分かれば他方の順位も決まる関係です。

# %%
print(f"{'レポ金利':>8} {'CTD':>8} {'CTDネットベーシス':>16} {'CTDインプライドレポ':>18}")
for repo in [0.001, 0.005]:
    # F はレポ0.1%基準に固定し、レポ変化がネットベーシスに与える効果だけを見る。
    df_r, _ = basis_table(basket, 0.0, repo=repo, futures=F2)
    idx = df_r.net_basis.idxmin()
    print(f"{repo:>8.3%} {df_r.loc[idx, 'bond_id']:>8} "
          f"{df_r.net_basis.min():>16.5f} {df_r.loc[idx, 'implied_repo']:>18.4%}")

# %% [markdown]
# レポ金利を上げると資金調達コストが増え、フォワード価格が上がるためネットベーシスは全体に
# 拡大します。ただし低クーポン JGB ではクーポン収入よりレポコストが勝ちキャリーがマイナスなので、
# レポ上昇の影響は各銘柄でほぼ平行に効き、CTD の顔ぶれ自体は（この小幅では）変わりません。
# レポと利回りの相対関係（キャリーの符号）が CTD 決定の隠れたドライバーになっています。
