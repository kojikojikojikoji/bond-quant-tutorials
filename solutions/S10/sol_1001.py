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
# # S10-1 演習 解答例
#
# 免疫化とALMの演習2問の解答例です。本文（`nb_1001_immunization.py`）と同じ
# 負債・国債ユニバース・割引カーブを再構築し、演習1（ツイスト診断とキーレート
# 免疫化）と演習2（CFマッチングとの費用対効果）を実装します。

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from bondlab.bond import FixedRateBond
from bondlab.analytics import duration_convexity, bump_curve
from bondlab.curve import bootstrap_par

np.random.seed(0)

SETTLEMENT = dt.date(2026, 7, 1)
FREQ = 2
Y0 = 0.015
BP = 1e-4


# %% [markdown]
# ## 共通の再構築
#
# 本文と同じ部品を組み直します。自作関数は次のとおりです。
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `make_bond(row)` | ユニバース1行 | `FixedRateBond` | 残存年数→満期日で債券生成 |
# | `cf_analytics(times, cfs, ytm, freq)` | 年数, CF, 利回り, 頻度 | dict(pv, modified, convexity) | 任意CF列の指標（`duration_convexity` と同一規約） |
# | `bond_curve_pv(bond, curve)` | 債券, カーブ | 額面100あたりPV | 割引カーブでの単一債券評価 |
# | `bond_krd(bond, curve, tenors, size, width)` | 債券, カーブ, キーレート | KRD配列 | 単一債券のキーレートデュレーション |
# | `year_bucket_cf(bond)` | 債券 | 年次CF配列(1..H年) | 決済日からの年バケット別CF合計 |

# %%
def make_bond(row):
    mat = SETTLEMENT + dt.timedelta(days=round(365.25 * row["maturity_years"]))
    return FixedRateBond(SETTLEMENT, mat, coupon=float(row["coupon"]),
                         frequency=FREQ, convention="ACT/ACT", face=100.0)


def cf_analytics(times, cfs, ytm, freq=FREQ):
    times = np.asarray(times, float)
    cfs = np.asarray(cfs, float)
    n = times * freq
    disc = (1.0 + ytm / freq) ** (-n)
    pv = float((cfs * disc).sum())
    macaulay = float((times * cfs * disc).sum() / pv)
    modified = macaulay / (1.0 + ytm / freq)
    d2 = (cfs * n * (n + 1.0) * (1.0 + ytm / freq) ** (-n - 2.0)).sum() / (freq ** 2)
    return dict(pv=pv, macaulay=macaulay, modified=modified, convexity=float(d2 / pv))


def bond_curve_pv(bond, curve):
    total = 0.0
    for d, c in bond.cashflows():
        if d > SETTLEMENT:
            total += c * curve.discount((d - SETTLEMENT).days / 365.25)
    return total


def bond_krd(bond, curve, tenors, size=BP, width=3.0):
    out = []
    for T in tenors:
        up = bump_curve(curve, T, size, width=width)
        dn = bump_curve(curve, T, -size, width=width)
        out.append(-(bond_curve_pv(bond, up) - bond_curve_pv(bond, dn)) / (2.0 * size))
    return np.array(out)


def year_bucket_cf(bond, horizon):
    buckets = np.zeros(horizon)
    for d, c in bond.cashflows():
        if d > SETTLEMENT:
            k = int(np.ceil((d - SETTLEMENT).days / 365.25))
            if 1 <= k <= horizon:
                buckets[k - 1] += c
    return buckets


# 負債・ユニバース・カーブ
liab_times = np.arange(1.0, 26.0)
_shape = np.exp(-0.5 * ((liab_times - 10.0) / 7.0) ** 2)
liab_cfs = 100.0 * _shape / _shape.sum() * 25.0
L = cf_analytics(liab_times, liab_cfs, Y0, FREQ)

universe = pd.read_csv("data/samples/synthetic_jgb_universe.csv")
bonds = [make_bond(r) for _, r in universe.iterrows()]
fair = dict(
    price=np.array([duration_convexity(b, Y0, SETTLEMENT)["dirty_price"] for b in bonds]),
    dur=np.array([duration_convexity(b, Y0, SETTLEMENT)["modified"] for b in bonds]),
    conv=np.array([duration_convexity(b, Y0, SETTLEMENT)["convexity"] for b in bonds]),
)
mkt_price = np.array([
    duration_convexity(b, Y0 + rc * BP, SETTLEMENT)["dirty_price"]
    for b, rc in zip(bonds, universe["rich_cheap_bp"].values)
])

tenors_grid = np.arange(1.0, 31.0)
par_rates = np.full_like(tenors_grid, Y0)   # フラット（本文と同じ）
curve = bootstrap_par(tenors_grid, par_rates, frequency=1, interp="log_linear")

key_tenors = np.array([2.0, 5.0, 10.0, 20.0, 30.0])
krd_l = np.zeros(len(key_tenors))
for j, T in enumerate(key_tenors):
    up = bump_curve(curve, T, BP, width=3.0)
    dn = bump_curve(curve, T, -BP, width=3.0)
    up_pv = float(np.sum([c * up.discount(t) for t, c in zip(liab_times, liab_cfs)]))
    dn_pv = float(np.sum([c * dn.discount(t) for t, c in zip(liab_times, liab_cfs)]))
    krd_l[j] = -(up_pv - dn_pv) / (2 * BP)

liab_curve_pv = float(np.sum([c * curve.discount(t) for t, c in zip(liab_times, liab_cfs)]))


# 本文のデュレーションマッチング解を再現
def solve_immunization():
    A_eq = np.vstack([fair["price"], fair["price"] * fair["dur"]])
    b_eq = np.array([L["pv"], L["pv"] * L["modified"]])
    A_ub = -(fair["price"] * fair["conv"])[None, :]
    b_ub = np.array([-L["pv"] * L["convexity"]])
    res = linprog(mkt_price, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * len(bonds), method="highs")
    return res.x


w_dur = solve_immunization()
print("デュレーションマッチング解 保有銘柄数:", int((w_dur > 1e-6).sum()))
print(f"市場コスト = {float(w_dur @ mkt_price):.2f}")


# %% [markdown]
# ## 演習1：ツイストでの破れをKRDで診断し、キーレート免疫化で対策
#
# まず本文のデュレーションマッチング解について、3つのツイスト（スティープナー・
# フラットナー・バタフライ）でサープラス変化を実測し、KRD近似
# $\Delta S\approx-\sum_i(\kappa^A_i-\kappa^L_i)\Delta z_i$ と突合します。

# %%
def port_curve_pv(w, cv):
    return float(np.sum([wi * bond_curve_pv(b, cv) for wi, b in zip(w, bonds) if wi > 1e-9]))


def liab_curve_pv_fn(cv):
    return float(np.sum([c * cv.discount(t) for t, c in zip(liab_times, liab_cfs)]))


def twist_curve(base, deltas, width=3.0):
    cv = base
    for T, dz in deltas:
        cv = bump_curve(cv, T, dz, width=width)
    return cv


def port_krd(w, cv, tenors):
    K = np.zeros(len(tenors))
    for wi, b in zip(w, bonds):
        if wi > 1e-9:
            K += wi * bond_krd(b, cv, tenors)
    return K


scenarios = {
    "スティープナー": [(2.0, -0.005), (5.0, -0.0025), (20.0, 0.0025), (30.0, 0.005)],
    "フラットナー": [(2.0, 0.005), (5.0, 0.0025), (20.0, -0.0025), (30.0, -0.005)],
    "バタフライ": [(2.0, -0.003), (10.0, 0.006), (30.0, -0.003)],
}

krd_a_dur = port_krd(w_dur, curve, key_tenors)
S0 = port_curve_pv(w_dur, curve) - liab_curve_pv_fn(curve)

print("=== デュレーションマッチング解のツイスト診断 ===")
print(f"基準サープラス S0 = {S0:+.3f}\n")
rows = []
for name, deltas in scenarios.items():
    cv = twist_curve(curve, deltas)
    dS = (port_curve_pv(w_dur, cv) - liab_curve_pv_fn(cv)) - S0
    dz = np.array([dict(deltas).get(T, 0.0) for T in key_tenors])
    dS_krd = -np.sum((krd_a_dur - krd_l) * dz)
    rows.append({"scenario": name, "ΔS_実測": dS, "ΔS_KRD近似": dS_krd})
diag = pd.DataFrame(rows)
print(diag.round(3).to_string(index=False))
print("\nKRDミスマッチ (資産 − 負債):")
print(pd.DataFrame({"key_tenor": key_tenors, "mismatch": (krd_a_dur - krd_l)}).round(2).to_string(index=False))

# %% [markdown]
# 実測とKRD近似が整合します。ミスマッチが正の長期バケットと負の中期バケットが、
# ツイストの符号に応じてサープラスを削ります（主因は両端 vs 中央の張り出し差）。
#
# 次に、各キーレートで $\kappa^A_i=\kappa^L_i$ を等式制約に加えた**キーレート
# 免疫化**を線形計画で解きます。決定変数は保有量、目的は市場コスト最小です。

# %%
# 各債券の (カーブPV, キーレートKRDベクトル) を事前計算
pv_i = np.array([bond_curve_pv(b, curve) for b in bonds])
K_mat = np.array([bond_krd(b, curve, key_tenors) for b in bonds])  # (n_bonds, n_tenors)

# 等式制約：カーブPV一致 ＋ 各キーレートで KRD 一致
A_eq_kr = np.vstack([pv_i[None, :], K_mat.T])
b_eq_kr = np.concatenate([[liab_curve_pv], krd_l])
res_kr = linprog(mkt_price, A_eq=A_eq_kr, b_eq=b_eq_kr,
                 bounds=[(0, None)] * len(bonds), method="highs")
assert res_kr.success, res_kr.message
w_kr = res_kr.x
print("キーレート免疫化解 保有銘柄数:", int((w_kr > 1e-6).sum()))
print(f"市場コスト = {float(w_kr @ mkt_price):.2f}")
print("KRDミスマッチ (キーレート免疫化後):")
print(pd.DataFrame({"key_tenor": key_tenors,
                    "mismatch": (port_krd(w_kr, curve, key_tenors) - krd_l)}).round(3).to_string(index=False))

# 同じ3シナリオで再評価
S0_kr = port_curve_pv(w_kr, curve) - liab_curve_pv_fn(curve)
print(f"\n基準サープラス S0 = {S0_kr:+.3f}")
rows2 = []
for name, deltas in scenarios.items():
    cv = twist_curve(curve, deltas)
    dS_dur = (port_curve_pv(w_dur, cv) - liab_curve_pv_fn(cv)) - S0
    dS_kr = (port_curve_pv(w_kr, cv) - liab_curve_pv_fn(cv)) - S0_kr
    rows2.append({"scenario": name, "ΔS_デュレーション": dS_dur, "ΔS_キーレート免疫": dS_kr})
cmp1 = pd.DataFrame(rows2)
print(cmp1.round(3).to_string(index=False))
print("\n各キーレートのミスマッチをゼロにしたので、ツイストでのサープラス変化が大きく縮む。")

fig, ax = plt.subplots(figsize=(8, 4.5))
xx = np.arange(len(scenarios))
ax.bar(xx - 0.2, cmp1["ΔS_デュレーション"], width=0.4, color="darkorange", label="デュレーション一致のみ")
ax.bar(xx + 0.2, cmp1["ΔS_キーレート免疫"], width=0.4, color="steelblue", label="キーレート免疫化")
ax.set_xticks(xx)
ax.set_xticklabels(list(scenarios.keys()))
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("サープラス変化 ΔS")
ax.set_title("ツイスト耐性：デュレーション一致 vs キーレート免疫化")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 演習2：CFマッチング（デディケーション） vs デュレーションマッチング
#
# 年次バケットで「各年までの累積資産CF ≥ 累積負債CF」を満たす**デディケーション**
# ポートフォリオを、市場コスト最小の線形計画で組みます（余剰現金は再投資利回り
# ゼロで翌年へ繰り越す保守的な設定）。

# %%
H = 25  # 負債ホライズン（年）
asset_bucket = np.array([year_bucket_cf(b, H) for b in bonds])   # (n_bonds, H)
liab_bucket = np.zeros(H)
for t, c in zip(liab_times, liab_cfs):
    liab_bucket[int(t) - 1] += c

# 累積制約：各 k で Σ_{y<=k} (assetCF w) >= Σ_{y<=k} liab
cum_asset = np.cumsum(asset_bucket, axis=1).T   # (H, n_bonds)
cum_liab = np.cumsum(liab_bucket)
res_cf = linprog(mkt_price, A_ub=-cum_asset, b_ub=-cum_liab,
                 bounds=[(0, None)] * len(bonds), method="highs")
assert res_cf.success, res_cf.message
w_cf = res_cf.x
print("CFマッチング解 保有銘柄数:", int((w_cf > 1e-6).sum()))

# 累積カバレッジ確認
cov = cum_asset @ w_cf
print("累積カバレッジ（資産 − 負債, 全年で非負のはず）最小値:", f"{(cov - cum_liab).min():.4f}")
assert (cov - cum_liab).min() > -1e-6

# %% [markdown]
# 費用対効果の比較：市場コストと、演習1の3ツイストでのサープラス変化の大きさ。

# %%
cost_dur = float(w_dur @ mkt_price)
cost_cf = float(w_cf @ mkt_price)

# CFマッチング解のツイスト耐性（カーブ評価のサープラス変化）
S0_cf = port_curve_pv(w_cf, curve) - liab_curve_pv_fn(curve)
tw_dur, tw_cf = [], []
for name, deltas in scenarios.items():
    cv = twist_curve(curve, deltas)
    tw_dur.append((port_curve_pv(w_dur, cv) - liab_curve_pv_fn(cv)) - S0)
    tw_cf.append((port_curve_pv(w_cf, cv) - liab_curve_pv_fn(cv)) - S0_cf)

summary = pd.DataFrame({
    "指標": ["市場コスト", "|ΔS| スティープナー", "|ΔS| フラットナー", "|ΔS| バタフライ"],
    "デュレーションM": [cost_dur, abs(tw_dur[0]), abs(tw_dur[1]), abs(tw_dur[2])],
    "CFマッチング": [cost_cf, abs(tw_cf[0]), abs(tw_cf[1]), abs(tw_cf[2])],
})
print(summary.round(3).to_string(index=False))
print(f"\n追加コスト（CF − デュレーション）= {cost_cf - cost_dur:+.2f}"
      f"（{(cost_cf / cost_dur - 1) * 100:+.2f}%）")

# %% [markdown]
# ### 考察
#
# - **コスト**：CFマッチングは各時点を賄う制約が強く、選べる銘柄が限られるため、
#   デュレーションマッチングより市場コストが高くなる。上表の追加コストがその対価。
# - **金利変動耐性**：CFマッチングは資産CFで負債を時点ごとに賄うので、ツイストを
#   含む非平行変形でもサープラス変化が小さい。デュレーションマッチングは平行シフト
#   にしか免疫がなく、ツイストで相対的に大きく振れる。
# - **結論**：低コストで平行シフト中心のリスクに備えるならデュレーション（＋定期
#   リバランス／キーレート免疫化）、コストを払ってでも非平行リスクまで消したいなら
#   CFマッチング。実務は負債の長さと超長期債の供給制約から、両者の中間
#   （キーレート免疫化）に落ち着くことが多い。

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
labels = ["デュレーションM", "CFマッチング"]
ax[0].bar(labels, [cost_dur, cost_cf], color=["darkorange", "steelblue"])
ax[0].set_title("市場コスト")
ax[0].set_ylabel("コスト")
xx = np.arange(len(scenarios))
ax[1].bar(xx - 0.2, np.abs(tw_dur), width=0.4, color="darkorange", label="デュレーションM")
ax[1].bar(xx + 0.2, np.abs(tw_cf), width=0.4, color="steelblue", label="CFマッチング")
ax[1].set_xticks(xx)
ax[1].set_xticklabels(list(scenarios.keys()))
ax[1].set_title("ツイストでの |ΔS|")
ax[1].set_ylabel("サープラス変化の絶対値")
ax[1].legend()
fig.tight_layout()
plt.show()
