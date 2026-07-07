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
# # S10-3 演習 解答例
#
# 債券ポートフォリオ最適化の演習2問の解答例です。本文の
# `bondlab.analytics.bump_curve` を使った KRD 因子リスクモデルと cvxpy の
# 制約付き最適化を再利用し、(1) 制約を締めたときのシャドープライス変化、
# (2) 平均分散法が債券で使いにくいことの数値的な確認、を行います。

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

from bondlab.bond import FixedRateBond
from bondlab.curve import bootstrap_par
from bondlab.analytics import bump_curve, duration_convexity

np.random.seed(0)

BP = 1e-4
SETTLEMENT = dt.date(2026, 1, 5)
KEY_TENORS = np.array([2.0, 5.0, 10.0, 20.0, 30.0])
KRD_WIDTH = 5.0
CURVE_BUMP = 1e-4


# %% [markdown]
# ## 共通の関数（本文からの再掲）
#
# カーブ構築・価格・KRD 因子・最適化ソルバを本文と同じ形で用意します。

# %%
def build_par_curve(tenors, level=0.020, floor=0.002, decay=8.0):
    tenors = np.asarray(tenors, dtype=float)
    par = floor + level * (1.0 - np.exp(-tenors / decay))
    return bootstrap_par(tenors, par, frequency=1, interp="log_linear")


def bond_from_row(row, settlement):
    mat_days = int(round(float(row["maturity_years"]) * 365.25))
    maturity = settlement + dt.timedelta(days=mat_days)
    return FixedRateBond(issue=settlement, maturity=maturity,
                         coupon=float(row["coupon"]), frequency=2,
                         convention="ACT/ACT", face=100.0)


def bond_cashflows(bond, settlement):
    flows = [(d, c) for d, c in bond.cashflows() if d > settlement]
    times = np.array([(d - settlement).days / 365.25 for d, _ in flows])
    cfs = np.array([c for _, c in flows])
    return times, cfs


def price_under_curve(times, cfs, curve):
    return float(np.sum(cfs * curve.discount(times)))


def krd_vector(times, cfs, curve, keys=KEY_TENORS, width=KRD_WIDTH, bump=CURVE_BUMP):
    p0 = price_under_curve(times, cfs, curve)
    krd = np.empty(keys.size)
    for k, t in enumerate(keys):
        c_up = bump_curve(curve, t, +bump, width=width)
        c_dn = bump_curve(curve, t, -bump, width=width)
        krd[k] = -(price_under_curve(times, cfs, c_up)
                   - price_under_curve(times, cfs, c_dn)) / (2.0 * bump * p0)
    return krd


def build_risk_model(universe, curve, settlement=SETTLEMENT, keys=KEY_TENORS):
    ids, mats, coupons, rc, ytm, moddur = [], [], [], [], [], []
    krd_rows = []
    for _, row in universe.iterrows():
        bond = bond_from_row(row, settlement)
        times, cfs = bond_cashflows(bond, settlement)
        p = price_under_curve(times, cfs, curve)
        clean = p - bond.accrued(settlement)
        y = bond.yield_from_price(clean, settlement)
        dc = duration_convexity(bond, y, settlement)
        ids.append(row["bond_id"]); mats.append(float(row["maturity_years"]))
        coupons.append(float(row["coupon"])); rc.append(float(row["rich_cheap_bp"]))
        ytm.append(y); moddur.append(dc["modified"])
        krd_rows.append(krd_vector(times, cfs, curve, keys))
    rc = np.array(rc); ytm = np.array(ytm)
    return dict(ids=np.array(ids), mats=np.array(mats), coupons=np.array(coupons),
                rc_bp=rc, ytm=ytm, exp_yield=ytm + rc * BP,
                mod_dur=np.array(moddur), krd=np.array(krd_rows))


def solve_max_yield(model, w0, krd_band=0.1, turnover=0.20):
    y = model["exp_yield"]; K = model["krd"]; n = y.size
    w = cp.Variable(n)
    krd_p = K.T @ w; krd_b = K.T @ w0
    c_krd_up = krd_p - krd_b <= krd_band
    c_krd_dn = krd_b - krd_p <= krd_band
    c_turn = cp.norm1(w - w0) <= 2.0 * turnover
    prob = cp.Problem(cp.Maximize(y @ w),
                      [cp.sum(w) == 1, w >= 0, c_krd_up, c_krd_dn, c_turn])
    prob.solve()
    return dict(status=prob.status, w=np.asarray(w.value).ravel(), obj=float(prob.value),
                c_krd_up=c_krd_up, c_krd_dn=c_krd_dn, c_turn=c_turn,
                krd_band=krd_band, turnover=turnover)


universe = pd.read_csv("data/samples/synthetic_jgb_universe.csv")
curve = build_par_curve(np.arange(1, 31))
model = build_risk_model(universe, curve)
w0 = np.full(model["ids"].size, 1.0 / model["ids"].size)
print("リスクモデル構築完了:", model["ids"].size, "銘柄")

# %% [markdown]
# ## 演習1：制約を締めたときのシャドープライス変化
#
# KRD 帯 `krd_band` を 0.10 → 0.05 → 0.02 と締めながら解き、最適利回り・回転率
# 制約の双対・バインドする KRD 制約の双対の大きさを比べます。KRD 帯を締めると
# 実行可能領域が狭まるため、最適利回りは単調に下がる（＝締める方向のシャドー
# プライスは正）はずです。

# %%
def turn_dual(sol):
    return float(np.atleast_1d(sol["c_turn"].dual_value)[0])


def krd_dual_abs_sum(sol):
    return float(np.sum(np.abs(np.atleast_1d(sol["c_krd_up"].dual_value)))
                 + np.sum(np.abs(np.atleast_1d(sol["c_krd_dn"].dual_value))))


def n_binding_krd(sol, model, w0):
    K = model["krd"]; krd_p = K.T @ sol["w"]; krd_b = K.T @ w0
    up = sol["krd_band"] - (krd_p - krd_b)
    dn = sol["krd_band"] - (krd_b - krd_p)
    return int(np.sum(np.abs(np.concatenate([up, dn])) < 1e-5))


bands = [0.10, 0.05, 0.02]
rows = []
for b in bands:
    s = solve_max_yield(model, w0, krd_band=b, turnover=0.20)
    rows.append({
        "KRD帯": b,
        "最適利回り%": model["exp_yield"] @ s["w"] * 100,
        "対ベンチ改善bp": (model["exp_yield"] @ s["w"] - model["exp_yield"] @ w0) * 1e4,
        "回転率双対": turn_dual(s),
        "KRD双対|和|": krd_dual_abs_sum(s),
        "バインドKRD数": n_binding_krd(s, model, w0),
    })
ex1 = pd.DataFrame(rows)
print(ex1.round(4).to_string(index=False))

# 目的値は帯を締めるほど単調に下がる（シャドープライス>0）。
assert ex1["最適利回り%"].is_monotonic_decreasing, "帯を締めても利回りが下がっていない"
print("\n確認: KRD帯を締めるほど最適利回りは単調に低下（締める方向の影の価格は正）")

# %%
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(ex1["KRD帯"], ex1["対ベンチ改善bp"], "o-", color="steelblue",
         label="対ベンチ改善 (bp)")
ax1.set_xlabel("KRD帯（±年）")
ax1.set_ylabel("対ベンチ利回り改善 (bp)", color="steelblue")
ax1.invert_xaxis()
ax2 = ax1.twinx()
ax2.plot(ex1["KRD帯"], ex1["KRD双対|和|"], "s--", color="indianred",
         label="KRD双対 絶対値和")
ax2.set_ylabel("KRD双対の絶対値和", color="indianred")
ax1.set_title("KRD帯を締めたときの利回り改善と KRD シャドープライス")
fig.tight_layout()
plt.show()

# %% [markdown]
# **解釈。** KRD 帯を締めると、割安な長期債を買ってもカーブ形状のずれが許されず、
# 割安銘柄への入れ替え余地が縮みます。結果として対ベンチ改善（利回り）は
# 単調に減り、これは「KRD 帯という制約のシャドープライスが正」であることを
# 意味します。同時に KRD 制約の双対（絶対値和）は帯を締めるほど大きくなり、
# 「あと少し帯を広げられれば利回りをどれだけ取り戻せるか」という限界価値が
# 高まっていることを示します。回転率制約の双対も、締めるほど KRD がボトルネックに
# 変わるため相対的に小さくなる傾向を確認できます。

# %% [markdown]
# ## 演習2：平均分散法が債券で使いにくいことを数値で示す
#
# 因子モデルから短いリターン履歴を合成し、標本平均・標本共分散で平均分散最適化を
# 解きます。国債リターンは水準・傾き・曲率という少数因子でほぼ説明されるため、
# 標本共分散はほぼ低ランク（悪条件）になります。短い履歴での推定誤差が最適化で
# 増幅され、極端かつ不安定な重みが出ることを確認します。
#
# ### リターン履歴の合成
#
# 各銘柄の日次リターンを $r_{i,t} = -\sum_k \text{KRD}_{i,k}\,f_{k,t}
# + \varepsilon_{i,t}$ とします。$f_{k,t}$ はキーテナーのゼロレート変化
# （少数の共通因子）、$\varepsilon$ は小さな固有ノイズです。KRD がリターンの
# 共通変動を支配するので、銘柄間相関は高く、共分散は悪条件になります。

# %%
def simulate_returns(model, n_days, rng):
    K = model["krd"]                                   # (N, key)
    n, m = K.shape
    # 因子（キーテナーのゼロレート変化, bp単位）は相関を持つ少数ショック。
    fac_vol = np.array([6.0, 5.0, 4.5, 4.0, 3.5]) * BP  # 日次 bp→小数
    common = rng.normal(size=(n_days, 1)) * 0.6         # 水準ショック（全期共通）
    fac = rng.normal(size=(n_days, m)) * fac_vol + common * fac_vol
    idio = rng.normal(size=(n_days, n)) * (0.5 * BP)    # 小さな固有ノイズ
    # リターン ≈ -KRD·Δzero + ノイズ（金利上昇で価格下落）
    ret = -(fac @ K.T) + idio
    return ret                                          # (n_days, N)


def mean_variance(mu, Sigma, gamma):
    n = mu.size
    w = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(Sigma))),
                      [cp.sum(w) == 1])
    prob.solve()
    return np.asarray(w.value).ravel()


# 履歴日数は銘柄数(40)よりやや長い60日にとり、標本共分散を full-rank に保つ。
# それでも金利因子が支配的で共分散は悪条件（条件数が大）になり、推定誤差が
# 最適化で増幅される様子を示す。
rng = np.random.default_rng(0)
N_DAYS = 60
gamma = 200.0

ret = simulate_returns(model, N_DAYS, rng)
mu_hat = ret.mean(axis=0)
Sigma_hat = np.cov(ret, rowvar=False)

cond = np.linalg.cond(Sigma_hat)
w_mv = mean_variance(mu_hat, Sigma_hat, gamma)
print(f"標本共分散の条件数 : {cond:.3e}   （悪条件ほど大）")
print(f"平均分散の重み  : 最大 {w_mv.max()*100:.0f}%  最小 {w_mv.min()*100:.0f}%")
print(f"ロング総額 {np.sum(w_mv[w_mv>0])*100:.0f}%  ショート総額 {np.sum(-w_mv[w_mv<0])*100:.0f}%")
print(f"（予算1に対しグロス {np.sum(np.abs(w_mv))*100:.0f}% ＝ 大きいほどレバレッジ過大）")

# 悪条件かつ極端なロング・ショートが出ること（実務では制御不能）を確認する。
assert cond > 1e4, "共分散が悪条件になっていない"
assert w_mv.min() < 0 and np.sum(np.abs(w_mv)) > 5.0, "平均分散が極端なレバレッジを生んでいない"

# %% [markdown]
# ### 履歴を再サンプルしたときの重みの不安定さ
#
# 独立な履歴を複数回引き直し、そのたびに平均分散最適化を解いて重みのばらつきを
# 見ます。推定誤差が最適化で増幅されるため、重みは履歴ごとに大きく振れます。
# 対照として、本文の因子制約アプローチ（KRD±0.1・回転率20%）の重みも並べます。

# %%
n_boot = 40
mv_weights = np.zeros((n_boot, model["ids"].size))
for b in range(n_boot):
    rb = simulate_returns(model, N_DAYS, np.random.default_rng(1000 + b))
    mv_weights[b] = mean_variance(rb.mean(axis=0), np.cov(rb, rowvar=False), gamma)

mv_std = mv_weights.std(axis=0)
sol_fac = solve_max_yield(model, w0, krd_band=0.1, turnover=0.20)

print(f"平均分散: 重みの銘柄別標準偏差 平均 {mv_std.mean()*100:.0f}%  最大 {mv_std.max()*100:.0f}%")
print(f"平均分散: 重みレンジ 平均 [{mv_weights.min()*100:.0f}%, {mv_weights.max()*100:.0f}%]")
print(f"因子制約: 重みレンジ [{sol_fac['w'].min()*100:.1f}%, {sol_fac['w'].max()*100:.1f}%]"
      f"（ロングオンリー・回転率制約で安定）")
print(f"因子制約: 重みの銘柄別標準偏差 {sol_fac['w'].std()*100:.2f}%（決定的で再サンプルに不変）")

# %%
fig, ax = plt.subplots(figsize=(10, 4))
order = np.argsort(model["mats"])
ax.errorbar(range(order.size), mv_weights.mean(axis=0)[order] * 100,
            yerr=mv_std[order] * 100, fmt="o", color="indianred", ms=3,
            capsize=2, label="平均分散（±1σ, 再サンプル）")
ax.plot(range(order.size), sol_fac["w"][order] * 100, "s-", color="steelblue",
        ms=4, label="因子制約（KRD±0.1・回転率20%）")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(range(order.size))
ax.set_xticklabels([f'{m:.0f}' for m in model["mats"][order]], fontsize=7)
ax.set_xlabel("満期（年, 昇順）")
ax.set_ylabel("ウェイト (%)")
ax.set_title("平均分散法（不安定・極端）と 因子制約アプローチ（安定）の比較")
ax.legend()
fig.tight_layout()
plt.show()

# 平均分散の重みは因子制約よりはるかにばらつく（不安定）。
assert mv_std.mean() > 1.0, "平均分散の重みが十分に不安定でない"
assert mv_std.mean() > 100.0 * sol_fac["w"].std(), "平均分散が因子制約より安定してしまった"

# %% [markdown]
# **解釈。** 標本共分散の条件数は大きく（銘柄が少数の金利因子でほぼ説明され
# 悪条件）、その逆行列を通す平均分散最適化は、短い履歴の推定誤差を増幅して
# 極端なロング・ショートを生みます。履歴を引き直すたびに重みが大きく振れ、
# 実運用に耐えません。これが本文で述べた「共分散の悪条件」「期待リターンの推定
# 誤差の増幅（error maximization）」の数値的な現れです。因子制約アプローチは
# 共分散を反転せず、KRD エクスポージャをベンチに固定しつつ回転率とロングオンリーで
# 解を安定させるため、入力の揺れに対して頑健な、実務的なポートフォリオを与えます。
