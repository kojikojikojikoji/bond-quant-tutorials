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
# # S10-2 演習 解答例
#
# インデックス複製の演習2問の解答例です。本文と同じ合成 JGB 母集団・KRD 行列・
# シナリオ生成を最小構成で組み直し、TE の逓減則とセル法／ファクター法の比較を
# 数値で確かめます。

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

from bondlab.curve import bootstrap_par
from bondlab.analytics import bump_curve

np.random.seed(0)

DATA = Path("data/samples/synthetic_jgb_universe.csv")
if not DATA.exists():
    for up in (Path(".."), Path("../.."), Path("../../..")):
        if (up / DATA).exists():
            DATA = up / DATA
            break

KEY_TENORS = np.array([2.0, 5.0, 10.0, 20.0, 30.0])
KRD_WIDTH = 4.0
KRD_SIZE = 1e-4
MAX_MAT = 39.5


# %% [markdown]
# ## 準備：本文と同じパイプラインを再構築
#
# パーカーブ・合成母集団・キャッシュフロー・KRD 行列・時価総額加重・シナリオを
# まとめて作ります。中身は本文の各関数と同一です。

# %%
def par_yield(t):
    return 0.001 + 0.021 * (1.0 - np.exp(-t / 8.0)) + 0.001 * (t / 40.0)


def build_universe(df, n_total, seed):
    rng = np.random.default_rng(seed)
    base = df.copy()
    base["maturity_years"] = base["maturity_years"].clip(upper=MAX_MAT)
    n_extra = n_total - len(base)
    mat = rng.uniform(1.05, MAX_MAT, size=n_extra)
    cpn = np.clip(par_yield(mat) + rng.normal(0.0, 0.0015, n_extra), 0.001, 0.025)
    cpn = np.round(cpn / 0.001) * 0.001
    rc = rng.normal(0.0, 4.0, n_extra)
    extra = pd.DataFrame({
        "bond_id": [f"JGB{100 + i:03d}" for i in range(n_extra)],
        "maturity_years": mat, "coupon": cpn, "rich_cheap_bp": rc,
    })
    univ = pd.concat([base, extra], ignore_index=True)
    anchor = np.exp(-((univ["maturity_years"].values - 10.0) / 12.0) ** 2)
    size = np.exp(rng.normal(0.0, 0.4, len(univ))) * (1.0 + 1.5 * anchor)
    univ["amount"] = np.round(size * 2.0, 3)
    return univ


def bond_cashflows(mat, cpn, freq=2):
    k = np.arange(0, int(np.ceil(mat * freq)) + 1)
    times = mat - k / freq
    times = np.sort(times[times > 1e-9])
    cfs = np.full_like(times, 100.0 * cpn / freq)
    cfs[-1] += 100.0
    return times, cfs


def price_on_curve(curve, times, cfs):
    return float(np.sum(cfs * curve.discount(times)))


def krd_matrix(curve, cfs_list, tenors, width, size):
    bumped = {(k, s): bump_curve(curve, float(tenors[k]), s * size, width)
              for k in range(len(tenors)) for s in (+1, -1)}
    K = np.empty((len(cfs_list), len(tenors)))
    for i, (times, cfs) in enumerate(cfs_list):
        p0 = price_on_curve(curve, times, cfs)
        for k in range(len(tenors)):
            pu = price_on_curve(bumped[(k, +1)], times, cfs)
            pd_ = price_on_curve(bumped[(k, -1)], times, cfs)
            K[i, k] = -(pu - pd_) / (2.0 * size * p0)
    return K


def stratified_select(univ, w, n, edges):
    buckets = np.digitize(univ["maturity_years"].values, edges)
    bucket_ids = np.unique(buckets)
    bw = np.array([w[buckets == b].sum() for b in bucket_ids])
    alloc = np.ones(len(bucket_ids), dtype=int)
    remaining = n - alloc.sum()
    if remaining > 0:
        frac = bw / bw.sum() * remaining
        add = np.floor(frac).astype(int)
        alloc += add
        for j in np.argsort(-(frac - add)):
            if alloc.sum() >= n:
                break
            alloc[j] += 1
    sel = []
    for b, a in zip(bucket_ids, alloc):
        pool = np.where(buckets == b)[0]
        a = min(a, len(pool))
        top = pool[np.argsort(-univ["amount"].values[pool])[:a]]
        sel.extend(top.tolist())
    return np.array(sorted(sel))


def cell_weights(univ, w, sel, edges):
    buckets = np.digitize(univ["maturity_years"].values, edges)
    x = np.zeros(len(sel))
    sel_buckets = buckets[sel]
    for b in np.unique(buckets):
        cell_w = w[buckets == b].sum()
        reps = np.where(sel_buckets == b)[0]
        if len(reps) > 0:
            x[reps] += cell_w / len(reps)
    return x / x.sum()


def match_krd(K_sel, b):
    n = K_sel.shape[0]
    x = cp.Variable(n, nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(K_sel.T @ x - b)), [cp.sum(x) == 1]).solve()
    xv = np.clip(np.array(x.value).ravel(), 0.0, None)
    return xv / xv.sum()


def scenario_returns(K, dur, n_scen, seed):
    rng = np.random.default_rng(seed)
    tenors = KEY_TENORS
    level = np.ones_like(tenors)
    slope = (tenors - tenors.mean()) / tenors.std()
    curv = -((tenors - 10.0) / 10.0) ** 2
    curv = curv - curv.mean()
    load = np.vstack([level, slope, curv])
    fac_vol = np.array([4.5e-4, 2.5e-4, 1.5e-4])
    dz = (rng.standard_normal((n_scen, 3)) * fac_vol) @ load
    r_sys = -dz @ K.T
    sigma_i = (1.2 / 1e4) * np.maximum(dur, 1.0)
    eps = rng.standard_normal((n_scen, len(dur))) * sigma_i
    return r_sys + eps


def to_full(x, sel, n_all):
    xf = np.zeros(n_all)
    xf[sel] = x
    return xf


def tracking_error(R, x_full, w):
    return float(np.std(R @ (x_full - w)) * np.sqrt(252) * 1e4)


grid = np.arange(1.0, 41.0, 1.0)
curve = bootstrap_par(grid, par_yield(grid), frequency=1)
univ = build_universe(pd.read_csv(DATA), n_total=300, seed=42)
cfs_list = [bond_cashflows(m, c) for m, c in zip(univ.maturity_years, univ.coupon)]
univ["price"] = [price_on_curve(curve, t, c) for t, c in cfs_list]
K = krd_matrix(curve, cfs_list, KEY_TENORS, KRD_WIDTH, KRD_SIZE)
dur = K.sum(axis=1)
w = (univ["price"].values * univ["amount"].values)
w = w / w.sum()
index_krd = w @ K
R_out = scenario_returns(K, dur, n_scen=3000, seed=2)
TENOR_EDGES = np.array([0.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 40.0])
print(f"母集団 {len(univ)} 銘柄, インデックス実効デュレーション {w @ dur:.2f}")

# %% [markdown]
# ## 演習1：トラッキングエラーの逓減則
#
# $n$ を 10〜200 で振り、アウトオブサンプル TE を測ります。両対数（log-log）で
# $\log \mathrm{TE}$ を $\log n$ に回帰し、傾きが $-1/2$ 前後になることを確かめます。
# 理論式の個別リスク項が $\sum_i (x_i-w_i)^2 \sigma_i^2 \sim 1/n$ に従い、TE は
# その平方根なので $n^{-1/2}$ が下限的な目安です。実際には抽出銘柄が少ないうちは
# KRD の残差（系統リスク）も一緒に減るため、傾きは $-1/2$ よりやや急になりえます。

# %%
def build_replication(n):
    sel = stratified_select(univ, w, n, TENOR_EDGES)
    x = match_krd(K[sel], index_krd)
    return tracking_error(R_out, to_full(x, sel, len(univ)), w)


n_grid = np.array([10, 15, 22, 32, 45, 65, 90, 120, 160, 200])
te = np.array([build_replication(int(n)) for n in n_grid])
for n, t in zip(n_grid, te):
    print(f"  n={n:4d}  TE={t:6.1f} bp")

slope, intercept = np.polyfit(np.log(n_grid), np.log(te), 1)
print(f"\nlog-log 回帰の傾き = {slope:.3f}（理論の目安 -0.5 前後）")
assert -1.0 < slope < -0.30, "TE は概ね n^(-1/2) 前後で逓減するはず"

# TE を 15bp 以下にする最小 n。
below = n_grid[te <= 15.0]
n_star = int(below[0]) if len(below) else None
print(f"TE ≤ 15bp を満たす最小 n = {n_star}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
axes[0].plot(n_grid, te, "o-")
axes[0].axhline(15.0, color="crimson", ls="--", lw=0.8, label="15bp 目標")
axes[0].set_title("演習1：TE の逓減（線形軸）")
axes[0].set_xlabel("n"); axes[0].set_ylabel("TE (bp)"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].loglog(n_grid, te, "o-", label="実測")
axes[1].loglog(n_grid, np.exp(intercept) * n_grid ** slope, "--",
               color="gray", label=f"傾き {slope:.2f}")
axes[1].set_title("演習1：TE の逓減（両対数）")
axes[1].set_xlabel("n"); axes[1].set_ylabel("TE (bp)"); axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
fig.tight_layout()
plt.show()

print(f"考察：TE は n^(-1/2) で減り、n={n_star} で 15bp を切る。以降は逓減が寝るため、")
print("15bp 前後を許容できるなら n_star 近傍が取引・管理コストとの妥当な妥協点になる。")

# %% [markdown]
# ## 演習2：セル法 vs KRD ファクター法（バケット細分化の影響）
#
# 年限バケットを粗く（4 バケット）／既定（7 バケット）／細かく（10 バケット）
# 変え、$n=30$ で両手法のアウトオブサンプル TE を比べます。セル法はセル内の
# 年限分布を代表銘柄へ丸めるため細分化で改善しますが、KRD ファクター法は
# 抽出銘柄の KRD さえ一致させられればバケット数に鈍感なはずです。

# %%
edge_sets = {
    "粗い(4)": np.array([0.0, 5.0, 10.0, 20.0, 40.0]),
    "既定(7)": TENOR_EDGES,
    "細かい(10)": np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 17.0, 25.0, 40.0]),
}

n = 30
rows = []
for label, edges in edge_sets.items():
    sel = stratified_select(univ, w, n, edges)
    x_cell = cell_weights(univ, w, sel, edges)
    x_fac = match_krd(K[sel], index_krd)
    rows.append({
        "バケット": label,
        "抽出n": len(sel),
        "セル法 TE(bp)": tracking_error(R_out, to_full(x_cell, sel, len(univ)), w),
        "ファクター法 TE(bp)": tracking_error(R_out, to_full(x_fac, sel, len(univ)), w),
    })
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

cell_span = res["セル法 TE(bp)"].max() - res["セル法 TE(bp)"].min()
fac_span = res["ファクター法 TE(bp)"].max() - res["ファクター法 TE(bp)"].min()
print(f"\nバケット数によるTE変動幅  セル法={cell_span:.1f}bp  ファクター法={fac_span:.1f}bp")
assert cell_span > fac_span, "セル法の方がバケット細分化に敏感なはず"

fig, ax = plt.subplots(figsize=(7.5, 4.3))
xpos = np.arange(len(res))
ax.bar(xpos - 0.2, res["セル法 TE(bp)"], width=0.4, label="セル法")
ax.bar(xpos + 0.2, res["ファクター法 TE(bp)"], width=0.4, label="ファクター法")
ax.set_xticks(xpos); ax.set_xticklabels(res["バケット"])
ax.set_title("演習2：バケット細分化と両手法の TE (n=30)")
ax.set_ylabel("アウトオブサンプル TE (bp)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
plt.show()

print("考察：セル法はバケットを細かくするほど代表銘柄の年限が指数へ寄り TE が下がる。")
print("ファクター法は KRD ベクトルを直接一致させるため、バケット数に対してほぼ横ばい。")
print("運用初期でセル法から入り、精度が要るならファクター法へ移すのが定石。")
