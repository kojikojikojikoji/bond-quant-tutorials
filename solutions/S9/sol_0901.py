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
# # S9-1 演習 解答例
#
# カーブ対比 rich/cheap 分析の演習2問の解答例です。本文と同じ
# `bondlab.curve` の NSS フィットを使い、残差パネル・Z スコア・AR(1) 半減期を
# 再構築します。

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bondlab.curve import fit_nss, nss

np.random.seed(0)

LAM_GRID = np.linspace(0.8, 8.0, 8)
BP = 1e4


def fit_date_residuals(mat, y, lam_grid=LAM_GRID):
    mat = np.asarray(mat, dtype=float)
    y = np.asarray(y, dtype=float)
    fit = fit_nss(mat, y, lam_grid=lam_grid)
    return (y - nss(mat, **fit)) * BP


def build_residual_panel(panel, lam_grid=LAM_GRID):
    rows = {}
    for date, grp in panel.groupby("date"):
        grp = grp.sort_values("maturity_years")
        resid = fit_date_residuals(grp["maturity_years"].values, grp["yield"].values, lam_grid)
        rows[date] = pd.Series(resid, index=grp["bond_id"].values)
    wide = pd.DataFrame(rows).T
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def residual_zscores(resid_wide):
    return (resid_wide - resid_wide.mean(axis=0)) / resid_wide.std(axis=0, ddof=1)


def ar1_halflife(series):
    r = np.asarray(series, dtype=float)
    r = r[~np.isnan(r)]
    y, x = r[1:], r[:-1]
    n = y.size
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (n - 2)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_phi = np.sqrt(cov[1, 1])
    phi = beta[1]
    halflife = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.nan
    return dict(phi=phi, halflife=halflife, se=se_phi, tstat=phi / se_phi)


universe = pd.read_csv("data/samples/synthetic_jgb_universe.csv")
panel = pd.read_csv("data/samples/synthetic_jgb_yield_panel.csv")
resid_panel = build_residual_panel(panel)
z_panel = residual_zscores(resid_panel)

# %% [markdown]
# ## 演習 1：過去日での cheapest 抽出と流動性込み考察
#
# 期間中央付近の営業日を1つ選び、その日の Z スコアで cheapest 上位5銘柄を
# 抽出します。各銘柄の半減期と残差水準から、一時的な割安か構造的な割安かを
# 論じます。

# %%
mid_idx = len(z_panel) // 2
target_date = z_panel.index[mid_idx]
z_day = z_panel.loc[target_date]
resid_day = resid_panel.loc[target_date]

cheapest5 = (
    pd.DataFrame({"z_score": z_day, "resid_bp": resid_day})
    .merge(universe.set_index("bond_id")[["maturity_years", "coupon"]],
           left_index=True, right_index=True)
    .sort_values("z_score", ascending=False)
    .head(5)
)

# 各銘柄の半減期を付す
hl = {b: ar1_halflife(resid_panel[b]) for b in cheapest5.index}
cheapest5 = cheapest5.assign(
    halflife_days=[hl[b]["halflife"] for b in cheapest5.index],
    phi=[hl[b]["phi"] for b in cheapest5.index],
)

print(f"対象営業日: {target_date.date()}")
print(cheapest5.round(3).to_string())

# %% [markdown]
# ### 考察
#
# - 半減期 `halflife_days` が短く Z スコアが大きい銘柄は、カーブから一時的に
#   外れただけで、収束を狙うロングが報われやすい「一時的な割安」です。
# - 半減期が長い、または NaN（$\phi$ が 0 以下や 1 以上で回帰が定義できない）
#   銘柄は、残差が縮まず高止まりしており、流動性の低さへの恒常的な対価
#   （off-the-run プレミアム）である可能性が高い「構造的な割安」です。
# - このデータには出来高・ビッド/アスク・on-the-run 区分といった流動性指標が
#   無いため、両者を確定的には切り分けられません。実務では流動性の代理変数で
#   残差を回帰し、流動性成分を除いた純粋な割安で判定します。

# %%
# 半減期と残差水準の簡易分類（参考）
def classify(row):
    hl_v = row["halflife_days"]
    if np.isnan(hl_v) or hl_v > 30:
        return "構造的な割安の疑い（回帰遅い/不能）"
    return "一時的な割安（収束期待）"

print("\n分類:")
for b, row in cheapest5.iterrows():
    print(f"  {b}: {classify(row)}  (半減期 {row['halflife_days']:.1f} 日)")

# %% [markdown]
# ## 演習 2：全銘柄の半減期分布と取引妙味
#
# 全銘柄について残差の AR(1) 半減期を推定し、分布を可視化します。半減期が短い
# 群と長い群を比較し、収束トレードの対象妥当性を $\phi$ の有意性込みで論じます。

# %%
records = []
for b in resid_panel.columns:
    est = ar1_halflife(resid_panel[b])
    records.append({
        "bond_id": b,
        "phi": est["phi"],
        "halflife_days": est["halflife"],
        "tstat": est["tstat"],
    })
hl_all = pd.DataFrame(records).set_index("bond_id")

valid = hl_all["halflife_days"].dropna()
print(f"半減期が定義できた銘柄数: {valid.size} / {len(hl_all)}")
print(f"半減期の中央値: {valid.median():.1f} 日,  平均: {valid.mean():.1f} 日")

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(valid, bins=15, color="steelblue", edgecolor="white")
ax.axvline(valid.median(), color="firebrick", ls="--", lw=1.5,
           label=f"中央値 {valid.median():.1f} 日")
ax.set_xlabel("平均回帰半減期 (日)")
ax.set_ylabel("銘柄数")
ax.set_title("残差の AR(1) 半減期の分布")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 短半減期群 vs 長半減期群

# %%
med = valid.median()
short_grp = hl_all.loc[valid.index][valid <= med]
long_grp = hl_all.loc[valid.index][valid > med]

print(f"短半減期群 (<= {med:.1f} 日): {len(short_grp)} 銘柄")
print(f"  phi 平均 {short_grp['phi'].mean():.3f},  |t値| 平均 {short_grp['tstat'].abs().mean():.1f}")
print(f"長半減期群 (>  {med:.1f} 日): {len(long_grp)} 銘柄")
print(f"  phi 平均 {long_grp['phi'].mean():.3f},  |t値| 平均 {long_grp['tstat'].abs().mean():.1f}")

# %% [markdown]
# ### 考察
#
# - 収束トレードの対象は**短半減期群**が妥当です。乖離が数日〜十数日で平均へ
#   戻るため、資本を長く拘束せず、回転を効かせられます。
# - 長半減期群は $\phi$ が 1 に近く、残差が縮む速度が遅い、あるいは回帰の
#   統計的裏付け（$t$ 値）が弱い銘柄を含みます。これらは流動性プレミアム等の
#   構造要因で恒常的に乖離している可能性があり、収束益を当てにくい対象です。
# - 実運用では、半減期の短さ・$\phi$ の有意性・現在の Z スコアの大きさ・
#   取引コスト（流動性）を組み合わせ、期待収束益が往復コストを上回る銘柄に
#   絞ります。
