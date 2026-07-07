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
# # S3-4 演習 解答例
#
# 本編 `notebooks/S03_risk/nb_0304_scenario_pca.py` の演習2問の解答。
# 実データからゼロレート変化の PCA を取り、負荷ベクトルの解釈（演習1）と、
# 複数の仮想シフトによるワースト損失の報告（演習2）を行う。

# %% [markdown]
# ## 準備：ゼロレート変化の PCA を再構築
#
# 本編と同じ手順で、パー利回りパネル → 日次ゼロレート → 変化行列 → PCA を作る。
# 自作 PCA が `bondlab.analytics.pca` と一致することも再確認する。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `pca_scratch(changes)` | 変化行列（行=日, 列=テナー） | `dict(eigenvalues, eigenvectors, explained_ratio)` | 共分散の固有値分解による PCA |
# | `zero_panel_from_par(par_panel, tenors, grid)` | パー利回りパネル, 報告テナー, 支払グリッド | ゼロレートパネル | 日次にパー→補間→ブートストラップ→ゼロ |
# | `present_value_on_curve(bond, curve, asof)` | 債券, 割引カーブ, 評価日 | 現在価値 | カーブ割引でキャッシュフローを評価 |
# | `shift_curve(tenors, base_zero, dz)` | テナー, 基準ゼロ, ゼロ変化 | `DiscountCurve` | ゼロを bp シフトした新カーブ |
# | `scenario_pnl(port, tenors, base_zero, dz, asof)` | ポートフォリオ, テナー, 基準ゼロ, シフト, 評価日 | シナリオ損益 | 全建玉を再評価し損益を合計 |

# %%
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bondlab
from bondlab import analytics
from bondlab import curve as blcurve
from bondlab.bond import FixedRateBond

SEED = 20260707
print("bondlab version:", bondlab.__version__)


def pca_scratch(changes: np.ndarray) -> dict:
    """変化行列（行=日, 列=テナー）の主成分分析を固有値分解で行う。"""
    x = np.asarray(changes, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return dict(eigenvalues=vals, eigenvectors=vecs, explained_ratio=vals / vals.sum())


def zero_panel_from_par(par_panel: pd.DataFrame, tenors: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """パー利回りパネルから、各日のゼロレートを報告テナー位置で読んだ行列を返す。"""
    rows = []
    for _date, par_row in par_panel.iterrows():
        par_on_grid = np.interp(grid, tenors, par_row.to_numpy(dtype=float))
        crv = blcurve.bootstrap_par(grid, par_on_grid, frequency=1)
        rows.append(crv.zero_rate(tenors))
    return np.asarray(rows)


panel = pd.read_csv("data/samples/synthetic_ust_par_panel.csv")
par_wide = panel.pivot(index="date", columns="tenor", values="par_yield").sort_index()
report_tenors = par_wide.columns.to_numpy(dtype=float)
n = report_tenors.size
payment_grid = np.arange(1.0, 31.0)

zero_panel = zero_panel_from_par(par_wide, report_tenors, payment_grid)
dz = np.diff(zero_panel, axis=0)

pca_real = pca_scratch(dz)
assert np.allclose(pca_real["eigenvalues"], analytics.pca(dz)["eigenvalues"], atol=1e-12)

# 符号規約（本編と同一）。
loads = pca_real["eigenvectors"][:, :3].copy()
if loads[:, 0].mean() < 0:
    loads[:, 0] = -loads[:, 0]
if loads[-1, 1] < 0:
    loads[:, 1] = -loads[:, 1]
if loads[n // 2, 2] < 0:
    loads[:, 2] = -loads[:, 2]

print("累積寄与率(第1-3):", round(pca_real["explained_ratio"][:3].sum(), 4))

# %% [markdown]
# ## 演習1：負荷ベクトルの可視化とレベル／スロープ／曲率の解釈
#
# 3つの負荷ベクトルを重ね描きし、符号と単調性から因子を同定する。

# %%
fig, ax = plt.subplots(figsize=(7, 4))
labels = [("PC1", "レベル"), ("PC2", "スロープ"), ("PC3", "曲率")]
for k, (name, jp) in enumerate(labels):
    ax.plot(report_tenors, loads[:, k], marker="o", label=f"{name} ({jp})")
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_xlabel("tenor (years)")
ax.set_ylabel("loading")
ax.set_title("S3-4 exercise1: PCA loadings")
ax.legend()
fig.tight_layout()

# 定量的な同定根拠。
same_sign = np.all(np.sign(loads[:, 0]) == np.sign(loads[0, 0]))
pc2_ends_flip = loads[0, 1] * loads[-1, 1] < 0
belly = n // 2
pc3_hump = (loads[belly, 2] - loads[0, 2]) * (loads[belly, 2] - loads[-1, 2]) > 0

print(f"PC1: 全テナー同符号 = {same_sign}  → 全期間が同方向に動くレベル（パラレルシフト）")
print(f"PC2: 両端で符号反転 = {pc2_ends_flip}  → 短長で逆向きのスロープ（スティープ/フラット）")
print(f"PC3: 中期が両端と逆 = {pc3_hump}  → 山（谷）型の曲率（バタフライ）")

# %% [markdown]
# **解釈**:
#
# - **PC1（レベル＝パラレル）**: 負荷が全テナーで同符号かつ概ね平坦。全期間の
#   ゼロレートが同方向へ動く成分で、寄与率が最も大きい。パラレルシフトに対応。
# - **PC2（スロープ＝スティープ／フラット）**: 短期側と長期側で符号が反転し、
#   テナーに対して単調。長短スプレッドの拡大・縮小を表し、正側がスティープ化、
#   負側がフラット化。
# - **PC3（曲率＝バタフライ）**: 中期（belly）が両端（wings）と逆符号の山谷型。
#   中期と両端が逆に動くバタフライに対応する。寄与率は小さい。

# %% [markdown]
# ## 演習2：複数の仮想シフトとワースト損失
#
# 仮想ポートフォリオを組み、パラレル・スティープ・フラット・バタフライを
# 複数の強度で当てて損益を計算し、ワースト損失を報告する。

# %%
asof = dt.date(2026, 3, 31)
base_zero = zero_panel[-1]


def present_value_on_curve(bond: FixedRateBond, curve: blcurve.DiscountCurve, asof: dt.date) -> float:
    """割引カーブでキャッシュフローを割り引いた債券の現在価値（dirty PV）。"""
    pv = 0.0
    for d, c in bond.cashflows():
        if d > asof:
            pv += c * curve.discount((d - asof).days / 365.0)
    return pv


def shift_curve(tenors: np.ndarray, base_zero: np.ndarray, dz_shift: np.ndarray) -> blcurve.DiscountCurve:
    """ゼロレートを dz_shift だけ動かした割引カーブを構築する。"""
    dfs = np.exp(-(base_zero + dz_shift) * tenors)
    return blcurve.DiscountCurve(tenors, dfs, interp="linear_zero")


def scenario_pnl(port, tenors, base_zero, dz_shift, asof) -> float:
    """ポートフォリオのシナリオ損益（変化後 − 基準、額面ウエイト加重）。"""
    base_curve = shift_curve(tenors, base_zero, np.zeros_like(base_zero))
    scen_curve = shift_curve(tenors, base_zero, dz_shift)
    return sum(
        w * (present_value_on_curve(b, scen_curve, asof) - present_value_on_curve(b, base_curve, asof))
        for b, w in port
    )


portfolio = [
    (FixedRateBond(dt.date(2025, 3, 31), dt.date(2028, 3, 31), 0.030, frequency=2), 40.0),
    (FixedRateBond(dt.date(2025, 3, 31), dt.date(2033, 3, 31), 0.038, frequency=2), 35.0),
    (FixedRateBond(dt.date(2025, 3, 31), dt.date(2046, 3, 31), 0.044, frequency=2), 25.0),
]
base_curve = shift_curve(report_tenors, base_zero, np.zeros_like(base_zero))
base_value = sum(w * present_value_on_curve(b, base_curve, asof) for b, w in portfolio)

# 型 × 強度のグリッドで仮想シフトを生成。
steep_unit = np.linspace(-1.0, 1.0, n)                       # スティープの単位形状
fly_unit = loads[:, 2] / np.max(np.abs(loads[:, 2]))          # バタフライの単位形状（PC3）

scenarios = {}
for bp in (50, 100, 150):
    scenarios[f"パラレル +{bp}bp"] = np.full(n, bp / 1e4)
    scenarios[f"パラレル -{bp}bp"] = np.full(n, -bp / 1e4)
    scenarios[f"スティープ ±{bp}bp"] = steep_unit * (bp / 1e4)
    scenarios[f"フラット ±{bp}bp"] = -steep_unit * (bp / 1e4)
    scenarios[f"バタフライ ±{bp}bp"] = fly_unit * (bp / 1e4)

rows = [(name, scenario_pnl(portfolio, report_tenors, base_zero, shift, asof))
        for name, shift in scenarios.items()]
pnl_df = pd.DataFrame(rows, columns=["シナリオ", "損益"]).sort_values("損益").reset_index(drop=True)
pnl_df["損益(bp of PV)"] = pnl_df["損益"] / base_value * 1e4

print(f"基準価値: {base_value:,.2f}\n")
print(pnl_df.round(4).to_string(index=False))

worst = pnl_df.iloc[0]
print(f"\nワースト損失: {worst['損益']:,.4f}  （{worst['シナリオ']}, {worst['損益(bp of PV)']:.1f} bp of PV）")

# %% [markdown]
# **考察**:
#
# - ワースト損失は**パラレル +150bp**で生じる。ポートフォリオは全建玉が正の
#   デュレーションを持つため、金利上昇（レベル因子の $+$ 側）が最も痛い。
#   同じ強度ならスティープ／フラット／バタフライよりパラレルの影響が大きく、
#   これは PC1（レベル）の寄与率が支配的であることと整合する。
# - スティープとフラットは損益の符号が逆だが、20年債のウエイトがあるため
#   長期金利が上がるスティープ側の損失がやや大きい。
# - バタフライの損益は最も小さい。中期と両端が相殺し、かつ曲率因子の変動が
#   小さいことの反映である。
#
# ストレステストの実務では、こうした「型 × 強度」の格子に加え、本編で見た
# ヒストリカル最悪日や PCA の $k$ シグマシナリオを併用し、想定の抜けを補い合う。

# %%
# 感応度の裏取り: パラレル微小シフトの数値デュレーションを確認する。
bump = 1e-4
pnl_up = scenario_pnl(portfolio, report_tenors, base_zero, np.full(n, bump), asof)
num_duration = -pnl_up / base_value / bump
print(f"ポートフォリオの数値デュレーション（パラレル+1bp から）: {num_duration:.3f} 年")
print("正のデュレーション → 金利上昇で損失、の符号がワースト結果と一致")
