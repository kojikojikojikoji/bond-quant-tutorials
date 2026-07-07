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
# # S5-4 演習 解答例

# %% [markdown]
# ## 準備
#
# 本編のキャリブレータを最小限だけ再掲する。合成カーブ、QuantLib カーブへの
# 写し、`SwaptionHelper` 生成、合成ボラ面（Hull-White 整合／市場風）、キャリブ
# レーション、再現誤差の各関数を手元に置く。

# %%
import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql

import bondlab
from bondlab.curve import bootstrap_par

print("bondlab version:", bondlab.__version__)

SEED = 20260707
rng = np.random.default_rng(SEED)
plt.rcParams["axes.unicode_minus"] = False

CAL_DC = ql.Actual365Fixed()
TODAY = ql.Date(1, 1, 2024)
ql.Settings.instance().evaluationDate = TODAY

par_tenors = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30], dtype=float)
par_rates = np.array(
    [0.030, 0.032, 0.034, 0.0355, 0.0365, 0.038, 0.0395, 0.0405, 0.0410, 0.0415]
)
curve = bootstrap_par(par_tenors, par_rates, frequency=1)

EXPIRIES = [1, 2, 3, 5, 7]
TENORS = [1, 2, 3, 5, 7]


def bondlab_to_ql_curve(curve, max_t=30.0, n_nodes=60):
    node_t = np.concatenate([[0.0], np.linspace(0.5, max_t, n_nodes)])
    dates = [TODAY + ql.Period(int(round(t * 365)), ql.Days) for t in node_t]
    dfs = [1.0] + [float(curve.discount(t)) for t in node_t[1:]]
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, CAL_DC))


ts = bondlab_to_ql_curve(curve)
float_index = ql.USDLibor(ql.Period(6, ql.Months), ts)


def make_helpers(grid, engine):
    helpers = []
    for (e, te, vol) in grid:
        h = ql.SwaptionHelper(
            ql.Period(e, ql.Years), ql.Period(te, ql.Years),
            ql.QuoteHandle(ql.SimpleQuote(vol)), float_index, ql.Period(1, ql.Years),
            CAL_DC, CAL_DC, ts,
        )
        h.setPricingEngine(engine)
        helpers.append(h)
    return helpers


def build_synthetic_surface(a_true, sigma_true):
    true_model = ql.HullWhite(ts, a_true, sigma_true)
    true_engine = ql.JamshidianSwaptionEngine(true_model)
    grid = []
    for e in EXPIRIES:
        for te in TENORS:
            h = ql.SwaptionHelper(
                ql.Period(e, ql.Years), ql.Period(te, ql.Years),
                ql.QuoteHandle(ql.SimpleQuote(0.20)), float_index, ql.Period(1, ql.Years),
                CAL_DC, CAL_DC, ts,
            )
            h.setPricingEngine(true_engine)
            vol = h.impliedVolatility(h.modelValue(), 1e-6, 200, 1e-4, 2.0)
            grid.append((e, te, float(vol)))
    return grid


def build_market_like_surface(hump_coef=0.030):
    """本編と同じだがコブ係数を引数化（演習1で振るため）。"""
    grid = []
    for e in EXPIRIES:
        for te in TENORS:
            base = 0.220
            hump = hump_coef * np.exp(-((np.log(e) - np.log(3.0)) ** 2) / 0.8)
            tenor_decay = -0.008 * np.log(te)
            grid.append((e, te, float(base + hump + tenor_decay)))
    return grid


def calibrate_hw(grid, a0=0.05, sigma0=0.01, max_iter=400):
    model = ql.HullWhite(ts, a0, sigma0)
    engine = ql.JamshidianSwaptionEngine(model)
    helpers = make_helpers(grid, engine)
    model.calibrate(helpers, ql.LevenbergMarquardt(), ql.EndCriteria(max_iter, 50, 1e-8, 1e-8, 1e-8))
    a, sigma = model.params()
    return float(a), float(sigma), helpers


def reproduction_errors(grid, helpers):
    n_e, n_t = len(EXPIRIES), len(TENORS)
    vol_err = np.zeros((n_e, n_t))
    for k, (e, te, mkt_vol) in enumerate(grid):
        h = helpers[k]
        model_vol = h.impliedVolatility(h.modelValue(), 1e-6, 200, 1e-4, 2.0)
        vol_err[EXPIRIES.index(e), TENORS.index(te)] = (model_vol - mkt_vol) * 1e4
    return vol_err


def rmse_bp(vol_err):
    return float(np.sqrt(np.mean(vol_err ** 2)))


def plot_error_heatmap(vol_err, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    vmax = max(1.0, np.abs(vol_err).max())
    im = ax.imshow(vol_err, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(TENORS)))
    ax.set_xticklabels([f"{t}y" for t in TENORS])
    ax.set_yticks(range(len(EXPIRIES)))
    ax.set_yticklabels([f"{e}y" for e in EXPIRIES])
    ax.set_xlabel("tenor")
    ax.set_ylabel("expiry")
    ax.set_title(title)
    for i in range(vol_err.shape[0]):
        for j in range(vol_err.shape[1]):
            ax.text(j, i, f"{vol_err[i, j]:.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="model - market (bp)")
    return ax


# %% [markdown]
# ## 演習1：再現誤差ヒートマップと1ファクターの限界
#
# 合成「市場」面へキャリブレーションし、再現誤差ヒートマップを描く。次に、
# 満期方向・テナー方向それぞれの平均絶対誤差を比べ、どちらに強い構造が残るかを
# 定量的に確かめる。

# %%
grid_mkt = build_market_like_surface()
a_mkt, sigma_mkt, helpers_mkt = calibrate_hw(grid_mkt)
vol_err_mkt = reproduction_errors(grid_mkt, helpers_mkt)

print(f"キャリブレート結果 : a = {a_mkt:.5f}, sigma = {sigma_mkt:.5f}")
print(f"再現誤差 RMSE      = {rmse_bp(vol_err_mkt):.1f} bp")

# 満期方向（行）・テナー方向（列）の平均絶対誤差。
by_expiry = np.abs(vol_err_mkt).mean(axis=1)
by_tenor = np.abs(vol_err_mkt).mean(axis=0)
print("\n満期別の平均絶対誤差 (bp):")
for e, v in zip(EXPIRIES, by_expiry):
    print(f"  満期 {e}y : {v:6.1f}")
print("テナー別の平均絶対誤差 (bp):")
for te, v in zip(TENORS, by_tenor):
    print(f"  テナー {te}y : {v:6.1f}")

plot_error_heatmap(vol_err_mkt, "Market-like surface: reproduction error (bp)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### コブの強さと RMSE の関係
#
# コブ係数 `hump_coef` を 0（＝Hull-White が完全再現しやすい平坦寄りの面）から
# 徐々に強めると、再現誤差 RMSE が単調に増える。1ファクターでは吸収できない
# 「面の曲がり」が誤差に直行するためである。

# %%
hump_grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
rmses = []
for hc in hump_grid:
    g = build_market_like_surface(hump_coef=hc)
    _, _, hh = calibrate_hw(g)
    rmses.append(rmse_bp(reproduction_errors(g, hh)))
    print(f"hump_coef = {hc:.2f} -> RMSE = {rmses[-1]:6.1f} bp")

# コブが強いほど RMSE は増える（単調増加）。
assert rmses[-1] > rmses[0]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hump_grid, rmses, "o-", color="steelblue")
ax.set_xlabel("hump coefficient")
ax.set_ylabel("reproduction RMSE (bp)")
ax.set_title("Stronger hump -> larger 1-factor misfit")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈（1ファクターの限界）**：Hull-White の自由パラメータは $a$ と $\sigma$ の
# 2つしかない。$\sigma$ はインプライドボラの全体水準を、$a$ は満期に対するボラの
# 減衰（平均回帰による長期ボラの縮み）を支配する。したがって Hull-White が表現
# できるのは、本質的に「水準」と「満期方向の単調な減衰」までである。
#
# ところが合成「市場」面は、満期 3y 付近にコブを持ち、テナー方向にも別の減衰を
# 持つ。$5\times5=25$ セルの情報を2パラメータへ押し込むため、コブの山と谷を
# 同時に合わせられず、誤差は満期方向に「山では負・裾では正」といった符号反転の
# 構造を残す。上のヒートマップと満期別平均誤差に、その系統性がはっきり出る。
# 実務では、テナー方向の追加自由度が欲しければ2ファクター（G2++）へ、スマイル・
# 満期構造まで欲しければ時間依存 $\sigma(t)$ や局所ボラ型へ拡張する。

# %% [markdown]
# ## 演習2：初期値・重み付けを変えた安定性評価

# %% [markdown]
# ### (a) 初期値の格子に対する収束先の散らばり
#
# 合成 Hull-White 面と合成「市場」面の両方について、初期値 $(a_0, \sigma_0)$ を
# 格子状に振り、収束後のパラメータの散らばりを比べる。合成 Hull-White 面は真の
# 最小がゼロ誤差で鋭いため識別性が高く、どこから出発しても同じ解へ戻る。市場風
# 面は谷が浅く広くなりうるので、散らばりが（相対的に）大きくなりやすい。

# %%
a0_grid = [0.005, 0.02, 0.05, 0.15, 0.40]
s0_grid = [0.002, 0.008, 0.02, 0.05]

grid_syn = build_synthetic_surface(0.06, 0.010)


def stability_scan(grid, label):
    res = []
    for a0 in a0_grid:
        for s0 in s0_grid:
            a, s, _ = calibrate_hw(grid, a0=a0, sigma0=s0)
            res.append((a, s))
    res = np.array(res)
    print(f"[{label}] 収束先 a: 平均 {res[:, 0].mean():.5f} 幅 {np.ptp(res[:, 0]):.2e}"
          f" | sigma: 平均 {res[:, 1].mean():.5f} 幅 {np.ptp(res[:, 1]):.2e}")
    return res


res_syn = stability_scan(grid_syn, "合成HW面")
res_mkt = stability_scan(grid_mkt, "合成『市場』面")

# 合成 HW 面は初期値によらずほぼ一点に収束する（高い識別性）。
assert np.ptp(res_syn[:, 0]) < 5e-3 and np.ptp(res_syn[:, 1]) < 5e-4

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, res, title in [
    (axes[0], res_syn, "HW-consistent surface"),
    (axes[1], res_mkt, "Market-like surface"),
]:
    ax.scatter(res[:, 0], res[:, 1], c="steelblue", s=40, alpha=0.7)
    ax.set_xlabel("calibrated a")
    ax.set_ylabel("calibrated sigma")
    ax.set_title(f"{title}: convergence points")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### (b) 重み付け（対象セルの選択）による最適パラメータの移動
#
# 部分キャリブレーション（対象外セルの重み 0 に相当）で、どのセルを重視するかを
# 変え、最適パラメータの移動量を定量化する。満期を重視する場合とテナーを重視
# する場合で、$a$（減衰）が特に大きく動く。

# %%
def calibrate_subset(grid, keep):
    sub = [(e, te, v) for (e, te, v) in grid if keep(e, te)]
    a, s, _ = calibrate_hw(sub)
    return a, s


scenarios = {
    "全セル均等": lambda e, te: True,
    "満期>=5y 重視": lambda e, te: e >= 5,
    "満期<=2y 重視": lambda e, te: e <= 2,
    "テナー>=5y 重視": lambda e, te: te >= 5,
    "テナー<=2y 重視": lambda e, te: te <= 2,
}
print(f"{'シナリオ':>16s} {'a':>10s} {'sigma':>10s}")
params = {}
for name, keep in scenarios.items():
    a, s = calibrate_subset(grid_mkt, keep)
    params[name] = (a, s)
    print(f"{name:>16s} {a:>10.5f} {s:>10.5f}")

a_vals = np.array([p[0] for p in params.values()])
s_vals = np.array([p[1] for p in params.values()])
print(f"\n重み変更による a の移動幅     = {np.ptp(a_vals):.5f}")
print(f"重み変更による sigma の移動幅 = {np.ptp(s_vals):.5f}")

# %% [markdown]
# **解釈（重み付けと目的関数）**：どのセルを重視するかで最適 $(a, \sigma)$ は
# 明確に動く。とくに満期の重み配分は $a$（ボラの満期減衰）を強く動かす。これは
# キャリブレーションが「面全体の最良近似」であり、重みがその近似の重心を決める
# ためである。
#
# **価格差ベース vs ボラ差ベース**：QuantLib の `SwaptionHelper` は既定で価格
# 残差を返す。価格は満期・テナーが長いセルほどアニュイティ $A$ が大きく本源的な
# 価格水準も上がるため、**価格差ベースは長い年限へ自然に重みが偏る**。一方、
# ボラ差ベースは水準を揃えた比較になり、短い年限の小さな価格差も相対的に効く。
# したがって同じ面でも、価格差最小化は長期の再現を、ボラ差最小化は面全体の
# バランスを優先しやすい、と予想できる。実務では、ヘッジしたい年限の重みを明示
# 的に上げる／`CalibrationHelper` の誤差種別を price から implied-vol に切り替える、
# といった設計判断でこの偏りをコントロールする。
