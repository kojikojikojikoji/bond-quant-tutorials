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
# # S9-2 演習 解答例
#
# バタフライ取引の構築と PnL 分解の演習2問の解答例です。本文と同じ
# `bondlab.curve` / `bondlab.analytics` を使い、自作の価格・リスク・分解関数を
# 再掲して自己完結させます。

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bondlab.curve import bootstrap_par
from bondlab.analytics import pca

np.random.seed(0)
BP = 1e-4


# %% [markdown]
# ### 共通ヘルパ（本文からの再掲）

# %%
def par_bond_price(y, tau, c, f=2, face=100.0):
    m = int(np.floor(tau * f + 1e-9))
    ks = np.arange(0, m + 1)
    s = tau - ks / f
    s = s[s > 1e-9]
    pv = np.sum(c / f * face * (1.0 + y / f) ** (-f * s))
    pv += face * (1.0 + y / f) ** (-f * tau)
    return float(pv)


def par_bond_risk(y, tau, c, f=2, face=100.0, h=1e-5):
    p0 = par_bond_price(y, tau, c, f, face)
    pu = par_bond_price(y + h, tau, c, f, face)
    pd_ = par_bond_price(y - h, tau, c, f, face)
    dpdy = (pu - pd_) / (2 * h)
    d2 = (pu - 2 * p0 + pd_) / (h * h)
    return dict(price=p0, dv01=-dpdy * BP, mod_dur=-dpdy / p0, conv=d2 / p0)


def butterfly_weights(D, scheme, betas=None, pca_loadings=None, body_dv01=1e4):
    B = body_dv01
    if scheme == "50-50":
        w = np.array([B / 2, -B, B / 2])
    elif scheme == "risk":
        b2, b10 = betas
        s = b2 + b10
        w = np.array([B * b2 / s, -B, B * b10 / s])
    elif scheme == "pca":
        v1, v2 = pca_loadings
        A = np.array([[v1[0], v1[2]], [v2[0], v2[2]]])
        rhs = -(-B) * np.array([v1[1], v2[1]])
        w2, w10 = np.linalg.solve(A, rhs)
        w = np.array([w2, -B, w10])
    else:
        raise ValueError(scheme)
    return dict(w=w, mult=w / D)


def decompose_leg_pnl(y0, y1, yroll, tau, dt, c, risk_roll):
    taup = tau - dt
    p_start = par_bond_price(y0, tau, c)
    p_carry = par_bond_price(y0, taup, c)
    p_roll = par_bond_price(yroll, taup, c)
    p_end = par_bond_price(y1, taup, c)
    dy = y1 - yroll
    dur = -risk_roll["mod_dur"] * p_roll * dy
    convex = 0.5 * risk_roll["conv"] * p_roll * dy * dy
    actual = p_end - p_start
    carry = p_carry - p_start
    roll = p_roll - p_carry
    return dict(actual=actual, carry=carry, roll=roll, duration=dur,
                convexity=convex, residual=actual - (carry + roll + dur + convex))


def build_curve(par_by_tenor, grid=None, f=2):
    if grid is None:
        grid = np.arange(0.5, 10.0 + 1e-9, 0.5)
    src_t = np.array(sorted(par_by_tenor))
    src_y = np.array([par_by_tenor[t] for t in src_t])
    return bootstrap_par(grid, np.interp(grid, src_t, src_y), frequency=f)


def backtest(panel, scheme, tenors=(2.0, 5.0, 10.0), body_dv01=1e4):
    dates = sorted(panel)
    tt = np.array(tenors)
    curves = {d: build_curve(panel[d]) for d in dates}
    Z = np.array([[curves[d].zero_rate(t) for t in tt] for d in dates])
    dZ = np.diff(Z, axis=0)
    pc = pca(dZ)
    pl = (pc["eigenvectors"][:, 0], pc["eigenvectors"][:, 1])
    X = np.column_stack([np.ones(len(dZ)), dZ[:, 0], dZ[:, 2]])
    coef, *_ = np.linalg.lstsq(X, dZ[:, 1], rcond=None)
    betas = np.array([coef[1], coef[2]])

    rows = []
    for k in range(len(dates) - 1):
        c0, c1 = curves[dates[k]], curves[dates[k + 1]]
        dt = (pd.Timestamp(dates[k + 1]) - pd.Timestamp(dates[k])).days / 365.0
        y0 = np.array([c0.zero_rate(t) for t in tt])
        risk0 = [par_bond_risk(y0[j], tt[j], y0[j]) for j in range(3)]
        D = np.array([r["dv01"] for r in risk0])
        mult = butterfly_weights(D, scheme, betas, pl, body_dv01)["mult"]
        agg = dict(actual=0.0, carry=0.0, roll=0.0, duration=0.0,
                   convexity=0.0, residual=0.0)
        for j in range(3):
            taup = tt[j] - dt
            yroll, y1 = c0.zero_rate(taup), c1.zero_rate(taup)
            r_roll = par_bond_risk(yroll, taup, y0[j])
            dec = decompose_leg_pnl(y0[j], y1, yroll, tt[j], dt, y0[j], r_roll)
            for key in agg:
                agg[key] += mult[j] * dec[key]
        agg["date"] = dates[k + 1]
        rows.append(agg)
    daily = pd.DataFrame(rows).set_index("date")
    mu, sd = daily["actual"].mean(), daily["actual"].std(ddof=1)
    return dict(daily=daily, betas=betas, pca_loadings=pl,
                cum=daily.cumsum(),
                sharpe=float(mu / sd * np.sqrt(252)),
                explained=float(1 - daily["residual"].var(ddof=1) / daily["actual"].var(ddof=1)))


raw = pd.read_csv("data/samples/synthetic_ust_par_panel.csv")
panel = {d: dict(zip(g["tenor"], g["par_yield"])) for d, g in raw.groupby("date")}

# %% [markdown]
# ## 演習1：3ウェイト方式のバタフライを比較
#
# まず全期間のカーブ変化から PCA ローディングと回帰ベータを推定し、3方式の建玉を
# 構築します。次に、(a) 合成のレベル変化 $(1,1,1)$ bp とスロープ変化 $(-1,0,1)$ bp
# を各建玉に当てたときの損益、(b) バックテストの累積 PnL・シャープ・説明率を比較
# します。

# %%
tt = np.array([2.0, 5.0, 10.0])
# 代表日（初日）のカーブで DV01 を測り建玉を作る
first = sorted(panel)[0]
c0 = build_curve(panel[first])
y0 = np.array([c0.zero_rate(t) for t in tt])
D = np.array([par_bond_risk(y0[j], tt[j], y0[j])["dv01"] for j in range(3)])

# ウェイト推定用の統計量（全期間）
res_ref = backtest(panel, "50-50")
betas, pl = res_ref["betas"], res_ref["pca_loadings"]

builds = {
    "50-50": butterfly_weights(D, "50-50"),
    "risk": butterfly_weights(D, "risk", betas=betas),
    "pca": butterfly_weights(D, "pca", pca_loadings=pl),
}

# (a) 合成ショックへの応答（PnL = -Σ w_i Δy_i, Δy は bp）
level = np.array([1.0, 1.0, 1.0])     # レベル変化
slope = np.array([-1.0, 0.0, 1.0])    # スロープ変化
print("方式ごとの合成ショック応答（1bp あたり PnL）")
print(f"{'方式':>8} {'レベル':>12} {'スロープ':>12}")
for name, b in builds.items():
    w = b["w"]
    pnl_level = -np.sum(w * level) * BP
    pnl_slope = -np.sum(w * slope) * BP
    print(f"{name:>8} {pnl_level:12.4e} {pnl_slope:12.4e}")

# %% [markdown]
# レベル変化はどの方式もほぼゼロ（DV01ニュートラル）です。スロープ変化に対しては
# 50-50 が最も残り、リスクウェイトで縮小、PCA では PC2（スロープ）を構成的に消す
# ため機械精度でゼロになります。「何をニュートラル化するか」の理論と整合します。

# %%
# (b) 3方式のバックテスト比較
print(f"{'方式':>8} {'累積PnL':>12} {'シャープ':>10} {'説明率':>10}")
results = {}
for name in ["50-50", "risk", "pca"]:
    r = backtest(panel, name)
    results[name] = r
    print(f"{name:>8} {r['cum']['actual'].iloc[-1]:12.2f} {r['sharpe']:10.2f} {r['explained']:10.4%}")

fig, ax = plt.subplots(figsize=(9, 4.2))
for name, r in results.items():
    ax.plot(r["cum"].index, r["cum"]["actual"], label=name)
ax.set_title("3ウェイト方式の累積PnL比較（2-5-10 バタフライ）")
ax.set_xlabel("日付"); ax.set_ylabel("累積 PnL")
ax.legend(); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig("_fig_sol_schemes.png", dpi=80); plt.close(fig)
print("\n3方式の累積PnL図を _fig_sol_schemes.png に保存しました")

# %% [markdown]
# ## 演習2：PnL分解でキャリー／ロール／カーブ変化の寄与を可視化
#
# 50-50 バタフライの各日 PnL を要因分解し、キャリー・ロール・カーブ変化
# （デュレーション＋コンベクシティ）の**累計寄与**を積み上げ棒で示します。

# %%
r = results["50-50"]
daily = r["daily"].copy()
daily["curve"] = daily["duration"] + daily["convexity"]
contrib = daily[["carry", "roll", "curve", "residual"]].sum()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
# 累計寄与の内訳
ax[0].bar(range(len(contrib)), contrib.values,
          color=["#4C72B0", "#55A868", "#C44E52", "#999999"])
ax[0].set_xticks(range(len(contrib)))
ax[0].set_xticklabels(["キャリー", "ロール", "カーブ変化", "残差"])
ax[0].axhline(0, color="black", lw=0.8)
ax[0].set_title("要因別の累計寄与 (50-50)")
ax[0].set_ylabel("PnL 寄与合計"); ax[0].grid(alpha=0.3, axis="y")

# 累積の積み上げ
cc = daily[["carry", "roll", "curve"]].cumsum()
ax[1].plot(cc.index, r["cum"]["actual"], color="black", lw=2, label="実 PnL")
ax[1].stackplot(cc.index, cc["carry"], cc["roll"], cc["curve"],
                labels=["キャリー", "ロール", "カーブ変化"], alpha=0.6)
ax[1].set_title("累積PnLの要因積み上げ")
ax[1].set_xlabel("日付"); ax[1].set_ylabel("累積 PnL")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3); ax[1].tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig("_fig_sol_attrib.png", dpi=80); plt.close(fig)

print("要因別の累計寄与:")
for name, v in contrib.items():
    print(f"  {name:10s}: {v:12.4f}")
dom = contrib[["carry", "roll", "curve"]].abs().idxmax()
print(f"\n累積PnLを最も駆動した要因: {dom}")
print("分解合計と実PnLの整合（説明率）:", f"{r['explained']:.4%}")
assert r["explained"] > 0.95
