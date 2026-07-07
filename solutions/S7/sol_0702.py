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
# # S7-2 演習 解答例
#
# CDS のハザードブートストラップと CS01 の演習 2 問の解答例です。本文と同じ
# 連続複利 2% フラット割引カーブを使い、`bondlab.credit` と突合します。

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from bondlab.curve import DiscountCurve
from bondlab.credit import bootstrap_hazard, cds_par_spread, HazardCurve

np.random.seed(0)

# 連続複利 2% フラット割引カーブ
tg = np.linspace(0.25, 30.0, 120)
disc = DiscountCurve(tg, np.exp(-0.02 * tg))


def survival(times, hazards, t):
    """区分定数ハザードから生存確率 S(t)。最終ノードより先は最後の λ で外挿する。"""
    times = np.asarray(times, dtype=float)
    hazards = np.asarray(hazards, dtype=float)
    edges = np.concatenate([[0.0], times])
    cumh = np.concatenate([[0.0], np.cumsum(np.diff(edges) * hazards)])

    def cum_one(tt):
        if tt <= 0:
            return 0.0
        i = int(np.searchsorted(times, tt))
        if i >= len(times):
            return cumh[-1] + (tt - times[-1]) * hazards[-1]
        return cumh[i] + (tt - edges[i]) * hazards[i]

    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        return float(np.exp(-cum_one(float(t))))
    return np.exp(-np.array([cum_one(float(x)) for x in t]))


def cds_legs_scratch(times, hazards, disc, maturity, recovery=0.4, freq=4):
    """リスキーアニュイティとプロテクション PV（経過保険料込み）。"""
    tau = 1.0 / freq
    pay_times = np.arange(1, int(round(maturity * freq)) + 1) / freq
    surv = survival(times, hazards, pay_times)
    annuity = float(np.sum(tau * disc.discount(pay_times) * surv))
    prev = np.concatenate([[0.0], pay_times[:-1]])
    s_prev = survival(times, hazards, prev)
    mid = 0.5 * (prev + pay_times)
    annuity += float(np.sum(0.5 * tau * disc.discount(mid) * (s_prev - surv)))
    n_int = int(round(maturity * 200))
    grid = np.linspace(0.0, maturity, n_int + 1)
    S = survival(times, hazards, grid)
    DF = disc.discount(0.5 * (grid[1:] + grid[:-1]))
    protection = float((1 - recovery) * np.sum(DF * (-(S[1:] - S[:-1]))))
    return annuity, protection


def par_spread_scratch(times, hazards, disc, maturity, recovery=0.4, freq=4):
    ann, prot = cds_legs_scratch(times, hazards, disc, maturity, recovery, freq)
    return prot / ann


def bootstrap_hazard_scratch(disc, tenors, spreads, recovery=0.4, freq=4):
    tenors = np.asarray(tenors, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    hazards = []
    for i, (T, s) in enumerate(zip(tenors, spreads)):
        def obj(lam, i=i, T=T, s=s):
            h = np.array(hazards + [lam])
            return par_spread_scratch(tenors[: i + 1], h, disc, T, recovery, freq) - s
        hazards.append(brentq(obj, 1e-8, 5.0, xtol=1e-12, maxiter=200))
    return tenors, np.array(hazards)


# %% [markdown]
# ## 演習 1：ハザード剥ぎ取りと再構築
#
# 6 テナーのパースプレッドからハザードをブートストラップし、再計算で入力に戻ること、
# `bondlab.credit.bootstrap_hazard` とハザードが一致することを確認します。

# %%
tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
quotes = np.array([70, 90, 110, 140, 155, 170]) / 1e4

times_s, haz_s = bootstrap_hazard_scratch(disc, tenors, quotes, recovery=0.4, freq=4)
hc_bl = bootstrap_hazard(disc, tenors, quotes, recovery=0.4, freq=4)

print(f"{'テナー':>6}  {'入力(bp)':>8}  {'ハザード':>10}  {'再構築(bp)':>10}  {'bondlab一致':>10}")
for T, q, hs, hb in zip(tenors, quotes, haz_s, hc_bl.hazards):
    recon = par_spread_scratch(times_s, haz_s, disc, T, 0.4, 4) * 1e4
    print(f"{T:6.0f}  {q * 1e4:8.2f}  {hs:10.6f}  {recon:10.4f}  {abs(hs - hb) < 1e-10!s:>10}")

recon_all = np.array([par_spread_scratch(times_s, haz_s, disc, T, 0.4, 4) for T in tenors])
assert np.allclose(recon_all, quotes, atol=1e-10)
assert np.allclose(haz_s, hc_bl.hazards, atol=1e-10)
print("\n再構築が入力に一致し、bondlab ともハザードが一致しました")

fig, ax = plt.subplots(figsize=(7, 4))
grid = np.linspace(0.0, 10.0, 200)
ax.plot(grid, survival(times_s, haz_s, grid), color="#1f77b4")
ax.set_xlabel("年数")
ax.set_ylabel("生存確率 S(t)")
ax.set_title("演習1：ブートストラップ後の生存確率カーブ")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察
#
# ブートストラップはテナーの短い順に、その区間のハザードを1つずつ求根で決めます。既に
# 確定した短いテナーのハザードは固定するので、上三角の逐次求解になり、剥ぎ取ったハザードで
# 再計算すると入力スプレッドに厳密に戻ります。`bondlab` も同じ区分定数・同じ経過近似を
# 使うため、ハザードは機械精度で一致します。

# %% [markdown]
# ## 演習 2：スプレッド拡大の MTM と CS01
#
# 5年 CDS を par=120bp で買った投資家（固定クーポン100bp、想定元本1,000万円）について、
# 120bp → 250bp の拡大に伴う MTM 変化と、250bp 時点の CS01 を計算します。さらに区間を
# 細かく刻んで CS01 を足し上げ、MTM 変化とほぼ一致することを確認します。

# %%
coupon = 0.01
notional = 10_000_000.0
maturity = 5.0
recovery = 0.4


def mtm_buyer(par_spread):
    """5年 CDS のプロテクション買い手 MTM（固定クーポン100bp、金額）。"""
    times_h, haz_h = bootstrap_hazard_scratch(disc, [maturity], [par_spread], recovery, 4)
    ann, prot = cds_legs_scratch(times_h, haz_h, disc, maturity, recovery, 4)
    return notional * (prot - coupon * ann)


def cs01(par_spread, bump=1e-4):
    """CS01 = MTM(s + 1bp) - MTM(s)。"""
    return mtm_buyer(par_spread + bump) - mtm_buyer(par_spread)


mtm_120 = mtm_buyer(0.0120)
mtm_250 = mtm_buyer(0.0250)
d_mtm = mtm_250 - mtm_120
cs01_250 = cs01(0.0250)

print(f"MTM（par=120bp）  = {mtm_120:>14,.2f} 円")
print(f"MTM（par=250bp）  = {mtm_250:>14,.2f} 円")
print(f"MTM 変化          = {d_mtm:>14,.2f} 円")
print(f"CS01（250bp時点） = {cs01_250:>14,.2f} 円/bp")

# CS01 を 1bp 刻みで足し上げて MTM 変化を近似
sweep = np.arange(0.0120, 0.0250, 1e-4)
cs01_sum = np.sum([cs01(s) for s in sweep])
print(f"\nCS01 の足し上げ   = {cs01_sum:>14,.2f} 円")
print(f"MTM 変化との差    = {cs01_sum - d_mtm:>14,.2f} 円")

# 相対誤差 0.5% 以内で一致
assert abs(cs01_sum - d_mtm) < abs(d_mtm) * 5e-3
print("\nCS01 の積み上げが MTM 変化と一致しました")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
levels = np.arange(0.0080, 0.0400, 1e-4)
ax.plot(levels * 1e4, [mtm_buyer(s) / 1e4 for s in levels], color="#1f77b4")
ax.axhline(0.0, ls="--", color="gray")
ax.scatter([120, 250], [mtm_120 / 1e4, mtm_250 / 1e4], color="#d62728", zorder=5)
ax.set_xlabel("5年パースプレッド（bp）")
ax.set_ylabel("買い手 MTM（万円）")
ax.set_title("演習2：スプレッド拡大と MTM（par=120bp で買った契約）")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察
#
# プロテクションの買い手は、安く買った保険（クーポン100bp）が市場で値上がりするので、
# スプレッド拡大で利益になります。MTM はスプレッドに対してほぼ直線で、その局所的な傾きが
# CS01 です。区間を 1bp 刻みで CS01 を足し上げると、曲線をリーマン和で近似したことになり、
# MTM 変化とごく小さな誤差で一致します。厳密には MTM はわずかに凹（スプレッドが高いほど
# アニュイティが縮み傾きが鈍る）なので、離散和と真の変化の差は小さな二次の項として残ります。
