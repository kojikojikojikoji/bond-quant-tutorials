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
# # S7-5 演習 解答例
#
# ## 準備
#
# 本編の合成カーブ・Hull-White パス生成・スワップ時価・エクスポージャー計算を最小限だけ再掲します。

# %%
import numpy as np
import matplotlib.pyplot as plt

import bondlab
from bondlab.curve import bootstrap_par
from bondlab.models import HullWhite, Vasicek
from bondlab.pricing import par_swap_rate
from bondlab.credit import HazardCurve

print("bondlab version:", bondlab.__version__)

SEED = 20260707
plt.rcParams["axes.unicode_minus"] = False

A_HW, SIGMA_HW = 0.05, 0.01
RECOVERY = 0.40
MATURITY = 10
FREQ = 1
N_STEPS = 120

par_tenors = np.arange(1, 31, dtype=float)
par_rates = 0.03 + 0.01 * (1.0 - np.exp(-par_tenors / 8.0))
curve = bootstrap_par(par_tenors, par_rates, frequency=1)
coupon_times = np.arange(1, MATURITY + 1, dtype=float)
K_par = par_swap_rate(curve, curve, coupon_times)
exposure_years = np.arange(0, MATURITY + 1)


def simulate_hw_short_rate(hw, T, n_steps, n_paths, seed):
    """本編と同一：Vasicek の OU 因子 + 確定ドリフトで HW 短期金利を再構成。"""
    a, sigma = hw.a, hw.sigma
    t_grid, x = Vasicek(a, 0.0, sigma, 0.0).simulate(T, n_steps, n_paths, seed=seed)
    fwd = np.array([hw._fm(max(t, 1e-4)) for t in t_grid])
    conv = sigma ** 2 / (2 * a ** 2) * (1.0 - np.exp(-a * t_grid)) ** 2
    return t_grid, fwd[None, :] + conv[None, :] + x


def swap_mtm(hw, t, r_t, K, coupon_times, side="payer"):
    """時点 t のスワップ時価（パス配列）。side=payer は固定を払う（本編と同一の符号）。"""
    rem = coupon_times[coupon_times > t + 1e-9]
    if rem.size == 0:
        return np.zeros_like(np.asarray(r_t, dtype=float))
    tau = 1.0 / FREQ
    p_last = hw.zcb(t, float(rem[-1]), r_t)
    fixed = np.zeros_like(np.asarray(r_t, dtype=float))
    for Ti in rem:
        fixed = fixed + tau * hw.zcb(t, float(Ti), r_t)
    payer = (1.0 - p_last) - K * fixed
    return payer if side == "payer" else -payer


def path_discount(r_full, t_grid):
    """本編と同一：D(t)=exp(-∫_0^t r)。"""
    dt = np.diff(t_grid)
    incr = 0.5 * (r_full[:, 1:] + r_full[:, :-1]) * dt[None, :]
    cum = np.concatenate([np.zeros((r_full.shape[0], 1)), np.cumsum(incr, axis=1)], axis=1)
    return np.exp(-cum)


def cva_discrete(exposure, pd_marginal, recovery=RECOVERY):
    return float((1.0 - recovery) * np.sum(exposure * pd_marginal))


def flat_hazard(spread, recovery=RECOVERY):
    return HazardCurve(np.array([float(MATURITY)]), np.array([spread / (1.0 - recovery)]))


# %% [markdown]
# ## 演習1：金利ボラ・信用スプレッドを振った CVA 感応度
#
# 金利ボラ $\sigma\in\{0.005,0.010,0.015\}$ と信用スプレッド $s\in\{50,100,200\}$ bp の 2 次元格子で CVA を計算します。
# $\sigma$ はエクスポージャー（割引 EE プロファイル）を、$s$ は限界 PD を動かします。$\sigma$ ごとに一度だけ
# モンテカルロを回して割引 EE を作り、$s$ の各値では PD を差し替えて足すだけにします（信用は再シミュレーション不要）。

# %%
sigmas = [0.005, 0.010, 0.015]
spreads_bp = [50, 100, 200]

# σ ごとに割引エクスポージャー dEE(t) を作る。
dEE_by_sigma = {}
for sig in sigmas:
    hw_s = HullWhite(A_HW, sig, curve)
    tg, rf = simulate_hw_short_rate(hw_s, float(MATURITY), N_STEPS, 8000, SEED + int(sig * 1e4))
    dsc = path_discount(rf, tg)
    step_per_year = int(round(N_STEPS / MATURITY))
    dEE = np.zeros(len(exposure_years))
    for j, yr in enumerate(exposure_years):
        if yr <= 0 or yr >= MATURITY:
            continue
        col = int(round(int(yr) * step_per_year))
        v = swap_mtm(hw_s, float(yr), rf[:, col], K_par, coupon_times)
        dEE[j] = float((dsc[:, col] * np.maximum(v, 0.0)).mean())
    dEE_by_sigma[sig] = dEE

# 2 次元 CVA 表。
table = np.zeros((len(sigmas), len(spreads_bp)))
for i, sig in enumerate(sigmas):
    for j, s_bp in enumerate(spreads_bp):
        hz = flat_hazard(s_bp * 1e-4)
        Sf = hz.survival(exposure_years.astype(float))
        pdm = Sf[:-1] - Sf[1:]
        table[i, j] = cva_discrete(dEE_by_sigma[sig][1:], pdm)

print("CVA 感応度表 [bp of notional]（行=σ、列=spread）")
header = "  σ \\ s |" + "".join(f"{s:>10d}bp" for s in spreads_bp)
print(header)
print("-" * len(header))
for i, sig in enumerate(sigmas):
    print(f"{sig:>7.3f} |" + "".join(f"{table[i, j]*1e4:>12.4f}" for j in range(len(spreads_bp))))

# %% [markdown]
# ### 二効果はおおむね「積」で効く
#
# CVA $=(1-R)\sum \mathrm{dEE}(t_k)\,\Delta\mathrm{PD}_k$ で、$\sigma$ は $\mathrm{dEE}$ を、$s$ は $\Delta\mathrm{PD}$ を線形近似で
# スケールさせます。スプレッドが小さければ $\Delta\mathrm{PD}\approx (s/(1-R))\,\Delta t$ なので、CVA はおよそ
# $(\text{エクスポージャーの大きさ}) \times (\text{スプレッド})$ の積になります。表を「$\sigma=0.010,\,s=100$bp」の基準セルで
# 割ると、行・列それぞれの比の積で他セルが近似できることが見えます。

# %%
base_i, base_j = 1, 1  # σ=0.010, s=100bp
base = table[base_i, base_j]
print("基準セル（σ=0.010, s=100bp）に対する比：実測 vs 行比×列比（積近似）")
print(f"{'σ':>7s}{'s[bp]':>8s}{'実測比':>10s}{'積近似':>10s}")
for i, sig in enumerate(sigmas):
    for j, s_bp in enumerate(spreads_bp):
        actual = table[i, j] / base
        approx = (table[i, base_j] / base) * (table[base_i, j] / base)
        print(f"{sig:>7.3f}{s_bp:>8d}{actual:>10.3f}{approx:>10.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.imshow(table * 1e4, cmap="Reds", aspect="auto", origin="lower")
ax.set_xticks(range(len(spreads_bp))); ax.set_xticklabels([f"{s}bp" for s in spreads_bp])
ax.set_yticks(range(len(sigmas))); ax.set_yticklabels([f"{s:.3f}" for s in sigmas])
ax.set_xlabel("credit spread s"); ax.set_ylabel("rate vol sigma")
ax.set_title("CVA sensitivity (bp of notional)")
for i in range(len(sigmas)):
    for j in range(len(spreads_bp)):
        ax.text(j, i, f"{table[i, j]*1e4:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, label="CVA [bp]")
plt.tight_layout()
plt.show()

print("解釈: σ を上げるとエクスポージャーが持ち上がって CVA 増（金利ベガ）、")
print("      s を上げると PD が増えて CVA 増（信用デルタ）。小スプレッド域では両者は積で効く。")

# %% [markdown]
# ## 演習2：ネッティングで CVA が減ることの確認
#
# 同じカウンターパーティと、**10y ペイヤースワップ**（固定 $K_1$ を払う）と**5y レシーバースワップ**（固定 $K_2$ を
# 受ける）の 2 本を持ちます。満期をずらしているので変動レッグは完全には打ち消し合わず、部分的なネッティングに
# なります。ネッティングあり・なしのエクスポージャーからそれぞれ CVA を出して比べます。

# %%
coupon_times_5 = np.arange(1, 6, dtype=float)     # 5y スワップの利払日
K1 = K_par                                        # 10y ペイヤーの固定金利（当初パー）
K2 = par_swap_rate(curve, curve, coupon_times_5)  # 5y レシーバーの固定金利（当初パー）
hw = HullWhite(A_HW, SIGMA_HW, curve)
tg, rf = simulate_hw_short_rate(hw, float(MATURITY), N_STEPS, 12000, SEED + 7)
dsc = path_discount(rf, tg)
step_per_year = int(round(N_STEPS / MATURITY))

hazard = flat_hazard(0.012)  # 120bp フラット
S = hazard.survival(exposure_years.astype(float))
pd_marginal = S[:-1] - S[1:]

dEE_net = np.zeros(len(exposure_years))    # ( V1 + V2 )^+
dEE_nonet = np.zeros(len(exposure_years))  # V1^+ + V2^+
for j, yr in enumerate(exposure_years):
    if yr <= 0 or yr >= MATURITY:
        continue
    col = int(round(int(yr) * step_per_year))
    v1 = swap_mtm(hw, float(yr), rf[:, col], K1, coupon_times, side="payer")
    v2 = swap_mtm(hw, float(yr), rf[:, col], K2, coupon_times_5, side="receiver")
    d = dsc[:, col]
    dEE_net[j] = float((d * np.maximum(v1 + v2, 0.0)).mean())
    dEE_nonet[j] = float((d * (np.maximum(v1, 0.0) + np.maximum(v2, 0.0))).mean())

cva_net = cva_discrete(dEE_net[1:], pd_marginal)
cva_nonet = cva_discrete(dEE_nonet[1:], pd_marginal)
reduction = (1.0 - cva_net / cva_nonet) * 100

print(f"CVA（ネッティングなし V1^+ + V2^+）   = {cva_nonet*1e4:.4f} bp")
print(f"CVA（ネッティングあり (V1+V2)^+）      = {cva_net*1e4:.4f} bp")
print(f"削減率                                 = {reduction:.1f} %")
assert cva_net <= cva_nonet + 1e-12
print("→ CVA_netted ≤ CVA_no-net（三角不等式 (V1+V2)^+ ≤ V1^+ + V2^+ の帰結）")

# %% [markdown]
# ### プロファイルの比較と、符号が逆なほど効く理由
#
# エクスポージャープロファイルを重ねると、ネッティングありのほうが常に下（または同じ）にあります。ペイヤーと
# レシーバーは金利に対して**逆向き**に時価が動くので、片方が正のとき他方は負になりやすく、合算すると打ち消し合います。
# 相殺の余地が大きいほど $(V_1+V_2)^+$ が $V_1^+ + V_2^+$ を大きく下回り、CVA 削減が効きます。

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(exposure_years, dEE_nonet * 1e4, "o-", color="crimson", lw=2,
        label="no netting: E[D(V1+ + V2+)]")
ax.plot(exposure_years, dEE_net * 1e4, "o-", color="steelblue", lw=2,
        label="netted: E[D(V1+V2)+]")
ax.fill_between(exposure_years, dEE_net * 1e4, dEE_nonet * 1e4,
                color="gray", alpha=0.2, label="netting benefit")
ax.set_xlabel("time (years)")
ax.set_ylabel("discounted exposure (bp of notional)")
ax.set_title("Netting lowers the exposure profile (payer + receiver offset)")
ax.legend()
plt.tight_layout()
plt.show()

# 完全相殺（K2 = K1）の極端ケースも確認：CVA はほぼ 0 になる。
dEE_net_full = np.zeros(len(exposure_years))
for j, yr in enumerate(exposure_years):
    if yr <= 0 or yr >= MATURITY:
        continue
    col = int(round(int(yr) * step_per_year))
    v1 = swap_mtm(hw, float(yr), rf[:, col], K1, coupon_times, side="payer")
    v2 = swap_mtm(hw, float(yr), rf[:, col], K1, coupon_times, side="receiver")
    dEE_net_full[j] = float((dsc[:, col] * np.maximum(v1 + v2, 0.0)).mean())
cva_full = cva_discrete(dEE_net_full[1:], pd_marginal)
print(f"完全相殺（K2=K1）のネッティング CVA = {cva_full*1e4:.4f} bp（同一条件の逆取引は互いに打ち消しほぼ 0）")
print("解釈: 逆向きの取引ほど (V1+V2) が 0 近傍に集まり、正のエクスポージャーが小さくなる。")
print("      これがネッティングセット単位で CVA を管理する実務上の動機。")
