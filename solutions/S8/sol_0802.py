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
# # S8-2 演習 解答例
#
# モンテカルロ OAS の演習2問の解答例です。本体 notebook
# `nb_0802_mc_oas.py` と同じ MC-OAS エンジン（Hull-White ×
# ロジスティック期限前償還）を再掲し、
#
# 1. 金利ボラ $\sigma$ を振ってオプションコストの変化を描く
# 2. パス数 $N$ を振って OAS 推定の標準誤差が $1/\sqrt{N}$ で縮むことを確認する
#
# を扱います。

# %% [markdown]
# ## エンジンの再掲
#
# 本体 notebook の自作関数を、この解答例内でも使えるよう最小構成で再定義します。
# 役割は本体の説明表と同じです。

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from bondlab.curve import bootstrap_par
from bondlab.models import HullWhite
from bondlab.mbs import cpr_to_smm

SEED = 20260802
BALANCE = 100.0
WAC = 0.05
WAM = 360
DT = 1.0 / 12.0
REFI_TENOR = 10.0


def market_forward(curve, t, eps=1e-4):
    """瞬間フォワード f^M(0,t)=-d/dt ln DF(t)（中心差分）。"""
    t = max(float(t), eps)
    return -(np.log(curve.discount(t + eps)) - np.log(curve.discount(t - eps))) / (2 * eps)


def simulate_hull_white(a, sigma, curve, wam, n_paths, seed):
    """HW 短期金利パス r_t=x_t+α(t)（対照変量つき厳密サンプリング）。"""
    rng = np.random.default_rng(seed)
    T = wam * DT
    times = np.linspace(0.0, T, wam + 1)
    conv = sigma ** 2 / (2 * a ** 2)
    alpha = np.array([market_forward(curve, t) + conv * (1 - np.exp(-a * t)) ** 2 for t in times])
    half = (n_paths + 1) // 2
    z = rng.standard_normal((half, wam))
    z = np.vstack([z, -z])[:n_paths]
    mrev = np.exp(-a * DT)
    sd = np.sqrt(sigma ** 2 / (2 * a) * (1 - np.exp(-2 * a * DT)))
    x = np.zeros(n_paths)
    out = np.empty((n_paths, wam + 1))
    out[:, 0] = x + alpha[0]
    for i in range(wam):
        x = x * mrev + sd * z[:, i]
        out[:, i + 1] = x + alpha[i + 1]
    return times, out


def path_discount_factors(rates, curve, wam):
    """月末への割引係数（台形則の積分＋市場カーブへの再基準化）。"""
    trap = (rates[:, :-1] + rates[:, 1:]) * 0.5 * DT
    integ = np.cumsum(trap, axis=1)
    disc = np.exp(-integ)
    tm = np.arange(1, wam + 1) * DT
    pm = np.asarray(curve.discount(tm))
    disc = disc * (pm / disc.mean(axis=0))
    return disc, tm, pm


def hw_refi_rate(hw, times, rates, tenor=REFI_TENOR):
    """各時点のモデル tenor 年ゼロレート（r_t にアフィン）。"""
    n = rates.shape[1]
    intercept = np.empty(n)
    slope = np.empty(n)
    for j, t in enumerate(times):
        z0 = -np.log(hw.zcb(t, t + tenor, 0.0)) / tenor
        z1 = -np.log(hw.zcb(t, t + tenor, 0.02)) / tenor
        slope[j] = (z1 - z0) / 0.02
        intercept[j] = z0
    return intercept[None, :] + slope[None, :] * rates


def prepay_cpr(age_months, incentive, ramp_months=30.0,
               base=0.05, ceiling=0.40, steepness=25.0, threshold=0.0):
    """年率 CPR = 経過ランプ × ロジスティック借換S字。"""
    age_months = np.asarray(age_months, dtype=float)
    incentive = np.asarray(incentive, dtype=float)
    ramp = np.minimum(1.0, age_months / ramp_months)
    refi = base + (ceiling - base) / (1.0 + np.exp(-steepness * (incentive - threshold)))
    return ramp * refi


def pool_cashflows(balance, wac, wam, smm_paths):
    """パス別の月次パススルー CF 行列（本体と同一ロジック）。"""
    r = wac / 12.0
    n = smm_paths.shape[0]
    bal = np.full(n, float(balance))
    cf = np.empty((n, wam))
    for m in range(wam):
        n_rem = wam - m
        interest = bal * r
        pmt = bal * r / (1.0 - (1.0 + r) ** (-n_rem))
        sched = np.minimum(pmt - interest, bal)
        prepay = (bal - sched) * smm_paths[:, m]
        total_prin = sched + prepay
        cf[:, m] = interest + total_prin
        bal = bal - total_prin
    return cf


def price_given_oas(cf, disc, tm, oas):
    return float(np.mean(np.sum(cf * disc * np.exp(-oas * tm), axis=1)))


def solve_oas(cf, disc, tm, price, lo=-0.05, hi=0.30):
    return brentq(lambda s: price_given_oas(cf, disc, tm, s) - price, lo, hi, xtol=1e-10)


def solve_zspread(cf_base, pm, tm, price, lo=-0.05, hi=0.30):
    return brentq(lambda z: float(np.sum(cf_base * pm * np.exp(-z * tm))) - price,
                  lo, hi, xtol=1e-10)


def build_prepay_smm(hw, times, rates, wac):
    refi = hw_refi_rate(hw, times, rates)
    age = np.arange(1, WAM + 1)
    incentive = wac - refi[:, 1:]
    return cpr_to_smm(prepay_cpr(age[None, :], incentive))


def base_case_smm(curve, wac):
    tm_local = np.arange(1, WAM + 1) * DT
    refi_det = np.array([-np.log(curve.discount(t + REFI_TENOR) / curve.discount(t)) / REFI_TENOR
                         for t in tm_local])
    age = np.arange(1, WAM + 1)
    return cpr_to_smm(prepay_cpr(age, wac - refi_det))[None, :]


# 共通の合成カーブと市場価格（本体と同じ設定）。
TENORS = [1, 2, 3, 5, 7, 10, 20, 30]
PAR_RATES = [0.030, 0.032, 0.034, 0.037, 0.039, 0.041, 0.044, 0.045]
curve = bootstrap_par(TENORS, PAR_RATES, frequency=1)
HW_A = 0.05
HW_SIGMA = 0.010

# 市場価格を「基準ボラで真の OAS=50bp」から逆算して固定する。
_hw = HullWhite(HW_A, HW_SIGMA, curve)
_t, _r = simulate_hull_white(HW_A, HW_SIGMA, curve, WAM, 3000, SEED)
_disc, _tm, _pm = path_discount_factors(_r, curve, WAM)
_smm = build_prepay_smm(_hw, _t, _r, WAC)
_cf = pool_cashflows(BALANCE, WAC, WAM, _smm)
PRICE_MARKET = price_given_oas(_cf, _disc, _tm, 0.0050)
print(f"固定した市場価格（額面100あたり）: {PRICE_MARKET:.4f}")


# %% [markdown]
# ## 演習1：金利ボラとオプションコスト
#
# $\sigma$ を 0.4%〜2.4% で振り、各 $\sigma$ で OAS・Zスプレッド・オプションコストを
# 求めます。市場価格は固定です。

# %%
def decompose(a, sigma, curve, wac, n_paths, seed, price):
    hw = HullWhite(a, sigma, curve)
    t, r = simulate_hull_white(a, sigma, curve, WAM, n_paths, seed)
    disc, tm, pm = path_discount_factors(r, curve, WAM)
    smm = build_prepay_smm(hw, t, r, wac)
    cf = pool_cashflows(BALANCE, wac, WAM, smm)
    oas = solve_oas(cf, disc, tm, price)
    cf_base = pool_cashflows(BALANCE, wac, WAM, base_case_smm(curve, wac))[0]
    z = solve_zspread(cf_base, pm, tm, price)
    return oas, z, z - oas


sigmas = np.array([0.004, 0.008, 0.012, 0.016, 0.020, 0.024])
rows = []
for sg in sigmas:
    oas, z, oc = decompose(HW_A, sg, curve, WAC, 3000, SEED, PRICE_MARKET)
    rows.append((sg, oas, z, oc))
    print(f"σ={sg*100:4.1f}%  OAS={oas*1e4:6.1f}bp  Z={z*1e4:6.1f}bp  optcost={oc*1e4:6.1f}bp")

rows = np.array(rows)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(rows[:, 0] * 100, rows[:, 3] * 1e4, "o-", label="オプションコスト")
ax.plot(rows[:, 0] * 100, rows[:, 2] * 1e4, "s--", color="gray", label="Zスプレッド")
ax.plot(rows[:, 0] * 100, rows[:, 1] * 1e4, "^--", color="crimson", label="OAS")
ax.set_xlabel("短期金利ボラ σ (%)"); ax.set_ylabel("スプレッド (bp)")
ax.set_title("金利ボラとオプションコスト"); ax.legend()
fig.tight_layout()

# %% [markdown]
# **考察。** オプションコストは $\sigma$ に対して単調に増加します。金利ボラが高いほど
# 「金利が下がって借り換えが起きる」確率質量が増え、借り手の繰上返済オプションの価値が
# 上がるためです。一方 Zスプレッドはほぼ一定です。これはフォワード金利に沿った確定
# シナリオの CF を市場カーブで割引くだけで、金利ボラの情報が入らないためです。ボラ上昇の
# インパクトはすべて OAS の低下として現れます。

# 単調性を軽く確認（多少の MC 揺らぎを許容）。
assert rows[-1, 3] > rows[0, 3]
assert np.all(np.diff(rows[:, 3]) > -2e-4)

# %% [markdown]
# ## 演習2：パス数と OAS の標準誤差
#
# パス数 $N$ を変え、各 $N$ でシード違いの OAS を複数回推定して、その標準偏差
# （OAS 推定の標準誤差）を測ります。共通乱数の効果を消すため、推定ごとに独立なシードを
# 使います。

# %%
def estimate_oas(n_paths, seed):
    hw = HullWhite(HW_A, HW_SIGMA, curve)
    t, r = simulate_hull_white(HW_A, HW_SIGMA, curve, WAM, n_paths, seed)
    disc, tm, _ = path_discount_factors(r, curve, WAM)
    smm = build_prepay_smm(hw, t, r, WAC)
    cf = pool_cashflows(BALANCE, WAC, WAM, smm)
    return solve_oas(cf, disc, tm, PRICE_MARKET)


n_reps = 12
Ns = [500, 1000, 2000, 4000]
se = []
for N in Ns:
    ests = np.array([estimate_oas(N, 1000 + k) for k in range(n_reps)])
    s = ests.std(ddof=1)
    se.append(s)
    print(f"N={N:5d}  meanOAS={ests.mean()*1e4:6.2f}bp  SE={s*1e4:5.2f}bp")

se = np.array(se)
Ns = np.array(Ns, dtype=float)

# 1/sqrt(N) の傾き（両対数で −0.5）を最小二乗で確認。
slope = np.polyfit(np.log(Ns), np.log(se), 1)[0]
print(f"log-log 傾き（理論値 -0.5 に近いはず）: {slope:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(Ns, se * 1e4, "o-", label="実測 SE")
ref = se[0] * 1e4 * np.sqrt(Ns[0] / Ns)     # 1/sqrt(N) 参照線（先頭に合わせる）
ax.loglog(Ns, ref, "--", color="gray", label=r"$\propto 1/\sqrt{N}$")
ax.set_xlabel("パス数 N"); ax.set_ylabel("OAS の標準誤差 (bp)")
ax.set_title("パス数と OAS 推定精度"); ax.legend()
fig.tight_layout()

# %% [markdown]
# **考察。** 標準誤差はパス数 $N$ の増加に対して概ね $1/\sqrt{N}$ で縮み、両対数の傾きは
# $-0.5$ 前後になります（対照変量を併用しているため厳密な $-0.5$ からは多少ぶれます）。
# したがって **精度を2bp→1bp と半減させるにはパス数を約4倍** にする必要があります。
# これがモンテカルロ評価の「精度とコストのトレードオフ」で、実務では対照変量・制御変量
# などの分散削減や、共通乱数によるソルバー安定化で計算量を抑えます。

# 傾きが -0.5 の周辺にあることを緩く確認。
assert -0.85 < slope < -0.25
