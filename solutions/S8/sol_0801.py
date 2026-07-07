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
# # S8-1 演習 解答例
#
# MBSの仕組みとプリペイメントモデルの演習2問の解答例です。本文と同じ
# `bondlab.mbs` / `bondlab.curve` を使います。

# %%
import numpy as np
import matplotlib.pyplot as plt

from bondlab.mbs import psa_cpr, cpr_to_smm, mbs_cashflows, weighted_average_life
from bondlab.curve import bootstrap_par

np.random.seed(0)

# %% [markdown]
# ## 演習 1：PSA 速度と WAL
#
# 残高100・WAC 5.5%・WAM 360 のプールで、PSA を変えて WAL を求めます。

# %%
BAL, WAC, WAM = 100.0, 0.055, 360
psa_list = [50, 100, 165, 250, 400, 600]
ages = np.arange(1, WAM + 1)

wals = []
for psa in psa_list:
    smm = cpr_to_smm(psa_cpr(ages, psa))
    cf = mbs_cashflows(BAL, WAC, WAM, smm)
    wals.append(weighted_average_life(cf))
wals = np.array(wals)

print(f"{'PSA':>6} {'WAL(年)':>9} {'対PSA100倍率':>12}")
wal_100 = wals[psa_list.index(100)]
for psa, wal in zip(psa_list, wals):
    print(f"{psa:>6} {wal:>9.3f} {wal / wal_100:>11.3f}x")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(psa_list, wals, "o-", color="#1f77b4")
ax.set_xlabel("PSA 速度")
ax.set_ylabel("WAL（年）")
ax.set_title("PSA 速度と加重平均年限")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察
#
# PSA を上げると WAL は単調に短くなります。ただし速度と WAL は反比例せず、短縮は逓減
# します。PSA を 100→200 と2倍にしても WAL は半分にならず、PSA を上げるほど1単位あたりの
# 短縮幅が小さくなります。これは、予定元本という金利非依存の底流が常にあること、そして
# 期限前償還が「残っている残高」に対して掛かるため残高が小さくなるほど実額が減ること
# によります。速度を上げても返せる元本には限りがあるので、WAL はゼロには漸近しません。

# %%
# WAL は PSA に対して単調減少、かつ短縮は逓減（隣接差の絶対値が縮む）。
assert np.all(np.diff(wals) < 0)                          # 単調減少
d1 = -np.diff(wals)                                       # 各ステップの短縮幅（正）
# 速度を上げるほど、同じ刻みでも短縮幅は小さくなる傾向（逓減）。
assert d1[-1] < d1[0]
print("演習1 チェックを通過しました")

# %% [markdown]
# ## 演習 2：金利±100bp とネガティブコンベクシティ
#
# 本文と同じ金利感応プリペイモデルで、金利シフト別の価格から実効デュレーション・
# 実効コンベクシティを求め、固定 PSA と比較します。

# %%
POOL_BAL, POOL_WAC, POOL_WAM = 100.0, 0.055, 358
BASE_MKT_RATE = 0.055                              # 現行クーポン（アット・ザ・マネー）
PAR_TENORS = np.arange(1.0, 31.0)                  # 等間隔テナー（bootstrap_par の前提）
BASE_PAR = 0.052 + 0.0005 * np.log(PAR_TENORS)     # 割引利回り（国債＋OAS）


def logistic_refi_cpr(incentive, refi_max=0.45, beta=300.0, mu=0.005):
    """金利インセンティブ I=WAC-c に対する借換 CPR（S字カーブ）。"""
    return refi_max / (1.0 + np.exp(-beta * (incentive - mu)))


def statistical_mbs_cashflows(balance, wac, wam, market_rate,
                              turnover=0.06, refi_max=0.45, beta=300.0,
                              mu=0.005, gamma=2.0):
    """金利感応・バーンアウト込みの計量プリペイメントで月次CFを生成する。"""
    r = wac / 12.0
    incentive = wac - market_rate
    bal, orig, cum_prepay = float(balance), float(balance), 0.0
    months, cashflows, prins = [], [], []
    for m in range(1, wam + 1):
        n_rem = wam - m + 1
        seasoning = min(1.0, m / 30.0)
        burnout = np.exp(-gamma * (cum_prepay / orig))
        refi = logistic_refi_cpr(incentive, refi_max, beta, mu) * burnout
        cpr = min(seasoning * (turnover + refi), 0.60)
        smm = 1.0 - (1.0 - cpr) ** (1.0 / 12.0)
        interest = bal * r
        pmt = bal * r / (1.0 - (1.0 + r) ** (-n_rem)) if bal > 0 else 0.0
        sched = min(pmt - interest, bal)
        prepay = (bal - sched) * smm
        prin = sched + prepay
        bal -= prin
        cum_prepay += prepay
        months.append(m)
        cashflows.append(interest + prin)
        prins.append(prin)
    return dict(month=np.array(months), cashflow=np.array(cashflows),
                total_principal=np.array(prins))


def price_responsive(rate_shift):
    disc = bootstrap_par(PAR_TENORS, BASE_PAR + rate_shift, frequency=1)
    cf = statistical_mbs_cashflows(POOL_BAL, POOL_WAC, POOL_WAM,
                                   market_rate=BASE_MKT_RATE + rate_shift)
    return float(np.sum(cf["cashflow"] * disc.discount(cf["month"] / 12.0)))


def price_fixed_psa(rate_shift, psa=160):
    disc = bootstrap_par(PAR_TENORS, BASE_PAR + rate_shift, frequency=1)
    smm = cpr_to_smm(psa_cpr(np.arange(1, POOL_WAM + 1), psa))
    cf = mbs_cashflows(POOL_BAL, POOL_WAC, POOL_WAM, smm)
    return float(np.sum(cf["cashflow"] * disc.discount(cf["month"] / 12.0)))


# %%
shifts_bp = np.array([-100, -50, 0, 50, 100])
shifts = shifts_bp / 1e4
p_resp = np.array([price_responsive(s) for s in shifts])
p_fix = np.array([price_fixed_psa(s) for s in shifts])

print(f"{'シフト':>7} {'金利感応価格':>12} {'固定PSA価格':>12}")
for bp, pr, pf in zip(shifts_bp, p_resp, p_fix):
    print(f"{bp:>5}bp {pr:>12.4f} {pf:>12.4f}")


# ±100bp（配列の両端）で実効デュレーション・実効コンベクシティを測る。
delta = 0.01
d_resp = (p_resp[0] - p_resp[-1]) / (2 * p_resp[2] * delta)
c_resp = (p_resp[0] + p_resp[-1] - 2 * p_resp[2]) / (p_resp[2] * delta ** 2)
d_fix = (p_fix[0] - p_fix[-1]) / (2 * p_fix[2] * delta)
c_fix = (p_fix[0] + p_fix[-1] - 2 * p_fix[2]) / (p_fix[2] * delta ** 2)

print(f"\n金利感応MBS : 実効デュレーション={d_resp:6.3f}  実効コンベクシティ={c_resp:+8.2f}")
print(f"固定PSA MBS : 実効デュレーション={d_fix:6.3f}  実効コンベクシティ={c_fix:+8.2f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(shifts_bp, p_resp, "o-", color="#d62728", label="金利感応MBS")
ax.plot(shifts_bp, p_fix, "s-", color="#1f77b4", alpha=0.7, label="固定PSA MBS")
ax.set_xlabel("金利シフト（bp）")
ax.set_ylabel("MBS 価格")
ax.set_title("金利シフト別の MBS 価格")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察
#
# 金利感応 MBS の実効コンベクシティは負（ネガティブコンベクシティ）です。金利が下がると
# 借換で期限前償還が加速し、元本が早く返って価格上昇が頭打ちになる（コンプレッション）
# 一方、金利が上がると期限前償還が減ってデュレーションが伸び、価格が余計に下がる
# （エクステンション）ためです。同じプールを金利に反応しない固定 PSA で評価すると、
# キャッシュフローが金利で動かないので普通の債券と同じく実効コンベクシティは正になります。
# 両者のコンベクシティの差が、借り手に無償で渡している期限前償還オプションの対価であり、
# MBS 投資家がスプレッド（OAS）で報われるべきリスクです。

# %%
assert c_resp < 0          # 金利感応MBSはネガティブコンベクシティ
assert c_fix > 0           # 固定PSAは正のコンベクシティ
assert d_resp > 0          # デュレーションは正
print("演習2 チェックを通過しました")
