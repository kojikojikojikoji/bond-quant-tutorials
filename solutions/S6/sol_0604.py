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
# # S6-4 演習 解答例

# %% [markdown]
# ## 準備
#
# 本編の `HullWhiteTree`・`CallableBond`・後退帰納評価器・OAS/Zスプレッドソルバーを
# 最小限だけ再掲する。カーブは本編と同じ合成パー利回りの年次補間・ブートストラップ。

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import bondlab
from bondlab.curve import bootstrap_par

print("bondlab version:", bondlab.__version__)


class HullWhiteTree:
    """Hull-White 三項ツリー（本編・S5-3 と同一）。"""

    def __init__(self, a, sigma, curve, T, n_steps):
        self.a, self.sigma, self.curve = a, sigma, curve
        self.T, self.N = T, n_steps
        self.dt = T / n_steps
        self.dx = sigma * np.sqrt(3.0 * self.dt)
        self.jmax = int(np.ceil(0.184 / (a * self.dt)))
        self.build()

    def _branch(self, j):
        m = j * (1.0 - self.a * self.dt)
        if j >= self.jmax:
            k = j - 1
        elif j <= -self.jmax:
            k = j + 1
        else:
            k = int(np.round(m))
        eta = m - k
        pu = 1.0 / 6.0 + (eta ** 2 + eta) / 2.0
        pm = 2.0 / 3.0 - eta ** 2
        pd = 1.0 / 6.0 + (eta ** 2 - eta) / 2.0
        return k, pu, pm, pd

    def build(self):
        Q = {0: 1.0}
        self.alpha = np.empty(self.N)
        self.Q_levels = [dict(Q)]
        for i in range(self.N):
            s = sum(q * np.exp(-j * self.dx * self.dt) for j, q in Q.items())
            pm_disc = self.curve.discount((i + 1) * self.dt)
            self.alpha[i] = (np.log(s) - np.log(pm_disc)) / self.dt
            Q_next = {}
            for j, q in Q.items():
                k, pu, pm, pd = self._branch(j)
                disc = np.exp(-(self.alpha[i] + j * self.dx) * self.dt)
                for jj, p in ((k + 1, pu), (k, pm), (k - 1, pd)):
                    Q_next[jj] = Q_next.get(jj, 0.0) + q * p * disc
            Q = Q_next
            self.Q_levels.append(dict(Q))

    def short_rate(self, i, j):
        a_i = self.alpha[i] if i < self.N else 2 * self.alpha[-1] - self.alpha[-2]
        return a_i + j * self.dx


class CallableBond:
    """固定利付コーラブル債の仕様（本編と同一）。"""

    def __init__(self, coupon, face, freq, maturity, calls, call_price):
        self.coupon, self.face, self.freq = coupon, face, freq
        self.maturity, self.calls, self.call_price = maturity, list(calls), call_price

    def coupon_amount(self):
        return self.coupon * self.face / self.freq

    def coupon_steps(self, tree):
        n = int(round(self.maturity * self.freq))
        return {int(round((k / self.freq) / tree.dt)): self.coupon_amount()
                for k in range(1, n + 1)}

    def call_steps(self, tree):
        return set(int(round(t / tree.dt)) for t in self.calls)

    def maturity_step(self, tree):
        return int(round(self.maturity / tree.dt))

    def scheduled_cashflows(self):
        n = int(round(self.maturity * self.freq))
        times = np.array([k / self.freq for k in range(1, n + 1)])
        cfs = np.full(n, self.coupon_amount())
        cfs[-1] += self.face
        return times, cfs


def price_on_tree(tree, bond, spread=0.0, callable_=True):
    cpn_steps = bond.coupon_steps(tree)
    call_steps = bond.call_steps(tree)
    mat_step = bond.maturity_step(tree)
    K = bond.call_price
    V = {j: bond.face + cpn_steps.get(mat_step, 0.0)
         for j in tree.Q_levels[mat_step].keys()}
    for i in range(mat_step - 1, -1, -1):
        c = cpn_steps.get(i, 0.0)
        is_call = callable_ and (i in call_steps) and (i > 0)
        V_next = {}
        for j in tree.Q_levels[i].keys():
            k, pu, pm, pd = tree._branch(j)
            disc = np.exp(-(tree.short_rate(i, j) + spread) * tree.dt)
            H = c + disc * (pu * V.get(k + 1, 0.0)
                            + pm * V.get(k, 0.0)
                            + pd * V.get(k - 1, 0.0))
            V_next[j] = min(H, K + c) if is_call else H
        V = V_next
    return V[0]


def solve_oas(tree, bond, price, lo=-0.05, hi=0.20):
    return brentq(lambda s: price_on_tree(tree, bond, spread=s) - price, lo, hi, xtol=1e-10)


def solve_zspread(curve, bond, price, lo=-0.05, hi=0.30):
    times, cfs = bond.scheduled_cashflows()
    base_df = np.array([curve.discount(t) for t in times])
    return brentq(lambda z: np.sum(cfs * base_df * np.exp(-z * times)) - price,
                  lo, hi, xtol=1e-10)


def effective_risk(price_fn, y0, dy):
    p0, pu, pd = price_fn(y0), price_fn(y0 + dy), price_fn(y0 - dy)
    return (pd - pu) / (2 * p0 * dy), (pu + pd - 2 * p0) / (p0 * dy ** 2)


par_df = pd.read_csv("data/samples/synthetic_ust_par_curve.csv")
annual = np.arange(1, 31)
par_annual = np.interp(annual, par_df["tenor"].values, par_df["par_yield"].values)
curve = bootstrap_par(annual, par_annual, frequency=1, interp="linear_zero")

a, sigma, Tb = 0.08, 0.015, 10.0

# %% [markdown]
# ## 演習1：クーポン別の負のコンベクシティ
#
# クーポンを $\{4\%,5\%,6\%\}$ に変え、金利平行シフト $-200$〜$+200$bp（25bp刻み）で
# コーラブル債の価格と実効デュレーション（$\pm100$bp 中心差分）を求める。段数は再構築が
# 多いので120段（コール日・クーポン日が段に一致）。

# %%
N_sweep = 120
levels = np.arange(-0.02, 0.0201, 0.0025)
dy = 0.01
coupons = [0.04, 0.05, 0.06]


def make_price_fn(coupon):
    bnd = CallableBond(coupon=coupon, face=100.0, freq=1, maturity=10.0,
                       calls=[float(y) for y in range(3, 10)], call_price=100.0)

    def price_fn(shift):
        c = bootstrap_par(annual, par_annual + shift, frequency=1, interp="linear_zero")
        tr = HullWhiteTree(a, sigma, c, T=Tb, n_steps=N_sweep)
        return price_on_tree(tr, bnd, callable_=True)
    return price_fn


results = {}
for cp in coupons:
    pf = make_price_fn(cp)
    P, D, C = [], [], []
    for lvl in levels:
        d, c = effective_risk(pf, lvl, dy)
        P.append(pf(lvl))
        D.append(d)
        C.append(c)
    results[cp] = (np.array(P), np.array(D), np.array(C))

# クーポンが高いほどコールが深くイン・ザ・マネーになり、(i) 低金利側の実効デュレーションが
# 短く、(ii) 実効コンベクシティの最小値がより深く負に振れる。
print("クーポン別：低金利側デュレーションと最小実効コンベクシティ")
for cp in coupons:
    P, D, C = results[cp]
    print(f"  クーポン {cp*100:.0f}%: Dur(-200bp)={D[0]:.2f}  "
          f"Dur(+200bp)={D[-1]:.2f}  min Conv={C.min():.2f}")
# 高クーポンほど低金利側デュレーションが短い（単調）。これがネガコンの直接の指標。
assert results[0.06][1][0] < results[0.05][1][0] < results[0.04][1][0]
# いずれのクーポンでも、どこかの水準で実効コンベクシティが負になる。
for cp in coupons:
    assert results[cp][2].min() < 0
print("→ 高クーポンほどコールがイン・ザ・マネーになりやすく、低金利側デュレーションが短い。")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
bp = levels * 1e4
colors = plt.cm.viridis(np.linspace(0, 0.8, len(coupons)))
for cp, col in zip(coupons, colors):
    P, D, C = results[cp]
    axes[0].plot(bp, P, "o-", color=col, ms=4, label=f"クーポン {cp*100:.0f}%")
    axes[1].plot(bp, D, "o-", color=col, ms=4, label=f"クーポン {cp*100:.0f}%")
axes[0].axhline(100.0, color="0.6", ls="--", lw=0.8, label="コール価格 100")
axes[0].set_xlabel("金利平行シフト (bp)")
axes[0].set_ylabel("価格")
axes[0].set_title("価格・利回り曲線（クーポン別）")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("金利平行シフト (bp)")
axes[1].set_ylabel("実効デュレーション")
axes[1].set_title("低金利側のデュレーション縮小＝負のコンベクシティ")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：クーポンが高いほど、金利が下がったときに継続価値がコール価格を上回りやすく、
# コール（期限前償還）が起きやすい。その結果、価格・利回り曲線は低金利側でより強く頭打ちに
# なり（上に凸が深まる）、実効デュレーションの落ち込みも大きい。高クーポンのコーラブル債は
# ディープ・イン・ザ・マネーのコールを内包した状態に近く、負のコンベクシティが顕著に出る。

# %% [markdown]
# ## 演習2：OAS・Zスプレッド・オプションコストの分解表
#
# 本編のコーラブル債（クーポン5%）について、市場価格をフェア価格から
# $\{+1.0, 0.0, -1.0, -2.0\}$ ずらした4通りで、OAS・Zスプレッド・オプションコストを逆算する。

# %%
bond = CallableBond(coupon=0.05, face=100.0, freq=1, maturity=10.0,
                    calls=[float(y) for y in range(3, 10)], call_price=100.0)
tree = HullWhiteTree(a, sigma, curve, T=Tb, n_steps=200)
p_fair = price_on_tree(tree, bond, callable_=True)

rows = []
for d in (1.0, 0.0, -1.0, -2.0):
    p_mkt = p_fair + d
    oas = solve_oas(tree, bond, p_mkt)
    z = solve_zspread(curve, bond, p_mkt)
    rows.append((d, p_mkt, z * 1e4, oas * 1e4, (z - oas) * 1e4))

tbl = pd.DataFrame(rows, columns=["フェアからの差", "市場価格", "Zスプレッド(bp)",
                                  "OAS(bp)", "オプションコスト(bp)"])
print(tbl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# 市場価格が下がると OAS は上がる（割安＝高スプレッド）。オプションコストはほぼ一定。
oas_col = tbl["OAS(bp)"].values
oc_col = tbl["オプションコスト(bp)"].values
assert oas_col[0] < oas_col[-1]                          # 価格低下で OAS 上昇
assert oc_col.max() - oc_col.min() < oas_col.max() - oas_col.min()
print(f"\nOAS のレンジ = {oas_col.max()-oas_col.min():.1f}bp、"
      f"オプションコストのレンジ = {oc_col.max()-oc_col.min():.1f}bp")

# %% [markdown]
# ### オプションコストは σ とクーポンで決まる（市場価格水準に鈍感）
#
# 同じ市場価格（フェア）で σ とクーポンを振り、オプションコストが主にこの2つで動くことを
# 確認する。

# %%
print(f"{'σ':>7}{'クーポン':>9}{'オプションコスト(bp)':>20}")
for s in (0.008, 0.015, 0.025):
    for cp in (0.04, 0.06):
        bnd = CallableBond(coupon=cp, face=100.0, freq=1, maturity=10.0,
                           calls=[float(y) for y in range(3, 10)], call_price=100.0)
        tr = HullWhiteTree(a, s, curve, T=Tb, n_steps=200)
        pf = price_on_tree(tr, bnd, callable_=True)
        oc = (solve_zspread(curve, bnd, pf) - solve_oas(tr, bnd, pf)) * 1e4
        print(f"{s:>7.3f}{cp*100:>8.0f}%{oc:>20.1f}")

# %% [markdown]
# **解釈**：OAS は「価格の割安さ」を測る量なので、市場価格が下がれば素直に上がる。一方
# オプションコスト（Z−OAS）は、内包するコールオプションの価値をスプレッド換算したもので、
# ボラティリティ $\sigma$（オプションのタイムバリューを増やす）とクーポン水準（コールの
# モネネス）で決まる。市場価格の小さな上下（信用・流動性要因）では、Zスプレッドと OAS が
# ほぼ同量だけ動くので差はほとんど変わらない。だから相対価値の比較には、価格水準に紛れる
# Zスプレッドではなく、オプションを切り分けた OAS を使う。
