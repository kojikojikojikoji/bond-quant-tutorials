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
# # S4-4 モンテカルロ法と分散削減
#
# ## 学習目標
#
# - モンテカルロ推定量の標準誤差を中心極限定理から導き、収束が $1/\sqrt{n}$ で
#   支配されることを説明できる
# - 対照変量法（antithetic）・制御変量法（control variate）・準乱数（Sobol）を
#   スクラッチで実装し、それぞれが分散を削減する仕組みを理解する
# - 制御変量の最適係数 $c^\* = \operatorname{Cov}/\operatorname{Var}$ を導出し、
#   実装で推定できる
# - 解析解のあるヨーロピアンコール（Black-Scholes）で MC 誤差の理論収束を検証できる
# - Greeks 計算のバンプ法と pathwise 法の違い（分散・バイアス）を説明できる
# - 各分散削減手法について「同一精度に必要なパス数」を比較し、費用対効果を判断できる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# モンテカルロ法は、パス依存やエキゾチックな商品を値付けする現場の主力手法です。銀行の金利デスクがアジアンやバミューダ、経路依存のストラクチャーを評価するとき、あるいは MC-OAS（S8）で証券化商品を評価するとき、価格は「割引後ペイオフの期待値をパス平均で近似したもの」です。誤差が $1/\sqrt{n}$ でしか縮まないという事実は、精度を1桁上げるのにパスを100倍にする必要があることを意味し、計算コスト・応答時間・夜間バッチの締め切りに直結します。ここで分散削減が効いてきます。
#
# 対照変量・制御変量・準乱数（Sobol）は、いずれも「同じ価格を、より小さい分散で推定する」ための道具です。制御変量の最適係数 $c^\*=\operatorname{Cov}/\operatorname{Var}$ を推定して、解析解を持つ近い商品（例えばヨーロピアン）を制御変量に使えば、同じ精度を出すのに必要なパス数を何倍も減らせます。これはそのまま計算費用の削減であり、電子取引のように応答速度が収益を左右する場面（S11-2）では、分散削減の巧拙が値付けの競争力になります。この節で「同一精度に必要なパス数」を手法ごとに比較するのは、その費用対効果を定量化する練習です。
#
# グリークス（Greeks）計算のバンプ法と pathwise 法の違いは、ヘッジの現場に直結します。デルタやベガを正しく・安定して出せなければ、ヘッジがずれて損益がぶれます。バンプ法は実装が容易ですが分散が大きく、pathwise 法はバイアスなく分散を抑えられる代わりに商品ごとの微分が要る、という取捨選択です。さらに、同じ MC のエンジンはリスク管理側の VaR/ES 推定（S3）にも使われます。値付けでもヘッジでもリスク計測でも、「必要な精度を、許される計算資源で出しきる」ことが、損失回避とコスト削減の両面で効きます。
# %% [markdown]
# ## 理論
#
# ### モンテカルロ推定量と標準誤差
#
# 求めたいのは期待値 $\theta = \mathbb{E}[X]$ である。ここで $X$ は割引後ペイオフ
# のような確率変数で、$X_1,\dots,X_n$ をその独立同分布な標本とする。標本平均
#
# $$ \hat{\theta}_n = \frac{1}{n}\sum_{i=1}^{n} X_i $$
#
# を推定量に使う。大数の法則より $\hat{\theta}_n \to \theta$（概収束）だが、実務で
# 効くのは「どれだけ速く近づくか」である。$\sigma^2 = \operatorname{Var}(X)$ とおくと、
# 標本平均の分散は独立性から
#
# $$ \operatorname{Var}(\hat{\theta}_n) = \frac{1}{n^2}\sum_{i=1}^{n}\operatorname{Var}(X_i)
#    = \frac{\sigma^2}{n}. $$
#
# 中心極限定理は、標準化した推定量が正規分布へ収束することを述べる。
#
# $$ \sqrt{n}\,(\hat{\theta}_n - \theta) \xrightarrow{d} \mathcal{N}(0,\sigma^2). $$
#
# したがって $\hat{\theta}_n$ の**標準誤差**（standard error）は
#
# $$ \operatorname{SE}(\hat{\theta}_n) = \frac{\sigma}{\sqrt{n}} \approx \frac{s}{\sqrt{n}},
#    \qquad s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i-\hat{\theta}_n)^2 $$
#
# となり、近似 $95\%$ 信頼区間は $\hat{\theta}_n \pm 1.96\,\operatorname{SE}$ で与えられる。
# 重要な帰結は二つある。第一に、誤差は $1/\sqrt{n}$ でしか縮まないので、精度を1桁
# 上げるにはパス数を100倍にする必要がある。第二に、誤差は分散 $\sigma^2$ に比例する
# ので、**同じ $\theta$ を推定するなら分散の小さい推定量に作り替える**方が、パスを
# 増やすより効率的なことが多い。これが分散削減の動機である。
#
# ### 対照変量法（antithetic variates）
#
# 標準正規乱数 $Z$ を使ってペイオフ $X = f(Z)$ を作るとき、$-Z$ も同じ分布に従う
# （対称性）。そこで一つの乱数から二つの標本
#
# $$ X^{+} = f(Z), \qquad X^{-} = f(-Z) $$
#
# を作り、その平均 $\bar{X} = \tfrac{1}{2}(X^{+}+X^{-})$ を1標本として使う。個々の
# 分散が等しく $\operatorname{Var}(X^{+})=\operatorname{Var}(X^{-})=\sigma^2$ なら
#
# $$ \operatorname{Var}(\bar{X}) = \frac{1}{4}\bigl(\sigma^2 + \sigma^2
#    + 2\operatorname{Cov}(X^{+},X^{-})\bigr)
#    = \frac{\sigma^2}{2}\bigl(1 + \rho\bigr), \qquad
#    \rho = \operatorname{Corr}(X^{+}, X^{-}). $$
#
# 独立に2標本取れば分散は $\sigma^2/2$ なので、$\rho < 0$ のとき対照変量が有利になる。
# $f$ が単調なら $f(Z)$ と $f(-Z)$ は負に相関し、削減が効く。逆に $f$ が偶関数に近い
# 領域では $\rho > 0$ になり、かえって悪化しうる点に注意する。
#
# ### 制御変量法（control variate）と最適係数
#
# 目標 $X$ と相関し、かつ**期待値 $\mu_Y = \mathbb{E}[Y]$ が既知**の確率変数 $Y$
# （制御変量）があるとする。任意の係数 $c$ に対し
#
# $$ X(c) = X - c\,(Y - \mu_Y) $$
#
# は $\mathbb{E}[X(c)] = \mathbb{E}[X] = \theta$ を満たす（不偏）。分散は
#
# $$ \operatorname{Var}(X(c)) = \operatorname{Var}(X) - 2c\operatorname{Cov}(X,Y)
#    + c^2 \operatorname{Var}(Y) $$
#
# で、$c$ について2次関数だから、$\partial/\partial c = 0$ より最適係数
#
# $$ c^\* = \frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(Y)} $$
#
# を得る。これを代入すると最小分散は
#
# $$ \operatorname{Var}(X(c^\*)) = \operatorname{Var}(X)\,\bigl(1 - \rho_{XY}^2\bigr),
#    \qquad \rho_{XY} = \operatorname{Corr}(X, Y) $$
#
# となる。すなわち削減率は相関の2乗で決まり、$|\rho_{XY}|$ が1に近い制御変量ほど
# 効く。実務では $c^\*$ を標本から推定するが、これは $X$ を $Y$ へ回帰したときの
# 回帰係数（傾き）に一致する。$c^\*$ を推定に使うことで微小なバイアスが入るが、
# $n$ が大きければ無視できる。
#
# ### 準乱数（quasi-random）と Sobol 系列
#
# 擬似乱数は独立性を模すため、点が固まったり空いたりする「むら」が残り、被覆の
# 質は $O(n^{-1/2})$ でしか改善しない。**準乱数**（low-discrepancy sequence、低食い違い量列）
# は、定義域 $[0,1)^d$ を意図的に均一に埋める決定的な点列である。埋まり具合の悪さを
# 測る**食い違い量**（discrepancy）$D_n^\*$ を使うと、Koksma-Hlawka の不等式
#
# $$ \left| \frac{1}{n}\sum_{i=1}^{n} g(u_i) - \int_{[0,1)^d} g\,du \right|
#    \le V(g)\, D_n^\* $$
#
# が積分誤差を上から抑える。Sobol 系列など主要な構成では $D_n^\* = O\!\bigl((\log n)^d / n\bigr)$
# なので、次元が小さければ実質 $O(n^{-1})$ に近く、$1/\sqrt{n}$ より速い。使い方は、
# 生成した一様点 $u_i$ を逆累積分布 $\Phi^{-1}(u_i)$ で正規乱数に変換して MC に流し込む。
# なお準乱数は決定的なので素朴には信頼区間が作れない。スクランブル（乱数化）した
# 系列を複数バッチ回し、そのばらつきから誤差を見積もる。
#
# ### Greeks：バンプ法と pathwise 法
#
# 価格 $\theta(S_0) = \mathbb{E}[e^{-rT} f(S_T)]$ のデルタ $\partial\theta/\partial S_0$
# を MC で求める方法は二つある。
#
# - **バンプ法**（有限差分）：$\hat{\Delta} = \dfrac{\hat{\theta}(S_0+h) - \hat{\theta}(S_0-h)}{2h}$。
#   実装は簡単だが、$O(h^2)$ の離散化バイアスが残り、$h$ を小さくすると分散が
#   $O(h^{-2})$ で爆発する。**同一乱数**（common random numbers）を両側に使うと
#   分散は劇的に下がる。
# - **pathwise 法**：微分と期待値の順序を交換し、$\Delta = \mathbb{E}\!\left[e^{-rT}
#   \dfrac{\partial f}{\partial S_T}\dfrac{\partial S_T}{\partial S_0}\right]$ を直接推定する。
#   GBM $S_T = S_0 e^{(r-\sigma^2/2)T+\sigma\sqrt{T}Z}$ では $\partial S_T/\partial S_0 = S_T/S_0$
#   なので、コールのデルタは
#
# $$ \hat{\Delta}_{\text{pw}} = e^{-rT}\,\mathbf{1}\{S_T > K\}\,\frac{S_T}{S_0}. $$
#
#   これは**不偏**（バイアスゼロ）で分散も小さいが、ペイオフが微分可能である必要が
#   ある（ヨーロピアンコールは連続なので可、ディジタルは不可）。

# %% [markdown]
# ## スクラッチ実装
#
# GBM 下のヨーロピアンコールを題材に、MC エンジンと3つの分散削減、および2種類の
# Greeks をスクラッチで書く。パラメータは全編を通じて $S_0=100,\,K=100,\,r=3\%,\,
# \sigma=20\%,\,T=1$ 年に固定する。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `bs_call_price(S0,K,r,sigma,T)` | 商品パラメータ | 価格 | Black-Scholes コールの解析解（真値） |
# | `gbm_terminal(S0,r,sigma,T,Z)` | パラメータ, 正規乱数配列 | $S_T$ 配列 | GBM の満期株価をベクトルで生成 |
# | `discounted_call_payoff(ST,K,r,T)` | 満期株価, 行使価格 | 割引ペイオフ配列 | $e^{-rT}\max(S_T-K,0)$ |
# | `mc_call_crude(...,n,seed)` | パラメータ, パス数 | dict(mean, stderr, samples) | 素朴 MC。標準誤差付きで返す |
# | `mc_call_antithetic(...,n,seed)` | 同上 | dict(...) | 対照変量 MC（$Z$ と $-Z$ の対平均） |
# | `mc_call_control(...,n,seed)` | 同上 | dict(..., beta) | 制御変量 MC（割引満期株価を制御に使う） |
# | `sobol_normals(n,seed)` | 点数, スクランブル種 | 正規乱数配列 | Sobol 一様点を $\Phi^{-1}$ で正規化 |
# | `mc_call_sobol(...,n,seed)` | パラメータ, 点数 | dict(mean, samples) | 準乱数 MC（1バッチ） |
# | `pathwise_delta(...,n,seed)` | パラメータ, パス数 | dict(mean, stderr) | pathwise デルタ推定量 |
# | `bump_delta(...,n,h,seed)` | パラメータ, パス数, バンプ幅 | dict(mean, stderr) | 同一乱数バンプ法デルタ |

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, qmc

import bondlab
from bondlab import sim

print("bondlab version:", bondlab.__version__)

# 全編共通のパラメータ。
S0, K, r, SIGMA, T = 100.0, 100.0, 0.03, 0.20, 1.0
SEED = 20260707


def bs_call_price(S0, K, r, sigma, T):
    """Black-Scholes ヨーロピアンコールの解析解（真値の基準）。"""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def gbm_terminal(S0, r, sigma, T, Z):
    """標準正規乱数 Z から GBM の満期株価 S_T をベクトルで生成する。"""
    return S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)


def discounted_call_payoff(ST, K, r, T):
    """割引後のコールペイオフ e^{-rT} max(S_T - K, 0)。"""
    return np.exp(-r * T) * np.maximum(ST - K, 0.0)


BS_TRUE = bs_call_price(S0, K, r, SIGMA, T)
print(f"Black-Scholes 解析解（真値）: {BS_TRUE:.6f}")

# %% [markdown]
# ### 素朴モンテカルロと標準誤差レポート
#
# まず素朴（crude）MC を実装する。`bondlab.sim.mc_stats` に標本を渡すと平均・標準誤差・
# 信頼区間が揃うので、レポート部分はそれに委譲する。

# %%
def mc_call_crude(S0, K, r, sigma, T, n, seed):
    """素朴モンテカルロ。標準誤差・信頼区間を mc_stats で付けて返す。"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    ST = gbm_terminal(S0, r, sigma, T, Z)
    samples = discounted_call_payoff(ST, K, r, T)
    stats = sim.mc_stats(samples)
    stats["samples"] = samples
    return stats


res_crude = mc_call_crude(S0, K, r, SIGMA, T, n=100_000, seed=SEED)
print(f"素朴MC 価格   : {res_crude['mean']:.6f}")
print(f"標準誤差      : {res_crude['stderr']:.6f}")
print(f"95%信頼区間   : [{res_crude['ci_low']:.6f}, {res_crude['ci_high']:.6f}]")
print(f"真値との差    : {res_crude['mean'] - BS_TRUE:+.6f}  "
      f"(≒ {(res_crude['mean'] - BS_TRUE) / res_crude['stderr']:+.2f} × 標準誤差)")

# 真値が 95% 信頼区間に入っていることを確認する。
assert res_crude["ci_low"] < BS_TRUE < res_crude["ci_high"]

# %% [markdown]
# ### 対照変量法：bondlab の antithetic ブラウン運動と一致させる
#
# 自前で $Z$ と $-Z$ を対にして平均を取る。1標本＝1対（正規乱数2個）なので、
# 同じ乱数消費で `bondlab.sim.brownian_paths(antithetic=True)` の終端増分から作った
# $Z$ と一致することを確認する。

# %%
def mc_call_antithetic(S0, K, r, sigma, T, n, seed):
    """対照変量 MC。n 対（= 2n 個の正規乱数）から対ごとの平均を1標本にする。"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    ST_plus = gbm_terminal(S0, r, sigma, T, Z)
    ST_minus = gbm_terminal(S0, r, sigma, T, -Z)
    pay_plus = discounted_call_payoff(ST_plus, K, r, T)
    pay_minus = discounted_call_payoff(ST_minus, K, r, T)
    samples = 0.5 * (pay_plus + pay_minus)  # 対平均が1標本
    stats = sim.mc_stats(samples)
    stats["samples"] = samples
    stats["rho"] = float(np.corrcoef(pay_plus, pay_minus)[0, 1])
    return stats


res_anti = mc_call_antithetic(S0, K, r, SIGMA, T, n=50_000, seed=SEED)
print(f"対照変量MC 価格 : {res_anti['mean']:.6f}")
print(f"標準誤差        : {res_anti['stderr']:.6f}")
print(f"対の相関 ρ      : {res_anti['rho']:+.4f}  （負なら削減が効く）")

# bondlab.sim.brownian_paths(antithetic=True) の終端が Z*sqrt(T) の対称化と一致することを確認。
n_chk = 8
_, W = sim.brownian_paths(n_chk, 1, T, seed=SEED, antithetic=True)
Z_from_bl = W[:, -1] / np.sqrt(T)
half = (n_chk + 1) // 2
# 後半が前半の符号反転になっている（対照変量の定義そのもの）。
assert np.allclose(Z_from_bl[:half], -Z_from_bl[half:2 * half])
print("bondlab の antithetic 終端増分が Z と -Z の対になっていることを確認しました")

# %% [markdown]
# ### 制御変量法：割引満期株価を制御に使い、bondlab.sim.control_variate と一致させる
#
# 制御変量 $Y = e^{-rT}S_T$ は既知の期待値 $\mu_Y = S_0$（リスク中立測度でのマルチンゲール性）
# を持ち、コールペイオフと強く相関する。自前で $c^\* = \operatorname{Cov}/\operatorname{Var}$
# を推定した結果が、`bondlab.sim.control_variate` の返す `estimate` / `beta` と一致することを
# 確認する。

# %%
def mc_call_control(S0, K, r, sigma, T, n, seed):
    """制御変量 MC。制御 Y = 割引満期株価（E[Y]=S0）で分散を削減する。"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    ST = gbm_terminal(S0, r, sigma, T, Z)
    target = discounted_call_payoff(ST, K, r, T)  # X
    control = np.exp(-r * T) * ST                  # Y, E[Y] = S0
    control_mean = S0
    cov = np.cov(target, control)
    c_star = cov[0, 1] / cov[1, 1]
    adjusted = target - c_star * (control - control_mean)
    stats = sim.mc_stats(adjusted)
    stats["beta"] = float(c_star)
    stats["rho"] = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    stats["samples"] = adjusted
    return stats


res_ctrl = mc_call_control(S0, K, r, SIGMA, T, n=100_000, seed=SEED)
print(f"制御変量MC 価格 : {res_ctrl['mean']:.6f}")
print(f"標準誤差        : {res_ctrl['stderr']:.6f}")
print(f"最適係数 c*     : {res_ctrl['beta']:.6f}")
print(f"target-control 相関 ρ : {res_ctrl['rho']:.4f}")

# bondlab.sim.control_variate と突き合わせる（同じ乱数列で再現）。
rng = np.random.default_rng(SEED)
Z = rng.standard_normal(100_000)
ST = gbm_terminal(S0, r, SIGMA, T, Z)
target = discounted_call_payoff(ST, K, r, T)
control = np.exp(-r * T) * ST
bl = sim.control_variate(target, control, S0)
print(f"\nbondlab.control_variate: estimate={bl['estimate']:.6f}, "
      f"beta={bl['beta']:.6f}, stderr={bl['stderr']:.6f}")
assert abs(bl["estimate"] - res_ctrl["mean"]) < 1e-10
assert abs(bl["beta"] - res_ctrl["beta"]) < 1e-10
assert abs(bl["stderr"] - res_ctrl["stderr"]) < 1e-10
print("自作の制御変量と bondlab.sim.control_variate が一致しました")

# %% [markdown]
# ### 準乱数（Sobol）：一様点を正規化して MC に流す
#
# `scipy.stats.qmc.Sobol` で $[0,1)$ の低食い違い量点を作り、$\Phi^{-1}$ で正規乱数に
# 変換する。Sobol はべき乗個（$2^m$）の点で均一性が最良になるので、`random_base2` を
# 使う。

# %%
def sobol_normals(n_log2, seed):
    """2^n_log2 個の Sobol 一様点を標準正規乱数へ変換して返す（1次元）。"""
    engine = qmc.Sobol(d=1, scramble=True, seed=np.random.default_rng(seed))
    u = engine.random_base2(m=n_log2).ravel()  # 2^m 点
    # 端点 0 を避けて Phi^{-1} の発散を防ぐ。
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return norm.ppf(u)


def mc_call_sobol(S0, K, r, sigma, T, n_log2, seed):
    """準乱数（Sobol）1バッチのコール価格。誤差は複数バッチのばらつきで測る。"""
    Z = sobol_normals(n_log2, seed)
    ST = gbm_terminal(S0, r, sigma, T, Z)
    samples = discounted_call_payoff(ST, K, r, T)
    return dict(mean=float(samples.mean()), n=samples.size, samples=samples)


res_sobol = mc_call_sobol(S0, K, r, SIGMA, T, n_log2=17, seed=SEED)  # 2^17 = 131072 点
print(f"Sobol MC 価格 ({res_sobol['n']}点): {res_sobol['mean']:.6f}")
print(f"真値との差              : {res_sobol['mean'] - BS_TRUE:+.6f}")

# 同じ点数の素朴 MC より真値に近いことを1例で確認（統計的な比較は後段で行う）。
res_crude_same = mc_call_crude(S0, K, r, SIGMA, T, n=res_sobol["n"], seed=SEED)
print(f"同点数の素朴MC の差     : {res_crude_same['mean'] - BS_TRUE:+.6f}")

# %% [markdown]
# ### Greeks：pathwise 法とバンプ法
#
# デルタを2通りで推定する。pathwise は不偏推定量 $e^{-rT}\mathbf{1}\{S_T>K\}S_T/S_0$、
# バンプは同一乱数の中心差分。真値は $\Delta_{\text{BS}} = e^{-qT}N(d_1) = N(d_1)$（無配当）。

# %%
def bs_call_delta(S0, K, r, sigma, T):
    """Black-Scholes コールのデルタ N(d1)（真値）。"""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def pathwise_delta(S0, K, r, sigma, T, n, seed):
    """pathwise デルタ推定量（不偏）。"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    ST = gbm_terminal(S0, r, sigma, T, Z)
    est = np.exp(-r * T) * (ST > K).astype(float) * ST / S0
    stats = sim.mc_stats(est)
    return dict(mean=stats["mean"], stderr=stats["stderr"], samples=est)


def bump_delta(S0, K, r, sigma, T, n, h, seed):
    """同一乱数を使う中心差分（バンプ法）デルタ。"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)  # 両側で共通の乱数
    ST_up = gbm_terminal(S0 + h, r, sigma, T, Z)
    ST_dn = gbm_terminal(S0 - h, r, sigma, T, Z)
    pay_up = discounted_call_payoff(ST_up, K, r, T)
    pay_dn = discounted_call_payoff(ST_dn, K, r, T)
    est = (pay_up - pay_dn) / (2.0 * h)
    stats = sim.mc_stats(est)
    return dict(mean=stats["mean"], stderr=stats["stderr"], samples=est)


DELTA_TRUE = bs_call_delta(S0, K, r, SIGMA, T)
pw = pathwise_delta(S0, K, r, SIGMA, T, n=100_000, seed=SEED)
bp = bump_delta(S0, K, r, SIGMA, T, n=100_000, h=1.0, seed=SEED)
print(f"デルタ真値 (N(d1))     : {DELTA_TRUE:.6f}")
print(f"pathwise デルタ        : {pw['mean']:.6f}  (SE={pw['stderr']:.6f})")
print(f"バンプ デルタ (h=1)     : {bp['mean']:.6f}  (SE={bp['stderr']:.6f})")
print(f"pathwise の標準誤差はバンプの {bp['stderr'] / pw['stderr']:.1f} 分の1")

# %% [markdown]
# ## QuantLib検証
#
# ここは「解析解との突合」と「$1/\sqrt{n}$ 収束の実証」を検証と位置づけるセクションである。
# GBM 下のヨーロピアンコールには Black-Scholes の閉じた解があるので、それを真値として
# (1) MC 価格が真値へ収束すること、(2) 標準誤差が理論どおり $1/\sqrt{n}$ で減衰すること、
# の2点を確認する。真値の独立な裏取りとして QuantLib の `AnalyticEuropeanEngine` も併記し、
# 自作 Black-Scholes 式と一致することを見る。

# %%
import QuantLib as ql

# QuantLib で同じヨーロピアンコールを組み、解析エンジンで評価する。
today = ql.Date(7, 7, 2026)
ql.Settings.instance().evaluationDate = today
day_count = ql.Actual365Fixed()
calendar = ql.NullCalendar()

spot = ql.QuoteHandle(ql.SimpleQuote(S0))
rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
vol_ts = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, calendar, SIGMA, day_count))
bsm = ql.BlackScholesMertonProcess(spot, div_ts, rate_ts, vol_ts)

maturity = today + ql.Period(int(round(T * 365)), ql.Days)
payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
exercise = ql.EuropeanExercise(maturity)
option = ql.VanillaOption(payoff, exercise)
option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm))

ql_price = option.NPV()
print(f"自作 Black-Scholes  : {BS_TRUE:.6f}")
print(f"QuantLib Analytic   : {ql_price:.6f}")
print(f"差                  : {abs(ql_price - BS_TRUE):.2e}")
# 満期の日数丸めによる ACT/365 のわずかな差を許容。
assert abs(ql_price - BS_TRUE) < 5e-3
print("自作解析解と QuantLib が一致（丸め誤差内）")

# %% [markdown]
# ### 検証1：MC 価格の真値への収束
#
# パス数 $n$ を増やすと素朴 MC 価格が Black-Scholes 真値へ収束することを、95% 信頼区間
# 付きで示す。

# %%
n_grid = np.array([250, 1000, 4000, 16000, 64000, 256000])
rng_master = np.random.default_rng(SEED)
conv = []
for n in n_grid:
    res = mc_call_crude(S0, K, r, SIGMA, T, n=int(n), seed=int(rng_master.integers(1 << 31)))
    conv.append((int(n), res["mean"], res["stderr"], res["ci_low"], res["ci_high"]))
conv_df = pd.DataFrame(conv, columns=["n", "MC価格", "標準誤差", "CI下限", "CI上限"])
conv_df["真値との差"] = conv_df["MC価格"] - BS_TRUE
print(conv_df.round(5).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(conv_df["n"], conv_df["MC価格"],
            yerr=1.96 * conv_df["標準誤差"], fmt="o-", capsize=4,
            color="#1f77b4", label="MC 価格 ±1.96·SE")
ax.axhline(BS_TRUE, color="#d62728", ls="--", label=f"BS 真値 {BS_TRUE:.4f}")
ax.set_xscale("log")
ax.set_xlabel("パス数 n（対数軸）")
ax.set_ylabel("コール価格")
ax.set_title("素朴MC 価格の真値への収束（95%信頼区間つき）")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 検証2：標準誤差の $1/\sqrt{n}$ 収束
#
# 標準誤差を $n$ に対して両対数でプロットすると、理論では傾き $-1/2$ の直線に乗る。
# log-log 回帰の傾きを数値で確認する。

# %%
log_n = np.log(conv_df["n"].to_numpy())
log_se = np.log(conv_df["標準誤差"].to_numpy())
slope, intercept = np.polyfit(log_n, log_se, 1)
print(f"標準誤差 ~ n^({slope:.3f})   （理論は -0.5）")
assert -0.6 < slope < -0.4

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(conv_df["n"], conv_df["標準誤差"], "o", color="#1f77b4", label="実測 標準誤差")
ax.loglog(conv_df["n"], np.exp(intercept) * conv_df["n"] ** slope, "-",
          color="#1f77b4", alpha=0.5, label=f"回帰 傾き {slope:.3f}")
ref = conv_df["標準誤差"].iloc[0] * (conv_df["n"] / conv_df["n"].iloc[0]) ** (-0.5)
ax.loglog(conv_df["n"], ref, "--", color="#d62728", alpha=0.6, label="理論 傾き -0.5")
ax.set_xlabel("パス数 n")
ax.set_ylabel("標準誤差")
ax.set_title("標準誤差の 1/√n 収束")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 実データ適用
#
# ここでは市場データではなく、乱数シードを固定した合成データを「実データ」に見立てる
# （MC の対象は本来すべて人工的な乱数である）。素朴・対照変量・制御変量・Sobol の
# 4手法について、**同一の総パス予算での標準誤差**と、そこから逆算した**同一精度に
# 必要なパス数**を比較する。
#
# 比較を公平にするため、いずれも「正規乱数を $N$ 個消費する」予算に揃える。対照変量は
# $N/2$ 対、制御変量・素朴は $N$ 標本、Sobol は $N$ 点（$2^m$）とする。準乱数は決定的で
# 素朴には標準誤差が測れないので、スクランブルを変えた $R$ バッチのばらつきから
# 標準誤差を見積もる。

# %%
def sobol_stderr(S0, K, r, sigma, T, n_log2, n_batches, seed):
    """スクランブル済み Sobol を n_batches 回まわし、バッチ推定のばらつきで標準誤差を測る。"""
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_batches)
    means = np.array([mc_call_sobol(S0, K, r, sigma, T, n_log2, s)["mean"]
                      for s in child_seeds])
    return float(means.mean()), float(means.std(ddof=1) / np.sqrt(n_batches)), means


# 共通のパス予算 N = 2^17 = 131072 に揃える。
N_LOG2 = 17
N = 2 ** N_LOG2

res_crude_b = mc_call_crude(S0, K, r, SIGMA, T, n=N, seed=SEED)
res_anti_b = mc_call_antithetic(S0, K, r, SIGMA, T, n=N // 2, seed=SEED)  # N/2 対 = N 乱数
res_ctrl_b = mc_call_control(S0, K, r, SIGMA, T, n=N, seed=SEED)
sob_mean, sob_se, sob_means = sobol_stderr(S0, K, r, SIGMA, T, N_LOG2, n_batches=32, seed=SEED)

# 標準誤差から「1パスあたりの分散」V1 = SE^2 · N を逆算し、目標 SE に必要なパス数を求める。
target_se = 0.005  # 目標：標準誤差 0.5 セント
rows = []
for name, mean_, se_, n_used in [
    ("素朴 (crude)", res_crude_b["mean"], res_crude_b["stderr"], N),
    ("対照変量 (antithetic)", res_anti_b["mean"], res_anti_b["stderr"], N),
    ("制御変量 (control variate)", res_ctrl_b["mean"], res_ctrl_b["stderr"], N),
    ("準乱数 (Sobol)", sob_mean, sob_se, N),
]:
    v1 = se_ ** 2 * n_used                 # パス予算あたりの実効分散
    n_req = int(np.ceil(v1 / target_se ** 2))
    rows.append(dict(手法=name, 価格=mean_, 標準誤差=se_,
                     実効分散V1=v1, 必要パス数=n_req))

cmp_df = pd.DataFrame(rows)
cmp_df["分散削減率"] = cmp_df["実効分散V1"].iloc[0] / cmp_df["実効分散V1"]
cmp_df["パス削減率"] = cmp_df["必要パス数"].iloc[0] / cmp_df["必要パス数"]
print(f"総パス予算 N = {N}、目標標準誤差 = {target_se}")
print(cmp_df.assign(
    価格=cmp_df["価格"].round(5),
    標準誤差=cmp_df["標準誤差"].round(6),
    実効分散V1=cmp_df["実効分散V1"].round(3),
    分散削減率=cmp_df["分散削減率"].round(1),
    パス削減率=cmp_df["パス削減率"].round(1),
).to_string(index=False))

# 全手法とも真値の近傍にあることを確認。
for m in cmp_df["価格"]:
    assert abs(m - BS_TRUE) < 0.05
# 分散削減が実際に効いている（素朴より小さいSE）ことを確認。
assert res_ctrl_b["stderr"] < res_crude_b["stderr"]
assert res_anti_b["stderr"] < res_crude_b["stderr"]

# %% [markdown]
# 表の読み方：**分散削減率**は「同じパス数でどれだけ分散が下がったか」、**パス削減率**は
# 「同じ精度に何分の1のパスで到達できるか」を表す（両者は理論上一致する）。制御変量は
# 割引満期株価との相関が非常に高いため、桁で分散が下がる。Sobol も $1/\sqrt{n}$ より速い
# ため、この予算では大きく削減される（ただしここでの必要パス数は $1/\sqrt{n}$ 換算の
# 保守的な見積もりで、実際の Sobol はさらに速く縮む）。

# %%
fig, ax = plt.subplots(figsize=(9, 5))
labels = cmp_df["手法"].tolist()
red = cmp_df["分散削減率"].to_numpy()
bars = ax.barh(labels, red, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"])
ax.set_xlabel("分散削減率（素朴MC=1、大きいほど効率的）")
ax.set_title(f"分散削減率の比較（総パス予算 N={N}）")
ax.set_xscale("log")
for b, v in zip(bars, red):
    ax.text(v, b.get_y() + b.get_height() / 2, f" ×{v:.1f}", va="center")
ax.grid(alpha=0.3, axis="x", which="both")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 演習
#
# 1. **分散削減率の比較表**：対照変量・制御変量・Sobol の3手法について、パス予算を
#    $N \in \{2^{12}, 2^{14}, 2^{16}\}$ と振り、それぞれの分散削減率（対 素朴MC）を
#    1枚の表にまとめよ。予算が増えると Sobol の優位が広がる（$1/\sqrt{n}$ との差が開く）
#    ことを確認し、理由を一言で述べよ。
# 2. **pathwise vs バンプの Greeks**：デルタについて、pathwise 推定量とバンプ法
#    （同一乱数・非同一乱数の両方）を比較せよ。バンプ幅 $h$ を $\{0.01, 0.1, 1, 5\}$ と
#    振り、各設定の推定値・バイアス（真値 $N(d_1)$ との差）・標準誤差を表にまとめ、
#    「pathwise は不偏で低分散」「非同一乱数バンプは $h$ を小さくすると分散爆発」を
#    数値で示せ。
#
# 解答例は `solutions/S4/sol_0404.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/04_stochastic.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | 標準誤差 | standard error | 推定量のばらつき $\sigma/\sqrt{n}$。MC 精度の尺度で $1/\sqrt{n}$ で縮む |
# | 対照変量法 | antithetic variates | $Z$ と $-Z$ を対にして負の相関で分散を下げる手法 |
# | 制御変量法 | control variate | 既知平均をもつ相関変量で補正し、$1-\rho^2$ 倍に分散を下げる手法 |
# | 準乱数 | quasi-random | 定義域を意図的に均一に埋める決定的点列。積分誤差が $1/\sqrt{n}$ より速い |
# | Sobol系列 | Sobol sequence | 代表的な低食い違い量列。食い違い量 $O((\log n)^d/n)$ |
