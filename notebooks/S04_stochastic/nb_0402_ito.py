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
# # S4-2 伊藤の公式と確率積分
#
# ## 学習目標
#
# - 確率積分（伊藤積分, Itô integral）を左端点評価で定義し、通常の
#   Riemann-Stieltjes 積分となぜ値が変わるのかを説明できる
# - 伊藤の公式（Itô's formula, 1次元・時間依存）を書き下し、$f(W_t)=W_t^2$ の
#   ような具体例に適用できる
# - 幾何ブラウン運動（GBM, geometric Brownian motion）と
#   Ornstein-Uhlenbeck 過程（OU 過程）の厳密解を、自作コードで数値解と突き合わせて
#   検証できる
# - OU 過程の定常分布（平均・分散）の理論値を、シミュレーションで確認できる
#
# S5 の短期金利モデルでは、SDE を書いて解の分布・モーメントを求める操作を
# 繰り返します。そこで実際に使うのは「伊藤の公式の使い方」であって導出そのもの
# ではありません。本ノートは理論を省かずに置きますが、導出スケッチは optional
# 節にまとめ、数値検証の節だけ追えば S5 に進める構成にしています。

# %% [markdown]
# ## 理論
#
# ### 確率積分（伊藤積分, Itô integral）の定義
#
# 標準ブラウン運動 $W_t$ に対し、被積分過程 $H_t$ の確率積分を、区間
# $[0,T]$ の分割 $0=t_0<t_1<\dots<t_n=T$ にそって
#
# $$
# \int_0^T H_t\, dW_t \;=\; \lim_{n\to\infty}\sum_{i=0}^{n-1} H_{t_i}\,\bigl(W_{t_{i+1}}-W_{t_i}\bigr)
# $$
#
# と定義します。要点は被積分過程を各小区間の**左端点** $t_i$ で評価することです。
# 左端点で止めることで $H_{t_i}$ は増分 $W_{t_{i+1}}-W_{t_i}$ と独立になり、
# 積分がマルチンゲール（平均が動かない過程）になります。金融でこの性質が
# 効いてくるのは、「その時点までの情報だけで賭ける（先読みしない）」という
# 取引の直感と一致するためです。
#
# ### Riemann-Stieltjes 積分との違い
#
# 通常の関数 $g(t)$ が有界変動なら、評価点を左端点にしても中点にしても
# Riemann-Stieltjes 積分の値は一致します。ところがブラウン運動は
# 有界変動ではなく、**二次変分**が
#
# $$
# \sum_{i=0}^{n-1}\bigl(W_{t_{i+1}}-W_{t_i}\bigr)^2 \;\xrightarrow[n\to\infty]{}\; T
# $$
#
# という 0 でない極限を持ちます。この $\sum (dW)^2 \to T$、記号で $(dW_t)^2=dt$
# こそが伊藤の理論の核です。評価点を左端点から中点へずらすと、ずれが二次変分に
# 比例して残り、積分値が変わります。中点評価に対応するのが後述の
# Stratonovich 積分で、両者の差はちょうど二次変分の項になります。
#
# ### 伊藤の公式（Itô's formula, 1次元・時間依存）
#
# $X_t$ が SDE
#
# $$
# dX_t = \mu(t,X_t)\,dt + \sigma(t,X_t)\,dW_t
# $$
#
# に従い、$f(t,x)$ が $t$ で1回・$x$ で2回連続微分可能なとき、$Y_t=f(t,X_t)$ は
#
# $$
# dY_t = \left(\frac{\partial f}{\partial t}
#   + \mu\,\frac{\partial f}{\partial x}
#   + \tfrac{1}{2}\sigma^2\,\frac{\partial^2 f}{\partial x^2}\right)dt
#   + \sigma\,\frac{\partial f}{\partial x}\,dW_t
# $$
#
# に従います。通常の連鎖律との違いは $\tfrac12\sigma^2 f_{xx}$ という項で、これは
# $(dX_t)^2=\sigma^2\,dt$ から生じます。ここで $\mu$ を**ドリフト（drift）**、
# $\sigma$ を**拡散係数（diffusion coefficient）** と呼びます。ドリフトが $dt$ の
# 係数（変化の平均的な向き）、拡散係数が $dW$ の係数（揺らぎの大きさ）です。
#
# 最小の例として $X_t=W_t$（$\mu=0,\ \sigma=1$）で $f(x)=x^2$ をとると
#
# $$
# d(W_t^2) = 2W_t\,dW_t + dt
# \quad\Longrightarrow\quad
# \int_0^T W_t\,dW_t = \frac{W_T^2 - T}{2}.
# $$
#
# 普通の微積分なら $\int_0^T W\,dW = W_T^2/2$ となるはずのところに、二次変分由来の
# $-T/2$ が付きます。この 1 本を数値で確かめるのが本ノートの中心です。
#
# ### 幾何ブラウン運動（GBM）の解
#
# $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$ に $f(x)=\ln x$ で伊藤の公式を使うと、
# $f_x=1/x,\ f_{xx}=-1/x^2$ より
#
# $$
# d(\ln S_t) = \left(\mu-\tfrac12\sigma^2\right)dt + \sigma\,dW_t
# $$
#
# となり、積分して指数を戻すと厳密解
#
# $$
# S_t = S_0\,\exp\!\left[\left(\mu-\tfrac12\sigma^2\right)t + \sigma W_t\right]
# $$
#
# を得ます。$-\tfrac12\sigma^2$ の補正が付くのが、単純な連鎖律との違いです。
#
# ### Ornstein-Uhlenbeck 過程（OU 過程）の解と定常分布
#
# $dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t$ は、水準 $\theta$ へ速さ
# $\kappa>0$ で引き戻される平均回帰過程です。$f(t,x)=x e^{\kappa t}$ に伊藤の公式を
# 使う（$\partial_t f=\kappa x e^{\kappa t},\ f_x=e^{\kappa t},\ f_{xx}=0$）と
# $d(X_t e^{\kappa t}) = \kappa\theta e^{\kappa t}dt + \sigma e^{\kappa t}dW_t$
# となり、積分して
#
# $$
# X_t = X_0 e^{-\kappa t} + \theta\bigl(1-e^{-\kappa t}\bigr)
#     + \sigma\int_0^t e^{-\kappa (t-s)}\,dW_s.
# $$
#
# 最後の項は平均 0 のガウス確率積分で、伊藤の等長性（Itô isometry）から分散は
# $\dfrac{\sigma^2}{2\kappa}\bigl(1-e^{-2\kappa t}\bigr)$ です。したがって
# 条件付き分布は
#
# $$
# X_t \mid X_0 \sim \mathcal{N}\!\left(X_0 e^{-\kappa t}+\theta(1-e^{-\kappa t}),\;
# \frac{\sigma^2}{2\kappa}\bigl(1-e^{-2\kappa t}\bigr)\right).
# $$
#
# $t\to\infty$ で初期値の記憶が消え、**定常分布**
#
# $$
# X_\infty \sim \mathcal{N}\!\left(\theta,\ \frac{\sigma^2}{2\kappa}\right)
# $$
#
# に収束します。とくに $\theta=0$ なら平均 0・分散 $\sigma^2/(2\kappa)$ です。
# この定常分散を後でシミュレーションと突き合わせます。
#
# ### （optional）Stratonovich 積分と伊藤積分の関係
#
# 中点評価の Stratonovich 積分 $\int_0^T H_t\circ dW_t$ は、被積分過程を
# $(W_{t_i}+W_{t_{i+1}})/2$ で評価します。$H_t=W_t$ のときは各小区間で
# $\tfrac12(W_{t_i}+W_{t_{i+1}})(W_{t_{i+1}}-W_{t_i})=\tfrac12(W_{t_{i+1}}^2-W_{t_i}^2)$
# と望遠鏡和になり、
#
# $$
# \int_0^T W_t\circ dW_t = \frac{W_T^2}{2}
# $$
#
# ちょうど普通の微積分の答えになります。伊藤積分との差は
# $\tfrac12\sum(dW)^2\to T/2$ で、これは二次変分そのものです。本ノートではこの
# 差 $T/2$ を数値で確認します。導出の詳細に立ち入らなくても、以降の数値検証節
# だけで S5 に進めます。

# %% [markdown]
# ## スクラッチ実装
#
# 確率積分（左端点＝伊藤、中点＝Stratonovich）と、GBM・OU の厳密解を自作します。
# ブラウン運動の生成と SDE の数値積分には `bondlab.sim` を使います。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `ito_integral(W)` | パス配列 `W` (n_paths, n_steps+1) | 各パスの $\int W\,dW$ | 左端点評価の確率積分 |
# | `stratonovich_integral(W)` | 同上 | 各パスの $\int W\circ dW$ | 中点評価の確率積分 |
# | `quadratic_variation(W)` | 同上 | 各パスの $\sum(dW)^2$ | 二次変分（$\to T$ を確認） |
# | `gbm_exact(S0, mu, sigma, times, W)` | 初期値・係数・時間・BM | パス配列 | GBM の厳密解 |
# | `ou_exact_paths(x0, kappa, theta, sigma, times, n_paths, seed)` | OU パラメータほか | (パス配列) | OU の厳密遷移サンプリング |
#
# ### 使用する bondlab の関数
#
# | 関数 | 役割 |
# |---|---|
# | `bondlab.sim.brownian_paths` | 標準ブラウン運動のパス生成 |
# | `bondlab.sim.simulate_sde` | 一般 SDE の数値積分（Euler-Maruyama / Milstein） |
#
# `brownian_paths` と `simulate_sde` は、同じ `seed` を渡すと同一の乱数増分
# $dW$ を使います。この性質を利用すると、同じ 1 本のブラウン運動に対する
# 厳密解と数値解をパスごとに比較できます。

# %%
import numpy as np
import matplotlib.pyplot as plt

import bondlab
from bondlab.sim import brownian_paths, simulate_sde

print("bondlab version:", bondlab.__version__)


def ito_integral(W):
    """左端点評価で各パスの ∫_0^T W dW を近似する。

    W[:, :-1]（各区間の左端の値）に増分 dW を掛けて和を取る。
    """
    dW = np.diff(W, axis=1)
    return np.sum(W[:, :-1] * dW, axis=1)


def stratonovich_integral(W):
    """中点評価で各パスの ∫_0^T W ∘ dW を近似する。

    区間端の平均 (W_i + W_{i+1})/2 を被積分値に使う。
    """
    dW = np.diff(W, axis=1)
    mid = 0.5 * (W[:, :-1] + W[:, 1:])
    return np.sum(mid * dW, axis=1)


def quadratic_variation(W):
    """各パスの二次変分 Σ (dW)^2 を返す（理論上 T に収束）。"""
    dW = np.diff(W, axis=1)
    return np.sum(dW ** 2, axis=1)


def gbm_exact(S0, mu, sigma, times, W):
    """幾何ブラウン運動の厳密解 S_t = S0 exp[(mu-σ²/2)t + σ W_t]。"""
    drift = (mu - 0.5 * sigma ** 2) * times[None, :]
    return S0 * np.exp(drift + sigma * W)


def ou_exact_paths(x0, kappa, theta, sigma, times, n_paths, seed):
    """OU 過程を厳密な遷移分布で1ステップずつサンプリングする。

    X_{t+dt} = X_t e^{-κdt} + θ(1-e^{-κdt}) + N(0, σ²/(2κ)(1-e^{-2κdt}))。
    Euler 近似と違い、任意の dt で分布が厳密。
    """
    rng = np.random.default_rng(seed)
    n_steps = times.size - 1
    X = np.empty((n_paths, times.size))
    X[:, 0] = x0
    for i in range(n_steps):
        dt = times[i + 1] - times[i]
        e = np.exp(-kappa * dt)
        var = sigma ** 2 / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * dt))
        mean = X[:, i] * e + theta * (1.0 - e)
        X[:, i + 1] = mean + np.sqrt(var) * rng.standard_normal(n_paths)
    return X

# %% [markdown]
# ### 確率積分：左端点と中点で結果が変わる
#
# 多数のパスで $\int W\,dW$（伊藤）と $\int W\circ dW$（Stratonovich）を計算し、
# 平均を理論値と比べます。理論は $E\!\left[\int W\,dW\right]=0$、
# $E\!\left[\int W\circ dW\right]=E[W_T^2]/2=T/2$、差の平均は $T/2$ です。

# %%
T = 1.0
n_steps = 2000
n_paths = 20000
seed = 20260707

times, W = brownian_paths(n_paths, n_steps, T, seed=seed)

ito = ito_integral(W)
strat = stratonovich_integral(W)
qv = quadratic_variation(W)

print(f"E[∫ W dW]       数値 = {ito.mean():+.5f}   理論 = 0")
print(f"E[∫ W ∘ dW]     数値 = {strat.mean():+.5f}   理論 = {T/2:.5f}")
print(f"E[差 Strat-Ito] 数値 = {(strat-ito).mean():+.5f}   理論 = {T/2:.5f}")
print(f"E[二次変分 ΣdW²] 数値 = {qv.mean():.5f}   理論 = {T:.5f}")

# 伊藤積分の恒等式を2通りで確認する。
# 離散で厳密に成り立つのは ∫W dW = (W_T² - ΣdW²)/2 の方（Stratonovich=W_T²/2 の望遠鏡和による）。
# 連続極限の教科書式 (W_T² - T)/2 は ΣdW² を T で置き換えたもので、差は二次変分の揺らぎぶん残る。
exact_rhs = 0.5 * (W[:, -1] ** 2 - qv)
approx_rhs = 0.5 * (W[:, -1] ** 2 - T)
print("離散で厳密 ∫W dW = (W_T²-ΣdW²)/2 のパス最大誤差:",
      float(np.max(np.abs(ito - exact_rhs))))
print("連続極限式 ∫W dW ≈ (W_T²-T)/2   のパス最大誤差:",
      float(np.max(np.abs(ito - approx_rhs))))

# %% [markdown]
# 伊藤積分の平均は 0、Stratonovich は $T/2$、差は二次変分ぶんの $T/2$ に一致します。
# 評価点を左端から中点へずらしただけで結果が変わるのが確率積分の本質です。
#
# ### 伊藤の公式の数値検証：$f(W_t)=W_t^2$
#
# 伊藤の公式は $d(W_t^2)=2W_t\,dW_t + dt$、積分形は $W_T^2 = 2\int_0^T W\,dW + T$
# です。左辺（終端値の二乗）と右辺（確率積分から組み立てた値）をパスごとに
# 突き合わせます。

# %%
lhs = W[:, -1] ** 2
rhs_exact = 2.0 * ito + qv    # 離散で厳密（W_T² = 2∫W dW + ΣdW²）
rhs_approx = 2.0 * ito + T    # 連続極限（ΣdW²→T）
print("離散で厳密  W_T² = 2∫W dW + ΣdW²  のパス最大誤差:",
      float(np.max(np.abs(lhs - rhs_exact))))
print("連続極限    W_T² ≈ 2∫W dW + T     のパス最大誤差:",
      float(np.max(np.abs(lhs - rhs_approx))))
print("両辺の平均:  E[W_T²] =", f"{lhs.mean():.5f}",
      " / E[2∫W dW + T] =", f"{rhs_approx.mean():.5f}", " / 理論 E[W_T²]=T=", T)

# %% [markdown]
# ### GBM：厳密解と数値解の比較
#
# 同じ `seed` を使うと `brownian_paths` の $W$ と `simulate_sde` の駆動ブラウン運動が
# 一致します。これを利用し、同じ 1 本のパス上で厳密解・Euler 解・Milstein 解を
# 比較します。

# %%
S0, mu, sigma = 100.0, 0.05, 0.2
n_steps_g, n_paths_g = 250, 5000
seed_g = 12345

times_g, W_g = brownian_paths(n_paths_g, n_steps_g, T, seed=seed_g)
S_exact = gbm_exact(S0, mu, sigma, times_g, W_g)

# drift(t,x)=mu*x, diffusion(t,x)=sigma*x。x はベクトル化された配列で渡る。
gbm_drift = lambda t, x: mu * x
gbm_diff = lambda t, x: sigma * x
gbm_diff_x = lambda t, x: sigma * np.ones_like(x)  # Milstein 用 ∂b/∂x

_, S_euler = simulate_sde(gbm_drift, gbm_diff, S0, T, n_steps_g, n_paths_g,
                          scheme="euler", seed=seed_g)
_, S_mil = simulate_sde(gbm_drift, gbm_diff, S0, T, n_steps_g, n_paths_g,
                        scheme="milstein", seed=seed_g, diffusion_x=gbm_diff_x)

err_euler = np.mean(np.abs(S_euler[:, -1] - S_exact[:, -1]))
err_mil = np.mean(np.abs(S_mil[:, -1] - S_exact[:, -1]))
print(f"GBM 終端の平均絶対誤差  Euler = {err_euler:.5f}  /  Milstein = {err_mil:.5f}")

# 終端分布の理論モーメント（対数正規）: E[S_T]=S0 e^{μT}
print(f"E[S_T]  厳密解サンプル = {S_exact[:, -1].mean():.4f}"
      f"   理論 S0 e^(μT) = {S0*np.exp(mu*T):.4f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k in range(4):
    ax[0].plot(times_g, S_exact[k], lw=1)
ax[0].set_title("GBM 厳密解のサンプルパス")
ax[0].set_xlabel("t")
ax[0].set_ylabel("S_t")

ax[1].plot(times_g, np.abs(S_euler - S_exact).mean(axis=0), label="Euler")
ax[1].plot(times_g, np.abs(S_mil - S_exact).mean(axis=0), label="Milstein")
ax[1].set_title("厳密解との平均絶対誤差の推移")
ax[1].set_xlabel("t")
ax[1].set_ylabel("mean |数値 - 厳密|")
ax[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# Milstein は拡散項の曲率補正を持つため、GBM のように拡散係数が状態に依存する
# 場合、Euler より終端誤差が小さくなります。
#
# ### OU：厳密解と数値解、そして定常分布
#
# OU 過程を、厳密な遷移分布によるサンプリング（`ou_exact_paths`）と
# Euler-Maruyama（`simulate_sde`）で走らせ、定常分散
# $\sigma^2/(2\kappa)$ と突き合わせます。$\theta=0$ にとるので定常平均は 0 です。

# %%
kappa, theta, sigma_ou = 1.5, 0.0, 0.3
x0_ou = 2.0                      # 定常平均から離れた初期値
T_ou = 6.0
n_steps_ou, n_paths_ou = 1200, 20000
seed_ou = 777

times_ou = np.linspace(0.0, T_ou, n_steps_ou + 1)
X_exact = ou_exact_paths(x0_ou, kappa, theta, sigma_ou, times_ou, n_paths_ou, seed_ou)

ou_drift = lambda t, x: kappa * (theta - x)
ou_diff = lambda t, x: sigma_ou * np.ones_like(x)
_, X_euler = simulate_sde(ou_drift, ou_diff, x0_ou, T_ou, n_steps_ou, n_paths_ou,
                          scheme="euler", seed=seed_ou)

stat_var_theory = sigma_ou ** 2 / (2.0 * kappa)
print(f"OU 定常分散 理論 σ²/(2κ) = {stat_var_theory:.6f}")
print(f"  終端分散  厳密サンプリング = {X_exact[:, -1].var():.6f}")
print(f"  終端分散  Euler           = {X_euler[:, -1].var():.6f}")
print(f"  終端平均  厳密 = {X_exact[:, -1].mean():+.5f}"
      f"  / Euler = {X_euler[:, -1].mean():+.5f}  （理論 0）")

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# 平均の減衰: 理論 X0 e^{-κt} へ収束
ax[0].plot(times_ou, X_exact.mean(axis=0), label="厳密サンプリング 平均")
ax[0].plot(times_ou, x0_ou * np.exp(-kappa * times_ou), "--", label="理論 X0 e^{-κt}")
ax[0].axhline(theta, color="gray", lw=0.8)
ax[0].set_title("OU：平均の減衰（平均回帰）")
ax[0].set_xlabel("t")
ax[0].legend()

# 分散の立ち上がり: 理論 σ²/(2κ)(1-e^{-2κt}) へ収束
var_theory_t = stat_var_theory * (1.0 - np.exp(-2.0 * kappa * times_ou))
ax[1].plot(times_ou, X_exact.var(axis=0), label="厳密サンプリング 分散")
ax[1].plot(times_ou, var_theory_t, "--", label="理論 σ²/(2κ)(1-e^{-2κt})")
ax[1].axhline(stat_var_theory, color="gray", lw=0.8, label="定常分散 σ²/(2κ)")
ax[1].set_title("OU：分散の立ち上がりと定常分散")
ax[1].set_xlabel("t")
ax[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# 平均は初期値 $X_0$ から水準 $\theta=0$ へ指数的に回帰し、分散は定常値
# $\sigma^2/(2\kappa)$ へ立ち上がります。Euler 解も十分細かい刻みでは厳密
# サンプリングとほぼ一致します。

# %% [markdown]
# ## QuantLib検証
#
# 確率積分そのものを直接返す関数は QuantLib にありませんが、OU 過程は
# `QuantLib.OrnsteinUhlenbeckProcess` として実装され、遷移分布の期待値と分散を
# 解析的に返します。ここでは **自作の厳密解（遷移モーメント）を QuantLib の
# 解析解と突き合わせることを「検証」と位置づけます**。両者が一致すれば、
# `ou_exact_paths` が正しい分布を再現していると確認できます。

# %%
import QuantLib as ql

# QuantLib の OU: speed=κ, volatility=σ, x0, level=θ
ql_ou = ql.OrnsteinUhlenbeckProcess(kappa, sigma_ou, x0_ou, theta)

# 各時刻 t について、[0,t] の遷移モーメントを QuantLib と理論式で比較する。
print(f"{'t':>4} {'QL 期待値':>12} {'理論 期待値':>12} {'QL 分散':>12} {'理論 分散':>12}")
for t in [0.5, 1.0, 2.0, 4.0]:
    ql_mean = ql_ou.expectation(0.0, x0_ou, t)
    ql_var = ql_ou.variance(0.0, x0_ou, t)
    th_mean = x0_ou * np.exp(-kappa * t) + theta * (1.0 - np.exp(-kappa * t))
    th_var = stat_var_theory * (1.0 - np.exp(-2.0 * kappa * t))
    print(f"{t:>4} {ql_mean:>12.6f} {th_mean:>12.6f} {ql_var:>12.6f} {th_var:>12.6f}")
    assert abs(ql_mean - th_mean) < 1e-9
    assert abs(ql_var - th_var) < 1e-9

# 定常分散（t→∞）も一致する。
print(f"\n定常分散  QuantLib(t=100) = {ql_ou.variance(0.0, x0_ou, 100.0):.6f}"
      f"   理論 σ²/(2κ) = {stat_var_theory:.6f}")
print("QuantLib と理論式が全時刻で一致（assert 通過）")

# %% [markdown]
# `ou_exact_paths` が使う遷移モーメントは QuantLib の解析解と機械精度で一致します。
# 定常分散も $\sigma^2/(2\kappa)$ に収束します。これで自作の OU 厳密解の
# 分布が正しいことを裏づけられました。

# %% [markdown]
# ## 実データ適用
#
# 確率積分は市場データを直接取りませんが、乱数シードを固定した合成ブラウン運動を
# 「データ」として扱い、教科書の等式を数値で確認します。
#
# 1. 伊藤積分 $\displaystyle\int_0^T W\,dW = \frac{W_T^2 - T}{2}$
# 2. Stratonovich との差 $\displaystyle\int W\circ dW - \int W\,dW = \frac{T}{2}$
#
# 刻みを細かくするほど、平均が理論値へ寄っていく様子を見ます。

# %%
T_d = 1.0
n_paths_d = 40000
print(f"{'n_steps':>8} {'E[∫W dW]':>12} {'E[Strat-Ito]':>14} {'E[ΣdW²]':>10}")
for ns in [50, 200, 1000, 5000]:
    td, Wd = brownian_paths(n_paths_d, ns, T_d, seed=4242)
    it = ito_integral(Wd)
    st = stratonovich_integral(Wd)
    qvd = quadratic_variation(Wd)
    print(f"{ns:>8} {it.mean():>12.5f} {(st-it).mean():>14.5f} {qvd.mean():>10.5f}")

print(f"\n理論値:  E[∫W dW]=0,  E[Strat-Ito]=T/2={T_d/2},  E[ΣdW²]=T={T_d}")

# 離散で厳密な恒等式 ∫W dW = (W_T²-ΣdW²)/2 をパスごとに確認する（機械精度）。
td, Wd = brownian_paths(n_paths_d, 5000, T_d, seed=4242)
it = ito_integral(Wd)
qvd = quadratic_variation(Wd)
assert np.max(np.abs(it - 0.5 * (Wd[:, -1] ** 2 - qvd))) < 1e-9
print("恒等式 ∫W dW = (W_T²-ΣdW²)/2 をパス単位で確認（assert 通過）")

# %% [markdown]
# 二次変分 $\sum(dW)^2$ は刻みによらず $T$ に張り付き、Stratonovich と伊藤の差は
# 常に $T/2$ です。この差は分割を細かくしても消えません。これが「ブラウン運動は
# 有界変動でない」ことの数値的な現れです。

# %% [markdown]
# ## 演習
#
# 1. **$f(W)=W^2$ の伊藤展開の数値検証（別ルート）**：
#    伊藤の公式より $W_T^2 = 2\int_0^T W\,dW + T$ です。本編とは別に、
#    区間分割 $n$ を $n=10,50,200,1000$ と変えて、各 $n$ で
#    $2\int W\,dW + T$ の**平均**と $E[W_T^2]=T$ の差（バイアス）を表にせよ。
#    伊藤積分の平均が 0 に近づく速さも併せて報告し、分割を細かくすると
#    どちらへ収束するかを述べよ。
#
# 2. **OU の平均回帰を初期値を変えて可視化**：
#    $\kappa=1.0,\ \theta=0,\ \sigma=0.5$ を固定し、初期値 $X_0\in\{-3,-1,0,2,4\}$
#    について `ou_exact_paths`（または `simulate_sde`）で平均パスを描け。
#    すべての初期値から平均が $\theta$ へ、分散が $\sigma^2/(2\kappa)$ へ
#    収束することを 1 枚の図で示し、回帰の速さが $\kappa$ でどう決まるかを
#    一言で述べよ。
#
# 解答例は `solutions/S4/sol_0402.py`（ペア notebook）に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/04_stochastic.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | 伊藤積分 | Itô integral | 被積分過程を各小区間の左端点で評価する確率積分。マルチンゲールになる |
# | 伊藤の公式 | Itô's formula | 確率過程 $f(t,X_t)$ の微分公式。$\tfrac12\sigma^2 f_{xx}$ の項が付く |
# | ドリフト | drift | SDE の $dt$ の係数。変化の平均的な向き |
# | 拡散係数 | diffusion coefficient | SDE の $dW$ の係数。揺らぎの大きさ |
# | Ornstein-Uhlenbeck過程 | Ornstein-Uhlenbeck process | 水準へ引き戻される平均回帰過程。定常分布は正規分布 |
