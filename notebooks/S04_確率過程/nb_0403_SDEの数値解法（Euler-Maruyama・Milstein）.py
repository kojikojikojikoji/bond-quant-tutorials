# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # S4-3 SDEの数値解法（Euler-Maruyama / Milstein）
#
# ## 学習目標
#
# - 確率微分方程式（SDE）を時間離散化して数値解を得る手順を、自分の手で実装できる
# - Euler-Maruyama 法と Milstein 法の更新式の違いと、追加項の由来（伊藤・テイラー展開）を説明できる
# - 強収束と弱収束の定義の違いを述べ、価格評価には弱収束で足りる理由を説明できる
# - 幾何ブラウン運動（GBM）の厳密解に対し、同一のブラウン運動を使って強収束次数を log-log 回帰で実測できる（EM≈0.5, Milstein≈1.0）
# - CIR過程で原点近傍の負値問題を再現し、full truncation 法で修正できる
#
# S4-1・S4-2 でブラウン運動と伊藤の補題を扱った。本節はその上に立ち、連続時間の
# SDE を計算機で解くための離散化を扱う。以降の S5（短期金利モデル）や S8（MC-OAS）
# は、ここで作るソルバーをそのまま土台にする。

# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# 多くのモデルは厳密解を持たないため、SDE を離散化して数値的に解くことになります。ここで作る Euler-Maruyama や Milstein のソルバーは、銀行の金利デスクが短期金利モデル（S5）でエキゾチックを値付けするとき、あるいは住宅ローン担保証券の MC-OAS（S8）でパスごとの割引を回すときの、文字どおりの計算エンジンです。日次のリスク再評価や大量の商品のプライシングでは、このソルバーを何万回も回します。
#
# 強収束と弱収束の区別は、実務での費用対効果の判断に効きます。個々のパスの精度（強収束）は EM で $\sqrt{\Delta t}$、Milstein で $\Delta t$ の速さでしか上がりませんが、価格評価で必要なのは期待値の精度（弱収束）で、多くの場合ここは EM でも $\Delta t$ の一次で足ります。つまり「値付けだけなら EM で刻みを粗くしても済むが、パス依存の経路精度が要るなら Milstein で拡散項の補正を入れる」という取捨選択が、計算コストと精度のバランスを決めます。夜間バッチの計算時間や、電子取引の応答遅延（S11-2）に直結するため、どの手法をどの刻みで使うかは実装者の判断が問われます。
#
# CIR 過程で原点近傍に負値が出る問題と full truncation による修正は、金利や分散が負にならないという商品性を数値解が壊さないための処置です。モデルが理論上は非負でも、離散化の副作用で負の金利・負の分散が出ると、割引係数やボラティリティが破綻し、価格やリスク量が無意味になります。この種の「離散化に固有の落とし穴」を潰せることは、フロントのモデル実装でもモデルバリデーション（S5-5）でも、成果物が信用されるための必須条件です。
#

# %% [markdown]
# ## 理論
#
# ### 対象とする SDE と離散化の考え方
#
# 1次元の伊藤 SDE
#
# $$ dX_t = a(t, X_t)\,dt + b(t, X_t)\,dW_t, \qquad X_0 = x_0 $$
#
# を考えます。$a$ をドリフト、$b$ を拡散係数、$W_t$ を標準ブラウン運動と呼びます。
# 解 $X_t$ は連続だが至る所で微分不可能な確率過程なので、常微分方程式のような
# 決定的な数値解法をそのまま使えません。そこで時間区間 $[0, T]$ を
# $0 = t_0 < t_1 < \dots < t_N = T$、刻み $\Delta t = T/N$ に分け、各小区間で
# ブラウン運動の増分
#
# $$ \Delta W_i = W_{t_{i+1}} - W_{t_i} \sim \mathcal{N}(0, \Delta t) $$
#
# を乱数で生成しながら、$X$ を1ステップずつ前進させます。$\Delta W_i$ の分散が
# $\Delta t$（標準偏差が $\sqrt{\Delta t}$）である点が、決定的な数値積分と本質的に
# 異なります。増分は $\sqrt{\Delta t}$ のオーダーで、$\Delta t$ より大きいことが
# 離散化誤差の解析を左右します。
#
# ### Euler-Maruyama 法
#
# 最も素直な離散化は、ドリフト項とテイラー展開の1次、拡散項を増分そのもので
# 置き換えるものです。**Euler-Maruyama 法（Euler-Maruyama method）** の更新式は
#
# $$ X_{i+1} = X_i + a(t_i, X_i)\,\Delta t + b(t_i, X_i)\,\Delta W_i $$
#
# です。決定的な Euler 法に確率項 $b\,\Delta W_i$ を足しただけの形で、実装が容易で
# あらゆる SDE に使えます。ただし後述するように、パスを個別に追う精度（強収束）は
# $\sqrt{\Delta t}$ の速さでしか上がりません。
#
# ### Milstein 法
#
# Euler-Maruyama の精度が頭打ちになる原因は、拡散係数 $b(t, X_t)$ 自身も小区間内で
# $X$ とともに揺らぐのに、その揺らぎを区間左端の値 $b(t_i, X_i)$ で固定してしまう点に
# あります。伊藤の補題で $b(t, X_t)$ の変化を1段展開すると、$b\,\partial_x b$ に
# $\int \Delta W\,dW = \tfrac12(\Delta W_i^2 - \Delta t)$ が掛かる補正項が現れます。
# これを加えたのが **Milstein 法（Milstein method）** です。
#
# $$ X_{i+1} = X_i + a\,\Delta t + b\,\Delta W_i
#    + \tfrac12\, b\,\frac{\partial b}{\partial x}\,\big(\Delta W_i^2 - \Delta t\big) $$
#
# 追加項は $\partial b/\partial x$、すなわち拡散係数の状態微分を要求します。拡散が
# 状態に依存しない加法ノイズ（$b$ が定数）のときは $\partial_x b = 0$ となり、
# Milstein は Euler-Maruyama に一致します。補正項が効くのは GBM や CIR のように
# $b$ が $X$ に依存する乗法ノイズの場合です。
#
# ### 強収束と弱収束
#
# 数値解 $X^{\Delta t}_N$ が真の解 $X_T$ に近づく速さは、二つの意味で測ります。
#
# - **強収束（strong convergence）**：パス単位の誤差の期待値
#   $\; \mathbb{E}\big[\,|X^{\Delta t}_N - X_T|\,\big] \le C\,\Delta t^{\gamma}$
#   を満たす最大の $\gamma$ を強収束次数と呼びます。真の解と数値解を**同一の
#   ブラウン運動**で走らせ、軌道そのものがどれだけ一致するかを見ます。
# - **弱収束（weak convergence）**：滑らかな関数 $g$ の期待値の誤差
#   $\; \big|\,\mathbb{E}[g(X^{\Delta t}_N)] - \mathbb{E}[g(X_T)]\,\big| \le C\,\Delta t^{\beta}$
#   を満たす最大の $\beta$ を弱収束次数と呼びます。分布（モーメント）が合えばよく、
#   個々のパスの一致は問いません。
#
# 次数は次のようにまとまります。
#
# | 手法 | 強収束次数 $\gamma$ | 弱収束次数 $\beta$ |
# |---|---|---|
# | Euler-Maruyama | 0.5 | 1.0 |
# | Milstein | 1.0 | 1.0 |
#
# ### どちらで足りるか：価格評価は弱収束
#
# デリバティブの価格は、割引ペイオフの期待値 $\mathbb{E}[\,e^{-rT} g(X_T)\,]$ です。
# ここで必要なのは期待値が正しいこと、つまり**弱収束**であって、個々のパスが真の
# 軌道に一致すること（強収束）ではありません。弱収束次数は Euler-Maruyama も
# Milstein も 1.0 なので、素朴なモンテカルロで価格を出すだけなら Euler-Maruyama で
# 十分です。強収束が問題になるのは、経路依存の量を厳密解と突き合わせて検証する、
# マルチレベル・モンテカルロで階層間のパスを結合する、感度をパス単位で評価する、
# といった軌道そのものの一致が効く場面です。本節では検証のために強収束を測ります。
#
# ### 時間刻みの選び方と離散化バイアス
#
# 数値解を使った推定量には二種類の誤差が乗ります。刻み $\Delta t$ に由来する
# **離散化バイアス（discretization bias）** と、有限のパス数に由来する統計誤差
# （標準誤差 $\propto 1/\sqrt{M}$、$M$ はパス数）です。弱収束次数 1 の手法では
# バイアスは $O(\Delta t)$ で、刻みを半分にすればバイアスも半減します。刻みを
# 細かくするほどバイアスは減りますが、ステップ数に比例して計算量が増えます。実務
# では「バイアスを統計誤差と同程度まで下げたら、あとはパス数を増やす」という配分が
# 目安になります。バイアスだけを消したいときは、刻み違いの推定量を外挿する
# リチャードソン外挿も使えます。

# %% [markdown]
# **数値例**（強収束次数の効き方）：刻みを半分 $\Delta t\to\Delta t/2$ にすると、強収束誤差 $C\,\Delta t^{\gamma}$ は Euler-Maruyama（$\gamma=0.5$）で $2^{-0.5}\approx0.71$ 倍、Milstein（$\gamma=1.0$）で $2^{-1}=0.5$ 倍になります。刻みを $1/4$ にすれば Euler は $0.5$ 倍、Milstein は $0.25$ 倍で、Milstein の方が速く精度が上がります。

# %% [markdown]
# ## スクラッチ実装
#
# 汎用の SDE ソルバーを自分で実装します。ドリフト $a(t,x)$ と拡散 $b(t,x)$ を関数
# として注入し、スキームを切り替えられる形にします。強収束の実測では厳密解と数値解を
# **同一のブラウン運動**で比較する必要があるため、ソルバーはブラウン増分
# `dW` を外部から受け取る設計にします（`bondlab.sim.simulate_sde` は内部で乱数を
# 生成するため、厳密解と増分を揃えるには自前ループが要る点に注意します）。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `make_increments(n_paths, n_steps, dt, seed)` | パス数, ステップ数, 刻み, シード | `dW` 配列 `(n_paths, n_steps)` | ブラウン増分を生成（`simulate_sde` と同じ生成規則） |
# | `solve_sde(drift, diffusion, x0, T, dW, scheme, diffusion_x)` | ドリフト, 拡散, 初期値, 満期, 増分, スキーム, `∂b/∂x` | `(times, X)` | 与えた増分で SDE を1ステップずつ前進 |
# | `strong_error(x_num, x_exact)` | 数値解終端, 厳密解終端 | 平均絶対誤差 | 強収束の誤差指標 $\mathbb{E}[|X_N-X_T|]$ |

# %%
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.font_manager as _fm
for _f in ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Noto Sans JP", "TakaoPGothic", "IPAPGothic"]:
    if any(_f == _n.name for _n in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
import bondlab
from bondlab.sim import simulate_sde

print("bondlab version:", bondlab.__version__)


def make_increments(n_paths, n_steps, dt, seed):
    """ブラウン増分 dW を生成する。

    bondlab.sim.simulate_sde と同じ生成規則（default_rng(seed) から
    standard_normal((n_paths, n_steps)) を引き、sqrt(dt) を掛ける）にそろえる。
    これにより同一シードで自作ソルバーと simulate_sde が一致する。
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    return np.sqrt(dt) * z


def solve_sde(drift, diffusion, x0, T, dW, scheme="euler", diffusion_x=None):
    """与えたブラウン増分 dW で 1次元 SDE を数値積分する自作ソルバー。

    dW を外部から受け取るので、同じ dW を厳密解にも使えば強収束を測れる。
    drift(t, x) / diffusion(t, x) はベクトル化された x を受ける前提。
    """
    if scheme == "milstein" and diffusion_x is None:
        raise ValueError("milstein には diffusion_x（∂b/∂x）が必要です")
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    for i in range(n_steps):
        t = times[i]
        x = X[:, i]
        a = drift(t, x)
        b = diffusion(t, x)
        step = a * dt + b * dW[:, i]
        if scheme == "milstein":
            bx = diffusion_x(t, x)
            step = step + 0.5 * b * bx * (dW[:, i] ** 2 - dt)
        X[:, i + 1] = x + step
    return times, X


def strong_error(x_num, x_exact):
    """強収束の誤差指標：終端でのパス単位平均絶対誤差 E[|X_N - X_T|]。"""
    return float(np.mean(np.abs(x_num - x_exact)))


# %% [markdown]
# ### bondlab.sim.simulate_sde との一致確認
#
# 自作 `solve_sde` が正しいかを、`bondlab.sim.simulate_sde` と突き合わせて確かめます。
# `simulate_sde` は内部で乱数を生成しますが、その生成規則は `make_increments` に写して
# あるので、同一シードなら増分が一致し、両者の解は機械精度で一致するはずです。GBM
# $dX = \mu X\,dt + \sigma X\,dW$ を題材にします（$b = \sigma x$ なので
# $\partial_x b = \sigma$）。

# %%
mu, sigma, x0 = 0.05, 0.2, 1.0
T, n_steps, n_paths, seed = 1.0, 200, 4000, 12345

drift = lambda t, x: mu * x
diffusion = lambda t, x: sigma * x
diffusion_x = lambda t, x: sigma * np.ones_like(x)

dt = T / n_steps
dW = make_increments(n_paths, n_steps, dt, seed)

# 自作ソルバー
t_mine, X_euler = solve_sde(drift, diffusion, x0, T, dW, scheme="euler")
_, X_milstein = solve_sde(drift, diffusion, x0, T, dW, scheme="milstein",
                          diffusion_x=diffusion_x)

# bondlab 実装（内部で同一規則の乱数を生成）
t_lib, X_euler_lib = simulate_sde(drift, diffusion, x0, T, n_steps, n_paths,
                                  scheme="euler", seed=seed)
_, X_milstein_lib = simulate_sde(drift, diffusion, x0, T, n_steps, n_paths,
                                 scheme="milstein", seed=seed,
                                 diffusion_x=diffusion_x)

err_euler = np.max(np.abs(X_euler - X_euler_lib))
err_milstein = np.max(np.abs(X_milstein - X_milstein_lib))
print(f"Euler-Maruyama  自作 vs bondlab の最大差: {err_euler:.2e}")
print(f"Milstein        自作 vs bondlab の最大差: {err_milstein:.2e}")
assert err_euler < 1e-12
assert err_milstein < 1e-12
print("一致を確認しました（機械精度）")

# %% [markdown]
# ## QuantLib検証
#
# 本節の検証には QuantLib の商品評価ではなく、**GBM の厳密解**を基準（オラクル）として
# 使います。GBM は解析解を持つ数少ない SDE で、離散化スキームの収束次数を厳密に
# 測れるため、数値解法の答え合わせに最適だからです。QuantLib の日付・評価機能は本節の
# 収束次数の検証には不要なので、ここでは GBM 厳密解との突き合わせをもって「検証」と
# 位置づけます（QuantLib による商品評価の検証は価格評価を扱う S6 以降で行います）。
#
# ### GBM の厳密解と同一ブラウン運動での比較
#
# GBM $dX = \mu X\,dt + \sigma X\,dW$ は、伊藤の補題から厳密解
#
# $$ X_T = X_0 \exp\!\Big(\big(\mu - \tfrac12\sigma^2\big) T + \sigma W_T\Big) $$
#
# を持ちます。$W_T = \sum_i \Delta W_i$ は数値解に使った増分の総和に等しいので、同じ
# `dW` を厳密解にも渡せば、真の軌道と数値軌道を同一のブラウン運動の上で比較できます。
# まず終端でのパス単位 RMSE を測ります。

# %%
def gbm_exact_terminal(x0, mu, sigma, T, dW):
    """同一の増分 dW から GBM の厳密な終端値 X_T を計算する。"""
    W_T = dW.sum(axis=1)  # W_T = Σ ΔW_i
    return x0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * W_T)


x_exact = gbm_exact_terminal(x0, mu, sigma, T, dW)
rmse_euler = np.sqrt(np.mean((X_euler[:, -1] - x_exact) ** 2))
rmse_milstein = np.sqrt(np.mean((X_milstein[:, -1] - x_exact) ** 2))
print(f"終端パス単位 RMSE（N={n_steps}）")
print(f"  Euler-Maruyama : {rmse_euler:.6e}")
print(f"  Milstein       : {rmse_milstein:.6e}")
# 同じ刻みなら Milstein の方がパス単位で厳密解に近い。
assert rmse_milstein < rmse_euler
print("同一刻みで Milstein < Euler（パス単位で厳密解に近い）を確認")

# %% [markdown]
# ### 強収束次数の log-log 回帰による実測
#
# 刻み $\Delta t$ を段階的に細かくし、各刻みで強収束誤差
# $\mathbb{E}[|X^{\Delta t}_N - X_T|]$ を測ります。理論では誤差 $\approx C\,\Delta t^{\gamma}$
# なので、両対数プロット上で傾き $\gamma$ の直線に乗るはずです。回帰で傾きを取り出し、
# Euler-Maruyama が $\approx 0.5$、Milstein が $\approx 1.0$ になることを確かめます。
# 各刻みで独立にブラウン増分を生成し、その総和から厳密解を作るので、刻みごとに厳密解と
# 数値解は同一のブラウン運動を共有します。

# %%
step_list = [8, 16, 32, 64, 128, 256, 512]
M = 20000  # パス数（統計誤差を抑える）

rows = []
for N in step_list:
    dtN = T / N
    dWN = make_increments(M, N, dtN, seed=2024 + N)
    xT = gbm_exact_terminal(x0, mu, sigma, T, dWN)
    _, Xe = solve_sde(drift, diffusion, x0, T, dWN, scheme="euler")
    _, Xm = solve_sde(drift, diffusion, x0, T, dWN, scheme="milstein",
                      diffusion_x=diffusion_x)
    rows.append((N, dtN, strong_error(Xe[:, -1], xT), strong_error(Xm[:, -1], xT)))

N_arr = np.array([r[0] for r in rows])
dt_arr = np.array([r[1] for r in rows])
err_e = np.array([r[2] for r in rows])
err_m = np.array([r[3] for r in rows])

# log-log 回帰で傾き（＝強収束次数）を推定
slope_e = np.polyfit(np.log(dt_arr), np.log(err_e), 1)[0]
slope_m = np.polyfit(np.log(dt_arr), np.log(err_m), 1)[0]
print(f"{'N':>5} {'dt':>10} {'EM強誤差':>14} {'Milstein強誤差':>16}")
for N, d, ee, em in rows:
    print(f"{N:>5} {d:>10.5f} {ee:>14.6e} {em:>16.6e}")
print(f"\n強収束次数（log-log 回帰の傾き）")
print(f"  Euler-Maruyama : {slope_e:.3f}  （理論 0.5）")
print(f"  Milstein       : {slope_m:.3f}  （理論 1.0）")
assert 0.4 < slope_e < 0.65
assert 0.85 < slope_m < 1.15

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(dt_arr, err_e, "o-", color="#1f77b4", label=f"Euler-Maruyama（傾き {slope_e:.2f}）")
ax.loglog(dt_arr, err_m, "s-", color="#d62728", label=f"Milstein（傾き {slope_m:.2f}）")
# 傾き 0.5・1.0 の参照線
ax.loglog(dt_arr, err_e[0] * (dt_arr / dt_arr[0]) ** 0.5, "--",
          color="#1f77b4", alpha=0.4, label="傾き 0.5 参照")
ax.loglog(dt_arr, err_m[0] * (dt_arr / dt_arr[0]) ** 1.0, "--",
          color="#d62728", alpha=0.4, label="傾き 1.0 参照")
ax.set_xlabel("時間刻み Δt")
ax.set_ylabel("強収束誤差  E[|X_N − X_T|]")
ax.set_title("GBM 厳密解に対する強収束次数の実測")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：両対数上で Euler-Maruyama は傾き約 0.5、Milstein は傾き約 1.0 の直線に
# 乗り、理論の強収束次数と一致します。同じ刻みでも Milstein の誤差が桁違いに小さいのは、
# 拡散係数の状態微分を補正項に取り込み、乗法ノイズの揺らぎまで捉えているためです。
# 一方で価格評価に必要な弱収束次数は両者とも 1.0 なので、期待値だけを求めるなら実装が
# 単純な Euler-Maruyama で足りる、という使い分けにつながります。

# %% [markdown]
# ## 実データ適用
#
# ### CIR 過程と原点近傍の負値問題
#
# 金利モデルで多用される **CIR 過程（Cox-Ingersoll-Ross）**
#
# $$ dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t $$
#
# を離散化します。$\kappa$ は平均回帰の速さ、$\theta$ は長期水準、$\sigma$ は
# ボラティリティです。連続時間では、フェラー条件 $2\kappa\theta \ge \sigma^2$ を満たせば
# $r_t$ は厳密に正に留まります。ところが Euler-Maruyama で離散化すると、$r_i$ が小さい
# ときに拡散項 $\sigma\sqrt{r_i}\,\Delta W_i$ が負の大きな値を取り、$r_{i+1}$ が負に
# なることがあります。すると次のステップで $\sqrt{r_{i+1}}$ が計算不能（NaN）になり、
# シミュレーションが破綻します。フェラー条件を破るパラメータで、この問題を再現します。
# ネットワークには接続せず、乱数シードを固定します。

# %% [markdown]
# **数値例**（フェラー条件の判定）：$\kappa=0.5$、$\theta=0.02$、$\sigma=0.30$ では $2\kappa\theta=2\cdot0.5\cdot0.02=0.02$ に対し $\sigma^2=0.09$ なので、$2\kappa\theta<\sigma^2$ で条件を破ります。したがって離散化した $r_t$ は原点近傍で負値に落ちやすく、full truncation が必要になります。

# %%
# フェラー条件 2κθ = σ² を破るパラメータ（2κθ < σ²）で負値を誘発する。
kappa, theta, sigma_r, r0 = 0.5, 0.02, 0.30, 0.02
feller = 2 * kappa * theta
print(f"2κθ = {feller:.4f},  σ² = {sigma_r**2:.4f}  →  フェラー条件 "
      f"{'成立' if feller >= sigma_r**2 else '不成立（負値が出やすい）'}")

T_cir, N_cir, M_cir = 1.0, 250, 5000
dt_cir = T_cir / N_cir
dW_cir = make_increments(M_cir, N_cir, dt_cir, seed=777)


def simulate_cir(kappa, theta, sigma, r0, T, dW, scheme="full_trunc"):
    """CIR 過程を Euler-Maruyama で離散化する。

    scheme="plain"      : 素朴に √r を使う。r が負に落ちると √(負) = NaN となり、
                          以降のステップに NaN が伝播してパスが破綻する。
    scheme="full_trunc" : full truncation 法。拡散に √max(r,0) を使い、負に振れても
                          平方根が実数のまま計算できる。負領域では拡散が 0 になり、
                          ドリフト κ(θ−r) が正へ引き戻す。
    戻り値は (パス配列, 各パスで1度でも負値に達したか, 各パスに NaN が出たか)。
    """
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    went_negative = np.zeros(n_paths, dtype=bool)
    for i in range(n_steps):
        ri = r[:, i]
        with np.errstate(invalid="ignore"):  # √(負) の警告を抑制（NaN の発生自体は観察対象）
            root = np.sqrt(np.maximum(ri, 0.0)) if scheme == "full_trunc" else np.sqrt(ri)
        r_next = ri + kappa * (theta - ri) * dt + sigma * root * dW[:, i]
        went_negative |= r_next < 0.0  # 最初の負値到達で立つ（以降 NaN でも記録は残る）
        r[:, i + 1] = r_next
    has_nan = np.isnan(r).any(axis=1)
    return r, went_negative, has_nan


# 素朴版（√r）：負値に達したパスは次のステップで NaN になり破綻する。
r_plain, neg_plain, nan_plain = simulate_cir(kappa, theta, sigma_r, r0, T_cir, dW_cir, scheme="plain")
print(f"素朴な √r")
print(f"  負値に達したパスの割合   : {neg_plain.mean()*100:.2f}%")
print(f"  NaN で破綻したパスの割合 : {nan_plain.mean()*100:.2f}%（負値到達がそのまま破綻に直結）")
# 負値に落ちたパスは NaN になって使えなくなる。
assert nan_plain.mean() > 0.0

# %% [markdown]
# ### full truncation 法による修正
#
# **full truncation（full truncation）** は、拡散項の平方根に $\sqrt{\max(r, 0)}$ を
# 使う修正です。状態 $r$ が負に振れても平方根が実数のまま計算でき、負の領域では拡散が
# 0 になってドリフト $\kappa(\theta - r)$ が正へ引き戻します。素朴版と full truncation は
# 「初めて負値に落ちるまで」は完全に同一の軌道をたどる（$r \ge 0$ の間は
# $\sqrt{\max(r,0)} = \sqrt{r}$）ため、**負値到達率そのものは両者で変わりません**。
# 違いは負値に落ちた後で、素朴版は NaN で破綻するのに対し、full truncation は平方根の
# 破綻を防いでシミュレーションを完走させ、浅い負値に留めて平均回帰で正へ戻します。

# %%
r_ft, neg_ft, nan_ft = simulate_cir(kappa, theta, sigma_r, r0, T_cir, dW_cir, scheme="full_trunc")
print(f"full truncation")
print(f"  負値に達したパスの割合   : {neg_ft.mean()*100:.2f}%（素朴版と同率）")
print(f"  NaN で破綻したパスの割合 : {nan_ft.mean()*100:.2f}%")
print(f"  r の最小値               : {r_ft.min():.5f}（浅い負値に留まる）")
# 負値到達率は素朴版と一致し、full truncation は NaN を1つも出さずに完走する。
assert neg_ft.mean() == neg_plain.mean()
assert not nan_ft.any()
print("full truncation は破綻せず完走（NaN ゼロ）を確認")

# %%
fig, ax = plt.subplots(figsize=(9, 5.5))
times_cir = np.linspace(0.0, T_cir, N_cir + 1)
dipped = np.where(neg_ft)[0][:15]      # 負値に落ちたパス（赤で強調）
positive = np.where(~neg_ft)[0][:35]   # 一度も負にならなかったパス
for k in positive:
    ax.plot(times_cir, r_ft[k], lw=0.6, alpha=0.4, color="#7f7f7f")
for k in dipped:
    ax.plot(times_cir, r_ft[k], lw=0.9, alpha=0.8, color="#d62728")
ax.axhline(0.0, color="black", lw=1.0, ls="--", label="原点 r=0")
ax.axhline(theta, color="green", lw=1.0, ls=":", label=f"長期水準 θ={theta}")
ax.set_xlabel("時間 t（年）")
ax.set_ylabel("短期金利 r")
ax.set_title(f"full truncation で完走した CIR パス（赤=負値に到達, 破綻率0%）")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# **解釈**：フェラー条件を破ると、離散化した CIR は原点近傍で頻繁に負値を出します。
# 負値到達率は素朴版と full truncation で変わりません（負に落ちるまでは同じ軌道だから
# です）。効くのはその後で、素朴な $\sqrt{r}$ は負値の瞬間に NaN となりシミュレーション
# 全体が使えなくなるのに対し、full truncation は $\sqrt{\max(r,0)}$ で破綻を防ぎ、浅い
# 負値に留めて平均回帰で正へ戻し、最後まで完走させます。金利の非負性を近似的に保ちつつ
# 計算を破綻させない軽い手当てなので、短期金利モデル（S5）でも標準的に使われます。
# なお厳密な非負性が必要なら、対数変換や非心カイ二乗分布からの厳密サンプリングへ進みます。

# %% [markdown]
# ## 演習
#
# 1. **EM と Milstein の強収束次数の実測**：GBM 以外に、平均回帰する
#    Ornstein-Uhlenbeck 過程 $dX = \kappa(\theta - X)\,dt + \sigma\,dW$（加法ノイズ）で、
#    Euler-Maruyama と Milstein の強収束次数を log-log 回帰で実測せよ。加法ノイズでは
#    $\partial_x b = 0$ なので両手法が一致し、ともに強収束次数が高くなる（この設定では
#    Euler-Maruyama も 1.0 に達する）ことを確認し、なぜ GBM のときと違うのかを述べよ。
# 2. **CIR の負値発生率の評価**：時間刻み $\Delta t$（ステップ数 $N$）とボラティリティ
#    $\sigma$ を変えて、負値到達率と、素朴な $\sqrt{r}$ の破綻（NaN）率・full truncation の
#    破綻率を表にまとめよ。刻みを細かくすると負値到達率がどう変わるか、full truncation が
#    どの領域で効く（破綻を防ぐ）かを論じよ。
#
# 解答例は `solutions/S4/sol_0403.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/04_stochastic.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | [Euler-Maruyama法](../../glossary/04_stochastic.md#euler-maruyama-method) | Euler-Maruyama method | SDE を $X_{i+1}=X_i+a\,\Delta t+b\,\Delta W_i$ で前進する最も基本的な離散化。強収束次数 0.5 |
# | [Milstein法](../../glossary/04_stochastic.md#milstein-method) | Milstein method | Euler に $\tfrac12 b\,\partial_x b(\Delta W^2-\Delta t)$ を加えた離散化。強収束次数 1.0 |
# | [強収束](../../glossary/04_stochastic.md#strong-convergence) | strong convergence | 同一ブラウン運動でのパス単位誤差 $\mathbb{E}[|X_N-X_T|]$ が $O(\Delta t^{\gamma})$ で減る速さ |
# | [弱収束](../../glossary/04_stochastic.md#weak-convergence) | weak convergence | 期待値の誤差 $|\mathbb{E}[g(X_N)]-\mathbb{E}[g(X_T)]|$ が $O(\Delta t^{\beta})$ で減る速さ。価格評価はこれで足りる |
# | [full truncation](../../glossary/04_stochastic.md#full-truncation) | full truncation | CIR 等の拡散項で $\sqrt{\max(r,0)}$ を使い、離散化で負に振れても破綻させない修正 |
