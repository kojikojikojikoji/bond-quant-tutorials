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
# # S4-1 ブラウン運動とシミュレーション
#
# ## 学習目標
#
# - ランダムウォークをスケーリング極限に飛ばすとブラウン運動になる、という筋道を
#   数値実験で確認できる
# - ブラウン運動の性質（独立増分・定常増分・連続だが至る所微分不可能・非有界変動）を、
#   自作したパスと突き合わせて説明できる
# - 2次変分が $t$ に収束する一方、1次変分（全変動）が発散することを数値で示し、
#   そこから $dW^2 = dt$ という置き換えが何を意味するかを言える
# - ベクトル化したパス生成器を自作し、`bondlab.sim.brownian_paths` と一致することを
#   確かめる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# ブラウン運動は、金利・為替・株価といった市場変数を連続時間で動かす標準的な部品です。銀行の金利デスクがスワップションやキャップを値付けするとき、あるいはヘッジファンドが金利のパスをまとめて振るとき、その裏で回っているのはここで自作したようなパス生成器です。したがって「ベクトル化して大量のパスを正しく・速く作れる」ことは、モデル評価とリスク計測の両方で土台になります。デリバティブ評価（S5・S6）は将来キャッシュフローの期待値を数千〜数万本のパス上で平均する作業であり、リスク管理の VaR/ES やシナリオ分析（S3）も、同じエンジンでポートフォリオの損益分布を作ります。パス生成が遅かったり分布が歪んでいたりすれば、そのまま値付けの誤差とリスク量の誤差に化けます。
#
# とりわけ二次変分が $\sum(\Delta W_i)^2 \to t$ に収束し、$(dW_t)^2 = dt$ と置ける事実は、実現ボラティリティ（realized variance）そのものです。オプションのインプライドボラティリティと実現ボラティリティの差を取りにいくボラ取引や、デルタヘッジの損益がこの二次変分で決まることを踏まえると、$dW^2=dt$ は抽象的な約束事ではなく、日々の損益を生む量だと分かります。一方で一次変分（全変動）が発散するという性質は、ブラウン運動の道を「取引コスト無限大の連続リバランス」で追うと破綻することを意味し、離散ヘッジの誤差や取引コストを見積もる感覚につながります。
#
# 就職の観点では、この節は銀行の金利デスククオンツ（S2・S5・S6）とリスク管理クオンツ（S3）の共通の入口です。派手なアルファは生みませんが、ここでの実装がずさんだと後段のモデルがすべて狂うため、フロントでもリスクでも「壊れないシミュレーション基盤」を作れることが評価されます。
# %% [markdown]
# ## 理論
#
# ### 記号
#
# 標準ブラウン運動（standard Brownian motion）を $W = (W_t)_{t \ge 0}$ と書く。
# 区間 $[0, T]$ を $n$ 等分し、刻みを $\Delta t = T/n$、格子点を
# $t_i = i\,\Delta t$（$i = 0, \dots, n$）とする。増分を
# $\Delta W_i = W_{t_{i+1}} - W_{t_i}$ と書く。
#
# ### ランダムウォークのスケーリング極限
#
# 平均 $0$・分散 $1$ の独立同分布な増分 $\xi_1, \xi_2, \dots$（例えば $\pm 1$ を
# 等確率でとるコイン）から、ランダムウォーク $M_k = \sum_{j=1}^{k} \xi_j$ を作る。
# これを時間・空間ともに縮小し、$[0,1]$ 上の連続関数へ折れ線補間したものを
#
# $$
# S^{(n)}_t = \frac{1}{\sqrt{n}}\, M_{\lfloor n t \rfloor}
# \qquad (0 \le t \le 1)
# $$
#
# と定義する。空間を $1/\sqrt{n}$ で縮めるのは、$M_k$ の分散が $k$ に比例して
# 増えるため（$\operatorname{Var}(M_k)=k$）、時間を $n$ 分の $1$ にするなら空間は
# $\sqrt{n}$ 分の $1$ にしないとスケールが釣り合わないからである。ドンスカーの
# 定理（Donsker's theorem, 関数版の中心極限定理）は、$n \to \infty$ で
# $S^{(n)}$ が標準ブラウン運動 $W$ に分布収束することを主張する。$\xi_j$ の分布に
# よらず極限が同じ $W$ になる点が、ブラウン運動が普遍的なモデルである理由になる。
#
# ### ブラウン運動の性質
#
# 標準ブラウン運動 $W$ は次で特徴づけられる。
#
# 1. **始点** $W_0 = 0$。
# 2. **独立増分（independent increments）**：$0 \le t_0 < t_1 < \dots < t_k$ に
#    対し、増分 $W_{t_1}-W_{t_0}, \dots, W_{t_k}-W_{t_{k-1}}$ は互いに独立。
# 3. **定常増分（stationary increments）**：増分の分布は始点によらず刻み幅だけで
#    決まり、$W_t - W_s \sim \mathcal{N}(0,\, t-s)$（$0 \le s < t$）。
# 4. **連続性**：$t \mapsto W_t$ の見本路はほとんど確実に連続。
#
# ここから直ちに $\mathbb{E}[W_t] = 0$、$\operatorname{Var}(W_t) = t$、
# $\mathbb{E}[W_t^2] = t$ が従う。共分散は $s \le t$ のとき、独立増分を使って
#
# $$
# \operatorname{Cov}(W_s, W_t)
# = \operatorname{Cov}(W_s,\, W_s + (W_t - W_s))
# = \operatorname{Var}(W_s) = s = \min(s, t).
# $$
#
# ### 連続だが至る所微分不可能・非有界変動
#
# 見本路は連続だが、ほとんど確実にどの点でも微分できない。直観的には、刻み $\Delta t$
# での増分の大きさが $|\Delta W| \sim \sqrt{\Delta t}$ のオーダーであり、差分商
# $\Delta W / \Delta t \sim 1/\sqrt{\Delta t}$ が $\Delta t \to 0$ で発散するためである。
#
# 同じ理由で見本路は**非有界変動（unbounded variation）**をもつ。1次変分（全変動）を
#
# $$
# V^{(1)}_n = \sum_{i=0}^{n-1} \lvert \Delta W_i \rvert
# $$
#
# とすると、各項の期待値は $\mathbb{E}\lvert \Delta W_i \rvert
# = \sqrt{2\Delta t/\pi}$ なので、
#
# $$
# \mathbb{E}\!\left[V^{(1)}_n\right]
# = n \sqrt{\tfrac{2\,\Delta t}{\pi}}
# = \sqrt{\tfrac{2}{\pi}}\, \frac{T}{\sqrt{\Delta t}}
# \;\xrightarrow[\;\Delta t \to 0\;]{}\; \infty .
# $$
#
# 刻みを細かくするほど 1次変分は $1/\sqrt{\Delta t}$ で増え、発散する。
#
# ### 2次変分が $t$ に収束すること
#
# 一方、増分の**2乗**の和である2次変分（quadratic variation）
#
# $$
# [W]^{(n)}_t = \sum_{i:\, t_{i+1} \le t} (\Delta W_i)^2
# $$
#
# は $n \to \infty$ で確定値 $t$ に収束する（$L^2$ かつほとんど確実に）。$[0,t]$ を
# $m$ 等分した場合を見る。各項は $(\Delta W_i)^2$ で、$\Delta W_i \sim
# \mathcal{N}(0, \Delta t)$ だから
#
# $$
# \mathbb{E}\big[(\Delta W_i)^2\big] = \Delta t, \qquad
# \operatorname{Var}\big[(\Delta W_i)^2\big] = 2\,(\Delta t)^2 .
# $$
#
# 独立性より期待値と分散は
#
# $$
# \mathbb{E}\big[[W]^{(m)}_t\big] = m\,\Delta t = t, \qquad
# \operatorname{Var}\big[[W]^{(m)}_t\big] = 2\,m\,(\Delta t)^2
# = \frac{2 t^2}{m} \;\xrightarrow[\;m \to \infty\;]{}\; 0 .
# $$
#
# 期待値は分割数によらず $t$、ばらつきは $m$ に反比例して消える。よって2次変分は
# ランダムな量でありながら、細分の極限では確定値 $t$ に収束する。
#
# ### 有界変動との対比 ── なぜ $dW^2 = dt$ か
#
# 滑らかな（有界変動の）関数 $f$ では逆の現象が起きる。$\lvert \Delta f_i \rvert
# \le C\,\Delta t$ 程度なので 1次変分は有限に収束し、2次変分は
# $\sum (\Delta f_i)^2 \le C^2 \Delta t \sum \lvert \Delta f_i \rvert \to 0$ で消える。
# 整理すると次の対比になる。
#
# | 量 | 滑らかな $f$ | ブラウン運動 $W$ |
# |---|---|---|
# | 1次変分 $\sum \lvert \Delta \cdot \rvert$ | 有限 | $\infty$（発散） |
# | 2次変分 $\sum (\Delta \cdot)^2$ | $0$ | $t$（有限・確定） |
#
# ブラウン運動では2次変分が消えずに $t$ として残る。増分1つあたりで見ると、
# $(\Delta W_i)^2$ は期待値 $\Delta t$・分散 $2(\Delta t)^2 = o(\Delta t)$ をもつ。
# 分散が $\Delta t$ より高位で消えるので、和をとると揺らぎが打ち消し合い、
# $(\Delta W_i)^2$ は平均的に $\Delta t$ で置き換えてよい。これを微分形で
#
# $$
# (dW)^2 = dt
# $$
#
# と書く。これは等式そのものではなく、「2次変分をとると確定的に $t$ が積み上がる」
# ことの略記である。伊藤の公式で $\tfrac12 f''(W)\,dt$ の項が現れるのはこの効果に
# よる（S4-2 以降で使う）。




# %% [markdown]
# **数値例**（共分散）：$s=0.5$、$t=2$ のとき $\operatorname{Cov}(W_s,W_t)=\min(s,t)=0.5$ です。相関は $\operatorname{Corr}(W_s,W_t)=\min(s,t)/\sqrt{st}=0.5/\sqrt{1}=0.5$ となります。
# %% [markdown]
# **数値例**（2次変分のばらつき）：$t=1$ で $[0,1]$ を $m=200$ 等分すると、$\operatorname{Var}\big[[W]^{(m)}_t\big]=2t^2/m=2/200=0.01$、標準偏差は $\sqrt{0.01}=0.1$ です。$m=50$ なら分散 $0.04$・標準偏差 $0.2$ で、分割を4倍細かくすると標準偏差は半分になります。
# %% [markdown]
# **数値例**（1次変分の発散）：$T=1$、$\Delta t=1/100$ とすると、増分1つの平均は $\mathbb{E}\lvert\Delta W_i\rvert=\sqrt{2\Delta t/\pi}=\sqrt{0.02/\pi}\approx0.0798$、全体では $\mathbb{E}[V^{(1)}_n]=\sqrt{2/\pi}\,\cdot\,T/\sqrt{\Delta t}=\sqrt{2/\pi}\cdot10\approx7.98$ です。刻みを $\Delta t=1/400$ に細かくすると $\sqrt{2/\pi}\cdot20\approx15.96$ と倍増し、発散へ向かいます。
# %% [markdown]
# ## スクラッチ実装
#
# ベクトル化したブラウン運動パス生成器と、2次変分を測る関数を自作する。生成器は
# `bondlab.sim.brownian_paths`（`antithetic=False`）と同じ手順を踏むので、出力が
# 一致することを後で確かめられる。
#
# ### 自作関数の仕様
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `bm_paths(n_paths, n_steps, T, seed)` | パス数, ステップ数, 満期, 乱数シード | `(times, W)`。`times` は `(n_steps+1,)`、`W` は `(n_paths, n_steps+1)` で `W[:,0]=0` | 標準ブラウン運動のパスをベクトル化生成する |
# | `quadratic_variation(W)` | パス配列 `(n_paths, n_steps+1)` | 累積2次変分 `(n_paths, n_steps+1)`、先頭列は $0$ | 各時点までの $\sum (\Delta W)^2$ を返す |
# | `total_variation(W)` | パス配列 `(n_paths, n_steps+1)` | 累積1次変分 `(n_paths, n_steps+1)`、先頭列は $0$ | 各時点までの $\sum \lvert \Delta W \rvert$ を返す |

# %%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import bondlab
from bondlab import sim

print("bondlab version:", bondlab.__version__)


def bm_paths(n_paths, n_steps, T, seed=None):
    """標準ブラウン運動のパスをベクトル化して生成する。

    刻み dt = T / n_steps。増分 ΔW = sqrt(dt) * Z（Z は標準正規）を累積和して
    W を組み立てる。W[:, 0] = 0。
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    incr = np.sqrt(dt) * z
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    times = np.linspace(0.0, T, n_steps + 1)
    return times, W


def quadratic_variation(W):
    """各時点までの累積2次変分 Σ(ΔW)^2 を返す。"""
    incr = np.diff(W, axis=1)
    qv = np.cumsum(incr ** 2, axis=1)
    return np.concatenate([np.zeros((W.shape[0], 1)), qv], axis=1)


def total_variation(W):
    """各時点までの累積1次変分（全変動）Σ|ΔW| を返す。"""
    incr = np.diff(W, axis=1)
    tv = np.cumsum(np.abs(incr), axis=1)
    return np.concatenate([np.zeros((W.shape[0], 1)), tv], axis=1)


# %% [markdown]
# ### bondlab との一致確認
#
# 同じシードなら、自作生成器と `bondlab.sim.brownian_paths` は同一の乱数手順を
# 踏むので、機械精度で一致する。

# %%
times_mine, W_mine = bm_paths(n_paths=500, n_steps=250, T=1.0, seed=42)
times_lib, W_lib = sim.brownian_paths(n_paths=500, n_steps=250, T=1.0, seed=42)

assert np.allclose(times_mine, times_lib)
assert np.allclose(W_mine, W_lib)
assert np.allclose(W_mine[:, 0], 0.0)
print("形状:", W_mine.shape, "  始点は全て0:", np.all(W_mine[:, 0] == 0.0))
print("bondlab との最大差:", np.max(np.abs(W_mine - W_lib)))

# %% [markdown]
# ## QuantLib検証
#
# ブラウン運動には、債券価格のような「別実装で計算し直せる基準値」は存在しない。
# 検証対象は解析的に分かっている分布そのもの ── $W_t \sim \mathcal{N}(0, t)$、
# すなわち $\mathbb{E}[W_t]=0$, $\operatorname{Var}(W_t)=t$, $\mathbb{E}[W_t^2]=t$ ──
# である。したがってここでの「検証」は、解析値との突合として位置づける。
#
# 突合の独立性を高めるため、numpy とは別系統の乱数源として QuantLib の
# ガウス乱数生成器（メルセンヌ・ツイスタ＋逆累積正規変換）で終端値
# $W_T = \sqrt{T}\,Z$ を生成し、標本モーメントとコルモゴロフ–スミルノフ検定で
# 解析分布 $\mathcal{N}(0, T)$ と照合する。

# %%
import QuantLib as ql
from scipy import stats

print("QuantLib version:", ql.__version__)

T = 2.0
n_draw = 200_000

# QuantLib のメルセンヌ・ツイスタ一様乱数を逆累積正規で標準正規へ変換する。
unif = ql.MersenneTwisterUniformRng(20240507)
inv_norm = ql.InverseCumulativeNormal(0.0, 1.0)
z_ql = np.fromiter((inv_norm(unif.next().value()) for _ in range(n_draw)),
                   dtype=float, count=n_draw)
W_T = np.sqrt(T) * z_ql  # W_T ~ N(0, T)

print(f"{'量':10s} {'解析値':>10s} {'標本値(QuantLib乱数)':>22s}")
print(f"{'E[W_T]':10s} {0.0:>10.4f} {W_T.mean():>22.4f}")
print(f"{'Var[W_T]':10s} {T:>10.4f} {W_T.var(ddof=1):>22.4f}")
print(f"{'E[W_T^2]':10s} {T:>10.4f} {np.mean(W_T ** 2):>22.4f}")

# コルモゴロフ–スミルノフ検定：H0 は W_T ~ N(0, T)。p 値が大きければ棄却されない。
ks = stats.kstest(W_T, "norm", args=(0.0, np.sqrt(T)))
print(f"\nKS 統計量 = {ks.statistic:.5f},  p 値 = {ks.pvalue:.3f}")

assert abs(W_T.mean()) < 0.02
assert abs(W_T.var(ddof=1) - T) < 0.05
assert ks.pvalue > 0.01
print("解析分布 N(0, T) と整合（帰無仮説は棄却されない）")

# %% [markdown]
# ### 2次変分が $t$ に収束することの検証
#
# 自作 `quadratic_variation` を使い、多数パスで2次変分の終端値
# $[W]_T$ の平均が $T$ に一致し、ステップ数を増やすとばらつき（標準偏差）が
# $\sqrt{2/m}\,T$ の理論線に沿って消えることを確かめる。

# %%
T = 1.0
n_paths = 4000
rows = []
for m in [50, 200, 800, 3200]:
    _, W = bm_paths(n_paths, m, T, seed=7)
    qv_T = quadratic_variation(W)[:, -1]  # 各パスの [W]_T
    theo_sd = np.sqrt(2.0 / m) * T        # 理論標準偏差
    rows.append((m, qv_T.mean(), qv_T.std(ddof=1), theo_sd))

print(f"{'ステップ数':>8s} {'平均[W]_T':>12s} {'標準偏差':>12s} {'理論SD':>12s}")
for m, mean, sd, theo in rows:
    print(f"{m:>8d} {mean:>12.5f} {sd:>12.5f} {theo:>12.5f}")

# 平均は T、標準偏差は理論線に一致するはず。
for m, mean, sd, theo in rows:
    assert abs(mean - T) < 0.02
    assert abs(sd - theo) < 0.15 * theo
print(f"\n平均は T={T} に一致し、標準偏差は sqrt(2/m)*T の理論線に沿って縮小")

# %% [markdown]
# ## 実データ適用
#
# 合成データ（乱数シード固定）で挙動を見る。市場の実データを持ち込む代わりに、
# 定義から生成したパスでスケーリング極限と2次変分の収束を観察する。

# %% [markdown]
# ### ランダムウォークのスケーリング極限を段階的に見る
#
# 同一の細かい $\pm 1$ 列を土台に、$n$ を段階的に増やしながら
# $S^{(n)}_t = M_{\lfloor n t \rfloor}/\sqrt{n}$ を折れ線で描く。$n$ が小さいうちは
# 角ばった階段だが、$n$ を上げると連続なブラウン運動の見本路へ近づいていく。

# %%
rng = np.random.default_rng(0)
n_max = 4096
xi = rng.choice([-1.0, 1.0], size=n_max)  # 土台となる ±1 列（共有）

ns = [16, 64, 256, 1024, 4096]
fig, axes = plt.subplots(1, len(ns), figsize=(15, 3), sharey=True)
for ax, n in zip(axes, ns):
    steps = xi[:n]
    walk = np.concatenate([[0.0], np.cumsum(steps)])
    t_grid = np.linspace(0.0, 1.0, n + 1)
    ax.plot(t_grid, walk / np.sqrt(n), lw=0.8)
    ax.set_title(f"n = {n}")
    ax.set_xlabel("t")
    ax.axhline(0.0, color="0.7", lw=0.6)
axes[0].set_ylabel(r"$S^{(n)}_t = M_{\lfloor nt\rfloor}/\sqrt{n}$")
fig.suptitle("ランダムウォークのスケーリング極限（共有した±1列を細分）")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 見本路・終端分布・2次変分
#
# 生成したパスの見本、終端 $W_T$ のヒストグラム（解析密度 $\mathcal{N}(0,T)$ を重ねる）、
# 各パスの累積2次変分（対角線 $y=t$ に張り付く）を並べる。

# %%
T = 1.0
times, W = bm_paths(n_paths=3000, n_steps=500, T=T, seed=123)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (1) 見本路
for k in range(8):
    axes[0].plot(times, W[k], lw=0.8)
axes[0].axhline(0.0, color="0.7", lw=0.6)
axes[0].set_title("ブラウン運動の見本路（8本）")
axes[0].set_xlabel("t")
axes[0].set_ylabel(r"$W_t$")

# (2) 終端分布
axes[1].hist(W[:, -1], bins=50, density=True, alpha=0.6, color="steelblue")
xs = np.linspace(-4 * np.sqrt(T), 4 * np.sqrt(T), 200)
axes[1].plot(xs, stats.norm.pdf(xs, 0.0, np.sqrt(T)), "r-", lw=1.5,
             label=r"$\mathcal{N}(0,T)$")
axes[1].set_title(r"終端 $W_T$ の分布")
axes[1].set_xlabel(r"$W_T$")
axes[1].legend()

# (3) 累積2次変分
qv = quadratic_variation(W)
for k in range(30):
    axes[2].plot(times, qv[k], color="steelblue", lw=0.5, alpha=0.4)
axes[2].plot(times, times, "r--", lw=1.5, label=r"$y=t$")
axes[2].set_title("累積2次変分 → t")
axes[2].set_xlabel("t")
axes[2].set_ylabel(r"$[W]_t$")
axes[2].legend()

plt.tight_layout()
plt.show()

print("終端の平均 =", round(float(W[:, -1].mean()), 4),
      " 分散 =", round(float(W[:, -1].var(ddof=1)), 4), " (理論: 0, ", T, ")")

# %% [markdown]
# ## 演習
#
# 1. **2次変分と1次変分の対比**：固定した満期 $T=1$ で、ステップ数 $m$ を
#    $\{25, 100, 400, 1600, 6400\}$ と変えて多数パスを生成し、各パスの終端
#    2次変分 $[W]_T$ と1次変分 $V^{(1)}_T = \sum \lvert \Delta W \rvert$ を測れ。
#    2次変分の平均が $m$ によらず $T$ に収束する一方、1次変分の平均が
#    $\sqrt{2/\pi}\,\sqrt{m}\,\sqrt{T}$ の理論線に沿って発散することを表と両対数
#    プロットで示せ。
# 2. **独立増分・定常性の確認**：ステップ数の大きいパスを生成し、増分
#    $\Delta W_i$ について、(a) ヒストグラムが $\mathcal{N}(0, \Delta t)$ に一致すること、
#    (b) 時間の前半と後半で増分分布が変わらないこと（定常性）、(c) 増分列の標本
#    自己相関がラグ $\ge 1$ でほぼ $0$ になること（独立性）を、図と数値で示せ。
#
# 解答例は `solutions/S4/sol_0401.py`。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/04_stochastic.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | ブラウン運動 | Brownian motion | 独立・定常な正規増分をもつ連続確率過程。$W_t \sim \mathcal{N}(0,t)$ |
# | 独立増分 | independent increments | 重ならない区間の増分が互いに独立という性質 |
# | 2次変分 | quadratic variation | 増分の2乗和の細分極限。ブラウン運動では $[W]_t = t$ |
# | スケーリング極限 | scaling limit | ランダムウォークを時間・空間ともに縮小した $n\to\infty$ の極限。ブラウン運動になる |
