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
# # S5-3 Hull-Whiteモデル① 理論と三項ツリー
#
# ## 学習目標
#
# - 市場カーブに整合する no-arbitrage モデル（no-arbitrage model）が、Vasicek に
#   時間依存の項 $\theta(t)$ を足すことで初期割引カーブを厳密に再現する仕組みを、
#   数式で説明できる
# - $\theta(t)$ が初期カーブを再現する条件を導き、短期金利を $r(t)=x(t)+\alpha(t)$ と
#   分解したときの決定論部分 $\alpha(t)$ の閉形式を書ける
# - Hull-White の三項ツリー（trinomial tree）を2段階法（対称格子→シフト）で
#   スクラッチ実装し、分枝の切替条件（branch switching）を含めて構築できる
# - ツリー上でゼロクーポン債を評価し、初期カーブの完全再現（全テナーで誤差 <1bp）と、
#   条件付き債券価格が `bondlab.models.HullWhite` の解析値・QuantLib の解析値へ
#   収束することを数値で確認できる

# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# Hull-White は、金利デスクの評価系で事実上の標準になっている無裁定モデルです。時間依存の $\theta(t)$ で今日の割引カーブ $P^M(0,\cdot)$ を厳密に通すため、「今日の債券価格すら合わないモデルで明日のデリバティブを語れるか」という批判を最初から回避できます。コーラブル債・バミューダン・スワップション・キャンセラブル・スワップといった経路依存商品を三項ツリーやツリー／PDE で評価する際、原資産の割引債価格が市場と一致していることは前提条件です。ここで組む2段階法（対称格子→シフト）の三項ツリーと、全テナー誤差 <1bp のカーブ再現、`bondlab.models.HullWhite`・QuantLib 解析値への収束確認は、本番のツリー評価エンジンの正しさを担保する検算です。
#
# 収益への効き方は、初期ミスプライスの排除に直結します。Vasicek や CIR は時間一様なパラメータでカーブの水準・傾き・曲率を同時に説明せねばならず、初期カーブに再現誤差が残ります。その誤差はオプション性の無い割引債ですら価格を歪め、マーケットメイクのクォートに紛れ込みます。Hull-White は $\theta(t)$ という関数自由度でカーブを吸収するので、残る $a,\sigma$ がボラティリティ構造だけを担い、値付けとヘッジ（デルタ・ベガ・バケット感応度）が市場整合になります。デスクはこの整合性を土台に、在庫リスクを管理しながらスプレッド収益を積み、相対価値やマクロのポジションを取ります。
#
# モデルバリデーション（S5-5、FRB SR 11-7）にとって、三項ツリーは独立実装によるベンチマーク比較の題材そのものです。検証担当は、開発担当の実装とは別経路（本 notebook のようなスクラッチのツリー、あるいは QuantLib）でカーブ再現誤差と収束を確かめ、分枝切替の条件やタイムステップ依存性まで検証してから承認します。ここが甘いと、ツリーの離散化誤差がバミューダン・スワップションの早期行使価値に効き、ヘッジ比率を通じて日々の損益に漏れ出します。
#

# %% [markdown]
# ## 理論
#
# ### 記号
#
# 短期金利（instantaneous short rate）を $r(t)$、時点 $t$ で満期 $T$ の
# ゼロクーポン債価格を $P(t,T)$ と書く。市場から観測される初期の割引カーブを
# $P^M(0,T)$、対応する市場の瞬間フォワードレートを
#
# $$
# f^M(0,T) = -\frac{\partial}{\partial T}\ln P^M(0,T)
# $$
#
# とする。平均回帰速度（mean reversion speed）$a>0$、ボラティリティ $\sigma>0$。
#
# ### Hull-White モデル（拡張 Vasicek）
#
# Vasicek モデル $dr = a(b-r)\,dt + \sigma\,dW$ は、長期水準 $b$ が定数のため
# 初期カーブを一般には再現できない。理論値と市場値がずれ、そのまま金利
# デリバティブを評価すると、原資産の債券価格の時点で市場と食い違う。
#
# Hull-White モデルは、この定数の水準項を時間の関数に置き換える。
#
# $$
# dr(t) = \bigl(\theta(t) - a\,r(t)\bigr)\,dt + \sigma\,dW(t)
# $$
#
# $\theta(t)$ を1つの自由な曲線として持つことで、初期カーブ $P^M(0,\cdot)$ を厳密に
# 通すよう調整できる。これが no-arbitrage モデルの要点である。モデルが出発点で
# 市場と一致するので、そこから評価するデリバティブに初期ミスプライスが混ざらない。
# アフィン構造（$dr$ のドリフトが $r$ の1次、拡散が定数）は Vasicek と同じなので、
# ゼロクーポン債価格は解析的に書ける。
#
# ### $\theta(t)$ によるカーブフィット
#
# 短期金利を決定論部分と確率部分に分解する。
#
# $$
# r(t) = x(t) + \alpha(t), \qquad dx(t) = -a\,x(t)\,dt + \sigma\,dW(t),\quad x(0)=0
# $$
#
# $x(t)$ は平均 $0$ の Ornstein-Uhlenbeck 過程（$\alpha=0$ の Vasicek）、$\alpha(t)$ は
# 確率項を含まない決定論的な関数で $\alpha(0)=r(0)$。この分解を SDE に代入すると、
# $\alpha$ が満たすべき常微分方程式
#
# $$
# \alpha'(t) = \theta(t) - a\,\alpha(t)
# $$
#
# を得る。あとは $\alpha(t)$ を初期カーブに一致させればよい。$T$-満期債の価格が
# $P^M(0,T)$ に等しくなる条件（アフィンモデルの ZCB 公式に $t=0$ を代入した式）から
# 逆算すると、$\alpha(t)$ は次の閉形式になる。
#
# $$
# \boxed{\;\alpha(t) = f^M(0,t) + \frac{\sigma^2}{2a^2}\bigl(1 - e^{-at}\bigr)^2\;}
# $$
#
# 第1項が市場のフォワードそのもの、第2項がボラティリティによる凸性
# （convexity）の補正である。これを ODE に戻すと $\theta(t)$ も定まる。
#
# $$
# \theta(t) = \frac{\partial f^M(0,t)}{\partial t} + a\,f^M(0,t)
#            + \frac{\sigma^2}{2a}\bigl(1 - e^{-2at}\bigr)
# $$
#
# ### ゼロクーポン債の解析価格
#
# アフィン構造から、時点 $t$・短期金利 $r(t)$ のもとで
#
# $$
# P(t,T) = A(t,T)\,e^{-B(t,T)\,r(t)}, \qquad B(t,T) = \frac{1 - e^{-a(T-t)}}{a}
# $$
#
# $$
# \ln A(t,T) = \ln\frac{P^M(0,T)}{P^M(0,t)} + B(t,T)\,f^M(0,t)
#             - \frac{\sigma^2}{4a}\bigl(1 - e^{-2at}\bigr)\,B(t,T)^2
# $$
#
# $t=0$ とおくと $B(0,0)$ の項が消え、$\ln A(0,T)=\ln P^M(0,T)$、すなわち
# $P(0,T)=P^M(0,T)$。初期カーブが厳密に再現される。この式が `bondlab.models.HullWhite`
# の `zcb` の実装そのものである。以降ではこの解析値を、自作ツリーの答え合わせに使う。
#
# ### 三項ツリーの構築（Hull の2段階法）
#
# 連続時間モデルを離散格子に落とす。二項ツリーではなく三項ツリー（各節点から3方向へ
# 分枝）を使う理由は、平均回帰のドリフトを分枝確率で表現しつつ、格子点の位置を
# 揃えて再結合（recombining）させるためである。分枝先を3つ持てば、平均と分散の
# 2条件を満たしたうえで確率に1つ自由度が残り、ドリフトを格子のずらしではなく
# 確率配分で吸収できる。
#
# **第1段階：対称な $x$ 格子。** まず決定論部分を外した過程
# $dx = -a\,x\,dt + \sigma\,dW$ を、時間刻み $\Delta t$、空間刻み
# $\Delta x = \sigma\sqrt{3\Delta t}$ の対称格子で表す。節点 $(i,j)$ の値は
# $x_{i,j} = j\,\Delta x$（$i$ は時間段、$j$ は空間位置の整数）。1段先の $x$ の条件付き
# 平均は Euler 近似で $x_{i,j}(1 - a\Delta t)$ なので、$\Delta x$ 単位で
#
# $$
# m_j = j\,(1 - a\Delta t)
# $$
#
# を中心とする。中心の分枝先 $k=\mathrm{round}(m_j)$ を選び、残差 $\eta = m_j - k$
# （$|\eta|\le 1/2$）を分枝確率に載せる。分枝先を $k+1,\,k,\,k-1$ とし、平均 $m_j\Delta x$ と
# 分散 $\sigma^2\Delta t = \Delta x^2/3$ を一致させると、確率は次で一意に決まる。
#
# $$
# p_u = \frac{1}{6} + \frac{\eta^2 + \eta}{2}, \quad
# p_m = \frac{2}{3} - \eta^2, \quad
# p_d = \frac{1}{6} + \frac{\eta^2 - \eta}{2}
# $$
#
# **分枝の切替条件。** 平均回帰があるので、$x$ が極端に大きく（小さく）なる状態には
# ほとんど到達しない。格子を無制限に広げず、
#
# $$
# j_{\max} = \bigl\lceil 0.184 / (a\Delta t) \bigr\rceil
# $$
#
# で幅を打ち切る。$j=j_{\max}$ の上端では中心を $k=j-1$ に取る下向き分枝
# （分枝先 $j,\,j-1,\,j-2$）、$j=-j_{\max}$ の下端では $k=j+1$ の上向き分枝
# （分枝先 $-j,\,-(j-1),\,-(j-2)$）へ切り替える。こうすると上端からさらに上、
# 下端からさらに下へは出られず、格子幅が $2j_{\max}+1$ に収まる。閾値 $0.184$ は、
# 切替後も $p_m=2/3-\eta^2>0$（$\eta^2<2/3$）を保つよう選んだものである。確率式自体は
# $\eta=m_j-k$ の定義を通じて上と同一で、変わるのは中心 $k$ の選び方だけである。
#
# **第2段階：時間スライスのシフト。** 第1段階の $x$ 格子はまだ市場カーブを知らない。
# 各時間段 $i$ を決定論シフト $\alpha_i$ だけ持ち上げ、短期金利を
# $r_{i,j} = \alpha_i + j\,\Delta x$ とする。$\alpha_i$ は、ツリーが割引債
# $P^M(0,(i{+}1)\Delta t)$ を再現するよう前向き帰納で決める。節点 $(i,j)$ に到達する
# Arrow-Debreu 価格（そこで1を払う証券の現在価値）を $Q_{i,j}$ とおくと、
# $Q_{0,0}=1$ から
#
# $$
# \alpha_i = \frac{1}{\Delta t}\Bigl[\ln\!\Bigl(\textstyle\sum_j Q_{i,j}\,e^{-j\Delta x\,\Delta t}\Bigr)
#            - \ln P^M(0,(i{+}1)\Delta t)\Bigr],
# $$
#
# $$
# Q_{i+1,j'} = \sum_j Q_{i,j}\,q_{j\to j'}\,e^{-(\alpha_i + j\Delta x)\Delta t}
# $$
#
# を交互に回す。$q_{j\to j'}$ は第1段階の分枝確率。この $\alpha_i$ は連続時間の
# $\alpha(t)$ の離散版であり、$\Delta t\to 0$ で
# $\alpha(t)=f^M(0,t)+\frac{\sigma^2}{2a^2}(1-e^{-at})^2$ に収束する。これで
# 「$\theta(t)$ によるカーブフィット」を格子上で実現したことになる。

# %% [markdown]
# #### 数値例
#
# **数値例**：$a=0.05$、残存 $T-t=7$ 年のとき $B(t,T)=\dfrac{1-e^{-0.05\times 7}}{0.05}=5.906$ です。
#
# **数値例**：$a=0.05,\ \sigma=0.01,\ t=10$ 年のとき、凸性補正 $\dfrac{\sigma^2}{2a^2}\bigl(1-e^{-at}\bigr)^2=3.10\times 10^{-3}$（約 31bp）だけ $\alpha(t)$ が市場フォワード $f^M(0,t)$ から持ち上がります。
#
# **数値例**：残差 $\eta=0.1$ の節点では分枝確率が $p_u=\tfrac16+\tfrac{\eta^2+\eta}{2}=0.222$、$p_m=\tfrac23-\eta^2=0.657$、$p_d=\tfrac16+\tfrac{\eta^2-\eta}{2}=0.122$（和 $=1$）となります。
#

# %% [markdown]
# ## スクラッチ実装
#
# 上の2段階法を1つのクラス `HullWhiteTree` にまとめる。前向き帰納で
# $\alpha_i$ と $Q_{i,j}$ を作り（`build`）、後ろ向き帰納でゼロクーポン債を評価する
# （`zcb`）。
#
# ### 使用する自作クラス・関数
#
# | メソッド | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `HullWhiteTree(a, sigma, curve, T, n_steps)` | パラメータ, 割引カーブ, 満期, 段数 | インスタンス | 格子の寸法を決め `build` を呼ぶ |
# | `_branch(j)` | 空間位置 $j$ | $(k, p_u, p_m, p_d)$ | 中心分枝先と3確率。上下端で切替 |
# | `build()` | なし | なし（属性を設定） | 第2段階の前向き帰納で $\alpha_i,\,Q_{i,j}$ を作る |
# | `short_rate(i, j)` | 時間段, 空間位置 | $r_{i,j}=\alpha_i+j\Delta x$ | 節点の短期金利 |
# | `reproduction_error()` | なし | 各グリッド満期のゼロレート誤差(bp) | 初期カーブ再現の検証 |
# | `zcb(mat_step, from_step)` | 満期段, 評価段 | 評価段の節点→価格 dict | 後ろ向き帰納で ZCB を評価 |
# | `reachable()` | なし | 各段の到達 $j$ 範囲 | 作図用に格子形状を返す |

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.font_manager as _fm
for _f in ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Noto Sans JP", "TakaoPGothic", "IPAPGothic"]:
    if any(_f == _n.name for _n in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
import bondlab
from bondlab.curve import bootstrap_par
from bondlab.models import HullWhite

print("bondlab version:", bondlab.__version__)

np.random.seed(0)  # 本編に乱数は使わないが、再現性のため固定する


class HullWhiteTree:
    """Hull-White 三項ツリー（Hull の2段階法）。

    第1段階で対称な x 格子（dx=-a x dt+σ dW）を組み、第2段階で各時間段を
    α_i だけシフトして初期割引カーブに整合させる。短期金利は r_{i,j}=α_i+jΔx。
    """

    def __init__(self, a, sigma, curve, T, n_steps):
        self.a, self.sigma, self.curve = a, sigma, curve
        self.T, self.N = T, n_steps
        self.dt = T / n_steps
        self.dx = sigma * np.sqrt(3.0 * self.dt)
        # 平均回帰で到達しない極端な状態を打ち切る幅。
        self.jmax = int(np.ceil(0.184 / (a * self.dt)))
        self.build()

    def _branch(self, j):
        """節点 j の中心分枝先 k と3確率。上下端で分枝を切り替える。"""
        m = j * (1.0 - self.a * self.dt)          # Δx 単位の条件付き平均
        if j >= self.jmax:
            k = j - 1                              # 上端：下向き分枝
        elif j <= -self.jmax:
            k = j + 1                              # 下端：上向き分枝
        else:
            k = int(np.round(m))                   # 通常：最近傍
        eta = m - k                                # 残差（|η|≤1/2、切替時はより大）
        pu = 1.0 / 6.0 + (eta ** 2 + eta) / 2.0
        pm = 2.0 / 3.0 - eta ** 2
        pd = 1.0 / 6.0 + (eta ** 2 - eta) / 2.0
        return k, pu, pm, pd

    def build(self):
        """前向き帰納で α_i と Arrow-Debreu 価格 Q_{i,j} を構築する。"""
        Q = {0: 1.0}                               # Q_{0,0}=1
        self.alpha = np.empty(self.N)
        self.Q_levels = [dict(Q)]                  # 各段の Q（作図・検証用）
        for i in range(self.N):
            # α_i：この段で割引債 P^M(0,(i+1)Δt) を再現するよう決める。
            s = sum(q * np.exp(-j * self.dx * self.dt) for j, q in Q.items())
            pm_disc = self.curve.discount((i + 1) * self.dt)
            self.alpha[i] = (np.log(s) - np.log(pm_disc)) / self.dt
            # Q を1段先へ伝播する。
            Q_next = {}
            for j, q in Q.items():
                k, pu, pm, pd = self._branch(j)
                disc = np.exp(-(self.alpha[i] + j * self.dx) * self.dt)
                for jj, p in ((k + 1, pu), (k, pm), (k - 1, pd)):
                    Q_next[jj] = Q_next.get(jj, 0.0) + q * p * disc
            Q = Q_next
            self.Q_levels.append(dict(Q))

    def short_rate(self, i, j):
        """節点 (i,j) の短期金利 r_{i,j}=α_i+jΔx。

        α_i は割引段 i=0..N-1 で定義される。最終段 i=N は割引に使わないが、
        作図のため直前2点から線形外挿した値を用いる。
        """
        a_i = self.alpha[i] if i < self.N else 2 * self.alpha[-1] - self.alpha[-2]
        return a_i + j * self.dx

    def reproduction_error(self):
        """各グリッド満期 kΔt について、ツリーが再現するゼロレートと
        市場ゼロレートの差(bp)を返す。P^M(0,kΔt)=Σ_j Q_{k,j} を使う。"""
        errs = []
        for k in range(1, self.N + 1):
            t = k * self.dt
            p_tree = sum(self.Q_levels[k].values())
            z_tree = -np.log(p_tree) / t
            errs.append((z_tree - self.curve.zero_rate(t)) * 1e4)
        return np.array(errs)

    def zcb(self, mat_step, from_step=0):
        """満期段 mat_step のゼロクーポン債を後ろ向き帰納で評価する。

        from_step の各節点における条件付き価格 P(t,T) の dict を返す。
        """
        V = {j: 1.0 for j in self.Q_levels[mat_step].keys()}   # 満期で額面1
        for i in range(mat_step - 1, from_step - 1, -1):
            V_next = {}
            for j in self.Q_levels[i].keys():
                k, pu, pm, pd = self._branch(j)
                disc = np.exp(-self.short_rate(i, j) * self.dt)
                V_next[j] = disc * (pu * V.get(k + 1, 0.0)
                                    + pm * V.get(k, 0.0)
                                    + pd * V.get(k - 1, 0.0))
            V = V_next
        return V

    def reachable(self):
        """各時間段で到達可能な j の (最小, 最大) を返す（作図用）。"""
        return [(min(d), max(d)) for d in self.Q_levels]


# %% [markdown]
# ### カーブの用意
#
# 実データ適用と同じ割引カーブを、ここでも使う。合成パー利回り
# `data/samples/synthetic_ust_par_curve.csv` を年次グリッド（1〜30 年）へ補間し、
# `bondlab.curve.bootstrap_par` で割引係数を剥ぎ取る。補間は割引係数のログ線形だと
# フォワードが年ごとに折れて $f^M$ が段状になり、離散シフト $\alpha_i$ の収束が
# 見えにくい。ここではゼロレート線形補間（`linear_zero`）を選び、フォワードを
# 滑らかにしておく。

# %%
par_df = pd.read_csv("data/samples/synthetic_ust_par_curve.csv")
annual = np.arange(1, 31)
par_annual = np.interp(annual, par_df["tenor"].values, par_df["par_yield"].values)
curve = bootstrap_par(annual, par_annual, frequency=1, interp="linear_zero")

print("補間後の年次パー利回り（一部）:")
for t in (1, 2, 5, 10, 30):
    print(f"  {t:2d}年: パー {par_annual[t-1]*100:.3f}%  "
          f"割引係数 {curve.discount(t):.5f}  ゼロ {curve.zero_rate(t)*100:.3f}%")

# %% [markdown]
# ### 2段階法をステップごとに図解
#
# 分枝の切替が見えるよう、あえて粗い小さなツリー（$a=0.30$, 満期2年, 6段）を組む。
# このとき $j_{\max}=2$ となり、上下端で分枝が切り替わる様子が観察できる。
# 左が第1段階の対称な $x$ 格子（0 を中心に上下対称）、右が第2段階でスライスごとに
# $\alpha_i$ だけ持ち上げた短期金利 $r$ 格子である。

# %%
demo = HullWhiteTree(a=0.30, sigma=0.01, curve=curve, T=2.0, n_steps=6)
print(f"デモツリー: Δt={demo.dt:.3f}, Δx={demo.dx:.5f}, j_max={demo.jmax}")


def draw_lattice(ax, tree, shift, title):
    """格子の節点と分枝（3本の枝）を描く。shift=True で r 格子（α_i シフト）。"""
    reach = tree.reachable()
    for i in range(tree.N + 1):
        lo, hi = reach[i]
        for j in range(lo, hi + 1):
            y = tree.short_rate(i, j) if shift else j * tree.dx
            ax.plot(i * tree.dt, y, "o", color="steelblue", ms=5, zorder=3)
            if i < tree.N:
                k, *_ = tree._branch(j)
                for jj in (k + 1, k, k - 1):
                    y2 = tree.short_rate(i + 1, jj) if shift else jj * tree.dx
                    ax.plot([i * tree.dt, (i + 1) * tree.dt], [y, y2],
                            "-", color="0.7", lw=0.7, zorder=1)
    ax.set_xlabel("時間 (年)")
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
draw_lattice(axes[0], demo, shift=False, title="第1段階：対称な x 格子")
axes[0].set_ylabel(r"$x = j\,\Delta x$")
axes[0].axhline(0.0, color="r", ls="--", lw=0.8)
draw_lattice(axes[1], demo, shift=True, title=r"第2段階：$\alpha_i$ シフト後の $r$ 格子")
axes[1].set_ylabel(r"短期金利 $r_{i,j}$")
axes[1].plot(np.arange(1, demo.N + 1) * demo.dt, demo.alpha,
             "r.-", lw=1.0, ms=6, label=r"シフト $\alpha_i$")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# 左図では上端 $j=+2$・下端 $j=-2$ で枝が内側へ折り返し、格子幅が
# $2j_{\max}+1=5$ 本で頭打ちになる（分枝の切替）。右図は各スライスを $\alpha_i$
# だけ持ち上げたもので、赤線がシフト量そのもの。これがカーブフィットを担う。

# %% [markdown]
# ### ツリー上でゼロクーポン債を評価する
#
# 本番の細かいツリー（10年満期・100段）を組み、$P(0,T)$ を後ろ向き帰納で評価して、
# `bondlab.models.HullWhite` の解析値と突き合わせる。無裁定モデルなので、初期カーブの
# 割引係数 $P^M(0,T)$ とも一致するはずである。

# %%
a, sigma = 0.10, 0.01
tree = HullWhiteTree(a, sigma, curve, T=10.0, n_steps=100)
hw = HullWhite(a, sigma, curve)

p_tree = tree.zcb(mat_step=tree.N, from_step=0)[0]   # 根の値 = P(0,10)
p_analytic = hw.zcb(0.0, 10.0)                        # 解析値（=P^M(0,10)）
p_market = curve.discount(10.0)                       # 市場割引係数

print(f"ツリー評価 P(0,10)   = {p_tree:.8f}")
print(f"解析値     P(0,10)   = {p_analytic:.8f}")
print(f"市場割引係数 P^M(0,10) = {p_market:.8f}")
print(f"ツリー vs 解析の差   = {abs(p_tree - p_analytic):.2e}")
assert abs(p_tree - p_analytic) < 1e-6

# %% [markdown]
# ## QuantLib検証
#
# 独立の実装として QuantLib の `ql.HullWhite` を使う。まず割引カーブを渡して
# 同じ $a,\sigma$ の Hull-White を作り、その解析式 `discountBond(t, T, r)` を
# ベンチマークにする。カーブのフォワード表現の差から `bondlab` とは $\sim10^{-4}$
# ずれるが、これは既知の範囲である。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

ref = ql.Date(1, 1, 2026)
dates = [ref] + [ref + ql.Period(int(t), ql.Years) for t in range(1, 11)]
dfs = [1.0] + [float(curve.discount(t)) for t in range(1, 11)]
ts = ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, ql.Actual365Fixed()))
qhw = ql.HullWhite(ts, a, sigma)

# 参考として短期金利 r=0.03、t=1→T=10 の解析 ZCB を3実装で並べる。
r_ref = 0.03
print(f"P(1,10; r={r_ref}) 解析比較:")
print(f"  bondlab  = {hw.zcb(1.0, 10.0, r_ref):.8f}")
print(f"  QuantLib = {qhw.discountBond(1.0, 10.0, r_ref):.8f}")
assert abs(hw.zcb(1.0, 10.0, r_ref) - qhw.discountBond(1.0, 10.0, r_ref)) < 5e-4

# %% [markdown]
# ### 検証1：初期カーブの完全再現（全テナーで誤差 <1bp）
#
# 無裁定モデルの核心は、初期カーブを厳密に通すことである。ツリーが各グリッド満期
# $k\Delta t$ で再現するゼロレートと市場ゼロレートの差を全段について測る。

# %%
err_bp = tree.reproduction_error()
print(f"グリッド満期数           = {err_bp.size}")
print(f"ゼロレート誤差の最大(bp)  = {np.max(np.abs(err_bp)):.6f}")
print(f"ゼロレート誤差の平均(bp)  = {np.mean(np.abs(err_bp)):.6f}")
assert np.max(np.abs(err_bp)) < 1.0   # 全テナーで 1bp 未満
print("→ 全テナーで誤差 <1bp。初期カーブを完全再現している。")

fig, ax = plt.subplots(figsize=(8, 3.4))
grid_t = np.arange(1, tree.N + 1) * tree.dt
ax.plot(grid_t, err_bp, "-", color="steelblue")
ax.axhline(0.0, color="0.6", lw=0.8)
ax.set_xlabel("満期 (年)")
ax.set_ylabel("ゼロレート誤差 (bp)")
ax.set_title("ツリーによる初期カーブ再現誤差（全テナーで <1bp）")
ax.set_ylim(-1.0, 1.0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 検証2：条件付き ZCB 価格の収束
#
# グリッド満期のゼロクーポン債は前向き帰納の構成上ほぼ厳密に一致するため、ツリーの
# 離散化誤差はそこには現れない。誤差が見えるのは、時点 $t_0>0$・状態 $r$ を条件とした
# 内部の債券価格 $P(t_0,T;r)$ である。段数 $N$ を増やしたとき、ツリーの後ろ向き帰納が
# 与える中央節点の価格が、`bondlab` と QuantLib の解析値へ収束するかを見る。

# %%
t0, T_mat = 1.0, 10.0
rows = []
for N in (20, 40, 80, 160):
    tr = HullWhiteTree(a, sigma, curve, T=T_mat, n_steps=N)
    i0 = int(round(t0 / tr.dt))
    v_tree = tr.zcb(mat_step=N, from_step=i0)[0]     # 中央節点 j=0
    r0 = tr.short_rate(i0, 0)                          # その節点の短期金利
    v_bond = hw.zcb(t0, T_mat, r0)
    v_ql = qhw.discountBond(t0, T_mat, r0)
    rows.append((N, r0, v_tree, v_bond, v_ql, abs(v_tree - v_bond)))

print(f"{'N':>5}{'r0':>9}{'ツリー':>12}{'bondlab':>12}{'QuantLib':>12}{'|差|':>11}")
for N, r0, vt, vb, vq, d in rows:
    print(f"{N:>5}{r0:>9.5f}{vt:>12.6f}{vb:>12.6f}{vq:>12.6f}{d:>11.2e}")

# 段数を倍にするごとに誤差がおよそ半減する（O(Δt) 収束）。
diffs = np.array([r[5] for r in rows])
ratios = diffs[:-1] / diffs[1:]
print("\n誤差比（隣接 N、2 に近ければ O(Δt)）:", np.round(ratios, 2))
assert diffs[-1] < diffs[0]              # 収束している
assert diffs[-1] < 2e-3                  # 最細段で解析値に十分近い
print("→ ツリー価格は解析値へ O(Δt) で収束する。")

# %% [markdown]
# ### 条件付き価格を状態 $r$ の関数として突合
#
# 同じ内部時点 $t_0$ で、到達する全節点の短期金利 $r_{i_0,j}$ とツリー価格の対を並べ、
# 解析価格 $P(t_0,T;r)$（`bondlab` と QuantLib）の曲線に重ねる。ツリー・2つの解析実装が
# 状態の全域で重なることを確認する。

# %%
tr = HullWhiteTree(a, sigma, curve, T=T_mat, n_steps=160)
i0 = int(round(t0 / tr.dt))
V = tr.zcb(mat_step=tr.N, from_step=i0)
js = np.array(sorted(V.keys()))
r_nodes = np.array([tr.short_rate(i0, j) for j in js])
v_nodes = np.array([V[j] for j in js])
v_bond = np.array([hw.zcb(t0, T_mat, r) for r in r_nodes])
v_ql = np.array([qhw.discountBond(t0, T_mat, r) for r in r_nodes])

print(f"状態全域での最大 |ツリー-bondlab| = {np.max(np.abs(v_nodes - v_bond)):.2e}")
print(f"状態全域での最大 |bondlab-QuantLib| = {np.max(np.abs(v_bond - v_ql)):.2e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(r_nodes, v_bond, "-", color="crimson", lw=1.6, label="解析 (bondlab)")
axes[0].plot(r_nodes, v_ql, "--", color="green", lw=1.2, label="解析 (QuantLib)")
axes[0].plot(r_nodes, v_nodes, "o", color="steelblue", ms=4, label="ツリー")
axes[0].set_xlabel(r"短期金利 $r_{t_0}$")
axes[0].set_ylabel(r"$P(t_0,T;r)$")
axes[0].set_title(r"条件付き ZCB 価格（$t_0=1$, $T=10$）")
axes[0].legend()

axes[1].plot(r_nodes, (v_nodes - v_bond) * 1e4, "-", color="steelblue")
axes[1].axhline(0.0, color="0.6", lw=0.8)
axes[1].set_xlabel(r"短期金利 $r_{t_0}$")
axes[1].set_ylabel("ツリー - 解析 (価格 bp)")
axes[1].set_title("残差（N=160）")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 実データ適用
#
# 上で構築した実カーブ（合成パー利回りを年次補間したもの）を用いて、ツリーの2つの
# 性質を数値で確かめる。ひとつはステップ数を変えたときの価格収束、もうひとつは
# 平均回帰速度 $a$ が格子形状に与える影響である。

# %% [markdown]
# ### ステップ数と価格収束
#
# 内部時点 $t_0=1.5$、満期 $T=10$、中央節点の条件付き ZCB 価格を段数 $N$ に対して
# 並べ、解析値からの誤差が $\Delta t$ に比例して縮むことを両対数で見る。$t_0$ は
# カーブの整数年ノードを避けて取る（整数年では補間フォワード $f^M$ が折れ、解析値
# 側の $f^M$ 評価に段差が入るため）。

# %%
t0b, Tb = 1.5, 10.0
Ns = np.array([20, 40, 80, 160])
err_conv = []
for N in Ns:
    tr = HullWhiteTree(a, sigma, curve, T=Tb, n_steps=N)
    i0 = int(round(t0b / tr.dt))
    v = tr.zcb(mat_step=N, from_step=i0)[0]
    r0 = tr.short_rate(i0, 0)
    err_conv.append(abs(v - hw.zcb(t0b, Tb, r0)))
err_conv = np.array(err_conv)
dts = Tb / Ns

print(f"{'N':>5}{'Δt':>9}{'誤差':>12}")
for N, dt, e in zip(Ns, dts, err_conv):
    print(f"{N:>5}{dt:>9.4f}{e:>12.2e}")

# 傾き（両対数）が 1 に近ければ O(Δt)。
slope = np.polyfit(np.log(dts), np.log(err_conv), 1)[0]
print(f"\n両対数の傾き = {slope:.2f}（1 に近いほど O(Δt)）")
assert 0.7 < slope < 1.3

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(dts, err_conv, "o-", color="steelblue", label="ツリー誤差")
ax.loglog(dts, err_conv[0] * (dts / dts[0]), "r--", lw=1.0, label=r"傾き1の基準線 $O(\Delta t)$")
ax.set_xlabel(r"時間刻み $\Delta t$")
ax.set_ylabel("解析値からの誤差")
ax.set_title("条件付き ZCB 価格の収束（実カーブ）")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 平均回帰速度 $a$ が格子形状に与える影響
#
# $a$ が大きいほど平均回帰が強く、極端な状態へ到達しにくいので打ち切り幅
# $j_{\max}=\lceil 0.184/(a\Delta t)\rceil$ が小さく、格子は細く（低く）なる。逆に $a$ が
# 小さいと格子は広がる。段数・満期を固定して、$a$ を変えたときの到達領域の輪郭を
# 重ねて描き、あわせて初期カーブ再現がどの $a$ でも保たれることを確認する。

# %%
a_list = [0.03, 0.05, 0.10, 0.20, 0.40]
N_shape = 60
colors = plt.cm.viridis(np.linspace(0, 0.85, len(a_list)))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
print(f"{'a':>6}{'j_max':>7}{'格子幅':>8}{'再現誤差最大(bp)':>18}")
for a_i, col in zip(a_list, colors):
    tr = HullWhiteTree(a_i, sigma, curve, T=10.0, n_steps=N_shape)
    reach = tr.reachable()
    times = np.arange(tr.N + 1) * tr.dt
    hi = np.array([tr.short_rate(i, reach[i][1]) for i in range(tr.N + 1)])
    lo = np.array([tr.short_rate(i, reach[i][0]) for i in range(tr.N + 1)])
    axes[0].plot(times, hi, "-", color=col, lw=1.3, label=f"a={a_i}")
    axes[0].plot(times, lo, "-", color=col, lw=1.3)
    max_err = np.max(np.abs(tr.reproduction_error()))
    width = 2 * tr.jmax + 1
    print(f"{a_i:>6.2f}{tr.jmax:>7d}{width:>8d}{max_err:>18.4f}")
    assert max_err < 1.0        # どの a でも初期カーブは <1bp で再現

axes[0].set_xlabel("時間 (年)")
axes[0].set_ylabel(r"短期金利 $r$（到達領域の上下端）")
axes[0].set_title("a が大きいほど格子は細い")
axes[0].legend(fontsize=8)

axes[1].bar([str(a_i) for a_i in a_list],
            [int(np.ceil(0.184 / (a_i * (10.0 / N_shape)))) for a_i in a_list],
            color=colors)
axes[1].set_xlabel("平均回帰速度 a")
axes[1].set_ylabel(r"打ち切り幅 $j_{\max}$")
axes[1].set_title(r"$j_{\max}=\lceil 0.184/(a\Delta t)\rceil$")
plt.tight_layout()
plt.show()

print("\n→ a が小さいほど格子は広がるが、初期カーブ再現はどの a でも <1bp で保たれる。")

# %% [markdown]
# ## 演習
#
# 1. **ツリーステップ数と価格収束。** 実カーブ・$a=0.1$, $\sigma=0.01$ のもとで、内部時点
#    $t_0=2.5$（整数年ノードを避ける）・満期 $T=10$ の中央節点の条件付き ZCB 価格を、
#    段数 $N\in\{20,40,80,160\}$ について求めよ。`bondlab` の解析値 `hw.zcb(t0,T,r0)`
#    （$r_0$ はその節点の短期金利）からの誤差を両対数でプロットし、収束の次数が
#    $O(\Delta t)$ になることを傾きの推定で示せ。あわせて、グリッド満期
#    $T=N\Delta t$ のゼロクーポン債では誤差がほぼ現れない（無裁定構成の帰結である）
#    ことを確認せよ。
# 2. **$a$ を変えてツリー形状と初期カーブ再現精度を評価。** $\sigma=0.01$ 固定、満期
#    10年・段数 80 のツリーを $a\in\{0.02,0.05,0.10,0.30\}$ で組み、(i) 打ち切り幅
#    $j_{\max}$ と格子幅、(ii) 各 $a$ での初期カーブ再現誤差の最大値(bp) を表にまとめよ。
#    $a$ を変えても再現誤差が <1bp に保たれることを確認し、$a$ が格子の広さと分枝の
#    切替の起こりやすさをどう変えるかを一言で述べよ。
#
# 解答例は `solutions/S5/sol_0503.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/05_rate_models.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | [no-arbitrageモデル](../../glossary/05_rate_models.md#no-arbitrage-model) | no-arbitrage model | 初期の市場カーブを厳密に再現するよう構成した金利モデル |
# | [三項ツリー](../../glossary/05_rate_models.md#trinomial-tree) | trinomial tree | 各節点から3方向へ分枝する再結合格子。平均回帰を確率で表す |
# | [平均回帰速度](../../glossary/05_rate_models.md#mean-reversion-speed) | mean reversion speed | 短期金利が長期水準へ引き戻される速さ $a$。大きいほど強い |
# | [θ(t)シフト](../../glossary/05_rate_models.md#theta-t-shift) | theta(t) shift | 時間依存の水準項。各時間段のシフト $\alpha_i$ で初期カーブに整合させる |
