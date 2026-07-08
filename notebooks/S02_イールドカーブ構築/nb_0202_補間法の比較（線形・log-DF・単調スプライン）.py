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
# # S2-2 補間法の比較（線形/log-DF/単調スプライン）
#
# ## 学習目標
#
# - 割引カーブの補間で「何を補間するか」（割引係数・その対数・ゼロレート・
#   フォワード）の選択が、フォワードレートの形にどう効くかを説明できる
# - ゼロ線形・log-DF 線形・自然三次スプライン・単調凸（Hagan-West）の4種を、
#   共通インターフェースでスクラッチ実装できる
# - 各補間がフォワードに与える影響を数式で導き、「滑らかさ」と「局所性」の
#   トレードオフとして整理できる
# - 単調凸法が考案された動機（log-DF のフォワード段差とスプラインの振動・
#   非局所性を同時に避ける）を説明できる
# - 自作実装を `bondlab.curve.DiscountCurve`・QuantLib・scipy と突合し、
#   一致（または想定どおりの差）を数値で確認できる
# - フォワードの「ギザギザ度」を2階差分ノルムで定量化し、補間法を推奨できる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# 補間法の選択は、地味に見えて収益とヘッジの精度に直結します。カーブ構築で気配が立つのは限られた年限だけで、その間を埋める補間が実質的にフォワードレートの形を決めます。フォワードは先渡し評価やスワップの各期のフィキシング推計、キャリーの期間配分に効くため、補間で生じるフォワードの段差や振動は、そのまま価格とヘッジ比率のノイズになります。ここを担うのはフロントのカーブ構築クオンツで、成果はマーケットメイクとRV双方のデスクが使います。
#
# 相対価値ファンドにとっては特に、補間の滑らかさが裁定判断の質を左右します。銘柄の rich/cheap は「フィット済みカーブからの乖離が平均回帰する」ことに賭ける取引ですが、補間が暴れると、本当は割安でも見かけ上のフォワードのギザギザに埋もれて割高・割安の判定を誤ります。逆に過度に平滑化すれば、局所的なゆがみを消してしまい取れるはずの機会を見落とします。単調凸法のように、log-DFの段差もスプラインの振動も避ける手法が好まれるのは、この判定を安定させるためです。イールドカーブ取引（スティープナー・バタフライ）でもフォワードの形が損益の源泉なので、補間の癖を把握しておく必要があります。
#
# 具体的には、オフザラン銘柄の値付けや、先スタートのスワップ・先渡し債の評価で、どの年限を何で補間したかが理論値を動かします。フォワードの「ギザギザ度」を2階差分で定量化して補間法を選ぶ習慣は、どのデスクでも通用する実務の型です。補間の選択を意識しないと、同じ入力から複数の理論値が出てしまい、値付けの一貫性も、乖離の計測も担保できません。
# %% [markdown]
# ## 理論
#
# ### 補間の対象量とフォワードレート
#
# 割引カーブは、少数の年限（ノード）で割引係数 $DF(t_i)$ が分かっている状態から
# 出発する。ノード間の値を埋めるのが補間である。ここで本質的なのは「$DF$ そのものを
# 補間するのか、その対数 $\ln DF$、ゼロレート $z(t)$、あるいは瞬間フォワード $f(t)$ を
# 補間するのか」という**対象量の選択**であって、これがフォワードの形を決める。
#
# 連続複利ゼロレートと割引係数の関係は
#
# $$
# DF(t) = e^{-z(t)\,t}, \qquad z(t) = -\frac{\ln DF(t)}{t}
# $$
#
# である。瞬間フォワードレート $f(t)$ は、割引係数の対数の傾きとして定義する。
#
# $$
# f(t) = -\frac{d}{dt}\ln DF(t) = \frac{d}{dt}\bigl(z(t)\,t\bigr)
# $$
#
# 区間 $[t_1, t_2]$ の（連続複利）フォワードは、その区間の瞬間フォワードの平均に等しい。
#
# $$
# F(t_1, t_2) = \frac{\ln DF(t_1) - \ln DF(t_2)}{t_2 - t_1}
#             = \frac{1}{t_2 - t_1}\int_{t_1}^{t_2} f(u)\,du
# $$
#
# フォワードは $\ln DF$ の**微分**なので、補間関数の滑らかさが1階分だけ落ちて現れる。
# ここが補間法の良し悪しが最も鋭く出る場所である。価格（＝割引係数）は滑らかに
# 見えても、その微分であるフォワードが段差だらけ・振動だらけになることは珍しくない。
#
# ### 各補間がフォワードに与える影響
#
# 記号を固定する。ノードを $0 = t_0 < t_1 < \dots < t_n$、各ノードの
# $y_i = \ln DF(t_i)$、区間幅 $\Delta_i = t_i - t_{i-1}$ とする。
#
# **(1) ゼロ線形（linear on zero）**：ゼロレート $z(t)$ を区間ごとに線形補間する。
# 区間 $[t_{i-1}, t_i]$ で傾きを $s_i = (z_i - z_{i-1})/\Delta_i$ とすると
# $z(t) = z_{i-1} + s_i (t - t_{i-1})$。フォワードは
#
# $$
# f(t) = \frac{d}{dt}\bigl(z(t)\,t\bigr) = z(t) + t\,s_i
# $$
#
# となり、区間内は $t$ の1次関数だが、ノードで傾き $s_i$ が切り替わるため
# フォワードは**ノードで不連続に跳ぶ**（鋸歯状）。
#
# **(2) log-DF 線形（log-linear on DF）**：$\ln DF$ を区間ごとに線形補間する。
# $\ln DF(t)$ は区間内で1次関数なので、その微分であるフォワードは
#
# $$
# f(t) = -\frac{d}{dt}\ln DF(t) = -\frac{y_i - y_{i-1}}{\Delta_i}
#      = \frac{\ln DF(t_{i-1}) - \ln DF(t_i)}{\Delta_i}
# \quad (\text{区間内で一定})
# $$
#
# すなわちフォワードは**区間ごとに一定の階段関数**になる。実装が単純で正の割引係数を
# 保ち、区間フォワードを厳密に再現する。反面、ノードごとに段差が立つ。これが
# `bondlab.curve.DiscountCurve` の既定（`log_linear`）である。
#
# **(3) 自然三次スプライン（natural cubic spline）**：$\ln DF$（またはゼロ）を、
# ノードで2階微分まで連続な3次多項式でつなぐ。$\ln DF$ が $C^2$ なのでフォワードは
# $C^1$、つまり**滑らか**になる。ただし3次スプラインは大域的な連立方程式で係数が
# 決まるため、あるノードの値を動かすと**遠くの区間まで影響が及ぶ（非局所）**。さらに
# 曲率を無理につなぐため、フォワードが入力にない**振動（オーバーシュート）**を起こし、
# 極端な場合はフォワードが負になりうる。端点条件（自然＝端の2階微分ゼロ）は端の
# フォワードを平坦化する副作用も持つ。
#
# **(4) 単調凸（monotone convex, Hagan-West）**：フォワードそのものを直接構成する。
# 区間 $i$ の離散フォワード
#
# $$
# f^{d}_i = \frac{z_i t_i - z_{i-1} t_{i-1}}{\Delta_i}
#         = \frac{-\ln DF(t_i) + \ln DF(t_{i-1})}{\Delta_i}
# $$
#
# を、その区間での瞬間フォワードの平均とみなす。まずノード上の瞬間フォワード $f_i$ を
# 隣り合う離散フォワードから重み付き平均で推定する。
#
# $$
# f_i = \frac{\Delta_{i+1}}{\Delta_i + \Delta_{i+1}} f^{d}_i
#     + \frac{\Delta_i}{\Delta_i + \Delta_{i+1}} f^{d}_{i+1}
# \quad(内部ノード)
# $$
#
# 各区間で $x = (t - t_{i-1})/\Delta_i \in [0,1]$ とし、離散フォワードからの
# 超過 $g(x) = f(t) - f^{d}_i$ を、両端 $g(0) = f_{i-1} - f^{d}_i$、
# $g(1) = f_i - f^{d}_i$ を通り、かつ $\int_0^1 g\,dx = 0$ を満たす関数で置く。基本形は
#
# $$
# g(x) = g(0)\,(1 - 4x + 3x^2) + g(1)\,(-2x + 3x^2)
# $$
#
# である（$\int_0^1(1-4x+3x^2)dx = \int_0^1(-2x+3x^2)dx = 0$ を確かめられる）。
# 積分がゼロという条件は、区間フォワードを**厳密に再現**することを保証する。
# Hagan-West はさらに、$g$ が単調性・正値性を壊す領域を4つに場合分けし、区間内で
# 放物線を張り替えて振動を抑える。結果としてフォワードは**連続・局所的・非振動**で、
# 入力を厳密に再現する。
#
# ### 動機：単調凸法はなぜ考案されたか
#
# 実務でフォワードカーブは、金利オプションやキャリー分析の一次入力になる。素朴な
# 補間には次の弱点があった。log-DF 線形はフォワードが階段状に段差を持ち、微分が
# ノードで不連続になる。3次スプラインは滑らかだが、(a) 局所性がなく1点の修正が
# カーブ全体を揺らす、(b) フォワードに振動が出て負値すら生じる。Hagan と West は
# 「フォワードが**正で、入力を厳密に再現し、単調性を保ち、しかも局所的**」という
# 条件を同時に満たす補間として単調凸法を提案した。滑らかさを最大化するのではなく、
# **振動を作らないことを優先**した設計である。
#
# ### 滑らかさと局所性のトレードオフ
#
# | 補間法 | 補間対象 | フォワードの形 | 滑らかさ | 局所性 | 正値・単調保存 |
# |---|---|---|---|---|---|
# | ゼロ線形 | $z(t)$ | ノードで不連続（鋸歯） | 低 | 高 | 中 |
# | log-DF 線形 | $\ln DF$ | 区間ごと一定（階段） | 低 | 高 | 高（正値保証） |
# | 自然三次スプライン | $\ln DF$ | $C^1$・振動しうる | 高 | 低 | 低 |
# | 単調凸 | $f(t)$ | 連続・非振動 | 中〜高 | 高 | 高 |
#
# 一般に「滑らかさ」と「局所性・非振動」は競合する。スプラインは滑らかさに全振り
# して局所性を失い、線形系は局所的だがフォワードが折れる。単調凸はその中間で、
# 局所性と非振動を保ちつつ実用的な滑らかさを得る折衷案にあたる。


# %% [markdown]
# **数値例**：ノードを $t=(1,2,3)$ 年、ゼロレート $z=(3.0\%,3.3\%,3.5\%)$ とすると $\ln DF=(-0.030,-0.066,-0.105)$ です。log-DF 線形補間では区間 $[2,3]$ のフォワードは $f=\dfrac{\ln DF(2)-\ln DF(3)}{1}=\dfrac{-0.066-(-0.105)}{1}=3.90\%$ で、区間内は一定の階段になります。一方ゼロ線形補間では同じノードでも区間始点で $f(2^+)=z(2)+2\,s_3=3.3\%+2\times0.20\%=3.70\%$ となり、対象量の選び方でフォワードが変わります。
#
# **数値例**：単調凸法では各区間の離散フォワードが $f^{d}_{[1,2]}=3.60\%,\ f^{d}_{[2,3]}=3.90\%$ となり、区間幅が等しいので内部ノード $t=2$ の瞬間フォワードは単純平均 $f_2=\tfrac12\times3.60\%+\tfrac12\times3.90\%=3.75\%$ に定まります。
# %% [markdown]
# ## スクラッチ実装
#
# 4種の補間法を、共通の基底クラス `CurveInterpolator` の上に実装する。各補間は
# 「$-\ln DF(t)$ を返す `neglogdf`」と「瞬間フォワード `inst_forward`」の2つを
# 実装すれば足り、割引係数・ゼロレートは基底クラスが導出する。この共通化により、
# 後段の比較を同じコードで回せる。
#
# ### 使用する自作クラス・関数
#
# | 名前 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `CurveInterpolator(times, dfs)` | ノード年数, 割引係数 | インスタンス | 共通基底。`discount`/`zero` を提供 |
# | `.discount(t)` | 年数 | 割引係数 | $\exp(-\text{neglogdf}(t))$ |
# | `.zero(t)` | 年数 | ゼロレート | $\text{neglogdf}(t)/t$ |
# | `LinearZero` | 同上 | 補間器 | ゼロレートを線形補間 |
# | `LogLinearDF` | 同上 | 補間器 | $\ln DF$ を線形補間（区間一定フォワード） |
# | `NaturalCubicDF` | 同上 | 補間器 | $\ln DF$ を自然三次スプライン補間 |
# | `MonotoneConvex` | 同上 | 補間器 | Hagan-West 単調凸フォワード補間 |
# | `_thomas_natural(x, y)` | ノード x, y | 2階微分列 $M$ | 自然スプラインの三重対角系を解く |

# %%
import numpy as np

import bondlab
from bondlab.curve import DiscountCurve, bootstrap_par

np.random.seed(0)
print("bondlab version:", bondlab.__version__)


class CurveInterpolator:
    """(times, dfs) ノードから割引・ゼロ・瞬間フォワードを与える共通基底。

    サブクラスは neglogdf(t)=−ln DF(t) と inst_forward(t) を実装する。
    """

    def __init__(self, times, dfs):
        self.t = np.asarray(times, dtype=float)
        self.df = np.asarray(dfs, dtype=float)
        self.rt = -np.log(self.df)  # = z(t) * t（ノード上の −ln DF）

    def neglogdf(self, t):
        raise NotImplementedError

    def inst_forward(self, t):
        raise NotImplementedError

    @staticmethod
    def _vec(fn, t):
        """スカラ実装 fn を配列入力へ広げる小さなヘルパ。"""
        arr = np.asarray(t, dtype=float)
        if arr.ndim == 0:
            return float(fn(float(arr)))
        return np.array([fn(float(x)) for x in arr])

    def discount(self, t):
        return np.exp(-self._vec(self.neglogdf, t))

    def zero(self, t):
        arr = np.asarray(t, dtype=float)
        return self._vec(self.neglogdf, arr) / arr

    def forward_curve(self, ts):
        """密なグリッド ts 上の瞬間フォワード列。"""
        return self._vec(self.inst_forward, ts)


# %% [markdown]
# **(1) ゼロ線形**：ゼロレート $z_i = (-\ln DF_i)/t_i$（$t>0$ のノード）を線形補間し、
# フォワードは $f(t) = z(t) + t\,s_i$ で評価する。

# %%
class LinearZero(CurveInterpolator):
    """ゼロレートを区間線形に補間する。"""

    def __init__(self, times, dfs):
        super().__init__(times, dfs)
        mask = self.t > 0
        self.tz = self.t[mask]
        self.z = self.rt[mask] / self.tz  # ゼロレート

    def _seg(self, t):
        i = int(np.searchsorted(self.tz, t, side="right"))
        i = min(max(i, 1), len(self.tz) - 1)
        s = (self.z[i] - self.z[i - 1]) / (self.tz[i] - self.tz[i - 1])
        return i, s

    def neglogdf(self, t):
        z = np.interp(t, self.tz, self.z)
        return z * t

    def inst_forward(self, t):
        i, s = self._seg(t)
        z = self.z[i - 1] + s * (t - self.tz[i - 1])
        return z + t * s  # f = z + t z'


# %% [markdown]
# **(2) log-DF 線形**：$\ln DF$ を線形補間する。フォワードは区間ごとに一定
# （$= (\,\ln DF_{i-1} - \ln DF_i\,)/\Delta_i$）になる。

# %%
class LogLinearDF(CurveInterpolator):
    """割引係数の対数を区間線形に補間する（区間一定フォワード）。"""

    def neglogdf(self, t):
        logdf = np.interp(t, self.t, -self.rt)  # ln DF を補間
        return -logdf

    def inst_forward(self, t):
        i = int(np.searchsorted(self.t, t, side="right"))
        i = min(max(i, 1), len(self.t) - 1)
        return (self.rt[i] - self.rt[i - 1]) / (self.t[i] - self.t[i - 1])


# %% [markdown]
# **(3) 自然三次スプライン**：$\ln DF$ を自然三次スプラインで補間する。まず
# 三重対角系（Thomas 法）でノードの2階微分 $M_i$ を解き、区間内は3次多項式で
# 値と傾きを評価する。フォワードは $-\dfrac{d}{dt}\ln DF$ として解析的に求める。

# %%
def _thomas_natural(x, y):
    """自然境界（端の2階微分 0）の三次スプラインの2階微分 M を解く。"""
    n = len(x)
    h = np.diff(x)
    M = np.zeros(n)
    if n < 3:
        return M
    # 内部ノード 1..n-2 についての三重対角系を Thomas 法で解く。
    lower = np.zeros(n - 2)
    diag = np.zeros(n - 2)
    upper = np.zeros(n - 2)
    rhs = np.zeros(n - 2)
    for k in range(1, n - 1):
        j = k - 1
        lower[j] = h[k - 1]
        diag[j] = 2.0 * (h[k - 1] + h[k])
        upper[j] = h[k]
        rhs[j] = 6.0 * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1])
    # 前進消去
    for j in range(1, n - 2):
        w = lower[j] / diag[j - 1]
        diag[j] -= w * upper[j - 1]
        rhs[j] -= w * rhs[j - 1]
    # 後退代入
    m = np.zeros(n - 2)
    m[-1] = rhs[-1] / diag[-1]
    for j in range(n - 4, -1, -1):
        m[j] = (rhs[j] - upper[j] * m[j + 1]) / diag[j]
    M[1:-1] = m
    return M


class NaturalCubicDF(CurveInterpolator):
    """割引係数の対数を自然三次スプラインで補間する。"""

    def __init__(self, times, dfs):
        super().__init__(times, dfs)
        self.x = self.t
        self.y = -self.rt  # ln DF
        self.M = _thomas_natural(self.x, self.y)

    def _eval(self, t, deriv=False):
        i = int(np.searchsorted(self.x, t, side="right"))
        i = min(max(i, 1), len(self.x) - 1)
        h = self.x[i] - self.x[i - 1]
        A = (self.x[i] - t) / h
        B = (t - self.x[i - 1]) / h
        yi_1, yi = self.y[i - 1], self.y[i]
        Mi_1, Mi = self.M[i - 1], self.M[i]
        if not deriv:
            return (A * yi_1 + B * yi
                    + ((A ** 3 - A) * Mi_1 + (B ** 3 - B) * Mi) * h ** 2 / 6.0)
        # d/dt の解析形
        return ((yi - yi_1) / h
                - (3 * A ** 2 - 1) / 6.0 * h * Mi_1
                + (3 * B ** 2 - 1) / 6.0 * h * Mi)

    def neglogdf(self, t):
        return -self._eval(t, deriv=False)

    def inst_forward(self, t):
        return -self._eval(t, deriv=True)  # f = -d/dt ln DF


# %% [markdown]
# **(4) 単調凸（Hagan-West）**：離散フォワード $f^d_i$ とノード上の瞬間フォワード
# $f_i$ を作り、区間内は超過フォワード $g(x)$ を4領域で場合分けして張る。$-\ln DF$ は
# $g$ の積分から得る（区間積分がゼロなのでノードのゼロレートを厳密に再現する）。
# フォワードの正値を保つため、ノード上の $f_i$ には $[0,\,2\min(f^d)]$ の枠をかける
# （Hagan-West の collar）。

# %%
class MonotoneConvex(CurveInterpolator):
    """Hagan-West の単調凸法でフォワードを直接補間する。"""

    def __init__(self, times, dfs):
        super().__init__(times, dfs)
        T, RT = self.t, self.rt
        self.fd = np.diff(RT) / np.diff(T)  # 離散フォワード（区間 i）
        n = len(self.fd)
        f = np.zeros(n + 1)
        for i in range(1, n):
            w_l = (T[i] - T[i - 1]) / (T[i + 1] - T[i - 1])
            w_r = (T[i + 1] - T[i]) / (T[i + 1] - T[i - 1])
            f[i] = w_l * self.fd[i] + w_r * self.fd[i - 1]
        f[0] = self.fd[0] - 0.5 * (f[1] - self.fd[0])
        f[n] = self.fd[n - 1] - 0.5 * (f[n - 1] - self.fd[n - 1])
        # 正値の collar
        f[0] = min(max(f[0], 0.0), 2.0 * self.fd[0])
        for i in range(1, n):
            f[i] = min(max(f[i], 0.0), 2.0 * min(self.fd[i - 1], self.fd[i]))
        f[n] = min(max(f[n], 0.0), 2.0 * self.fd[n - 1])
        self.fnode = f
        self.n = n

    def _locate(self, t):
        i = int(np.searchsorted(self.t, t, side="right"))
        return min(max(i, 1), self.n)

    def _g_params(self, i):
        g0 = self.fnode[i - 1] - self.fd[i - 1]
        g1 = self.fnode[i] - self.fd[i - 1]
        return g0, g1

    @staticmethod
    def _G(x, g0, g1):
        """超過フォワード g(x)（Hagan-West の4領域）。"""
        if g0 == 0.0 and g1 == 0.0:
            return 0.0
        # 領域1：基本の放物線
        if ((g0 < 0 and -0.5 * g0 <= g1 <= -2.0 * g0)
                or (g0 > 0 and -0.5 * g0 >= g1 >= -2.0 * g0)
                or (g0 * g1 < 0 and 0.5 * abs(g0) <= abs(g1) <= 2.0 * abs(g0))):
            return g0 * (1 - 4 * x + 3 * x ** 2) + g1 * (-2 * x + 3 * x ** 2)
        # 領域2
        if (g0 < 0 and g1 > -2.0 * g0) or (g0 > 0 and g1 < -2.0 * g0):
            eta = (g1 + 2.0 * g0) / (g1 - g0)
            return g0 if x <= eta else g0 + (g1 - g0) * ((x - eta) / (1 - eta)) ** 2
        # 領域3
        if (g0 > 0 and 0 > g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
            eta = 3.0 * g1 / (g1 - g0)
            return g1 + (g0 - g1) * ((eta - x) / eta) ** 2 if x < eta else g1
        # 領域4
        eta = g1 / (g1 + g0)
        A = -g0 * g1 / (g0 + g1)
        if x <= eta:
            return A + (g0 - A) * ((eta - x) / eta) ** 2
        return A + (g1 - A) * ((x - eta) / (1 - eta)) ** 2

    @staticmethod
    def _int_G(x, g0, g1):
        """0 から x までの g の積分（−ln DF の区間内の増分に使う）。"""
        if g0 == 0.0 and g1 == 0.0:
            return 0.0
        if ((g0 < 0 and -0.5 * g0 <= g1 <= -2.0 * g0)
                or (g0 > 0 and -0.5 * g0 >= g1 >= -2.0 * g0)
                or (g0 * g1 < 0 and 0.5 * abs(g0) <= abs(g1) <= 2.0 * abs(g0))):
            return g0 * (x - 2 * x ** 2 + x ** 3) + g1 * (-x ** 2 + x ** 3)
        if (g0 < 0 and g1 > -2.0 * g0) or (g0 > 0 and g1 < -2.0 * g0):
            eta = (g1 + 2.0 * g0) / (g1 - g0)
            if x <= eta:
                return g0 * x
            return g0 * x + (g1 - g0) * (x - eta) ** 3 / (3 * (1 - eta) ** 2)
        if (g0 > 0 and 0 > g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
            eta = 3.0 * g1 / (g1 - g0)
            if x <= eta:
                return g1 * x + (g0 - g1) / (eta ** 2) * (eta ** 3 - (eta - x) ** 3) / 3.0
            return g1 * eta + (g0 - g1) * eta / 3.0 + g1 * (x - eta)
        eta = g1 / (g1 + g0)
        A = -g0 * g1 / (g0 + g1)
        if x <= eta:
            return A * x + (g0 - A) / (eta ** 2) * (eta ** 3 - (eta - x) ** 3) / 3.0
        Ieta = A * eta + (g0 - A) * eta / 3.0
        return Ieta + A * (x - eta) + (g1 - A) * (x - eta) ** 3 / (3 * (1 - eta) ** 2)

    def neglogdf(self, t):
        if t <= 0:
            return 0.0
        i = self._locate(t)
        dt = self.t[i] - self.t[i - 1]
        x = (t - self.t[i - 1]) / dt
        g0, g1 = self._g_params(i)
        return self.rt[i - 1] + self.fd[i - 1] * (t - self.t[i - 1]) + dt * self._int_G(x, g0, g1)

    def inst_forward(self, t):
        i = self._locate(t)
        dt = self.t[i] - self.t[i - 1]
        x = (t - self.t[i - 1]) / dt
        g0, g1 = self._g_params(i)
        return self.fd[i - 1] + self._G(x, g0, g1)


# %% [markdown]
# ### bondlab との一致確認
#
# 合成パーカーブをブートストラップしてノードを作り、自作の `LogLinearDF` /
# `LinearZero` が `bondlab.curve.DiscountCurve` の `log_linear` / `linear_zero` と
# 一致することを確認する。パーカーブのテナーは 0.5〜30 年と等間隔でないため、
# ここでは年次グリッド（1〜30 年）へパー利回りを線形に載せ替えてから
# `bootstrap_par(frequency=1)` を適用する（`bootstrap_par` は等間隔・年次払いを
# 前提とするため。詳細は「実データ適用」で述べる）。

# %%
import pandas as pd

par_df = pd.read_csv("data/samples/synthetic_ust_par_curve.csv")
grid = np.arange(1, 31, dtype=float)
par_on_grid = np.interp(grid, par_df["tenor"].to_numpy(), par_df["par_yield"].to_numpy())

curve = bootstrap_par(grid, par_on_grid, frequency=1)  # log_linear の DiscountCurve
node_t = curve.times          # [0, 1, 2, ..., 30]
node_df = curve.dfs

ll = LogLinearDF(node_t, node_df)
lz = LinearZero(node_t, node_df)
cub = NaturalCubicDF(node_t, node_df)
mc = MonotoneConvex(node_t, node_df)

dc_lz = DiscountCurve(grid, node_df[1:], interp="linear_zero")

check_t = np.array([1.5, 2.75, 4.0, 8.5, 15.0, 23.0])
print("t       scratch_ll    bondlab_ll    scratch_lz    bondlab_lz")
for t in check_t:
    print(f"{t:5.2f}  {ll.discount(t):.10f}  {float(curve.discount(t)):.10f}  "
          f"{lz.discount(t):.10f}  {float(dc_lz.discount(t)):.10f}")

assert np.max(np.abs(ll.discount(check_t) - curve.discount(check_t))) < 1e-12
assert np.max(np.abs(lz.discount(check_t) - dc_lz.discount(check_t))) < 1e-12
print("bondlab との log_linear / linear_zero 一致を確認しました")

# %% [markdown]
# 単調凸法がノードのゼロレートを厳密に再現すること（区間積分ゼロの帰結）も確認する。

# %%
node_zero_mc = mc.zero(node_t[1:])
node_zero_true = curve.zero_rate(node_t[1:])
assert np.max(np.abs(node_zero_mc - node_zero_true)) < 1e-12
# フォワードが全区間で正であること（単調凸の設計目標）も確認する。
dense = np.linspace(1.0, 30.0, 2000)
assert np.all(mc.forward_curve(dense) > 0)
print("単調凸：ノード再現誤差", f"{np.max(np.abs(node_zero_mc - node_zero_true)):.2e}",
      "／ フォワード最小値", f"{mc.forward_curve(dense).min()*100:.3f}%")

# %% [markdown]
# ## QuantLib検証
#
# 同じ $\ln DF$ ノードに QuantLib の各補間器を当て、自作実装と突合する。
#
# - `LogLinearInterpolation`（$\ln DF$ の線形）↔ 自作 `LogLinearDF`
# - `CubicNaturalSpline`（自然三次）↔ 自作 `NaturalCubicDF` と `scipy.CubicSpline`
# - `MonotonicCubicNaturalSpline`（Hyman 単調フィルタ）は、Hagan-West の単調凸とは
#   別系統の単調化なので、値の一致ではなく**フォワードの質**を並べて比較する。

# %%
import QuantLib as ql
from scipy.interpolate import CubicSpline

logdf_nodes = np.log(node_df)
x_arr = ql.Array(list(node_t))
y_arr = ql.Array(list(logdf_nodes))

ql_loglin = ql.LinearInterpolation(x_arr, y_arr)
ql_cubic = ql.CubicNaturalSpline(x_arr, y_arr)
ql_mono = ql.MonotonicCubicNaturalSpline(x_arr, y_arr)
sp_cubic = CubicSpline(node_t, logdf_nodes, bc_type="natural")

grid_chk = np.array([1.5, 3.3, 6.0, 8.5, 14.0, 22.5, 28.0])
print("t       自作LL−QL      自作Cubic−QL   自作Cubic−scipy")
for t in grid_chk:
    ll_scratch = np.log(ll.discount(t))
    cub_scratch = np.log(cub.discount(t))
    print(f"{t:5.2f}  {ll_scratch - ql_loglin(t, True):+.2e}   "
          f"{cub_scratch - ql_cubic(t, True):+.2e}   "
          f"{cub_scratch - sp_cubic(t):+.2e}")

# 数値一致の主張
for t in grid_chk:
    assert abs(np.log(ll.discount(t)) - ql_loglin(t, True)) < 1e-12
    assert abs(np.log(cub.discount(t)) - ql_cubic(t, True)) < 1e-10
    assert abs(np.log(cub.discount(t)) - sp_cubic(t)) < 1e-10
print("QuantLib LogLinear / CubicNaturalSpline・scipy CubicSpline と一致を確認しました")

# %% [markdown]
# QuantLib の `CubicNaturalSpline` と scipy の `CubicSpline(bc_type='natural')`、
# および自作 `NaturalCubicDF` は同じ自然境界の三次スプラインなので、機械精度で
# 一致する。一方 `MonotonicCubicNaturalSpline` は Hyman フィルタで単調性を課すため
# 値が異なる。次節でフォワードを描いて質の違いを見る。

# %% [markdown]
# ## 実データ適用
#
# `data/samples/synthetic_ust_par_curve.csv` の米国債パーカーブ（0.5〜30 年）を
# ブートストラップし、4種の補間でフォワードカーブを描いて比較する。
#
# 補足（`bootstrap_par` の前提）：`bondlab.curve.bootstrap_par` は「テナーが
# 年次払いの等間隔グリッドに一致する」ことを前提に、各テナーを1回のクーポン期間と
# みなして割引係数を剥ぎ取る。CSV のテナーは 0.5, 1, 2, 3, 5, 7, 10, 20, 30 年と
# 等間隔でないため、そのまま `frequency=1` で渡すと短期側のゼロレートが大きく歪む
# （0.5 年を1年分のクーポン期間として扱ってしまう）。本 notebook では、パー利回りを
# 年次グリッド（1〜30 年）へ線形補間してから `bootstrap_par` に渡し、金融的に妥当な
# ノード列を得ている。ここでの主題はノード列を**どう補間するか**なので、入力整形は
# 単純な線形補間に固定した。

# %%
node_zero = curve.zero_rate(node_t[1:]) * 100
print("ブートストラップ後のノード（年次グリッド）")
print(pd.DataFrame({"tenor": node_t[1:], "zero%": np.round(node_zero, 3),
                    "DF": np.round(node_df[1:], 5)}).to_string(index=False))

# %%
import matplotlib.pyplot as plt

import matplotlib.font_manager as _fm
for _f in ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Noto Sans JP", "TakaoPGothic", "IPAPGothic"]:
    if any(_f == _n.name for _n in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
interps = {
    "ゼロ線形": lz,
    "log-DF 線形": ll,
    "自然三次スプライン": cub,
    "単調凸 (Hagan-West)": mc,
}
tt = np.linspace(1.0, 30.0, 1200)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for label, obj in interps.items():
    axes[0].plot(tt, obj.zero(tt) * 100, label=label, lw=1.3)
    axes[1].plot(tt, obj.forward_curve(tt) * 100, label=label, lw=1.3)
axes[0].scatter(node_t[1:], node_zero, color="k", s=18, zorder=5, label="ノード")
axes[0].set_title("ゼロレート")
axes[0].set_xlabel("年限")
axes[0].set_ylabel("ゼロレート (%)")
axes[1].set_title("瞬間フォワード")
axes[1].set_xlabel("年限")
axes[1].set_ylabel("フォワード (%)")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ゼロレート（左）はどの補間もノードを通り、目視ではほぼ重なる。ところが右の
# フォワードでは差が際立つ。log-DF 線形は階段状に段差が立ち、ゼロ線形はノードで
# 折れて鋸歯状になる。自然三次スプラインは滑らかだが端やノード付近で振れやすい。
# 単調凸は連続かつ振動が小さく、正値を保っている。「価格は似ていてもフォワードは
# 大きく違う」という補間の本質がここに表れる。

# %% [markdown]
# QuantLib の `MonotonicCubicNaturalSpline`（Hyman 単調フィルタ）と自作の単調凸
# （Hagan-West）を並べ、単調化の系統差を見る。

# %%
def ql_forward(interp_obj, t, h=1e-4):
    """QL 補間器（ln DF）から中心差分で瞬間フォワードを作る。"""
    lo = max(t - h, node_t[1] + 1e-6)
    hi = min(t + h, node_t[-1] - 1e-6)
    return -(interp_obj(hi, True) - interp_obj(lo, True)) / (hi - lo)


fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tt, [ql_forward(ql_cubic, t) * 100 for t in tt], label="QL 自然三次", lw=1.2)
ax.plot(tt, [ql_forward(ql_mono, t) * 100 for t in tt], label="QL Hyman 単調三次", lw=1.2)
ax.plot(tt, mc.forward_curve(tt) * 100, label="自作 単調凸 (Hagan-West)", lw=1.6)
ax.set_title("単調化の系統差：Hyman 三次 vs Hagan-West 単調凸")
ax.set_xlabel("年限")
ax.set_ylabel("フォワード (%)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Hyman フィルタは三次スプラインを土台に単調性を後付けするため、振動は抑えるが
# ノード付近の形は三次寄りになる。Hagan-West はフォワードそのものを構成するので、
# より素直で非振動なフォワードを与える。目的（滑らかさ優先か非振動優先か）で
# 使い分けるのが実務の勘所である。

# %% [markdown]
# ## 演習
#
# 1. **フォワードのギザギザ度**：4種の補間について、密なグリッド上の瞬間
#    フォワードの**2階差分ノルム** $\lVert \Delta^2 f \rVert_2$ を求めて比較せよ。
#    値が小さいほどフォワードが滑らか（ギザギザが少ない）である。どの補間を
#    推奨するか、滑らかさ・局所性・正値保存のトレードオフを踏まえて理由とともに
#    述べよ。
# 2. **局所性の実験**：1点のノード（たとえば 10 年）のゼロレートを $+20$bp だけ
#    動かし、各補間でフォワードカーブがどこまで変化するかを調べよ。変化が有意
#    （たとえば $0.5$bp 超）な年限の範囲を測り、「局所的な補間」と「大域的な補間」を
#    数値で区別せよ。
#
# 解答例は `solutions/S2/sol_0202.py` に置く。以下は演習1のさわりだけ示す。

# %%
def forward_roughness(obj, ts):
    """瞬間フォワード列の2階差分の L2 ノルム（ギザギザ度）。"""
    f = obj.forward_curve(ts)
    return float(np.linalg.norm(np.diff(f, n=2)))


rough_grid = np.linspace(1.0, 30.0, 600)
print("補間法                     フォワード2階差分ノルム")
for label, obj in interps.items():
    print(f"{label:26s} {forward_roughness(obj, rough_grid)*1e4:12.4f}  (×1e-4)")

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/02_curves.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | [補間](../../glossary/02_curves.md#interpolation) | interpolation | ノード間の値を関数で埋める操作。対象量の選択がフォワードの形を決める |
# | [スプライン](../../glossary/02_curves.md#spline) | spline | 区間ごとの低次多項式を、ノードで滑らかさを保ってつないだ関数 |
# | [単調性](../../glossary/02_curves.md#monotonicity) | monotonicity | 入力の増減を補間が保つ性質。フォワードの正値・非振動に効く |
# | [局所性](../../glossary/02_curves.md#locality) | locality | 1点の入力変化が近傍の区間だけに影響し、遠くへ波及しない性質 |
# | [Hagan-West](../../glossary/02_curves.md#hagan-west-monotone-convex) | Hagan-West (monotone convex) | フォワードを直接構成し、正値・非振動・局所性・入力厳密再現を両立する補間 |
