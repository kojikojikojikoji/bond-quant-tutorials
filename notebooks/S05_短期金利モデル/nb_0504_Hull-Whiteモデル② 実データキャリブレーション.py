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
# # S5-4 Hull-Whiteモデル② 実データキャリブレーション
#
# ## 学習目標
#
# - スワップションのクォート慣行（満期×テナーのグリッド、ATM、Blackボラ表示）を
#   説明できる
# - Hull-White モデルのパラメータ $a, \sigma$ を、スワップション市場価格へ
#   キャリブレーションする手順を、目的関数の設計から実行できる
# - QuantLib の `JamshidianSwaptionEngine` と Levenberg-Marquardt 法を使って
#   キャリブレータを組み、満期×テナー別の再現誤差ヒートマップで診断できる
# - 合成ボラ面で既知パラメータの復元を確認し、初期値に対する安定性を評価できる
# - $a$ と $\sigma$ の識別性・局所解という、キャリブレーション特有の落とし穴を
#   理解する
#
# S5-3 では Hull-White モデルの構造（$\theta(t)$ を初期カーブに合わせる無裁定
# モデル）を扱った。本 notebook は続編として、残る2つのパラメータ $a$（平均回帰
# 速度）と $\sigma$（ボラティリティ）を市場から決める作業に充てる。ここが実務で
# Hull-White を「使える」状態にする前半の山場になる。


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# キャリブレーションは、金利デスクのクオンツが Hull-White を「実際に使える状態」にする工程です。S5-3 で $\theta(t)$ が今日のカーブを通すことを保証しても、ボラティリティ構造を決める $a$（平均回帰速度）と $\sigma$ を市場から決めなければ、スワップションやコーラブル債のオプション価値は計算できません。スワップション市場の Black ボラ面（満期×テナーのグリッド、ATM クォート）へモデル価格を合わせ込む本 notebook の作業が、そのままデスクの日次キャリブレーションの縮図です。`JamshidianSwaptionEngine` と Levenberg-Marquardt でキャリブレータを組み、満期×テナー別の再現誤差ヒートマップで診断する手順は、本番の値付けエンジンに毎営業日与えるパラメータを作る流れと同じです。
#
# 収益への効き方は、面の再現精度とヘッジの安定性に表れます。キャリブレーション後の $a,\sigma$ でベガ（各グリッドセルへの感応度）とバケット・デルタが決まり、それに沿ってスワップションでベガを、スワップや先物でデルタを相殺すると、マーケットメイクのスプレッドやストラクチャー組成のマージンだけが残ります。面の再現が悪ければ、ヘッジが噛み合わずに市場が動くたびに損益が振れます。$a$ が小さいほど遠いテナーのボラが効くなど、パラメータが面のどこを説明するかを理解しておくことが、どの商品にどのキャリブレーション対象を選ぶかの判断に直結します。
#
# モデルバリデーション（S5-5、FRB SR 11-7）の中心論点は、キャリブレーション特有の落とし穴です。$a$ と $\sigma$ は識別性が弱く、目的関数が平坦な谷や複数の局所解を持つため、初期値次第で違う解に収束します。本 notebook で合成ボラ面から既知パラメータの復元を確認し、初期値に対する安定性を評価するのは、まさに検証担当が要求する「復元テスト」です。検証担当は、キャリブレーションが安定に真値へ戻るか、面の外挿・時点間のパラメータ推移が滑らかか、選んだ目的関数と重みが恣意的でないかを独立に確かめて承認します。ここが甘いと、日によってパラメータが跳ね、同じポジションの評価とリスク指標が不安定になって損益を汚染します。
# %% [markdown]
# ## 理論
#
# ### スワップションのクォート慣行
#
# スワップション（swaption）は、将来の一定期日に、あらかじめ決めた固定金利で
# 金利スワップに入る権利である。区別すべき2つの年限がある。
#
# - **満期（expiry）**：オプションを行使できるまでの年数。例：2年後に判断する。
# - **テナー（tenor）**：行使したときに入るスワップの年数。例：入ると5年スワップ。
#
# この「満期 $\times$ テナー」を格子状に並べたものがスワップション市場である。
# 「2y5y（2年満期・5年テナー）」のように呼ぶ。各セルに1つのボラティリティが
# クォートされ、面（surface）を成す。
#
# クォートは通常 **ATM（at-the-money）** で行う。ATM とは、原スワップのフォワード
# スワップレートに等しい行使レートを指す。このとき本源的価値がゼロで、価格は
# 時間価値だけになるため、ボラティリティの情報が最も素直に表れる。
#
# 価格そのものではなく **Blackボラティリティ** で表示するのが慣行である。
# 市場価格 $V^{\mathrm{mkt}}$ を、次の Black 公式でボラ $\sigma_{\mathrm{B}}$ に
# 逆算した値をクォートする。
#
# ### Black によるスワップション評価（最小限）
#
# ペイヤー・スワップション（固定を払う権利）の価値は、フォワードスワップレート
# $S_0$ を対数正規と仮定した Black モデルで次のように書ける。
#
# $$
# V = A \cdot \left[ S_0\, N(d_1) - K\, N(d_2) \right], \qquad
# d_{1,2} = \frac{\ln(S_0/K) \pm \tfrac{1}{2}\sigma_{\mathrm{B}}^2 T}{\sigma_{\mathrm{B}}\sqrt{T}}
# $$
#
# ここで $T$ は満期、$K$ は行使レート、$N(\cdot)$ は標準正規分布関数である。
# $A$ は **アニュイティ（annuity, PV01）** と呼ぶ割引係数の和で、テナー区間の
# スワップ固定脚のキャッシュフローを現在価値化する重みである。
#
# $$
# A = \sum_{j=1}^{n} \tau_j\, P(0, T_j)
# $$
#
# $\tau_j$ は各利払区間の年数、$P(0, T_j)$ はゼロクーポン債価格（割引係数）。
# ATM では $K = S_0$ なので $d_1 = \tfrac{1}{2}\sigma_{\mathrm{B}}\sqrt{T}$、
# $d_2 = -d_1$ となり、$V = A\, S_0\,[2N(d_1)-1]$ と簡潔になる。
#
# 重要なのは向きである。**市場は $\sigma_{\mathrm{B}}$ を与え、Black公式が価格を
# 与える**。一方、Hull-White モデルは自分の $a, \sigma$ からスワップション価格を
# 計算する。両者の価格（あるいは価格から逆算したボラ）を一致させるのが
# キャリブレーションである。本 notebook では S6-3 のデリバティブ評価に依存せず、
# この対応関係だけで話を閉じる。
#
# ### Hull-White でのスワップション評価と Jamshidian 分解
#
# Hull-White は1ファクターのガウシアンモデルなので、スワップションを
# **Jamshidian 分解（Jamshidian decomposition）** で解析的に評価できる。要点は、
# スワップは固定脚キャッシュフローを持つ「クーポン債」とみなせること、そして
# 1ファクターモデルでは短期金利が単調にキャッシュフロー現在価値を動かすため、
# クーポン債オプションを **個々のゼロクーポン債オプションの和** に厳密分解できる、
# という点にある。行使境界となる短期金利 $r^{*}$ を1次元求根で1つ求めれば、
# 各ゼロクーポン債オプションは Hull-White の閉じた式（$B(t,T)$、$\sigma_P$ による
# ガウシアン公式）で評価できる。QuantLib の `JamshidianSwaptionEngine` がこれを
# 実装している。
#
# ### キャリブレーションの目的関数設計
#
# キャリブレーションとは、モデル価格を市場に最も近づけるパラメータを探す最適化
# である。設計上の論点は3つある。
#
# **(1) 価格差かボラ差か。** 目的関数を市場価格とモデル価格の差（price error）で
# 組むか、Blackボラの差（implied-vol error）で組むかを選ぶ。価格差は満期・テナー
# が長いセルほど価格の絶対水準が大きく、自然に重みが偏る。ボラ差は水準を揃えた
# 比較になり、面全体をバランスよく合わせやすい。QuantLib の `SwaptionHelper` は
# 既定で価格ベースの残差を返すが、後述のとおり診断は必ずボラ差でも見る。
#
# **(2) 重み付け。** セルごとに重み $w_i$ を掛けて
# $\min_{a,\sigma} \sum_i w_i\, (V_i^{\mathrm{model}} - V_i^{\mathrm{mkt}})^2$
# を解く。ヘッジ対象の年限を厚く、流動性の低いセルを薄く、といった調整に使う。
#
# **(3) 最小化アルゴリズム。** 残差が2乗和なので **Levenberg-Marquardt法**
# （ガウス・ニュートンと最急降下を補間する非線形最小二乗法）が定番である。
# 少数パラメータ・滑らかな残差で収束が速い。
#
# ### $a, \sigma$ の識別性と局所解
#
# **識別性（identifiability）** とは、データからパラメータを一意に決められる度合い
# である。Hull-White の $a$（平均回帰速度）と $\sigma$（瞬間ボラ）は、
# スワップションのインプライドボラに対して似た方向に効きやすく、単一のオプション
# だけでは分離しにくい。満期・テナーの異なる複数セルを同時に使うことで、
# $a$ が支配する「ボラの期間構造の減衰」と $\sigma$ が支配する「全体水準」を
# 分離できる。逆に、面が平坦だと識別性は落ち、$a$ が定まりにくくなる。
#
# **局所解（local minimum）** とは、目的関数の谷のうち大域最小でないものである。
# 非線形最小化は初期値によって局所解に落ちうる。対策は、初期値を変えて再現性を
# 見ること、パラメータに妥当な範囲を与えること、面の情報量を確保することである。
# 本 notebook では初期値を変えた安定性確認を検証の柱に据える。


# %% [markdown]
# #### 数値例
#
# **数値例**：ATM（$K=S_0$）で $S_0=2\%$、Blackボラ $\sigma_{\mathrm{B}}=30\%$、満期 $T=5$ 年のとき $d_1=\tfrac12\sigma_{\mathrm{B}}\sqrt{T}=0.335$、$2N(d_1)-1=0.263$ となり、アニュイティ $A=4.0$ ならスワップション価値は $V=A\,S_0\,[2N(d_1)-1]=0.0210$ です。
#
# **数値例**：1年刻み5年・割引率 $2\%$（連続複利）のフラットカーブでは、アニュイティは $A=\sum_{j=1}^{5}1\cdot e^{-0.02\,T_j}=4.711$ です。
# %% [markdown]
# ## スクラッチ実装
#
# キャリブレーションの前段として、Black スワップション評価とアニュイティを自作
# する。目的は2つ。第一に、ATM でボラと価格が可逆に結び付くことを手を動かして
# 確認する。第二に、後段の QuantLib による再現誤差を「価格差」だけでなく
# 「ボラ差」でも読めるよう、価格 $\to$ ボラの逆算を自前で持っておく。
#
# 割引カーブは、合成パー利回りから `bondlab.curve.bootstrap_par` で構築する。
# ネットワークは使わず、面もカーブもすべて手元で生成する。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `swap_annuity(curve, expiry, tenor, freq)` | 割引カーブ, 満期, テナー, 頻度 | アニュイティ $A$ | 固定脚 PV01 |
# | `forward_swap_rate(curve, expiry, tenor, freq)` | 同上 | フォワードスワップレート $S_0$ | ATM 行使レート |
# | `black_payer_swaption(S0, K, vol, T, annuity)` | フォワード, 行使, ボラ, 満期, $A$ | 価格 | Black 価格 |
# | `implied_black_vol(price, S0, K, T, annuity)` | 価格, フォワード, 行使, 満期, $A$ | ボラ | 価格→ボラ逆算 |

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.optimize import brentq
from scipy.stats import norm

import bondlab
from bondlab.curve import bootstrap_par

print("bondlab version:", bondlab.__version__)

SEED = 20260707
rng = np.random.default_rng(SEED)

# 日本語フォントが無い環境でも文字化けさせないため、記号を ASCII に寄せる。
plt.rcParams["axes.unicode_minus"] = False


def swap_annuity(curve, expiry, tenor, freq=1):
    """固定脚のアニュイティ A = Σ τ_j P(0, T_j) を返す。

    満期 expiry から始まり tenor 年続くスワップの固定脚を想定する。
    freq は年間利払回数。等間隔グリッドで割引係数を積み上げる。
    """
    tau = 1.0 / freq
    pay_times = expiry + tau * np.arange(1, int(round(tenor * freq)) + 1)
    return float(np.sum(tau * curve.discount(pay_times)))


def forward_swap_rate(curve, expiry, tenor, freq=1):
    """フォワードスワップレート S0 = (P(0,T_s) - P(0,T_e)) / A を返す。

    分子は変動脚の価値（区間端の割引係数の差）、分母はアニュイティ。
    """
    p_start = curve.discount(expiry)
    p_end = curve.discount(expiry + tenor)
    return float((p_start - p_end) / swap_annuity(curve, expiry, tenor, freq))


def black_payer_swaption(S0, K, vol, T, annuity):
    """ペイヤー・スワップションの Black 価格。vol は年率の対数正規ボラ。"""
    if vol <= 0 or T <= 0:
        return annuity * max(S0 - K, 0.0)
    sqT = vol * np.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * sqT ** 2) / sqT
    d2 = d1 - sqT
    return annuity * (S0 * norm.cdf(d1) - K * norm.cdf(d2))


def implied_black_vol(price, S0, K, T, annuity):
    """Black 価格から対数正規ボラを逆算する（1次元求根）。"""
    intrinsic = annuity * max(S0 - K, 0.0)
    if price <= intrinsic + 1e-14:
        return 0.0
    f = lambda v: black_payer_swaption(S0, K, v, T, annuity) - price
    return float(brentq(f, 1e-6, 5.0, xtol=1e-10))


# %% [markdown]
# 合成パー利回りから割引カーブを作り、代表的な ATM スワップションで、ボラ→価格→
# ボラの往復が一致することを確認する。ATM なので行使レート $K$ はフォワード
# スワップレート $S_0$ に等しく置く。

# %%
par_tenors = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30], dtype=float)
par_rates = np.array(
    [0.030, 0.032, 0.034, 0.0355, 0.0365, 0.038, 0.0395, 0.0405, 0.0410, 0.0415]
)
curve = bootstrap_par(par_tenors, par_rates, frequency=1)

expiry, tenor = 2.0, 5.0
S0 = forward_swap_rate(curve, expiry, tenor)
A = swap_annuity(curve, expiry, tenor)
K = S0  # ATM

vol_in = 0.24
price = black_payer_swaption(S0, K, vol_in, expiry, A)
vol_back = implied_black_vol(price, S0, K, expiry, A)

print(f"フォワードスワップレート S0 = {S0:.6f}")
print(f"アニュイティ A            = {A:.6f}")
print(f"入力ボラ                  = {vol_in:.6f}")
print(f"Black 価格                = {price:.8f}")
print(f"逆算ボラ                  = {vol_back:.6f}")

# 往復誤差はゼロに近いはず。
assert abs(vol_back - vol_in) < 1e-8
print("ボラ→価格→ボラ の往復が一致（ATM評価の整合を確認）")

# %% [markdown]
# ## QuantLib検証
#
# 本題のキャリブレータを組む。方針は「既知パラメータ $(a^{*}, \sigma^{*})$ の
# Hull-White からスワップション ATM ボラ面を合成し、その面へ改めてキャリブレー
# ションして $(a^{*}, \sigma^{*})$ を復元できるか」を確かめることである。正解が
# 手元にあるので、キャリブレータの正しさを厳密に検証できる。
#
# 合成ボラ面は、`SwaptionHelper` に既知モデルの価格を入れ、`impliedVolatility()`
# で ATM Blackボラへ変換して作る。グリッドは満期・テナーとも $5\times5$（実行時間
# を抑えるため）とする。QuantLib のカーブは、`bondlab` の割引カーブを同じノードで
# 写し取って `DiscountCurve` に渡す。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

CAL_DC = ql.Actual365Fixed()
TODAY = ql.Date(1, 1, 2024)
ql.Settings.instance().evaluationDate = TODAY


def bondlab_to_ql_curve(curve, max_t=30.0, n_nodes=60):
    """bondlab の DiscountCurve を QuantLib の割引カーブハンドルへ写す。

    0 から max_t まで等間隔のノードで割引係数をサンプルし、Actual365Fixed で
    日付に変換する。以降の QuantLib 評価はこのハンドルを土台に使う。
    """
    node_t = np.concatenate([[0.0], np.linspace(0.5, max_t, n_nodes)])
    dates = [TODAY + ql.Period(int(round(t * 365)), ql.Days) for t in node_t]
    dfs = [1.0] + [float(curve.discount(t)) for t in node_t[1:]]
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, CAL_DC))


ts = bondlab_to_ql_curve(curve)
float_index = ql.USDLibor(ql.Period(6, ql.Months), ts)

EXPIRIES = [1, 2, 3, 5, 7]
TENORS = [1, 2, 3, 5, 7]


def make_helpers(grid, engine):
    """(expiry, tenor, vol) のグリッドから SwaptionHelper のリストを作る。

    各 helper は与えたエンジンで価格付けするよう設定する。ATM を仮定し、
    固定脚は年1回・Actual365Fixed とする（合成面なので規約は自己完結でよい）。
    """
    helpers = []
    for (e, te, vol) in grid:
        h = ql.SwaptionHelper(
            ql.Period(e, ql.Years),
            ql.Period(te, ql.Years),
            ql.QuoteHandle(ql.SimpleQuote(vol)),
            float_index,
            ql.Period(1, ql.Years),
            CAL_DC,
            CAL_DC,
            ts,
        )
        h.setPricingEngine(engine)
        helpers.append(h)
    return helpers


def build_synthetic_surface(a_true, sigma_true):
    """既知 (a, sigma) の Hull-White から ATM スワップション Blackボラ面を作る。

    ダミーボラで helper を作り、真モデルのエンジンでモデル価格を計算し、その価格
    を implied vol へ逆算して「市場ボラ」とみなす。既知パラメータの復元検証用。
    """
    true_model = ql.HullWhite(ts, a_true, sigma_true)
    true_engine = ql.JamshidianSwaptionEngine(true_model)
    grid = []
    for e in EXPIRIES:
        for te in TENORS:
            h = ql.SwaptionHelper(
                ql.Period(e, ql.Years),
                ql.Period(te, ql.Years),
                ql.QuoteHandle(ql.SimpleQuote(0.20)),
                float_index,
                ql.Period(1, ql.Years),
                CAL_DC,
                CAL_DC,
                ts,
            )
            h.setPricingEngine(true_engine)
            vol = h.impliedVolatility(h.modelValue(), 1e-6, 200, 1e-4, 2.0)
            grid.append((e, te, float(vol)))
    return grid


def calibrate_hw(grid, a0=0.05, sigma0=0.01, max_iter=400):
    """スワップション ATM ボラ面へ Hull-White の (a, sigma) をキャリブレート。

    Jamshidian エンジン + Levenberg-Marquardt。初期値 (a0, sigma0) から出発する。
    返り値は (a, sigma, helpers)。helpers は誤差診断に再利用する。
    """
    model = ql.HullWhite(ts, a0, sigma0)
    engine = ql.JamshidianSwaptionEngine(model)
    helpers = make_helpers(grid, engine)
    opt = ql.LevenbergMarquardt()
    end = ql.EndCriteria(max_iter, 50, 1e-8, 1e-8, 1e-8)
    model.calibrate(helpers, opt, end)
    a, sigma = model.params()
    return float(a), float(sigma), helpers


# %% [markdown]
# ### 既知パラメータの復元
#
# 正解 $(a^{*}, \sigma^{*}) = (0.06, 0.010)$ の Hull-White からボラ面を合成し、
# 既定の初期値でキャリブレーションする。復元されたパラメータが正解と一致すれば、
# キャリブレータは正しく動いている。

# %%
A_TRUE, SIGMA_TRUE = 0.06, 0.010
grid_syn = build_synthetic_surface(A_TRUE, SIGMA_TRUE)

vols_syn = np.array([v for (_, _, v) in grid_syn]).reshape(len(EXPIRIES), len(TENORS))
print("合成 ATM ボラ面（%表示）：")
print("           " + "".join(f"{te:>7d}y" for te in TENORS) + "  ← テナー")
for i, e in enumerate(EXPIRIES):
    print(f"満期 {e:>2d}y : " + "".join(f"{100 * vols_syn[i, j]:>7.2f} " for j in range(len(TENORS))))

a_hat, sigma_hat, helpers_syn = calibrate_hw(grid_syn)
print(f"\n復元パラメータ : a = {a_hat:.5f}, sigma = {sigma_hat:.5f}")
print(f"正解パラメータ : a = {A_TRUE:.5f}, sigma = {SIGMA_TRUE:.5f}")

# 合成面なので正解を高精度に復元できるはず。
assert abs(a_hat - A_TRUE) < 5e-3
assert abs(sigma_hat - SIGMA_TRUE) < 5e-4
print("既知パラメータを復元（キャリブレータの正しさを確認）")

# %% [markdown]
# ### 再現誤差の診断
#
# キャリブレーション後、各セルの **モデルボラ** と **市場ボラ** の差、および
# 価格差を集計する。合成面（正解が1ファクター Hull-White）では、再現誤差は
# 数値誤差の範囲に収まるはずである。満期×テナーの誤差を関数化しておく。

# %%
def reproduction_errors(grid, helpers):
    """各セルの (vol差[bp], 価格差) を満期×テナーの行列で返す。

    モデルボラは helper のモデル価格を implied vol へ逆算して得る。市場ボラは
    グリッドが持つクォート。差はモデル - 市場（bp）。
    """
    n_e, n_t = len(EXPIRIES), len(TENORS)
    vol_err = np.zeros((n_e, n_t))
    price_err = np.zeros((n_e, n_t))
    for k, (e, te, mkt_vol) in enumerate(grid):
        h = helpers[k]
        model_vol = h.impliedVolatility(h.modelValue(), 1e-6, 200, 1e-4, 2.0)
        i, j = EXPIRIES.index(e), TENORS.index(te)
        vol_err[i, j] = (model_vol - mkt_vol) * 1e4  # bp
        price_err[i, j] = h.modelValue() - h.marketValue()
    return vol_err, price_err


def rmse_bp(vol_err):
    return float(np.sqrt(np.mean(vol_err ** 2)))


vol_err_syn, price_err_syn = reproduction_errors(grid_syn, helpers_syn)
print(f"合成面の再現誤差 RMSE = {rmse_bp(vol_err_syn):.3f} bp")
print(f"最大絶対ボラ誤差       = {np.abs(vol_err_syn).max():.3f} bp")
print(f"最大絶対価格誤差       = {np.abs(price_err_syn).max():.2e}")

# 正解が1ファクター HW なので、再現誤差は数値精度レベル（< 5 bp）に収まる。
assert rmse_bp(vol_err_syn) < 5.0
print("再現誤差は数値精度レベル → 許容判定を通過")

# %% [markdown]
# ### 誤差ヒートマップ
#
# 満期×テナー別の再現誤差（ボラ差, bp）をヒートマップにする。合成面では全セルが
# ほぼゼロで、モデルが面を完全に説明できていることが目で見て分かる。

# %%
def plot_error_heatmap(vol_err, title, ax=None, vlim=None):
    """満期×テナーの再現誤差ヒートマップを描く。値をセルに数値表示する。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    vmax = vlim if vlim is not None else max(1.0, np.abs(vol_err).max())
    im = ax.imshow(vol_err, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(TENORS)))
    ax.set_xticklabels([f"{t}y" for t in TENORS])
    ax.set_yticks(range(len(EXPIRIES)))
    ax.set_yticklabels([f"{e}y" for e in EXPIRIES])
    ax.set_xlabel("tenor")
    ax.set_ylabel("expiry")
    ax.set_title(title)
    for i in range(vol_err.shape[0]):
        for j in range(vol_err.shape[1]):
            ax.text(j, i, f"{vol_err[i, j]:.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="model - market (bp)")
    return ax


plot_error_heatmap(vol_err_syn, "Synthetic HW surface: reproduction error (bp)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 初期値を変えた安定性確認
#
# 非線形最小化は初期値によって局所解に落ちうる。極端に離れた初期値をいくつか
# 与え、同じ最適解へ収束するかを見る。合成面では識別性が高く、どの初期値からでも
# 正解に戻るはずである。

# %%
init_points = [(0.01, 0.001), (0.20, 0.050), (0.50, 0.020), (0.005, 0.030)]
print(f"{'初期値 (a0, sigma0)':>22s} -> {'収束 a':>9s} {'収束 sigma':>11s}")
recovered = []
for a0, s0 in init_points:
    a_i, s_i, _ = calibrate_hw(grid_syn, a0=a0, sigma0=s0)
    recovered.append((a_i, s_i))
    print(f"  ({a0:.3f}, {s0:.3f})".rjust(22) + f" -> {a_i:>9.5f} {s_i:>11.5f}")

recovered = np.array(recovered)
spread_a = recovered[:, 0].max() - recovered[:, 0].min()
spread_s = recovered[:, 1].max() - recovered[:, 1].min()
print(f"\n収束先のばらつき : a の幅 = {spread_a:.2e}, sigma の幅 = {spread_s:.2e}")

# 局所解に散らばらず、単一解へ収束することを確認する。
assert spread_a < 5e-3 and spread_s < 5e-4
print("どの初期値からも同一解へ収束 → 局所解の心配は小さい（識別性が高い）")

# %% [markdown]
# ## 実データ適用
#
# スワップション ATM ボラは無料公開が乏しいため、ここでは合成の「市場」ボラ面を
# 一つの土台として使う。ただし今度は **Hull-White とは別の形** で面を作る。満期
# 方向にコブ（hump）を持ち、テナー方向に緩く減衰する、より現実に近い期間構造で
# ある。1ファクター Hull-White はこの面を完全には説明できないため、再現誤差に
# 構造が残る。ここに **1ファクターモデルの限界** が現れる。

# %%
def build_market_like_surface():
    """Hull-White 非整合な合成「市場」ATM ボラ面（bp単位ではなく小数）。

    満期方向に 3y 付近をピークとするコブ、テナー方向に対数的な減衰を与える。
    現実のスワップションボラに見られる期間構造の癖を模す（値はすべて合成）。
    """
    grid = []
    for e in EXPIRIES:
        for te in TENORS:
            base = 0.220
            hump = 0.030 * np.exp(-((np.log(e) - np.log(3.0)) ** 2) / 0.8)
            tenor_decay = -0.008 * np.log(te)
            grid.append((e, te, float(base + hump + tenor_decay)))
    return grid


grid_mkt = build_market_like_surface()
vols_mkt = np.array([v for (_, _, v) in grid_mkt]).reshape(len(EXPIRIES), len(TENORS))
print("合成『市場』ATM ボラ面（%表示）：")
print("           " + "".join(f"{te:>7d}y" for te in TENORS) + "  ← テナー")
for i, e in enumerate(EXPIRIES):
    print(f"満期 {e:>2d}y : " + "".join(f"{100 * vols_mkt[i, j]:>7.2f} " for j in range(len(TENORS))))

a_mkt, sigma_mkt, helpers_mkt = calibrate_hw(grid_mkt)
print(f"\nキャリブレート結果 : a = {a_mkt:.5f}, sigma = {sigma_mkt:.5f}")

# %% [markdown]
# キャリブレーション後の再現誤差を診断する。合成 Hull-White 面と違い、ボラ差の
# RMSE は大きく、しかも満期・テナー方向に系統的なパターンが残る。目的関数（価格
# 2乗和）は最善を尽くしているが、2パラメータでは面のコブと減衰を同時に再現でき
# ない。これが1ファクターモデルの構造的な限界である。

# %%
vol_err_mkt, price_err_mkt = reproduction_errors(grid_mkt, helpers_mkt)
print(f"合成『市場』面の再現誤差 RMSE = {rmse_bp(vol_err_mkt):.1f} bp")
print(f"最大絶対ボラ誤差              = {np.abs(vol_err_mkt).max():.1f} bp")

# 合成面（数値精度レベル）と比べ、桁違いに大きな構造的誤差が残る。
assert rmse_bp(vol_err_mkt) > 3 * rmse_bp(vol_err_syn)
print("1ファクター HW ではコブ付き面を説明しきれず、系統的な再現誤差が残る")

# %% [markdown]
# 2つの面（合成 Hull-White／合成「市場」）の再現誤差を並べて可視化する。左は
# ほぼ真っ白（誤差ゼロ）、右は満期方向のコブと符号の反転がはっきり残る。

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_error_heatmap(vol_err_syn, "HW-consistent surface (bp)", ax=axes[0])
plot_error_heatmap(vol_err_mkt, "Market-like surface (bp)", ax=axes[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 重み付けの効果
#
# 目的関数の重みを変えると、キャリブレーションが「どのセルを優先して合わせるか」
# が動く。ここでは満期 $\ge 5$ 年のセルを厚く重み付けし、長い満期の再現を改善
# する代わりに短い満期の誤差が増えることを見る。`SwaptionHelper` の重みは
# コンストラクタでは直接与えられないため、`CalibrationHelper` の重みを別に
# 積むのではなく、ここでは重み付けの「考え方」を、対象セルを絞った部分キャリブ
# レーションで示す。

# %%
def calibrate_subset(grid, keep):
    """keep(e, te) が真になるセルだけを使ってキャリブレートする。

    重み付けの極端な形（対象外セルの重み 0）に相当し、どのセルを重視するかで
    最適パラメータが動くことを示すための補助関数。
    """
    sub = [(e, te, v) for (e, te, v) in grid if keep(e, te)]
    a, s, _ = calibrate_hw(sub)
    return a, s


a_long, s_long = calibrate_subset(grid_mkt, lambda e, te: e >= 5)
a_short, s_short = calibrate_subset(grid_mkt, lambda e, te: e <= 2)
print(f"満期>=5y 重視 : a = {a_long:.5f}, sigma = {s_long:.5f}")
print(f"満期<=2y 重視 : a = {a_short:.5f}, sigma = {s_short:.5f}")
print(f"全セル均等    : a = {a_mkt:.5f}, sigma = {sigma_mkt:.5f}")
print("重み（どのセルを重視するか）で最適パラメータが動く → 目的関数設計が結果を左右する")

# %% [markdown]
# ## 演習
#
# 1. **満期・テナー別の再現誤差ヒートマップと1ファクターの限界。**
#    合成「市場」面（`build_market_like_surface`）へキャリブレーションし、
#    再現誤差ヒートマップ（ボラ差 bp）を描け。誤差が満期方向・テナー方向の
#    どちらに強い構造を持つかを読み取り、「なぜ1ファクター Hull-White では
#    この面を再現しきれないのか」を、$a$ と $\sigma$ の自由度（2つ）と
#    面の情報量（$5\times5$ セル）の関係から、自分の言葉で説明せよ。
#    さらに、コブの強さ（`hump` の係数）を変えたとき RMSE がどう動くかを調べよ。
#
# 2. **初期値・重み付けを変えたキャリブの安定性評価。**
#    (a) 合成 Hull-White 面と合成「市場」面の両方について、初期値 $(a_0, \sigma_0)$
#    を格子状に振り、収束後の $(a, \sigma)$ の散らばりを比較せよ。どちらの面で
#    局所解・不安定が出やすいかを述べよ。
#    (b) 満期を重視する／テナーを重視する、といった部分キャリブレーション
#    （`calibrate_subset` の一般化）で最適パラメータがどれだけ動くかを定量化し、
#    「価格差ベース」と「ボラ差ベース」で重みの効き方がどう違うと予想されるかを
#    考察せよ。
#
# 解答例は `solutions/S5/sol_0504.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/05_rate_models.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | キャリブレーション | calibration | モデルパラメータを市場価格に最も近づけるよう推定する作業 |
# | Levenberg-Marquardt法 | Levenberg-Marquardt | ガウス・ニュートンと最急降下を補間する非線形最小二乗法 |
# | 識別性 | identifiability | データからパラメータを一意に決められる度合い |
# | 局所解 | local minimum | 目的関数の谷のうち大域最小でないもの。初期値依存で落ちうる |
# | ATMボラ | at-the-money volatility | 行使レートをフォワードに一致させたときのインプライドボラ |
