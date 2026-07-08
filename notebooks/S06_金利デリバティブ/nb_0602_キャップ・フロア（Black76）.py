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
# # S6-2 キャップ/フロア（Black'76）
#
# ## 学習目標
#
# - キャップ（cap）をキャップレット（caplet）のポートフォリオへ分解し、1本ずつ
#   Black'76 で評価して合算できる
# - Black'76 をフォワード測度で正当化する導出を、S4-5 のニュメレール変換を実際に
#   使いながら追える
# - フラットボラ（flat vol）とスポットボラ（spot vol）の違いを説明し、フラット
#   ボラからスポットボラを剥ぎ取る（strip）計算を実装できる
# - 対数正規ボラと正規（Bachelier）ボラを相互変換でき、低金利・負金利で正規ボラが
#   好まれる理由を数値で示せる
# - `bondlab.pricing` の Black'76 / Bachelier / インプライドボラ / キャップレットを
#   スクラッチ実装と突合し、さらに QuantLib の `blackFormula` /
#   `bachelierBlackFormula` / `CapFloor` エンジンと機械精度で一致させられる
#
# S6-1 で金利スワップの評価を扱った。本 notebook はその延長として、金利の上限を
# 買う保険であるキャップ／フロアを、標準の Black'76 で値付けする。キャップレット
# 分解・ボラ表示・測度変換という、金利オプション実務の共通言語がここで揃う。


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# キャップ／フロアは、変動金利で借りている事業法人や住宅ローン提供者が金利上昇・低下に備えて買う保険です。銀行のレートデスクはこれを組成して売り、受け取ったプレミアムに対して、原資産である一連のフォワード金利のデルタを金利先物やスワップで、ボラティリティのリスクをスワップションで相殺します。デスクの収益は、顧客に提示するボラ（オファー）とヘッジに使う市場ボラ（ミッド）の差、およびヘッジ後に残るガンマ・ベガの見立てから生まれます。キャップをキャップレットに分解して1本ずつ Black'76 で値付けするのは、満期の異なる各オプションに別々のボラを当てて在庫リスクを満期ごとに管理するためで、フラットボラからスポットボラを剥ぎ取る計算は、この満期別リスク把握の前処理そのものです。
#
# ボラティリティを商品として売買するのがこのデスクの本質です。キャップ・スワップション・フロアはいずれもインプライドボラを介して価格が決まり、デスクは「市場が織り込むボラ」対「実現しそうなボラ」の差に賭けます。インプライドが高いと見ればオプションを売ってデルタヘッジしながらタイムディケイ（セータ）を取り、安いと見れば買ってガンマを取る。この売買はマーケットメイクのスプレッド収益にも、ボラの方向観に賭けるマクロ的なポジションにもなります。フラットボラ／スポットボラの区別を誤ると、どの満期にボラのロング・ショートが乗っているかを取り違え、意図しないベガのポジションを抱えます。
#
# 低金利・負金利環境で対数正規ボラから正規（Bachelier）ボラへ表示を切り替える論点は、実務では死活問題でした。フォワードがゼロ近傍・マイナスになると対数正規ボラは発散・定義不能になり、クォート・リスク集計・キャリブレーションが破綻します。日欧のマイナス金利局面で市場慣行が正規ボラ建てに移行したのはこのためで、両ボラの相互変換を数値で押さえておくことは、値付けだけでなくリスク管理システムの表示単位を選ぶ判断にも直結します。
# %% [markdown]
# ## 理論
#
# ### キャップ・フロア・キャップレット
#
# **キャップ（cap）** は、変動金利の借入に対する金利上限保険である。参照金利
# （例：3か月フォワード金利）が行使レート $K$ を上回った分だけ、各利払期に
# キャッシュフローを受け取る。1回の利払期に対応する1本のオプションを
# **キャップレット（caplet）** と呼ぶ。期間 $[T_{i-1}, T_i]$（日数比 $\tau_i$）の
# キャップレットのペイオフは、期末 $T_i$ に
#
# $$
# \text{caplet}_i = \tau_i \,\bigl(L_i - K\bigr)^{+}
# $$
#
# である。ここで $L_i$ は $T_{i-1}$ に確定する単利フォワード金利。**フロア
# （floor）** は下限保険で、ペイオフは $\tau_i (K - L_i)^{+}$ とプット型になる。
#
# キャップは各利払期のキャップレットを足し合わせたポートフォリオである。
#
# $$
# \text{Cap} = \sum_{i=1}^{n} \text{caplet}_i .
# $$
#
# 分解できることが実務上の要になる。各キャップレットは満期（フォワード金利の
# 確定時点）が異なる別々のオプションなので、原理的には満期ごとに違うボラを
# 当てられる。これが後述するスポットボラの話につながる。
#
# ### フォワード測度による Black'76 の正当化
#
# キャップレットは「$T_{i-1}$ で確定し $T_i$ で支払う」ため、割引の扱いが
# やや込み入る。ここで S4-5 の **ニュメレール変換** を実際に使う。
#
# リスク中立測度 $\mathbb{Q}$（ニュメレールは資産勘定 $B_t$）では、キャップレットの
# 現在価値は
#
# $$
# V_0 = \mathbb{E}^{\mathbb{Q}}\!\left[ \frac{B_0}{B_{T_i}} \,\tau_i (L_i - K)^{+} \right].
# $$
#
# 割引項 $1/B_{T_i}$ と $(L_i-K)^{+}$ が同じ期待値の中にあり、両者は一般に相関する
# ので、このままでは扱いにくい。そこで**ニュメレールを割引債 $P(t, T_i)$ に
# 取り替える**。$T_i$ 満期割引債をニュメレールとする測度を **$T_i$-フォワード
# 測度** $\mathbb{Q}^{T_i}$ と呼ぶ。測度変換の基本式（S4-5）は、任意のペイオフ
# $X_{T_i}$ について
#
# $$
# \mathbb{E}^{\mathbb{Q}}\!\left[\frac{B_0}{B_{T_i}} X_{T_i}\right]
# = P(0, T_i)\; \mathbb{E}^{\mathbb{Q}^{T_i}}\!\left[ X_{T_i} \right]
# $$
#
# である。ニュメレールを $B$ から $P(\cdot, T_i)$ へ替えると、割引係数が期待値の
# **外**に $P(0,T_i)$ として括り出せる。よって
#
# $$
# V_0 = P(0, T_i)\, \tau_i \, \mathbb{E}^{\mathbb{Q}^{T_i}}\!\left[ (L_i - K)^{+} \right].
# $$
#
# この測度変換の効き目は決定的である。**フォワード金利 $L_i$ は $T_i$-フォワード
# 測度の下でマルチンゲール**になる。実際 $L_i = (P(t,T_{i-1}) - P(t,T_i)) /
# (\tau_i P(t,T_i))$ は、$P(\cdot,T_i)$ を分母に持つ（トレード可能資産÷ニュメレール
# の）比なので、$\mathbb{Q}^{T_i}$ でドリフトを持たない。$L_i$ を対数正規、すなわち
#
# $$
# \frac{dL_t}{L_t} = \sigma \, dW_t^{T_i}
# $$
#
# と仮定すれば、$L_{T_i}$ は対数正規分布に従い、期待値 $\mathbb{E}^{\mathbb{Q}^{T_i}}[L_{T_i}]
# = L_0$（フォワードそのもの）となる。あとは Black-Scholes と同型の計算で
#
# $$
# \mathbb{E}^{\mathbb{Q}^{T_i}}[(L_{T_i}-K)^{+}]
# = L_0 \,\Phi(d_1) - K\,\Phi(d_2),\quad
# d_{1,2} = \frac{\ln(L_0/K) \pm \tfrac12 \sigma^2 T}{\sigma\sqrt{T}}
# $$
#
# を得る。これが **Black'76** である。株式の Black-Scholes と違い、スポット価格の
# 代わりにフォワード $L_0$ が入り、割引は測度変換で括り出した $P(0,T_i)$ が担う。
# ここで $T$ はフォワードが確定する時点（キャップレットの満期）$T_{i-1}$ を使う。
# したがってキャップレット1本は
#
# $$
# \text{caplet}_i = \tau_i \, P(0, T_i)\,\bigl[L_0^{(i)}\Phi(d_1) - K\,\Phi(d_2)\bigr].
# $$
#
# 実装では `black76(forward, strike, expiry, vol, df)` が $\Phi$ の部分を、
# `caplet(...)` が $\tau_i$ 倍と割引をまとめて担う。
#
# ### フラットボラ vs スポットボラ
#
# 市場は個々のキャップレットではなく、**キャップ全体**を1つのボラでクォートする。
# ある満期のキャップを、全キャップレットに**同一の**ボラ $\sigma^{\text{flat}}$ を
# 当てて価格が合うように決めた値が **フラットボラ（flat volatility）** である。
# 満期の違うキャップごとに1つずつ与えられ、期間構造をなす。
#
# 一方、キャップレットを正しく評価するには、満期ごとに異なる **スポットボラ
# （spot / forward-forward volatility）** $\sigma_i^{\text{spot}}$ が必要になる。両者は
# 「同じキャップ価格を再現する」点で結ばれる。満期 $T_m$ のキャップについて
#
# $$
# \sum_{i=1}^{m} \text{caplet}_i\bigl(\sigma^{\text{flat}}(T_m)\bigr)
# = \sum_{i=1}^{m} \text{caplet}_i\bigl(\sigma_i^{\text{spot}}\bigr).
# $$
#
# 左辺（フラット）から右辺（スポット）を、満期の短い順に1本ずつ解いていく操作が
# **ボラの剥ぎ取り（vol stripping）** である。カーブのブートストラップ（S2）と同じ
# 逐次求解の構造を持つ。フラットボラは「平均的な」ボラなので、スポットボラが
# 期間方向に傾いていると、フラットボラとスポットボラは体系的にずれる。
#
# ### 正規（Bachelier）ボラとの変換
#
# Black'76 は金利を**対数正規**（$dL/L = \sigma dW$）と仮定する。対して
# **Bachelier モデル（正規モデル）** は金利を**正規**（$dL = \sigma_N dW$、加法的）と
# 仮定する。正規ボラ $\sigma_N$（bp 単位）でのキャップレットは
#
# $$
# \text{caplet}_i^{N} = \tau_i P(0,T_i)\,\Bigl[(L_0 - K)\Phi(d) + \sigma_N\sqrt{T}\,\phi(d)\Bigr],
# \quad d = \frac{L_0 - K}{\sigma_N\sqrt{T}} .
# $$
#
# 同じ市場価格は、対数正規ボラでも正規ボラでも表せる。ATM 近傍では両者は
#
# $$
# \sigma_N \approx \sigma_{\text{LN}} \cdot L_0
# $$
#
# という素朴な関係で結ばれる（フォワード水準を掛けるだけ）。正確な変換は、片方の
# ボラで価格を出し、もう片方のモデルでその価格に一致するボラを逆算（インプライド）
# すればよい。本 notebook ではこの「価格経由」の変換を実装する。
#
# ### 負金利下で Black が破綻する様子
#
# 対数正規の仮定は $\ln(L_0/K)$ を含むため、フォワード $L_0$ か行使 $K$ が
# **負またはゼロ**になると定義できない。$d_1, d_2$ が発散・NaN になり、Black'76 は
# 破綻する。日欧の低金利・負金利局面では、これが現実の障害になった。正規モデルは
# $L_0 - K$ という**差**しか使わないため、金利が負でも滑らかに機能する。そのため
# 低金利環境では**正規ボラでのクォートが標準**になった。この破綻と代替を、後段で
# 数値により示す。

# %% [markdown]
# ## スクラッチ実装
#
# まず Black'76・Bachelier・インプライドボラ・キャップレットを numpy/scipy だけで
# 自作し、その後 `bondlab.pricing` と突合する。自作関数の仕様を表で示す。
#
# ### 自作関数の仕様
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `my_black76(F, K, T, vol, df, opt)` | フォワード, 行使, 満期, 対数正規ボラ, 割引係数, "call"/"put" | オプション価格 | Black'76 の閉形式 |
# | `my_bachelier(F, K, T, nvol, df, opt)` | フォワード, 行使, 満期, 正規ボラ, 割引係数, "call"/"put" | オプション価格 | 正規モデルの閉形式 |
# | `my_implied_vol(price, F, K, T, df, opt)` | 価格, フォワード, 行使, 満期, 割引係数, "call"/"put" | 対数正規ボラ | Black 価格を反転してボラを求める |
# | `my_caplet(F, K, T, vol, tau, df, model)` | フォワード, 行使, 満期, ボラ, 日数比, 割引係数, "black"/"bachelier" | キャップレット価格 | $\tau$ 倍＋割引でキャップレット化 |

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy import stats
from scipy.optimize import brentq

import bondlab
from bondlab import pricing
from bondlab.curve import bootstrap_par

print("bondlab version:", bondlab.__version__)

SEED = 20260707
rng = np.random.default_rng(SEED)
plt.rcParams["axes.unicode_minus"] = False


def my_black76(F, K, T, vol, df=1.0, opt="call"):
    """Black'76 の閉形式。フォワードを対数正規と仮定したコール/プット価格。"""
    if T <= 0 or vol <= 0:
        payoff = max(F - K, 0.0) if opt == "call" else max(K - F, 0.0)
        return df * payoff
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if opt == "call":
        return df * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    return df * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))


def my_bachelier(F, K, T, nvol, df=1.0, opt="call"):
    """Bachelier（正規）モデルの閉形式。金利を正規と仮定した価格。"""
    if T <= 0 or nvol <= 0:
        payoff = max(F - K, 0.0) if opt == "call" else max(K - F, 0.0)
        return df * payoff
    d = (F - K) / (nvol * np.sqrt(T))
    sign = 1.0 if opt == "call" else -1.0
    return df * (sign * (F - K) * stats.norm.cdf(sign * d) + nvol * np.sqrt(T) * stats.norm.pdf(d))


def my_implied_vol(price, F, K, T, df=1.0, opt="call"):
    """Black'76 価格を反転して対数正規ボラを求める（Brent 法）。"""
    intrinsic = df * (max(F - K, 0.0) if opt == "call" else max(K - F, 0.0))
    if price <= intrinsic + 1e-14:
        return 0.0

    def objective(s):
        return my_black76(F, K, T, s, df, opt) - price

    return brentq(objective, 1e-8, 5.0, xtol=1e-10, maxiter=200)


def my_caplet(F, K, T, vol, tau, df, model="black"):
    """キャップレット価格。tau は日数比、df は期末までの割引係数。"""
    fn = my_black76 if model == "black" else my_bachelier
    return tau * fn(F, K, T, vol, df, opt="call")


# 代表点で bondlab と一致するか確認する。
F, K, T, vol, df = 0.030, 0.032, 2.0, 0.25, 0.95
print("Black'76      自作 =", my_black76(F, K, T, vol, df), " bondlab =", pricing.black76(F, K, T, vol, df))
print("Bachelier     自作 =", my_bachelier(F, K, T, 0.008, df), " bondlab =", pricing.bachelier(F, K, T, 0.008, df))
print("implied vol   自作 =", my_implied_vol(pricing.black76(F, K, T, vol, df), F, K, T, df),
      " bondlab =", pricing.implied_vol_black(pricing.black76(F, K, T, vol, df), F, K, T, df))
print("caplet        自作 =", my_caplet(F, K, T, vol, 0.5, df), " bondlab =", pricing.caplet(F, K, T, vol, 0.5, df))

# %% [markdown]
# 代表点だけでなく、行使レートの帯全体で機械精度の一致を `assert` で守る。プット
# コール・パリティ（$C - P = df\,(F-K)$）も同時に確認する。

# %%
strikes = np.linspace(0.010, 0.060, 21)
for kk in strikes:
    assert abs(my_black76(F, kk, T, vol, df) - pricing.black76(F, kk, T, vol, df)) < 1e-14
    assert abs(my_bachelier(F, kk, T, 0.008, df) - pricing.bachelier(F, kk, T, 0.008, df)) < 1e-14
    # インプライドボラの往復：価格→ボラ→価格が元に戻る
    price = pricing.black76(F, kk, T, vol, df)
    iv = my_implied_vol(price, F, kk, T, df)
    assert abs(iv - vol) < 1e-8
    # プットコール・パリティ
    call = my_black76(F, kk, T, vol, df)
    put = my_black76(F, kk, T, vol, df, "put")
    assert abs((call - put) - df * (F - kk)) < 1e-12

print("行使レート帯全体で bondlab と一致・パリティ・ボラ往復を確認しました")

# %% [markdown]
# ### キャップをキャップレットに分解して評価
#
# 四半期利払いの2年キャップを想定する。各期のフォワードは、合成カーブから
# 単利フォワード $L_i = (P(T_{i-1})/P(T_i)-1)/\tau_i$ で作る。キャップ価格は、
# キャップレットを1本ずつ Black'76 で評価して合算したものになる。

# %%
par_tenors = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
par_rates = np.array([0.030, 0.032, 0.034, 0.0355, 0.0365, 0.038, 0.0395])
curve = bootstrap_par(par_tenors, par_rates, frequency=1)

# 四半期グリッド（フォワードスタートで最初のキャップレットも将来確定にする）
tau = 0.25
reset_times = np.arange(0.5, 2.5, tau)          # 各キャップレットのフォワード確定時点
pay_times = reset_times + tau                    # 支払時点


def forward_rates(curve, reset_times, tau):
    """単利フォワード L_i = (P(T_{i-1})/P(T_i) - 1)/tau を各期について返す。"""
    p_start = curve.discount(reset_times)
    p_end = curve.discount(reset_times + tau)
    return (p_start / p_end - 1.0) / tau


fwds = forward_rates(curve, reset_times, tau)
dfs_pay = curve.discount(pay_times)
K_cap = 0.035
flat_vol = 0.30

caplets = np.array([
    pricing.caplet(f, K_cap, T_exp, flat_vol, tau, dfp, "black")
    for f, T_exp, dfp in zip(fwds, reset_times, dfs_pay)
])
cap_price = caplets.sum()
print(f"フォワード金利（%）: {np.round(fwds * 100, 3)}")
print(f"キャップレット価格 : {np.round(caplets * 1e4, 3)} (×1e4)")
print(f"2年キャップ価格    : {cap_price:.6f}  (= キャップレットの合計)")

# %%
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(reset_times, caplets * 1e4, width=0.18)
ax.set_xlabel("キャップレット満期（年）")
ax.set_ylabel("価格 ×1e4")
ax.set_title("2年キャップのキャップレット分解（K=3.5%, flat vol=30%）")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## QuantLib検証
#
# `blackFormula` / `bachelierBlackFormula` で単一オプションを、`CapFloor` エンジンで
# キャップ全体を突合する。いずれも機械精度で一致することを確認する。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

# 単一オプション：df を掛けた blackFormula / bachelierBlackFormula と一致するか
for kk in [0.025, 0.035, 0.045]:
    b_mine = pricing.black76(F, kk, T, vol, df, "call")
    b_ql = df * ql.blackFormula(ql.Option.Call, kk, F, vol * np.sqrt(T))
    n_mine = pricing.bachelier(F, kk, T, 0.008, df, "call")
    n_ql = df * ql.bachelierBlackFormula(ql.Option.Call, kk, F, 0.008 * np.sqrt(T))
    print(f"K={kk:.3f}  black diff={abs(b_mine - b_ql):.2e}  bachelier diff={abs(n_mine - n_ql):.2e}")
    assert abs(b_mine - b_ql) < 1e-14
    assert abs(n_mine - n_ql) < 1e-14

# %% [markdown]
# ### キャップ全体：QuantLib の `CapFloor` エンジン
#
# QuantLib で同じ合成カーブから割引カーブを組み、`Cap` を `BlackCapFloorEngine` /
# `BachelierCapFloorEngine` で評価する。bondlab のキャップレット分解と突合する。

# %%
DC = ql.Actual365Fixed()
TODAY = ql.Date(1, 1, 2024)
ql.Settings.instance().evaluationDate = TODAY


def bondlab_to_ql_curve(curve, max_t=12.0, n_nodes=48):
    """bondlab の DiscountCurve を QuantLib の割引カーブハンドルへ写す。"""
    node_t = np.concatenate([[0.0], np.linspace(0.25, max_t, n_nodes)])
    dates = [TODAY + ql.Period(int(round(t * 365)), ql.Days) for t in node_t]
    node_dfs = [1.0] + [float(curve.discount(t)) for t in node_t[1:]]
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, node_dfs, DC))


ts = bondlab_to_ql_curve(curve)
index = ql.USDLibor(ql.Period(3, ql.Months), ts)

start = TODAY + ql.Period(6, ql.Months)          # フォワードスタート（既定済み固定を避ける）
end = start + ql.Period(2, ql.Years)
schedule = ql.Schedule(
    start, end, ql.Period(3, ql.Months), ql.NullCalendar(),
    ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
)
ibor_leg = ql.IborLeg([1.0], schedule, index)
ql_cap = ql.Cap(ibor_leg, [K_cap])


def caplet_decomposition(leg, ts, strike, vol, model="black"):
    """QuantLib のクーポン脚から、bondlab のキャップレット和でキャップを評価する。"""
    total = 0.0
    for cf in map(ql.as_floating_rate_coupon, leg):
        if cf is None or cf.fixingDate() <= TODAY:
            continue
        f = cf.indexFixing()
        tau_i = cf.accrualPeriod()
        T_exp = DC.yearFraction(TODAY, cf.fixingDate())
        df_pay = ts.discount(cf.date())
        total += pricing.caplet(f, strike, T_exp, vol, tau_i, df_pay, model)
    return total


# Black
ql_cap.setPricingEngine(ql.BlackCapFloorEngine(ts, ql.QuoteHandle(ql.SimpleQuote(flat_vol)), DC))
ql_black = ql_cap.NPV()
bl_black = caplet_decomposition(ibor_leg, ts, K_cap, flat_vol, "black")
print(f"[Black]     QuantLib = {ql_black:.10f}  bondlab分解 = {bl_black:.10f}  diff = {abs(ql_black - bl_black):.2e}")
assert abs(ql_black - bl_black) < 1e-12

# Bachelier
nvol_cap = 0.010
ql_cap.setPricingEngine(ql.BachelierCapFloorEngine(ts, ql.QuoteHandle(ql.SimpleQuote(nvol_cap))))
ql_bach = ql_cap.NPV()
bl_bach = caplet_decomposition(ibor_leg, ts, K_cap, nvol_cap, "bachelier")
print(f"[Bachelier] QuantLib = {ql_bach:.10f}  bondlab分解 = {bl_bach:.10f}  diff = {abs(ql_bach - bl_bach):.2e}")
assert abs(ql_bach - bl_bach) < 1e-12
print("キャップ全体で QuantLib と機械精度一致を確認しました")

# %% [markdown]
# ## 実データ適用
#
# 合成カーブ上の同一キャップを、対数正規ボラと正規ボラの**両方**で表現する。
# さらに、低金利のカーブへ差し替えて「なぜ正規ボラが好まれるか」を数値で示し、
# フラットボラからスポットボラを剥ぎ取る計算を実装する。
#
# ### 同一価格を対数正規ボラ・正規ボラで表現する
#
# キャップを対数正規フラットボラ 30% で評価し、その価格に一致する正規ボラを
# 逆算する。ATM 近傍の目安 $\sigma_N \approx \sigma_{\text{LN}} \cdot L_0$ と比べる。

# %%
def cap_price_black(fwds, K, expiries, vol, tau, dfs_pay):
    return sum(pricing.caplet(f, K, T, vol, tau, dp, "black")
               for f, T, dp in zip(fwds, expiries, dfs_pay))


def cap_price_bachelier(fwds, K, expiries, nvol, tau, dfs_pay):
    return sum(pricing.caplet(f, K, T, nvol, tau, dp, "bachelier")
               for f, T, dp in zip(fwds, expiries, dfs_pay))


target = cap_price_black(fwds, K_cap, reset_times, flat_vol, tau, dfs_pay)
# その価格に一致する正規フラットボラを Brent 法で逆算する
implied_normal = brentq(
    lambda nv: cap_price_bachelier(fwds, K_cap, reset_times, nv, tau, dfs_pay) - target,
    1e-6, 0.05, xtol=1e-12,
)
approx_normal = flat_vol * fwds.mean()
print(f"対数正規フラットボラ : {flat_vol:.4f}")
print(f"同価格の正規フラットボラ（逆算） : {implied_normal:.6f}  ({implied_normal * 1e4:.1f} bp)")
print(f"素朴な目安 σ_LN·L0            : {approx_normal:.6f}  ({approx_normal * 1e4:.1f} bp)")
print(f"キャップ価格（両表現で一致）  : {target:.8f}")

# %% [markdown]
# ### 低金利・負金利で正規ボラが好まれる理由
#
# フォワード水準を段階的に下げ、キャップ価格を一定に保ったまま、対数正規ボラと
# 正規ボラがどう振る舞うかを見る。対数正規ボラは水準の低下とともに跳ね上がり、
# フォワードが $K$ に近づくと不安定化する。正規ボラは水準に対してほぼ平坦で、
# 負金利へ入っても定義され続ける。

# %%
levels = np.array([0.030, 0.020, 0.010, 0.005, 0.000, -0.005])
K_atmish = 0.005                                   # 低水準でも意味を持つよう低めの行使
records = []
for lv in levels:
    shifted = fwds - fwds.mean() + lv              # フォワードの平均水準を lv にずらす
    # 価格は正規ボラ 60bp を真値として作る（負金利でも必ず定義できる）
    true_nvol = 0.0060
    px = cap_price_bachelier(shifted, K_atmish, reset_times, true_nvol, tau, dfs_pay)
    # 対数正規ボラを逆算（フォワード>0 かつ価格>本源的価値のときのみ可能）
    try:
        if np.all(shifted > 0):
            ln_vol = brentq(
                lambda v: cap_price_black(shifted, K_atmish, reset_times, v, tau, dfs_pay) - px,
                1e-6, 5.0, xtol=1e-10,
            )
        else:
            ln_vol = np.nan
    except (ValueError, ZeroDivisionError):
        ln_vol = np.nan
    records.append((lv, px, ln_vol, true_nvol))

print(f"{'平均F':>8} {'キャップ価格':>14} {'対数正規ボラ':>14} {'正規ボラ(bp)':>12}")
for lv, px, ln_vol, nv in records:
    ln_disp = "破綻(NaN)" if np.isnan(ln_vol) else f"{ln_vol:12.4f}"
    print(f"{lv:8.3f} {px:14.8f} {ln_disp:>14} {nv * 1e4:12.1f}")

# %%
fig, ax1 = plt.subplots(figsize=(7.2, 3.6))
lv_arr = np.array([r[0] for r in records])
ln_arr = np.array([r[2] for r in records])
nv_arr = np.array([r[3] for r in records]) * 1e4
ax1.plot(lv_arr, ln_arr, "o-", color="C3", label="対数正規ボラ（Black）")
ax1.set_xlabel("フォワードの平均水準")
ax1.set_ylabel("対数正規ボラ", color="C3")
ax1.axvline(0.0, color="gray", ls=":", lw=1)
ax2 = ax1.twinx()
ax2.plot(lv_arr, nv_arr, "s--", color="C0", label="正規ボラ（bp）")
ax2.set_ylabel("正規ボラ（bp）", color="C0")
ax1.set_title("低金利で対数正規ボラは発散し、負金利では破綻する")
fig.tight_layout()
plt.show()

# %% [markdown]
# 価格は一定なのに、対数正規ボラだけが水準低下とともに膨れ上がり、フォワードが
# ゼロ以下に入ると値そのものが定義できなくなる（表の「破綻」）。正規ボラは
# 一定水準を保つ。低金利・負金利では正規ボラが標準になる理由がこれである。
#
# ### フラットボラ⇔スポットボラ変換
#
# 満期の異なるキャップに与えたフラットボラから、キャップレット1本ずつの
# スポットボラを剥ぎ取る。満期の短い順に、直前の満期までのキャップとの差
# （＝その満期に新しく増えたキャップレット群）が、スポットボラで再現できる
# ように逐次求解する。

# %%
def strip_spot_vols(fwds, expiries, pay_dfs, tau, strike, flat_vols):
    """フラットボラ列からスポットボラ列を剥ぎ取る（キャップレット単位）。

    キャップ満期 = 各キャップレット満期とみなし、i 番目のキャップと i-1 番目の
    キャップの価格差を、i 番目のスポットボラで一致させる。
    """
    n = len(fwds)
    spot = np.empty(n)
    cum_price = 0.0                                # スポットボラで積み上げたキャップ価格
    for i in range(n):
        # フラットボラで測ったキャップ(T_i)の価格（先頭から i 本すべてを同一ボラで）
        cap_i = sum(
            pricing.caplet(fwds[j], strike, expiries[j], flat_vols[i], tau, pay_dfs[j], "black")
            for j in range(i + 1)
        )
        target_caplet = cap_i - cum_price          # i 番目キャップレットが埋めるべき価格
        f, T_exp, dp = fwds[i], expiries[i], pay_dfs[i]
        spot[i] = brentq(
            lambda v: pricing.caplet(f, strike, T_exp, v, tau, dp, "black") - target_caplet,
            1e-6, 5.0, xtol=1e-10,
        )
        cum_price += pricing.caplet(f, strike, T_exp, spot[i], tau, dp, "black")
    return spot


# 期間方向に傾いたフラットボラ（短期高め・長期低め）を与える
flat_curve = 0.34 - 0.05 * (reset_times - reset_times.min())
spot_curve = strip_spot_vols(fwds, reset_times, dfs_pay, tau, K_cap, flat_curve)

# 検証：剥ぎ取ったスポットボラで各満期キャップを組み直すと、フラットボラ価格に戻る
for i in range(len(fwds)):
    cap_flat = sum(pricing.caplet(fwds[j], K_cap, reset_times[j], flat_curve[i], tau, dfs_pay[j], "black")
                   for j in range(i + 1))
    cap_spot = sum(pricing.caplet(fwds[j], K_cap, reset_times[j], spot_curve[j], tau, dfs_pay[j], "black")
                   for j in range(i + 1))
    assert abs(cap_flat - cap_spot) < 1e-10
print("剥ぎ取り後のスポットボラで全満期キャップが再現されることを確認しました")

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(reset_times, flat_curve * 100, "o-", label="フラットボラ（入力）")
ax.plot(reset_times, spot_curve * 100, "s--", label="スポットボラ（剥ぎ取り）")
ax.set_xlabel("キャップレット満期（年）")
ax.set_ylabel("ボラ（%）")
ax.set_title("フラットボラからスポットボラを剥ぎ取る")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# スポットボラはフラットボラより傾きが強く出る。フラットボラが「先頭からの平均」
# であるのに対し、スポットボラは各満期の限界的な情報だからである。カーブが右下がり
# のとき、スポットボラはフラットボラの下側へ、より急に伸びる。

# %% [markdown]
# ## 演習
#
# 1. **フラットボラからスポットボラを剥ぎ取る。**
#    本編の `strip_spot_vols` を出発点に、以下を行え。
#    (a) 右上がり（短期低め・長期高め）のフラットボラ曲線を与え、剥ぎ取った
#    スポットボラがフラットボラの**上側**に、より急に伸びることを図で示せ。
#    (b) 剥ぎ取ったスポットボラで各満期のキャップを組み直し、入力のフラットボラで
#    評価したキャップ価格と機械精度で一致することを `assert` で確認せよ。
#    (c) フラットボラが**平坦**（全満期同一）なとき、スポットボラも平坦になるか。
#    数値で確かめ、その理由をキャップレット分解の観点から一言で述べよ。
#
# 2. **低金利・負金利で Black が壊れ Bachelier が機能する。**
#    ATM キャップレット1本について、フォワード $L_0$ を $+3\%$ から $-1\%$ まで
#    動かし、次を示せ。
#    (a) 正規ボラ 60bp を真値として価格を作り、その価格から対数正規ボラを逆算する。
#    $L_0 \le 0$ で Black が定義できず（NaN・例外）、$L_0 \to 0^{+}$ で対数正規ボラが
#    発散することを図示せよ。
#    (b) 同じ価格系列に対し、Bachelier の正規ボラ逆算は負金利まで滑らかに定義され
#    続けることを示せ。
#    (c) 「なぜ正規モデルは負金利で壊れないのか」を、価格式が $L_0$ と $K$ を
#    **差** $(L_0-K)$ でしか使わない点から説明せよ。
#
# 解答例は `solutions/S6/sol_0602.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/06_derivatives.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | Black'76 | Black'76 | フォワードを対数正規と仮定した金利オプションの標準評価式 |
# | キャップレット | caplet | キャップを構成する1利払期分のコール型金利オプション |
# | インプライドボラティリティ | implied volatility | 市場価格をモデルに反転して得るボラ。フラット／スポットの別がある |
# | Bachelierモデル | Bachelier model | 金利を正規（加法的）と仮定する評価モデル。低・負金利で機能する |
# | フォワード測度 | forward measure | 満期 T の割引債をニュメレールとする測度。フォワードがマルチンゲール |
