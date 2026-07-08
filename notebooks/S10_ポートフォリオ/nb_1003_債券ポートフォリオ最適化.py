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
# # S10-3 債券ポートフォリオ最適化
#
# ## 学習目標
#
# - 平均分散法（mean-variance）を債券にそのまま当てはめる難しさを、リターン
#   分布の非対称性・期待リターンの推定誤差・共分散行列の悪条件の3点から説明できる
# - 少数の因子（KRD・スプレッドDV01）で銘柄のリスクを張る**ファクターリスク
#   モデル**を組み、フル共分散の代わりにベンチマークとの因子エクスポージャ差を
#   制約する設計を理解する
# - 取引コスト・回転率制約とロバスト最適化の考え方を、線形計画の枠内で表現できる
# - 制約付き最適化を cvxpy で解き、**バインドしている制約**と**シャドープライス
#   （双対変数）**を読み取って、最適解を経済的に説明できる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# アクティブ債券運用会社は、デュレーション・カーブ形状・クレジットスプレッドについて自分の見立てを持ち、その見立てをベンチマーク対比のポジションに落として超過リターン（アルファ）を狙います。稼ぎの源泉は、当てた見立てが実現したときの超過リターンと、それに対して受け取る運用報酬です。ただし年金・保険・投信の受託運用では、ベンチマークから大きく離れること自体がリスク（顧客との約束違反）なので、「見立ての表現」と「制約の遵守」を同時に成立させる必要があります。ここで最適化が中身のある道具になります。素朴な平均分散法を債券にそのまま当てると、期待リターンの推定誤差が最適化で増幅され（error maximization）、同一カーブ上の利回りが強く相関するために共分散が悪条件になって、わずかな入力差で極端なロング・ショートが噴き出します。実務でこれをそのまま資産に張れば、見立てではなく推定ノイズに賭けることになります。
#
# そこで、フル共分散を推定・反転する代わりに、リスクを少数の観測可能な因子——キーレートデュレーション（KRD）とスプレッドDV01——のエクスポージャで表すファクターリスクモデルを使います。ポートフォリオの因子エクスポージャ $B^\top w$ をベンチマークの水準からどれだけ離すかを制約すれば、「10年の金利は下がると読むから 10年 KRD をベンチマーク＋αにする」「クレジットは締まると読むからスプレッドDV01を積む」といった見立てを、他の年限やセクターのリスクを取らずにピンポイントで表現できます。アクティブリスクを因子の言葉で明示的に管理できることが、単なる山勘との違いです。
#
# さらに、売買には取引コストと回転率の制約がかかり、推定リターンの不確実性に対するロバスト最適化も要ります。これらを線形計画・凸計画の枠内で表現し cvxpy で解くと、最適解だけでなく**どの制約がバインドしているか**と**シャドープライス（双対変数）**が得られます。シャドープライスは「回転率上限を 1 単位緩めれば期待効用がいくら増えるか」を示すので、制約を緩めて取りに行くべきアルファと、コストに見合わないアルファを切り分けられます。これは運用会社が超過リターンとコスト・リスク予算を天秤にかける意思決定そのものであり、免疫化（S10-1）や複製（S10-2）が「合わせる」制約最適化だったのに対し、こちらは「制約の中でリターンを最大化する」最適化として同じ枠組みの延長にあります（docs/債券ファンドの業務.md「運用会社・ALM」の枠組み）。
# %% [markdown]
# ## 理論
#
# ### 平均分散法（mean-variance optimization）を債券に当てる難しさ
#
# Markowitz の平均分散法は、期待リターン $\mu\in\mathbb{R}^N$ と共分散行列
# $\Sigma\in\mathbb{R}^{N\times N}$ を所与として
#
# $$ \max_{w}\; \mu^\top w - \tfrac{\gamma}{2}\, w^\top \Sigma w
#    \quad\text{s.t.}\quad \mathbf{1}^\top w = 1 $$
#
# を解く。株式では標準的でも、債券にそのまま持ち込むと次の3つで破綻しやすい。
#
# **(1) リターン分布の非対称性。** 債券価格は満期に額面へ収束する
# （pull-to-par）。価格の上振れには「額面＋残りクーポン」という上限があり、
# 下振れ（デフォルト・急激な金利上昇）の裾のほうが厚い。さらに価格・利回り関係の
# 凸性（convexity）が非線形性を生む。分散という2次モーメントは、こうした歪んだ
# 分布の裾リスクを取り逃す。
#
# **(2) 期待リターンの推定誤差。** $\mu$ の推定誤差は最適化を通じて増幅される。
# 最適化は「推定 $\hat\mu$ が高い資産」へ荷重を寄せるが、その高さが推定誤差
# 由来なら、最適化は誤差そのものを最大化する（Michaud のいう
# *error maximization*）。債券の期待超過リターンはキャリー・ロールダウン・
# スプレッド収束など小さな源泉の積み上げで、$\hat\mu$ の相対誤差は大きい。
#
# **(3) 共分散行列の悪条件。** 同一カーブ上の国債利回りは強く相関し、変動の
# ほとんどは水準・傾き・曲率の少数因子（S3-4 の主成分）で説明される。すると
# $\Sigma$ はほぼ低ランクで**悪条件**（near-singular）になり、$\Sigma^{-1}$ が
# 暴れて、わずかな入力差で極端なロング・ショートが出る。標本共分散を短い履歴で
# 推定すればなおさらで、これは演習2で数値的に確認する。
#
# ### ファクターリスクモデル（factor risk model）
#
# そこで、フル共分散を推定・反転する代わりに、リスクを少数の**観測可能な因子**の
# エクスポージャで表す。各銘柄 $i$ の因子負荷ベクトルを $b_i\in\mathbb{R}^K$ と
# し、負荷行列を $B=[b_1,\dots,b_N]^\top\in\mathbb{R}^{N\times K}$ とおく。
# ポートフォリオ $w$ の因子エクスポージャは $B^\top w$ である。本 notebook では
# 因子として次を使う。
#
# - **キーレート・デュレーション（KRD）**：カーブ上の代表テナー（2,5,10,20,30年）
#   のゼロレートを局所的に動かしたときの価格感応度。カーブの水準だけでなく
#   **形状変化**（スティープ化・フラット化・曲率）への感応を分解して測る。
# - **スプレッドDV01（spread DV01）**：カーブ全体に対する平行なスプレッド拡大
#   1bp あたりの価格変化。信用・セクターのスプレッドリスクを1本で捉える因子。
#
# リスク管理は「$\Sigma$ を最小化する」ではなく「因子エクスポージャ $B^\top w$ を
# ベンチマーク $B^\top w_b$ に十分近づける」に置き換わる。因子でマッチすれば、
# 因子で説明される変動はベンチマークと相殺して消え、残るのは因子外の残差
# リスクだけになる。共分散の反転を避けられるので、悪条件・推定誤差の問題を
# 大きく緩和できる。
#
# ### 取引コストと回転率制約（turnover constraint）
#
# 現ポートフォリオ $w_0$ から目標 $w$ へ組み替えるとき、売買には手数料・
# ビッドアスクがかかる。線形の取引コストは $\sum_i \kappa_i |w_i - w_{0,i}|$ で、
# 目的関数に $\ell_1$ ペナルティとして足せる。取引総額そのものを縛るなら
# **回転率**（turnover）を制約にする。片道回転率を
#
# $$ \text{turnover}(w) = \tfrac{1}{2}\sum_i |w_i - w_{0,i}| $$
#
# と定義し、$\text{turnover}(w)\le \tau$ とする。$\tfrac12$ は買い増し総額と
# 売り減らし総額が（予算制約 $\mathbf 1^\top w=\mathbf 1^\top w_0=1$ の下で）
# 等しいことによる。$\ell_1$ は凸なので、この制約は凸のまま扱える。
#
# ### ロバスト最適化（robust optimization）入門
#
# 期待リターン $\mu$ が誤差を含むなら、点推定 $\hat\mu$ を信じずに、$\mu$ が
# ある不確実性集合 $\mathcal U$ を動くとして**最悪ケース**を最大化する。箱型
# $\mathcal U=\{\mu:\ |\mu_i-\hat\mu_i|\le\delta_i\}$ なら
#
# $$ \max_w\ \min_{\mu\in\mathcal U}\ \mu^\top w
#    = \max_w\ \big(\hat\mu^\top w - \delta^\top |w|\big) $$
#
# となり、推定に自信のない銘柄（$\delta_i$ 大）への荷重にペナルティが付く。
# 楕円型 $\mathcal U=\{\mu:\ (\mu-\hat\mu)^\top\Omega^{-1}(\mu-\hat\mu)\le\rho^2\}$
# なら最悪ケース項は $-\rho\,\lVert \Omega^{1/2} w\rVert_2$ で、これは分散
# ペナルティと同じ形の縮約になる。いずれも凸のまま解ける点が実務的な利点である。
#
# ### 双対変数（シャドープライス）の読み方
#
# 制約 $g(w)\le b$ 付きの最大化で最適値を $p^\star(b)$ とすると、対応する
# 双対変数（ラグランジュ乗数）$\lambda\ge 0$ は
#
# $$ \lambda \;=\; \frac{\partial p^\star}{\partial b} $$
#
# である（包絡線定理, envelope theorem）。すなわち $\lambda$ は「制約の右辺を
# 1単位ゆるめたら目的関数がどれだけ増えるか」を表す**シャドープライス**（影の
# 価格）である。使い方の要点は次の3つ。
#
# - **相補スラック**（complementary slackness）：$\lambda_j\,(b_j-g_j(w^\star))=0$。
#   バインドしていない制約（スラックが正）の双対はゼロ、バインドしている制約
#   （スラック≈0）だけが正の双対を持つ。よって**どの制約が効いているか**は
#   双対の非ゼロで判別できる。
# - **大きさ**：$\lambda$ が大きい制約ほど、そこを1単位ゆるめたときの利回り
#   改善が大きい。制約緩和・銘柄追加の優先順位付けに使える。
# - **符号の規約**：数値ソルバ（cvxpy）が返す `dual_value` の符号は問題の
#   立て方（最大化/最小化, 制約の向き）に依存する。本 notebook では符号を
#   決め打ちせず、右辺を微小に動かして再求解し、$\Delta p^\star \approx
#   \lambda\,\Delta b$ が数値的に成り立つことで整合を確かめる。



# %% [markdown]
# **数値例**：回転率制約 $\tfrac12\lVert w-w_0\rVert_1\le\tau$ の右辺は $b=2\tau$ です。上限を $\tau=20\%\to21\%$ に緩めると $\Delta b=2\Delta\tau=0.02$、双対 $\lambda=0.05$ なら期待利回りは $\Delta p^\star\approx\lambda\,\Delta b=0.05\times0.02=0.001$（$+10\text{bp}$）改善します。バインドしていない制約は $\lambda=0$ で、緩めても改善はゼロです。
# %% [markdown]
# **数値例**：2銘柄で現保有 $w_0=(0.5,0.5)$ から $w=(0.62,0.38)$ へ組み替えると、片道回転率は $\tfrac12\sum_i|w_i-w_{0,i}|=\tfrac12(|{+}0.12|+|{-}0.12|)=0.12$（12%）です。買い増し総額 $0.12$ と売り減らし総額 $0.12$ が等しく、$\tfrac12$ を掛ける理由がそのまま確認できます。
# %% [markdown]
# ## スクラッチ実装
#
# 割引カーブを1本組み、合成JGBユニバースの各銘柄について価格・利回りを求め、
# KRD とスプレッドDV01 の因子負荷を作る。最後に cvxpy で制約付き最適化を解く
# ソルバを用意する。KRD 因子は `bondlab.analytics.bump_curve` でカーブを
# バンプして作る。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `build_par_curve(tenors, params)` | 整数テナー, カーブ形状パラメータ | `DiscountCurve` | 合成パー利回りをブートストラップして割引カーブを作る |
# | `bond_from_row(row, settlement)` | ユニバース1行, 決済日 | `FixedRateBond` | 満期年数・クーポンから半年払い固定利付債を組む |
# | `bond_cashflows(bond, settlement)` | 債券, 決済日 | (年数配列, 金額配列) | 決済日以降のキャッシュフロー（割引前）を取り出す |
# | `price_under_curve(times, cfs, curve)` | 年数配列, 金額配列, カーブ | dirty price | 任意カーブ下でのキャッシュフロー現在価値 |
# | `krd_vector(times, cfs, curve, keys, width, bump)` | CF, カーブ, キーテナー | KRD配列(年) | キーレート毎のバンプ再評価でKRDを測る |
# | `spread_dv01(times, cfs, curve, bump)` | CF, カーブ | スプレッドDV01 | 平行スプレッド1bp当たりの価格変化 |
# | `build_risk_model(universe, curve, settlement, keys)` | ユニバース, カーブ | dict（価格・利回り・KRD行列・sDV01） | 全銘柄の因子リスクモデルを組み立てる |
# | `solve_max_yield(model, w0, krd_band, turnover)` | リスクモデル, 現保有, 制約 | dict（最適解・制約オブジェクト） | 利回り最大化の制約付き最適化を解く |

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
for _f in ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Noto Sans JP", "TakaoPGothic", "IPAPGothic"]:
    if any(_f == _n.name for _n in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
import cvxpy as cp

from bondlab.bond import FixedRateBond
from bondlab.curve import bootstrap_par, DiscountCurve
from bondlab.analytics import bump_curve, duration_convexity

np.random.seed(0)

BP = 1e-4                       # 1 ベーシスポイント
SETTLEMENT = dt.date(2026, 1, 5)
KEY_TENORS = np.array([2.0, 5.0, 10.0, 20.0, 30.0])  # KRD のキーテナー（年）
KRD_WIDTH = 5.0                 # 三角バンプの裾幅（年）
CURVE_BUMP = 1e-4               # KRD・sDV01 のバンプ幅（1bp）


def build_par_curve(tenors, level=0.020, floor=0.002, decay=8.0):
    """合成パー利回りをブートストラップして割引カーブを作る。

    パー利回りは floor から level+floor へ滑らかに立ち上がる右肩上がりの形
    par(t) = floor + level * (1 - exp(-t/decay)) とする。tenors は年1回払いの
    等間隔グリッド（1,2,...）を渡す前提で、bootstrap_par の等間隔仮定に沿う。
    """
    tenors = np.asarray(tenors, dtype=float)
    par = floor + level * (1.0 - np.exp(-tenors / decay))
    return bootstrap_par(tenors, par, frequency=1, interp="log_linear")


def bond_from_row(row, settlement):
    """ユニバース1行（満期年数・クーポン）から半年払い固定利付債を組む。"""
    mat_days = int(round(float(row["maturity_years"]) * 365.25))
    maturity = settlement + dt.timedelta(days=mat_days)
    return FixedRateBond(
        issue=settlement,
        maturity=maturity,
        coupon=float(row["coupon"]),
        frequency=2,
        convention="ACT/ACT",
        face=100.0,
    )


def bond_cashflows(bond, settlement):
    """決済日以降のキャッシュフローを (年数配列, 金額配列) で返す（割引前）。"""
    flows = [(d, c) for d, c in bond.cashflows() if d > settlement]
    times = np.array([(d - settlement).days / 365.25 for d, _ in flows])
    cfs = np.array([c for _, c in flows])
    return times, cfs


def price_under_curve(times, cfs, curve):
    """割引カーブ下でのキャッシュフロー現在価値（dirty price）。"""
    return float(np.sum(cfs * curve.discount(times)))


def krd_vector(times, cfs, curve, keys=KEY_TENORS, width=KRD_WIDTH, bump=CURVE_BUMP):
    """キーレート・デュレーション（年）を、キーテナー毎のバンプ再評価で測る。

    各キーテナー t_k について、その近傍を三角形に持ち上げた（下げた）カーブで
    価格を評価し、中心差分で KRD_k = -(P_up - P_dn)/(2*bump*P0) を得る。単位は
    年（金利1単位変化に対する価格の対数感応度）。
    """
    p0 = price_under_curve(times, cfs, curve)
    krd = np.empty(keys.size)
    for k, t in enumerate(keys):
        c_up = bump_curve(curve, t, +bump, width=width)
        c_dn = bump_curve(curve, t, -bump, width=width)
        p_up = price_under_curve(times, cfs, c_up)
        p_dn = price_under_curve(times, cfs, c_dn)
        krd[k] = -(p_up - p_dn) / (2.0 * bump * p0)
    return krd


def spread_dv01(times, cfs, curve, bump=CURVE_BUMP):
    """スプレッドDV01：カーブ全体を平行に bump だけ持ち上げたときの価格変化。

    ゼロレートへの一様シフトを、割引係数側で DF(t)->DF(t)*exp(-bump*t) と等価に
    掛けて表現する。返り値は正で、スプレッド拡大1bp当たりの価格下落幅。
    """
    p0 = price_under_curve(times, cfs, curve)
    df = cfs * curve.discount(times) * np.exp(-bump * times)
    p_up = float(np.sum(df))
    return -(p_up - p0) / bump * BP


def build_risk_model(universe, curve, settlement=SETTLEMENT, keys=KEY_TENORS):
    """全銘柄の因子リスクモデルを組む。

    返り値 dict:
      ids, mats, coupons, rc_bp : 識別子・満期・クーポン・割安割高(bp)
      price      : dirty price（額面100）
      ytm        : カーブ整合の最終利回り
      exp_yield  : ytm に割安分(rich_cheap_bp)を上乗せした期待利回り
      mod_dur    : 修正デュレーション（年）
      krd        : KRD 行列 (N x K)
      sdv01      : スプレッドDV01 (N,)
    """
    ids, mats, coupons, rc = [], [], [], []
    price, ytm, moddur, sdv01 = [], [], [], []
    krd_rows = []
    for _, row in universe.iterrows():
        bond = bond_from_row(row, settlement)
        times, cfs = bond_cashflows(bond, settlement)
        p = price_under_curve(times, cfs, curve)
        clean = p - bond.accrued(settlement)
        y = bond.yield_from_price(clean, settlement)
        dc = duration_convexity(bond, y, settlement)
        ids.append(row["bond_id"])
        mats.append(float(row["maturity_years"]))
        coupons.append(float(row["coupon"]))
        rc.append(float(row["rich_cheap_bp"]))
        price.append(p)
        ytm.append(y)
        moddur.append(dc["modified"])
        krd_rows.append(krd_vector(times, cfs, curve, keys))
        sdv01.append(spread_dv01(times, cfs, curve))
    rc = np.array(rc)
    ytm = np.array(ytm)
    return dict(
        ids=np.array(ids),
        mats=np.array(mats),
        coupons=np.array(coupons),
        rc_bp=rc,
        price=np.array(price),
        ytm=ytm,
        exp_yield=ytm + rc * BP,           # 割安(正)なら期待利回りを上乗せ
        mod_dur=np.array(moddur),
        krd=np.array(krd_rows),
        sdv01=np.array(sdv01),
    )


def solve_max_yield(model, w0, krd_band=0.1, turnover=0.20):
    """利回り最大化 s.t. KRD±band・回転率≤turnover・ロングオンリー・予算1。

    KRD 帯制約は上側 (K^T w - K^T w0 <= band) と下側 (K^T w0 - K^T w <= band) の
    2本に分け、キーテナー毎の双対（シャドープライス）を読めるようにする。
    返り値 dict に最適重み・目的値・制約オブジェクト（dual_value 参照用）を格納。
    """
    y = model["exp_yield"]
    K = model["krd"]                        # (N, key)
    n = y.size
    w = cp.Variable(n)
    krd_p = K.T @ w                         # ポートフォリオKRD (key,)
    krd_b = K.T @ w0                        # ベンチマークKRD (key,)

    c_budget = cp.sum(w) == 1
    c_long = w >= 0
    c_krd_up = krd_p - krd_b <= krd_band
    c_krd_dn = krd_b - krd_p <= krd_band
    c_turn = cp.norm1(w - w0) <= 2.0 * turnover   # 片道回転率 = (1/2)||w-w0||_1

    prob = cp.Problem(
        cp.Maximize(y @ w),
        [c_budget, c_long, c_krd_up, c_krd_dn, c_turn],
    )
    prob.solve()
    return dict(
        status=prob.status,
        w=np.asarray(w.value).ravel(),
        obj=float(prob.value),
        c_krd_up=c_krd_up,
        c_krd_dn=c_krd_dn,
        c_turn=c_turn,
        krd_band=krd_band,
        turnover=turnover,
    )


# %% [markdown]
# 割引カーブを組み、リスクモデルを構築します。カーブは1〜30年の等間隔グリッドの
# 合成パー利回りからブートストラップします（`bootstrap_par` の等間隔仮定に沿う
# ため整数グリッドを使います）。

# %%
universe = pd.read_csv("data/samples/synthetic_jgb_universe.csv")
curve = build_par_curve(np.arange(1, 31))

grid = np.linspace(0.5, 30, 60)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(grid, curve.zero_rate(grid) * 100, color="steelblue", label="ゼロレート")
ax.plot(grid, [curve.forward_rate(t, t + 0.5) * 100 for t in grid], color="darkorange",
        lw=1, ls="--", label="6か月フォワード")
ax.set_xlabel("年数")
ax.set_ylabel("レート (%)")
ax.set_title("合成JGB割引カーブ")
ax.legend()
fig.tight_layout()
plt.show()

model = build_risk_model(universe, curve)
print("銘柄数:", model["ids"].size)
print("期待利回り レンジ (%):", round(model["exp_yield"].min() * 100, 3),
      "〜", round(model["exp_yield"].max() * 100, 3))

# %% [markdown]
# リスクモデルの中身を表で確認します。KRD 行列は各行が銘柄、各列がキーテナー
# （2,5,10,20,30年）の感応度です。短い銘柄は 2〜5年の列に、長い銘柄は 20〜30年の
# 列にエクスポージャが集中します。

# %%
krd_cols = [f"KRD_{int(t)}y" for t in KEY_TENORS]
table = pd.DataFrame({
    "満期": model["mats"],
    "クーポン%": model["coupons"] * 100,
    "利回り%": model["ytm"] * 100,
    "割安bp": model["rc_bp"],
    "期待利回り%": model["exp_yield"] * 100,
    "修正Dur": model["mod_dur"],
    "sDV01": model["sdv01"],
}, index=model["ids"])
krd_df = pd.DataFrame(model["krd"], index=model["ids"], columns=krd_cols)
show = pd.concat([table, krd_df], axis=1)
display(show.iloc[[0, 10, 20, 30, 39]].round(3))

# %% [markdown]
# 代表銘柄の KRD プロファイルを描きます。満期の異なる3銘柄で、感応度がどの
# キーテナーに立つかを比べます。三角バンプの裾が広いため隣接テナーへ少し
# にじみますが、KRD の山はおおむね各銘柄の満期に対応します。

# %%
pick = [5, 20, 39]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(KEY_TENORS.size)
w_bar = 0.25
for j, idx in enumerate(pick):
    ax.bar(x + (j - 1) * w_bar, model["krd"][idx], width=w_bar,
           label=f'{model["ids"][idx]} ({model["mats"][idx]:.1f}年)')
ax.set_xticks(x)
ax.set_xticklabels([f"{int(t)}年" for t in KEY_TENORS])
ax.set_xlabel("キーテナー")
ax.set_ylabel("KRD (年)")
ax.set_title("代表銘柄のキーレート・デュレーション")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## QuantLib検証
#
# ここで扱う最適化は、QuantLib の守備範囲（商品評価）の外にあります。制約付き
# 凸最適化と双対性は QuantLib に対応部品が無いため、本節は QuantLib との突合では
# なく、**(A) KRD 因子生成エンジンの数値恒等式**と、**(B) バインド制約の特定と
# シャドープライス（双対変数）の整合**の2つを検証と位置づけます。いずれも
# 「実装が正しく組まれていること」を確かめるもので、実データでの収益を保証する
# ものではありません。
#
# ### 検証A：KRD の総和と平行デュレーションの一致
#
# カーブの全ノードを1点ずつ 1bp 持ち上げたときの KRD をすべて足し合わせると、
# カーブ全体を平行に持ち上げたときの実効デュレーションに一致するはずです
# （1次の重ね合わせ）。両者を同じ補間経路（ノードのゼロレートをバンプして
# 再構築）で計算し、`bump_curve` と再評価による KRD 計算が正しいことを確認します。

# %%
def parallel_bump(curve, size):
    """全ノードのゼロレートを一様に size だけずらした平行シフト後のカーブ。

    bump_curve と同じ再構築経路（ノードのゼロレート→割引係数）を通すので、
    単一ノードバンプの総和と1次で厳密に対応する。
    """
    times = curve.times.copy()
    zeros = np.array([curve.zero_rate(t) if t > 0 else 0.0 for t in times])
    mask = times > 0
    dfs = np.exp(-(zeros[mask] + size) * times[mask])
    return DiscountCurve(times[mask], dfs, interp=curve.interp)


bond0 = bond_from_row(universe.iloc[20], SETTLEMENT)
times0, cfs0 = bond_cashflows(bond0, SETTLEMENT)
all_nodes = curve.times[curve.times > 0]                 # カーブの全ノード
krd_all = krd_vector(times0, cfs0, curve, keys=all_nodes, width=None)  # 単一ノードバンプ
p0 = price_under_curve(times0, cfs0, curve)
p_up = price_under_curve(times0, cfs0, parallel_bump(curve, +CURVE_BUMP))
p_dn = price_under_curve(times0, cfs0, parallel_bump(curve, -CURVE_BUMP))
eff_dur = -(p_up - p_dn) / (2.0 * CURVE_BUMP * p0)        # 平行シフトの実効Dur
print(f"全ノードKRDの総和 = {krd_all.sum():.6f} 年")
print(f"平行シフト実効Dur = {eff_dur:.6f} 年")
assert abs(krd_all.sum() - eff_dur) < 1e-4, "KRDの総和が平行デュレーションと不一致"
print("検証A 合格: KRDの分解が平行デュレーションを回収している")

# %% [markdown]
# ### 検証B：バインド制約とシャドープライスの整合
#
# 制約付き最適化を1度解き、各制約のスラック（余裕）を見て**どの制約がバインド
# しているか**を特定します。相補スラック性より、バインドした制約だけが非ゼロの
# 双対（シャドープライス）を持つはずです。さらに、バインドした制約の右辺を微小に
# 動かして再求解し、目的関数の変化 $\Delta p^\star$ が $\lambda\,\Delta b$ と
# 一致することで、双対変数がシャドープライスとして正しいことを数値的に確かめます。

# %%
w0 = np.full(model["ids"].size, 1.0 / model["ids"].size)   # 等ウェイトのベンチマーク
sol = solve_max_yield(model, w0, krd_band=0.1, turnover=0.20)
print("最適化ステータス:", sol["status"])

# 各制約のスラックを計算してバインドを判定する。
K = model["krd"]
krd_p = K.T @ sol["w"]
krd_b = K.T @ w0
slack_up = sol["krd_band"] - (krd_p - krd_b)     # c_krd_up の余裕
slack_dn = sol["krd_band"] - (krd_b - krd_p)     # c_krd_dn の余裕
slack_turn = 2.0 * sol["turnover"] - float(np.sum(np.abs(sol["w"] - w0)))

TOL = 1e-6
bind = pd.DataFrame({
    "制約": [f"KRD上限 {int(t)}年" for t in KEY_TENORS]
             + [f"KRD下限 {int(t)}年" for t in KEY_TENORS] + ["回転率上限"],
    "スラック": np.concatenate([slack_up, slack_dn, [slack_turn]]),
    "双対値": np.concatenate([
        np.atleast_1d(sol["c_krd_up"].dual_value),
        np.atleast_1d(sol["c_krd_dn"].dual_value),
        [float(np.atleast_1d(sol["c_turn"].dual_value)[0])],
    ]),
})
bind["バインド"] = bind["スラック"].abs() < 1e-5
display(bind.round(5))

# 相補スラック：バインドしていない制約の双対はほぼゼロ。
non_binding = bind.loc[~bind["バインド"], "双対値"].abs().max()
print(f"\n非バインド制約の双対の最大絶対値: {non_binding:.2e}（≈0 なら相補スラック整合）")
assert non_binding < 1e-4, "非バインド制約の双対が0でない（相補スラック違反）"

# %% [markdown]
# 回転率制約の右辺を微小に動かして再求解し、シャドープライスの整合を確かめます。
# $\Delta p^\star \approx \lambda\,\Delta b$（$b$ は回転率制約の右辺 $2\tau$）が
# 数値的に成り立てば、双対値をシャドープライスとして読んでよいことになります。

# %%
lam_turn = float(np.atleast_1d(sol["c_turn"].dual_value)[0])
d_tau = 1e-4
sol_up = solve_max_yield(model, w0, krd_band=0.1, turnover=0.20 + d_tau)
d_obj = sol_up["obj"] - sol["obj"]
d_b = 2.0 * d_tau                                  # 右辺 2τ の変化
predicted = lam_turn * d_b
print(f"回転率制約の双対 λ = {lam_turn:.5f}")
print(f"再求解の目的変化 Δp* = {d_obj:.3e}")
print(f"双対予測  λ·Δb     = {predicted:.3e}")
assert abs(d_obj - predicted) < 1e-6, "シャドープライスと再求解の目的変化が不一致"
print("検証B 合格: 双対変数がシャドープライス（∂p*/∂b）として整合している")

# %% [markdown]
# ## 実データ適用
#
# 合成JGBユニバースで「**期待利回り最大化 subject to KRD±0.1年・回転率20%以下**」
# を解きます。ベンチマーク（現保有）は等ウェイトとし、ロングオンリー・予算1を
# 課します。期待利回りは各銘柄のカーブ整合利回りに割安分（`rich_cheap_bp`）を
# 上乗せした値で、割安（cheap）な銘柄ほど高くなります。

# %%
sol = solve_max_yield(model, w0, krd_band=0.1, turnover=0.20)
w_opt = sol["w"]
active = w_opt - w0                                # アクティブ・ウェイト

port_yield = float(model["exp_yield"] @ w_opt)
bench_yield = float(model["exp_yield"] @ w0)
realized_turnover = 0.5 * float(np.sum(np.abs(active)))
print(f"ベンチ期待利回り: {bench_yield*100:.4f}%")
print(f"最適期待利回り  : {port_yield*100:.4f}%  (+{(port_yield-bench_yield)*1e4:.2f} bp)")
print(f"実現回転率      : {realized_turnover*100:.2f}%  (上限 20%)")
print(f"ポートKRD  : {np.round(K.T @ w_opt, 3)}")
print(f"ベンチKRD  : {np.round(K.T @ w0, 3)}")
print(f"KRD差(年)  : {np.round(K.T @ w_opt - K.T @ w0, 4)}  (帯 ±0.1)")

# %% [markdown]
# アクティブ・ウェイト（最適 − ベンチマーク）を満期順に描きます。買い増した
# （正）銘柄と減らした（負）銘柄の並びから、最適化がどこにリスク配分を移したかを
# 読み取ります。

# %%
order = np.argsort(model["mats"])
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["seagreen" if a >= 0 else "indianred" for a in active[order]]
ax.bar(range(order.size), active[order] * 100, color=colors)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(range(order.size))
ax.set_xticklabels([f'{m:.0f}' for m in model["mats"][order]], fontsize=7)
ax.set_xlabel("満期（年, 昇順）")
ax.set_ylabel("アクティブ・ウェイト (%)")
ax.set_title("最適ポートフォリオのアクティブ配分")
fig.tight_layout()
plt.show()

# %% [markdown]
# 買い増した上位銘柄と減らした上位銘柄を、割安度（`rich_cheap_bp`）とともに
# 一覧します。

# %%
detail = pd.DataFrame({
    "満期": model["mats"],
    "クーポン%": model["coupons"] * 100,
    "割安bp": model["rc_bp"],
    "期待利回り%": model["exp_yield"] * 100,
    "ベンチw%": w0 * 100,
    "最適w%": w_opt * 100,
    "アクティブ%": active * 100,
}, index=model["ids"])
print("=== 買い増し上位5 ===")
display(detail.sort_values("アクティブ%", ascending=False).head(5).round(3))
print("\n=== 削減上位5 ===")
display(detail.sort_values("アクティブ%").head(5).round(3))

# %% [markdown]
# ### 最適解の直感的な説明
#
# 目的は期待利回りの最大化で、割安（`rich_cheap_bp` が正）な銘柄ほど期待利回りが
# 高く、優先的に買い増したい対象になります。ただし無条件では長期・高クーポン債に
# 荷重が偏り、デュレーションが跳ね上がります。そこを2つの制約が押さえます。
#
# - **KRD±0.1年**：ポートフォリオの各キーテナー感応度をベンチマークから
#   ±0.1年に収めます。割安な長期債を買うなら、KRD を保つために同年限の割高債を
#   減らす、あるいは他テナーで相殺する動きが必要になり、**カーブ形状のリスクを
#   ベンチに固定したまま**割安銘柄へ入れ替えます。上の KRD 差がすべて帯 ±0.1 の
#   内側に収まっていることが、これを示します。
# - **回転率20%**：組み替え総額を資本の20%までに縛ります。最も割安な銘柄から
#   順に買える量が制限されるため、最適化は「1bp の割安あたりの改善が大きい入れ替え」
#   から優先して回転を使い切ります。実現回転率が20%に張り付いているのは、
#   もっと動かせば利回りを上げられる（＝回転率制約がバインドしている）ことの表れ
#   です。検証Bで見たとおり、この制約のシャドープライスは正で、回転率を1単位
#   ゆるめれば利回りがその分だけ改善します。
#
# つまり最適解は「回転率の予算内で、KRD をベンチに固定しつつ、割高銘柄を割安銘柄へ
# 置き換える」入れ替えとして解釈できます。どの制約が効いているかはシャドープライスの
# 非ゼロで判別でき、次に資源を割くべき制約（回転率を上げる/KRD帯を広げる）の
# 優先順位もその大きさで比較できます。

# %% [markdown]
# ## 演習
#
# 1. **制約を締めたときのシャドープライス変化。** `solve_max_yield` の
#    `krd_band` を 0.10 → 0.05 → 0.02 と締めながら解き、(a) 最適期待利回り、
#    (b) 回転率制約の双対、(c) バインドする KRD 制約の本数と双対の大きさ、が
#    どう動くかを表と図にまとめよ。制約を締めると目的値が下がること（シャドー
#    プライスが正であること）と、締めるほど KRD 帯の双対が大きくなることを
#    確認し、経済的に説明せよ。解答例は `solutions/S10/sol_1003.py`。
# 2. **平均分散法が債券で使いにくいことを数値で示す。** 因子モデル＋ノイズから
#    短い（例：30営業日）リターン履歴を合成し、標本平均 $\hat\mu$・標本共分散
#    $\hat\Sigma$ を推定して平均分散最適化 $\max_w \hat\mu^\top w -
#    \tfrac{\gamma}{2}w^\top\hat\Sigma w$（$\mathbf 1^\top w=1$）を解け。
#    (a) $\hat\Sigma$ の条件数、(b) 得られる重みの極端さ（最大・最小ウェイト、
#    ロング・ショート総額）、(c) 履歴を再サンプルしたときの重みの不安定さ、を
#    示し、本 notebook の因子制約アプローチと対比して、なぜ債券で平均分散法が
#    素直に使えないかを論じよ。解答例は `solutions/S10/sol_1003.py`。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/10_portfolio.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | [ファクターリスクモデル](../../glossary/10_portfolio.md#factor-risk-model) | factor risk model | リスクを少数の観測可能な因子（KRD・sDV01等）のエクスポージャで表す枠組み |
# | [凸最適化](../../glossary/10_portfolio.md#convex-optimization) | convex optimization | 凸目的・凸制約の最適化。局所最適が大域最適で、双対と一括して解ける |
# | [回転率制約](../../glossary/10_portfolio.md#turnover-constraint) | turnover constraint | 現保有からの売買総額（片道 $\tfrac12\lVert w-w_0\rVert_1$）を上限で縛る制約 |
# | [シャドープライス](../../glossary/10_portfolio.md#shadow-price) | shadow price | 制約の右辺を1単位ゆるめたときの目的関数の変化。双対変数 $\partial p^\star/\partial b$ |
