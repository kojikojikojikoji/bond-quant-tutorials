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
# # S10-2 インデックス複製（サンプリング最適化）
#
# ## 学習目標
#
# - 債券インデックスを少数銘柄で追う理由を、株式との違い（流動性・最低ロット・
#   銘柄数）から説明できる
# - 層化サンプリング（stratified sampling）に基づくセル法を、セクション×年限×格付
#   のセルへ母集団を割り付ける形で実装できる
# - キーレートデュレーション（KRD）を `bondlab.analytics.bump_curve` で計算し、
#   複製ポートフォリオの KRD をインデックスへ一致させる最適化を `cvxpy` で解ける
# - トラッキングエラー（tracking error）をファクター残差と個別リスクへ分解し、
#   事前（ex-ante）に予測できる
# - インサンプルとアウトオブサンプルでトラッキングエラーを測り分け、構造的な
#   ファクター整合が過剰適合に強い理由を数値で確かめられる
#
# 本 notebook は S10-1 のポートフォリオ最適化に続き、「ベンチマークを持たずに
# ベンチマークを再現する」という運用側の中核問題を扱います。土台には S3-3 の
# KRD、S2 のカーブ構築を使います。


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# パッシブ運用会社が債券インデックスファンドや ETF を提供するとき、ベンチマークを完全にそのまま保有する完全法は現実には使えません。総合社債指数は構成銘柄が数千から一万を超え、その多くは発行直後に満期保有投資家へ吸収されて市場に出回らなくなります。売り気配すら立たない銘柄を額面単位で買い集めることは不可能で、仮にできても数千銘柄を薄く持つ事務コストと取引コストが指数連動の価値を食い潰します。そこで運用側は、少数の代表銘柄でインデックスの**リスク特性だけ**を再現します。ここでの稼ぎ方は超過リターンではなく、「ベンチマークをどれだけ安く・小さな誤差で追えるか」という複製の質そのものです。低コストで低トラッキングエラーを実現できることが、パッシブファンドの競争力であり運用報酬（低廉だが大量の資産にかかる）の正当化になります。
#
# 複製の中核は、どのリスク軸で一致させるかの設計です。層化サンプリングはセクション（国債・社債・セクター）×年限×格付でセルを切り、各セルから流動性の高い代表銘柄を抽出してセル単位でリスクを近似します。さらにカーブ形状への感応まで揃えるため、キーレートデュレーション（KRD）を計算し、複製ポートフォリオの KRD をインデックスへ一致させる最適化を解きます。これにより、水準シフトだけでなくスティープ化・フラット化といった非平行変形に対しても指数と同じように動くポートフォリオが得られます。
#
# 複製の良し悪しはトラッキングエラーで測ります。実務では、事前（ex-ante）にファクター残差と個別リスクへ分解して予測し、少数銘柄でも構造的なファクター（KRD・格付・セクター）を揃えておけば、個別銘柄のノイズが平均化されてアウトオブサンプルの誤差が抑えられます。逆に、過去リターンへ表面的にフィットさせただけの複製は、構成が毎月動く債券市場ではすぐに剥がれます。インサンプルとアウトオブサンプルでトラッキングエラーを測り分けるのは、この過剰適合を避けるためです。負債を守る ALM（S10-1）が「負債に合わせる」問題であるのに対し、複製は「ベンチマークに合わせる」問題で、いずれも運用会社が顧客に約束したリスク特性を再現する技術という点で地続きです（docs/債券ファンドの業務.md「運用会社・ALM」の枠組み）。
# %% [markdown]
# ## 理論
#
# ### 債券インデックスは株式インデックスと何が違うか
#
# 株式インデックスであれば、構成銘柄を時価総額比でそのまま買えば完全複製
# （full replication、完全法）が成立します。銘柄数は数百から精々数千、各銘柄は
# 取引所で連続的に売買され、1株単位で端数も持てます。債券インデックスでは、この
# 前提がことごとく崩れます。
#
# | 論点 | 株式 | 債券 |
# |---|---|---|
# | 銘柄数 | 数百〜数千 | 総合社債指数で数千〜1万超 |
# | 流動性 | 取引所で連続約定 | 店頭取引。多くの銘柄は日々の約定が無い |
# | 最低ロット | 1株 | 額面が大きく端数を持てない |
# | 構成の安定性 | 入替は年数回 | 毎月の新発・償還で構成が動く |
#
# とりわけ効くのが**流動性**です。指数採用銘柄の多くは発行直後に投資家の
# 満期保有ポートフォリオへ吸収され、市場に出回らなくなります（いわゆる
# タンス預金化）。売り気配すら立たない銘柄を完全法で買い集めることは、
# 現実には不可能です。加えて銘柄数が数千に達すると、各銘柄をわずかずつ持つ
# こと自体が取引コストと事務負担で割に合いません。
#
# そこで運用側は、**少数の代表銘柄でインデックスのリスク特性だけを再現する**
# 方針を採ります。これを部分複製（サンプリング）と呼びます。

# %% [markdown]
# ### 完全法・層化サンプリング・セル法・ファクター法
#
# 複製手法は、どこまで銘柄単位の一致を諦め、どのリスク軸で一致させるかで並びます。
#
# - **完全法（full replication）**: 全構成銘柄を指数ウェイトで保有する。誤差は
#   理論上ゼロだが、債券では流動性と銘柄数の壁で成立しない。
# - **層化サンプリング（stratified sampling）**: 母集団を互いに素なセルへ分割し、
#   各セルから代表銘柄を抽出して、セル単位でリスクを近似する。抽出銘柄数を絞る
#   ほど個別リスクは残るが、母集団全体を機械的に薄く買うより偏りが小さい。
# - **セル法（cell approach）**: 層化の軸を具体化した実装。債券では
#   「セクション（国債／社債の業種）× 年限バケット × 格付」の3次元格子でセルを
#   切り、各セルの指数内ウェイトを、そのセルの代表銘柄へ集約する。直感的で頑健
#   だが、セル内の年限分布までは合わせられないため KRD は近似一致に留まる。
# - **リスクファクターマッチング（factor matching, ファクター法）**: KRD ベクトル
#   やスプレッドデュレーションといった感応度を、複製ポートフォリオとインデックスで
#   明示的に一致させる。最適化で感応度差を最小化するため整合は精密だが、限られた
#   データに過剰適合しやすく、アウトオブサンプルでの検証が欠かせない。
#
# 本 notebook では、JGB のみの合成インデックス（格付・セクションは実質1種）を
# 対象とするため、セルは年限バケットへ縮約されます。ただし実装はセクション×年限
# ×格付の一般形で書き、社債指数へそのまま拡張できる形にします。

# %% [markdown]
# ### キーレートデュレーションとトラッキングエラーの予測
#
# ポートフォリオのカーブ感応度は、少数のキー年限 $\{\tau_k\}$ における
# ゼロレート $z(\tau_k)$ への感応度、すなわちキーレートデュレーション
# （key rate duration, KRD）で表します。銘柄 $i$、キー年限 $\tau_k$ の KRD は
#
# $$ \mathrm{KRD}_{i,k} = -\frac{1}{P_i}\frac{\partial P_i}{\partial z(\tau_k)} \approx -\frac{P_i(z+\delta e_k)-P_i(z-\delta e_k)}{2\,\delta\,P_i} $$
#
# で、カーブを $\tau_k$ 近傍だけ $\pm\delta$ 動かした中心差分で数値評価します。
# この「カーブの一部だけを動かす」操作を `bondlab.analytics.bump_curve` が担います。
#
# 時価総額加重インデックス（ウェイト $w_i$）と複製ポートフォリオ（ウェイト
# $x_i$）の日次リターン差は、一次で次のように分解できます。
#
# $$ r_p - r_b \;=\; -\sum_{k}\Big(\underbrace{\textstyle\sum_i (x_i-w_i)\,\mathrm{KRD}_{i,k}}_{\Delta k_k:\ \text{KRD ミスマッチ}}\Big)\Delta z_k \;+\; \sum_i (x_i - w_i)\,\varepsilon_i $$
#
# 第1項はカーブファクター $\Delta z_k$ に対する感応度の食い違い、第2項は
# 銘柄固有（スプレッド・流動性）の残差 $\varepsilon_i$ です。両者が独立とすれば、
# 事前トラッキングエラー（ex-ante tracking error）は
#
# $$ \mathrm{TE}^2 \;=\; \Delta k^\top \Sigma_z\, \Delta k \;+\; \sum_i (x_i-w_i)^2 \sigma_i^2 $$
#
# となります。$\Sigma_z$ はカーブファクターの共分散、$\sigma_i$ は銘柄 $i$ の
# 個別リスクです。ここから複製の設計方針が読めます。第1項は KRD を一致させて
# ($\Delta k \to 0$) 消す。第2項は各銘柄を薄く広く持つほど小さくなり、抽出銘柄数
# $n$ に対しておよそ $1/\sqrt{n}$ で逓減します。トラッキングエラーは通常、年率
# ベーシスポイントで表示します（$\mathrm{TE}_{\text{年率}} = \mathrm{TE}_{\text{日次}}\sqrt{252}$）。

# %% [markdown]
# ## スクラッチ実装
#
# ### 使用する自作関数
#
# 複製の中核ロジックは自作します。各関数の役割は次のとおりです。
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `build_universe(df, n_total, seed)` | 種データ, 目標銘柄数, 乱数種 | DataFrame | CSV を核に母集団を実 JGB 規模へ拡張 |
# | `bond_cashflows(mat, cpn, freq)` | 残存年, クーポン率, 頻度 | (times, cfs) | 半年払いの (時点, 金額) 配列を生成 |
# | `price_on_curve(curve, times, cfs)` | カーブ, 時点, 金額 | float | $P=\sum cf_j\,DF(t_j)$ で現在価値 |
# | `krd_matrix(curve, cfs_list, tenors, width, size)` | カーブ, 各銘柄CF, キー年限, 幅, 幅δ | ndarray | 全銘柄×キー年限の KRD 行列 |
# | `stratified_select(univ, w, n, edges)` | 母集団, 指数W, 銘柄数, 年限境界 | index | 層化サンプリングで n 銘柄を抽出 |
# | `match_krd(K_sel, b)` | 抽出KRD, 指数KRD | ndarray | cvxpy で KRD を一致させるウェイト |
# | `cell_weights(univ, w, sel, edges)` | 母集団, 指数W, 抽出, 年限境界 | ndarray | セル法：セル指数Wを代表銘柄へ集約 |
# | `scenario_returns(K, dur, univ, n_scen, seed)` | KRD, デュレーション, 母集団, 本数, 種 | ndarray | カーブ＋個別の日次リターンを生成 |
# | `tracking_error(R, x_full, w)` | リターン, 複製W, 指数W | float | 年率トラッキングエラー（bp） |

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

from bondlab.curve import bootstrap_par
from bondlab.analytics import bump_curve
from bondlab.bond import FixedRateBond

np.random.seed(0)

# 母集団 CSV の探索（notebook をどこから実行しても見つかるようにする）。
DATA = Path("data/samples/synthetic_jgb_universe.csv")
if not DATA.exists():
    for up in (Path(".."), Path("../.."), Path("../../..")):
        if (up / DATA).exists():
            DATA = up / DATA
            break

# キー年限（KRD を測る節点）と数値微分の設定。
KEY_TENORS = np.array([2.0, 5.0, 10.0, 20.0, 30.0])
KRD_WIDTH = 4.0       # 三角バンプの片側幅（年）
KRD_SIZE = 1e-4       # カーブバンプ量（1bp）
MAX_MAT = 39.5        # カーブ節点(1..40年)内に収めるための残存上限

# %% [markdown]
# ### パーイールドカーブの構築
#
# 割引カーブは `bootstrap_par` で作ります。剥ぎ取りが歪まないよう、年次
# （等間隔）グリッドのパー利回りを入力します。形状は短期ほぼゼロ・長期 2% 台
# という近年の JGB を模した合成値です。

# %%
def par_yield(t):
    """年限 t（年）の合成パー利回り。短期ほぼゼロ→長期 2% 台の右肩上がり。"""
    return 0.001 + 0.021 * (1.0 - np.exp(-t / 8.0)) + 0.001 * (t / 40.0)


grid = np.arange(1.0, 41.0, 1.0)             # 1〜40年の年次グリッド
curve = bootstrap_par(grid, par_yield(grid), frequency=1, interp="log_linear")

print("パーカーブ節点（抜粋）:")
for t in (1.0, 5.0, 10.0, 20.0, 30.0, 40.0):
    print(f"  {t:4.0f}年  par={par_yield(t)*100:5.3f}%  DF={curve.discount(t):.5f}  z={curve.zero_rate(t)*100:5.3f}%")

# %% [markdown]
# ### 合成母集団の生成
#
# 母集団 CSV は 40 銘柄です。実際の JGB は残存1年超だけで約 300 銘柄あるため、
# CSV の 40 銘柄を核に、残存・クーポン・リッチチープの分布を引き継いで約 300
# 銘柄へ拡張します。時価総額加重に必要な発行残高は、ベンチマーク年限ほど大きく
# なるよう対数正規で合成します（すべて乱数種固定）。

# %%
def build_universe(df, n_total, seed):
    """CSV を核に n_total 銘柄の合成母集団を作る。CSV の 40 銘柄はそのまま残す。"""
    rng = np.random.default_rng(seed)
    base = df.copy()
    base["maturity_years"] = base["maturity_years"].clip(upper=MAX_MAT)
    n_extra = n_total - len(base)
    mat = rng.uniform(1.05, MAX_MAT, size=n_extra)
    # クーポンはパー利回りに緩く連動＋端数刻み（実際の JGB も年限別に発行される）。
    cpn = np.clip(par_yield(mat) + rng.normal(0.0, 0.0015, n_extra), 0.001, 0.025)
    cpn = np.round(cpn / 0.001) * 0.001
    rc = rng.normal(0.0, 4.0, n_extra)          # リッチ/チープ（bp）
    extra = pd.DataFrame({
        "bond_id": [f"JGB{100 + i:03d}" for i in range(n_extra)],
        "maturity_years": mat,
        "coupon": cpn,
        "rich_cheap_bp": rc,
    })
    univ = pd.concat([base, extra], ignore_index=True)
    # 発行残高（兆円）：10年・20年など主要年限を厚めに。対数正規で分散させる。
    anchor = np.exp(-((univ["maturity_years"].values - 10.0) / 12.0) ** 2)
    size = np.exp(rng.normal(0.0, 0.4, len(univ))) * (1.0 + 1.5 * anchor)
    univ["amount"] = np.round(size * 2.0, 3)     # 発行残高（相対単位）
    return univ


raw = pd.read_csv(DATA)
univ = build_universe(raw, n_total=300, seed=42)
print(f"母集団 CSV: {len(raw)} 銘柄 → 合成母集団: {len(univ)} 銘柄")
print(univ.head(3).to_string(index=False))
print("残存レンジ:", round(univ.maturity_years.min(), 2), "〜", round(univ.maturity_years.max(), 2), "年")

# %% [markdown]
# ### キャッシュフローと現在価値
#
# 各銘柄を半年利付債として扱い、満期を起点に利払時点を刻みます。価格は
# 割引係数の加重和 $P=\sum_j cf_j\,DF(t_j)$ です。

# %%
def bond_cashflows(mat, cpn, freq=2):
    """半年払い固定利付債の (時点[年], キャッシュフロー) を返す。満期に額面を上乗せ。"""
    k = np.arange(0, int(np.ceil(mat * freq)) + 1)
    times = mat - k / freq
    times = np.sort(times[times > 1e-9])
    cfs = np.full_like(times, 100.0 * cpn / freq)
    cfs[-1] += 100.0                              # 満期（最大時点）で額面償還
    return times, cfs


def price_on_curve(curve, times, cfs):
    """カーブ上での現在価値 P = Σ cf_j DF(t_j)。"""
    return float(np.sum(cfs * curve.discount(times)))


cfs_list = [bond_cashflows(m, c) for m, c in zip(univ.maturity_years, univ.coupon)]
prices = np.array([price_on_curve(curve, t, c) for t, c in cfs_list])
univ["price"] = prices
print("価格レンジ:", round(prices.min(), 2), "〜", round(prices.max(), 2))

# %% [markdown]
# `FixedRateBond` によるクロスチェックで、自作のキャッシュフロー生成が
# ずれていないことを確かめます。ある銘柄について、`FixedRateBond` が生成する
# 利払スケジュールの本数・総償還額が、`bond_cashflows` と一致することを見ます。

# %%
import datetime as dt

val_date = dt.date(2026, 1, 15)
m0, c0 = float(univ.maturity_years.iloc[7]), float(univ.coupon.iloc[7])
mat_date = val_date + dt.timedelta(days=int(round(m0 * 365.25)))
qbond = FixedRateBond(issue=val_date, maturity=mat_date, coupon=c0, frequency=2)
future = [(d, c) for d, c in qbond.cashflows() if d > val_date]

t_mine, c_mine = bond_cashflows(m0, c0)
print(f"FixedRateBond の将来CF本数 = {len(future)},  自作 = {len(t_mine)}")
print(f"総償還額(額面込)  FixedRateBond = {sum(c for _, c in future):.2f},  自作 = {c_mine.sum():.2f}")
assert abs(len(future) - len(t_mine)) <= 1
assert abs(sum(c for _, c in future) - c_mine.sum()) < 0.6

# %% [markdown]
# ### KRD 行列の構築
#
# `bump_curve` は「カーブ全体」ではなく「キー年限近傍のゼロレートだけ」を
# 三角状に動かした新カーブを返します。バンプ済みカーブは全銘柄で共通なので、
# キー年限×上下の 10 本を先に作り置きしてから、全銘柄を一括で再評価します。

# %%
def krd_matrix(curve, cfs_list, tenors, width, size):
    """全銘柄×キー年限の KRD 行列（形状 N×K）を中心差分で求める。"""
    bumped = {(k, s): bump_curve(curve, float(tenors[k]), s * size, width)
              for k in range(len(tenors)) for s in (+1, -1)}
    K = np.empty((len(cfs_list), len(tenors)))
    for i, (times, cfs) in enumerate(cfs_list):
        p0 = price_on_curve(curve, times, cfs)
        for k in range(len(tenors)):
            pu = price_on_curve(bumped[(k, +1)], times, cfs)
            pd_ = price_on_curve(bumped[(k, -1)], times, cfs)
            K[i, k] = -(pu - pd_) / (2.0 * size * p0)
    return K


K = krd_matrix(curve, cfs_list, KEY_TENORS, KRD_WIDTH, KRD_SIZE)
dur = K.sum(axis=1)          # KRD 合計 ≈ 実効デュレーション
univ["eff_dur"] = dur
print("KRD 行列 形状:", K.shape)
print("\nキー年限別 KRD（残存 ~10年の銘柄の例）:")
idx10 = int((univ.maturity_years - 10.0).abs().idxmin())
for tau, kk in zip(KEY_TENORS, K[idx10]):
    print(f"  {tau:4.0f}年: {kk:6.3f}")
print(f"  合計(≈実効デュレーション): {dur[idx10]:.3f}  （残存 {univ.maturity_years.iloc[idx10]:.2f}年）")

# %% [markdown]
# ### 時価総額加重インデックスの定義
#
# 各銘柄の時価総額を「価格 × 発行残高」とし、その比率を指数ウェイト $w_i$ と
# します。インデックスの KRD プロファイルは $\sum_i w_i\,\mathrm{KRD}_{i}$ です。
# これが複製の一致目標になります。

# %%
mv = univ["price"].values * univ["amount"].values
w = mv / mv.sum()
univ["weight"] = w
index_krd = w @ K
index_dur = float(w @ dur)

print(f"インデックス銘柄数: {len(univ)}")
print(f"インデックス実効デュレーション: {index_dur:.3f}")
print("インデックス KRD プロファイル:")
for tau, kk in zip(KEY_TENORS, index_krd):
    print(f"  {tau:4.0f}年: {kk:6.3f}")

# %% [markdown]
# ### セル法とファクター法の複製構築
#
# まず層化サンプリングで候補銘柄を絞ります。年限バケット（＝JGB でのセル）の
# 指数ウェイトに比例して抽出枠 $n$ を配分し、各バケットで発行残高最大の銘柄を
# 代表として選びます。社債指数ではこの `edges` にセクションと格付の軸を足すだけで、
# 同じ関数がセクション×年限×格付のセルへ一般化します。

# %%
TENOR_EDGES = np.array([0.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 40.0])


def stratified_select(univ, w, n, edges):
    """年限バケット（セル）の指数Wに比例して n 銘柄を層化抽出する。"""
    buckets = np.digitize(univ["maturity_years"].values, edges)
    bucket_ids = np.unique(buckets)
    bw = np.array([w[buckets == b].sum() for b in bucket_ids])
    # 各バケットへ最低1、残りを指数Wへ比例配分（端数は大きい順に加える）。
    alloc = np.ones(len(bucket_ids), dtype=int)
    remaining = n - alloc.sum()
    if remaining > 0:
        frac = bw / bw.sum() * remaining
        add = np.floor(frac).astype(int)
        alloc += add
        for j in np.argsort(-(frac - add)):
            if alloc.sum() >= n:
                break
            alloc[j] += 1
    sel = []
    for b, a in zip(bucket_ids, alloc):
        pool = np.where(buckets == b)[0]
        a = min(a, len(pool))
        top = pool[np.argsort(-univ["amount"].values[pool])[:a]]  # 残高上位＝流動的
        sel.extend(top.tolist())
    return np.array(sorted(sel))


def cell_weights(univ, w, sel, edges):
    """セル法：各セルの指数W合計を、そのセルの代表銘柄へ均等集約する。"""
    buckets = np.digitize(univ["maturity_years"].values, edges)
    x = np.zeros(len(sel))
    sel_buckets = buckets[sel]
    for b in np.unique(buckets):
        cell_w = w[buckets == b].sum()
        reps = np.where(sel_buckets == b)[0]
        if len(reps) > 0:
            x[reps] += cell_w / len(reps)
    return x / x.sum()


def match_krd(K_sel, b):
    """KRD ファクター法：Σ x KRD = b に最も近い長期のみ・満額投資ウェイトを解く。"""
    n = K_sel.shape[0]
    x = cp.Variable(n, nonneg=True)
    obj = cp.Minimize(cp.sum_squares(K_sel.T @ x - b))
    cons = [cp.sum(x) == 1]
    cp.Problem(obj, cons).solve()
    xv = np.array(x.value).ravel()
    return np.clip(xv, 0.0, None) / np.clip(xv, 0.0, None).sum()


# 例として n=30 で両手法を構築し、KRD の一致度を比べる。
n_demo = 30
sel = stratified_select(univ, w, n_demo, TENOR_EDGES)
x_cell = cell_weights(univ, w, sel, TENOR_EDGES)
x_fac = match_krd(K[sel], index_krd)

krd_cell = x_cell @ K[sel]
krd_fac = x_fac @ K[sel]
comp = pd.DataFrame({
    "キー年限": KEY_TENORS,
    "指数": index_krd,
    "セル法": krd_cell,
    "ファクター法": krd_fac,
})
print(f"抽出銘柄数 n={len(sel)}")
print(comp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nKRD 二乗誤差  セル法={np.sum((krd_cell-index_krd)**2):.4f}  "
      f"ファクター法={np.sum((krd_fac-index_krd)**2):.2e}")

# %% [markdown]
# ファクター法は KRD 二乗誤差を最適化で潰すため、5 本のキー年限をほぼ完全に
# 一致させます。セル法はセル内の年限分布を代表銘柄1本へ丸めるため、KRD は
# 近似一致に留まります。次節では、この差が実際のトラッキングエラーにどう効くかを
# 見ます。

# %% [markdown]
# ## QuantLib検証
#
# 複製ポートフォリオには、YTM 価格のように突き合わせるべき閉形式のベンチマークが
# ありません。そこで本節は QuantLib による突合の代わりに、**トラッキングエラーの
# インサンプル／アウトオブサンプル計測を検証と位置づけます**。狙いは二つです。
#
# 1. 構築時に使った以外のカーブシナリオ（アウトオブサンプル）でも、KRD を一致
#    させた複製の追随誤差が小さいままかを確かめる。
# 2. シナリオへ直接あてはめた「過剰適合する複製」と比べ、KRD ファクター法が
#    インサンプルとアウトオブサンプルで成績が乖離しない（頑健である）ことを示す。
#
# ### シナリオ生成
#
# 日次リターンを、カーブファクター（水準・傾き・曲率）と銘柄固有ショックの和で
# 生成します。リターン差の第1項（KRD ミスマッチ）と第2項（個別リスク）を、
# 理論式のとおり再現するためです。

# %%
def scenario_returns(K, dur, univ, n_scen, seed):
    """日次リターン行列（n_scen×N）を生成する。カーブ3ファクター＋銘柄固有ショック。"""
    rng = np.random.default_rng(seed)
    tenors = KEY_TENORS
    # 3ファクターのキー年限ローディング：水準・傾き・曲率。
    level = np.ones_like(tenors)
    slope = (tenors - tenors.mean()) / tenors.std()
    curv = -((tenors - 10.0) / 10.0) ** 2
    curv = curv - curv.mean()
    load = np.vstack([level, slope, curv])                 # 3×K
    fac_vol = np.array([4.5e-4, 2.5e-4, 1.5e-4])           # 日次のファクターゆらぎ
    f = rng.standard_normal((n_scen, 3)) * fac_vol
    dz = f @ load                                          # n_scen×K（キー年限のゼロ変化）
    r_sys = -dz @ K.T                                      # 系統リターン（n_scen×N）
    # 銘柄固有：スプレッド/流動性の日次ノイズ。デュレーションでリターン換算。
    idio_bp = 1.2
    sigma_i = (idio_bp / 1e4) * np.maximum(dur, 1.0)
    eps = rng.standard_normal((n_scen, len(dur))) * sigma_i
    return r_sys + eps


R_in = scenario_returns(K, dur, univ, n_scen=3000, seed=1)     # インサンプル
R_out = scenario_returns(K, dur, univ, n_scen=3000, seed=2)    # アウトオブサンプル


def tracking_error(R, x_full, w):
    """年率トラッキングエラー（bp）。x_full と w は全銘柄長のウェイト。"""
    diff = R @ (x_full - w)
    return float(np.std(diff) * np.sqrt(252) * 1e4)


def to_full(x, sel, n_all):
    """抽出銘柄のウェイトを全銘柄長のベクトルへ埋め込む。"""
    xf = np.zeros(n_all)
    xf[sel] = x
    return xf


# %% [markdown]
# ### 構造的な複製 vs 過剰適合する複製
#
# 同じ抽出銘柄集合（n=30）に対し、二通りのウェイトを作ります。KRD ファクター法
# （構造的：シナリオを見ない）と、インサンプルのリターンへ最小二乗であてはめた
# 直接あてはめ（過剰適合）です。両者をインサンプルとアウトオブサンプルで測ります。

# %%
def fit_returns(R_sel, r_idx):
    """過剰適合の例：インサンプルのリターン差分散を直接最小化するウェイト。"""
    n = R_sel.shape[1]
    x = cp.Variable(n, nonneg=True)
    obj = cp.Minimize(cp.sum_squares(R_sel @ x - r_idx))
    cp.Problem(obj, [cp.sum(x) == 1]).solve()
    xv = np.clip(np.array(x.value).ravel(), 0.0, None)
    return xv / xv.sum()


r_idx_in = R_in @ w
x_fac_full = to_full(x_fac, sel, len(univ))
x_fit = fit_returns(R_in[:, sel], r_idx_in)
x_fit_full = to_full(x_fit, sel, len(univ))

rows = [
    ["KRD ファクター法（構造的）", tracking_error(R_in, x_fac_full, w), tracking_error(R_out, x_fac_full, w)],
    ["直接あてはめ（過剰適合）", tracking_error(R_in, x_fit_full, w), tracking_error(R_out, x_fit_full, w)],
    ["セル法", tracking_error(R_in, to_full(x_cell, sel, len(univ)), w),
     tracking_error(R_out, to_full(x_cell, sel, len(univ)), w)],
]
val = pd.DataFrame(rows, columns=["手法", "インサンプルTE(bp)", "アウトオブサンプルTE(bp)"])
val["過学習ギャップ"] = val["アウトオブサンプルTE(bp)"] - val["インサンプルTE(bp)"]
print(val.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

# 直接あてはめはインサンプルで低く出るが、アウトで悪化するはず（過学習）。
assert val.loc[1, "インサンプルTE(bp)"] < val.loc[0, "インサンプルTE(bp)"]
assert val.loc[1, "過学習ギャップ"] > val.loc[0, "過学習ギャップ"]

# %% [markdown]
# 直接あてはめはインサンプルで最小の追随誤差を示しますが、これは 3000 本の
# ノイズにウェイトを合わせ込んだ結果で、アウトオブサンプルでは悪化します
# （過学習ギャップが大きい）。KRD ファクター法はシナリオを一切見ずに感応度だけを
# 合わせるため、インサンプルとアウトオブサンプルでほぼ同じ成績になります。
# **複製の評価はアウトオブサンプルのトラッキングエラーで行うべき**という、
# 本節の検証上の結論です。以降の実データ適用でも、TE はアウトオブサンプルで測ります。

# %% [markdown]
# ## 実データ適用
#
# 合成 JGB 母集団（残存1年超・時価総額加重）を複製対象とし、抽出銘柄数 $n$ を
# 10・30・100 と変えたときのアウトオブサンプル・トラッキングエラーを測ります。
# 各 $n$ で層化サンプリング→KRD ファクター法の順に複製を構築します。

# %%
def build_replication(univ, w, K, n, edges):
    """層化抽出 → KRD ファクター法で複製を作り、全銘柄長ウェイトを返す。"""
    sel = stratified_select(univ, w, n, edges)
    x = match_krd(K[sel], w @ K)
    return sel, to_full(x, sel, len(univ))


n_grid = [10, 20, 30, 50, 75, 100, 150]
te_curve = []
for n in n_grid:
    sel_n, xf_n = build_replication(univ, w, K, n, TENOR_EDGES)
    te = tracking_error(R_out, xf_n, w)
    te_curve.append(te)
    print(f"n={len(sel_n):4d}  アウトオブサンプルTE = {te:6.1f} bp")

te_curve = np.array(te_curve)

# %% [markdown]
# ### 逓減カーブと実務的妥協点
#
# トラッキングエラーは銘柄数に対しておよそ $1/\sqrt{n}$ で逓減します。理論式の
# 第2項（個別リスク）が $\sum_i (x_i-w_i)^2\sigma_i^2 \sim 1/n$ に従うためです。
# 実務では「これ以上増やしても TE がほとんど減らない」肩の位置を妥協点にします。

# %%
key_ns = {10: None, 30: None, 100: None}
for n, te in zip(n_grid, te_curve):
    if n in key_ns:
        key_ns[n] = te

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(n_grid, te_curve, "o-", label="アウトオブサンプル TE")
# 1/sqrt(n) の理論減衰を n=10 の実測へ合わせて重ねる。
c = te_curve[0] * np.sqrt(n_grid[0])
ax.plot(n_grid, c / np.sqrt(n_grid), "--", color="gray", label=r"$\propto 1/\sqrt{n}$ 目安")
for n in (10, 30, 100):
    ax.annotate(f"{key_ns[n]:.0f}bp", (n, key_ns[n]),
                textcoords="offset points", xytext=(0, 10), ha="center")
ax.set_title("複製銘柄数とトラッキングエラーの逓減")
ax.set_xlabel("抽出銘柄数 n")
ax.set_ylabel("アウトオブサンプル TE (bp, 年率)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

drop_10_30 = key_ns[10] - key_ns[30]
drop_30_100 = key_ns[30] - key_ns[100]
print(f"TE: n=10 → {key_ns[10]:.0f}bp,  n=30 → {key_ns[30]:.0f}bp,  n=100 → {key_ns[100]:.0f}bp")
print(f"改善幅  10→30: {drop_10_30:.0f}bp,  30→100: {drop_30_100:.0f}bp")
print("30→100 の改善幅は 10→30 より小さく、逓減が寝ている。")
print("実務的妥協点：TE 低減が鈍る n=50〜75 前後。取引・管理コストと追随精度の釣り合いから、")
print("総合 JGB インデックスなら 50〜80 銘柄程度で実装するのが妥当と判断できる。")

# %% [markdown]
# ### セル法とファクター法のアウトオブサンプル比較
#
# 同じ抽出銘柄数で、セル法（最適化なし）と KRD ファクター法（最適化あり）の
# アウトオブサンプル TE を比べます。ファクター法は KRD ミスマッチを潰すぶん、
# カーブ由来の追随誤差を減らせます。

# %%
rows = []
for n in [10, 30, 100]:
    sel_n = stratified_select(univ, w, n, TENOR_EDGES)
    x_cell_n = cell_weights(univ, w, sel_n, TENOR_EDGES)
    x_fac_n = match_krd(K[sel_n], index_krd)
    rows.append({
        "n": len(sel_n),
        "セル法 TE(bp)": tracking_error(R_out, to_full(x_cell_n, sel_n, len(univ)), w),
        "ファクター法 TE(bp)": tracking_error(R_out, to_full(x_fac_n, sel_n, len(univ)), w),
    })
cmp = pd.DataFrame(rows)
cmp["削減率(%)"] = (1 - cmp["ファクター法 TE(bp)"] / cmp["セル法 TE(bp)"]) * 100
print(cmp.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

# %% [markdown]
# ## 演習
#
# 1. **銘柄数とトラッキングエラーの逓減**: `build_replication` を使い、$n$ を
#    5 から 200 まで細かく振ってアウトオブサンプル TE を測れ。TE を $n$ で
#    両対数プロットし、傾きが $-1/2$ に近い（$\mathrm{TE}\propto n^{-1/2}$）ことを
#    回帰で確かめよ。次に、TE を 15bp 以下に抑えるのに必要な最小の $n$ を求め、
#    その水準が実務的妥協点として妥当かを一言で述べよ。
# 2. **セル法 vs KRD ファクター法**: 年限バケットの切り方（`TENOR_EDGES`）を
#    粗く（例 4 バケット）／細かく（例 10 バケット）変え、$n=30$ で両手法の
#    アウトオブサンプル TE がどう動くかを比較せよ。セル法が細分化の恩恵を強く
#    受ける一方、ファクター法は KRD さえ合えばバケット数に鈍感なことを、数値で
#    説明せよ。
#
# 解答例は `solutions/S10/sol_1002.py` に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/10_portfolio.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | トラッキングエラー | tracking error | 複製とベンチマークのリターン差の標準偏差。追随の精度指標 |
# | 完全法 | full replication | 全構成銘柄を指数ウェイトで保有する複製。債券では流動性の壁で困難 |
# | 層化サンプリング | stratified sampling | 母集団をセルへ分割し各セルから代表を抽出する部分複製 |
# | セル法 | cell approach | セクション×年限×格付のセルへ層化し代表銘柄へ集約する複製法 |
