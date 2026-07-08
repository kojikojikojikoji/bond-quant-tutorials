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
# # S8-3 日本の証券化市場（RMBS・機構債）
#
# ## 学習目標
#
# - 住宅金融支援機構MBS（機構MBS）の仕組みを、超過担保・団体信用生命保険・
#   月次公表データの3点から説明できる
# - 日本のプリペイメント（期限前償還）が米国より金利感応度が低い理由を、
#   制度要因から述べられる
# - 日本市場の規模と投資家構成の特徴を把握し、それが商品性・価格形成に
#   与える含意を説明できる
# - 米国のPSA/計量モデル（S8-1）を日本の実績データに当てはめ、フィットの
#   悪さを定量化できる（本回の「検証」の位置づけ）
# - `bondlab.mbs` のキャッシュフロー生成器で日本型プールを評価し、団信・
#   超過担保が投資家キャッシュフローに与える影響を数値で示せる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# 機構MBS（JHF）は、米系のプリペイメントモデルをそのまま日本に持ち込むと外れる、という点にこそ運用上の妙味があります。米系運用会社が S8-1・S8-2 で組んだ金利感応度の高い S字カーブと OAS の枠組みは、日本では借換コストの高さ・金利環境の平坦さ・生活イベント主導の借換行動によって傾きが寝てしまい、そのままでは実績 CPR に合いません。本回で行う「米国モデルを日本の実績に当てはめてフィットの悪さを定量化する」作業は、実務でいえば市場標準モデルの当てはまりを疑い、日本固有のパラメータで引き直す最初の一歩です。この地味な作り込みが、海外勢が取りこぼす銘柄選別のアルファを生みます。
#
# 団信（団体信用生命保険）による非自発的償還が、金利にほとんど反応しない期限前償還の床（floor）を作る点は、投資家にとって二重の意味を持ちます。第一に、金利が上がっても最低限の元本が額面で返ってくるので、米国 MBS ほど強いネガティブコンベクシティを負わずに済み、金利ボラティリティが上がった局面での価格下落が相対的に軽く出ます。第二に、超過担保と機構保証で信用リスクが抑えられているぶん、値付けはやはりプリペイメントのタイミング読みが中心になります。国内の生保・年金・銀行など、負債に対して資産を組む ALM 主体にとっては、団信の床で期限前償還の不確実性が小さいこと自体が、長期の固定利回りを確保しやすい魅力になります。
#
# 稼ぎ方は米国型のボラティリティ取引より、キャリーと ALM 適合に寄ります。国内投資家は機構MBS を国債対比のスプレッド商品として保有し、金利に鈍感で予見しやすいキャッシュフローから安定したキャリーを取ります。月次公表データ（残高・期限前償還率・延滞率）を継続的に追ってモデルをキャリブレーションし直せば、市場コンセンサスとのわずかなプリペイメント観の差を、割安シリーズの選別に変えられます。米系運用会社の側から見れば、日本の低い金利感応度と分厚い非自発的成分を正しくモデル化できるかどうかが、日本の証券化に踏み込むかどうかの分かれ目になり、本回のフィット検証はその意思決定の土台を与えます。
# %% [markdown]
# ## 理論
#
# ### 住宅金融支援機構MBS（機構MBS）の仕組み
#
# 機構MBS（Agency RMBS of JHF）は、独立行政法人 住宅金融支援機構
# （JHF: Japan Housing Finance Agency）が、買取型「フラット35」等の長期・
# 固定金利の住宅ローン債権をプールし、それを裏付けに発行する債券です。
# 米国のエージェンシーMBS（Ginnie Mae / Fannie Mae / Freddie Mac）に対応する、
# 日本で最大規模の証券化商品です。投資家は住宅ローンの元利金を裏付けとした
# パススルー型のキャッシュフローを受け取ります。
#
# 投資家保護の中核は次の3点です。
#
# **超過担保（over-collateralization, OC）**
# 発行債券の残高よりも、裏付けとなる住宅ローンプールの残高を多めに置く構造です。
# 貸倒れが生じても、まず超過担保部分が損失を吸収するため、投資家の元本が
# 毀損しにくくなります。機構MBSでは信用補完の一環としてこの構造がとられ、
# 実務上は投資家が受け取る元本が額面どおり守られる度合いを高めます。
#
# **団体信用生命保険（団信, group credit life insurance）**
# 借入人が死亡・高度障害となった場合に、保険金でローン残高が一括弁済される
# 仕組みです。投資家からみると、借入人の死亡が「額面での期限前償還」として
# 現れます。これは金利水準とほとんど無関係に発生する**非自発的償還**であり、
# 日本のプリペイメントに、金利に鈍感な床（floor）を与えます。
#
# **月次公表データ（ディスクロージャー, disclosure）**
# 機構は各シリーズについて、残高・期限前償還率・延滞率などを毎月公表します。
# 投資家はこの月次データからプールの期限前償還実績を追跡し、モデルの
# キャリブレーションや評価に用います。本回ではこの公表系列を模した合成データを
# 使います。
#
# ### 日本のプリペイメント特性 — なぜ金利感応度が低いのか
#
# プリペイメントは、大きく **自発的（借換・繰上返済）** と **非自発的（売却・
# 転居・団信・デフォルト）** に分かれます。金利感応度を生むのは主に自発的な
# 借換です。米国では、市場金利がローン金利（WAC）を十分に下回ると、借換
# インセンティブ $I = \text{WAC} - r_{\text{mkt}}$ が正になり、借換の波が立ち、
# CPR（年率条件付き期限前償還率）が急峻なS字を描いて跳ね上がります。
#
# 日本では、同じインセンティブに対する反応が構造的に弱くなります。制度要因を
# 3つ挙げます。
#
# | 制度要因 | 内容 | 借換行動への効果 |
# |---|---|---|
# | 借換コストの相対的高さ | 抵当権抹消・再設定の登録免許税、司法書士報酬、事務手数料、保証料の負担 | 小さな金利差では借換が採算に乗らず、感応度を鈍らせる |
# | 金利環境の平坦さ | 長期にわたる低金利で、そもそも新旧金利差（インセンティブ）が小さい | インセンティブの分布が0近傍に集中し、S字の急峻部に届きにくい |
# | 商慣行・情報摩擦 | 借換の意思決定が生活イベント（転居・繰上）主導で、金利最適化の主体性が弱い | 反応がなだらかになり、非自発的要因の比重が上がる |
#
# 結果として日本のCPRは、(1) 水準が低め、(2) インセンティブに対する
# 傾き（感応度）が緩やか、(3) 団信・住み替えなどの金利に鈍感な非自発的成分の
# 比重が相対的に大きい、という特徴を持ちます。本回の「検証」は、この日米差を
# 実証的に定量化することにあります。
#
# ### 市場規模と投資家構成
#
# 機構MBSは日本の証券化市場で最大の残高を持つ資産クラスです。投資家層は、
# 国内の預金取扱金融機関（銀行・信用金庫等）、生命保険会社、年金、公的セクター
# が中心で、満期保有目的の**バイ・アンド・ホールド**志向が強いのが特徴です。
# トレーディング目的の短期売買が主体の米国市場に比べ、価格発見はゆるやかで、
# 流動性プレミアムやプリペイメントの読み違いが価格に緩やかに反映されます。
# バイ・アンド・ホールド投資家にとって、プリペイメントの安定性（読みやすさ）は
# それ自体が価値であり、日本のプリペイメントの低感応度は「予測可能性」という
# 面ではプラスに働きます。
#
# ### 商品性が投資家キャッシュフローに与える影響
#
# - **超過担保** → 信用損失をOC部分が吸収し、投資家の受取元本を額面に近づける。
# - **団信** → 金利に鈍感な非自発的償還の床を作り、CPRの下限を押し上げる一方で
#   金利感応度をさらに薄める。
# - **低い金利感応度** → 金利が下がってもWAL（加重平均年限）が米国ほど縮まない。
#   すなわち**ネガティブ・コンベクシティ（負の凸性）が米国MBSより弱い**。金利が
#   下がっても早期償還で持っていかれる度合いが小さく、価格の頭打ちが緩やかです。
#
# 以下では、これらを合成データと `bondlab.mbs` の評価で数値に落とします。


# %% [markdown]
# **数値例**：日本型モデルの団信の床は年率 $0.6\%$ です。シーズニング済み（$\text{age}=60$）でインセンティブ $I=-40\text{bp}$（借換不利）でも $\text{CPR}=5.6\%$ と 0 に落ちず、非自発的償還が下支えします。
# %% [markdown]
# ## スクラッチ実装
#
# ここでは、(1) 日本の月次ディスクロージャーを模した合成CPR系列を生成し、
# (2) 日本型・米国型それぞれのプリペイメント関数を自作し、(3) `bondlab.mbs` の
# キャッシュフロー生成器で日本型プールを評価します。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `seasoning_ramp(age, ramp, plateau)` | 経過月, 立上り月, 上限 | 係数(0〜plateau) | 新規プールの償還が立ち上がる季節性ランプ |
# | `refi_s_curve(incentive, k, center)` | 借換インセンティブ, 傾き, 中心 | 0〜1 | 借換反応のロジスティックS字 |
# | `cpr_japan(age, incentive)` | 経過月, インセンティブ | 年率CPR | 日本型（低感応度・団信の床あり） |
# | `cpr_us(age, incentive)` | 経過月, インセンティブ | 年率CPR | 米国型（高感応度・急峻S字） |
# | `fit_psa_multiple(ages, cpr)` | 経過月配列, 実績CPR | PSA倍率 | 実績CPRへPSA曲線を最小二乗で当てる |
# | `price_pool(cf, curve)` | CF辞書, 割引カーブ | 現在価値 | 月次CFを割引いてプール価格を出す |
#
# `bondlab.mbs` からは `psa_cpr` / `cpr_to_smm` / `smm_to_cpr` / `mbs_cashflows` /
# `weighted_average_life` を利用します（実装の詳細はS8-1で扱いました）。

# %%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import bondlab
from bondlab.mbs import (
    psa_cpr,
    cpr_to_smm,
    smm_to_cpr,
    mbs_cashflows,
    weighted_average_life,
)
from bondlab.curve import bootstrap_par

print("bondlab version:", bondlab.__version__)

SEED = 20260707
rng = np.random.default_rng(SEED)


# %% [markdown]
# 自作関数を定義します。インセンティブ $I=\text{WAC}-r_{\text{mkt}}$ は小数
# （例 $I=0.01$ は100bp）で扱います。日本型と米国型の差は、S字の**傾き $k$**、
# **借換で上乗せされるCPRの振幅**、そして日本型だけが持つ**団信の床**に集約
# されます。

# %%
def seasoning_ramp(age, ramp=30.0, plateau=1.0):
    """新規プールの償還立ち上がりを表す季節性ランプ。

    経過月 age が 0 から ramp までは線形に立ち上がり、以降は plateau で一定。
    PSA の 30ヶ月ランプと同じ発想。
    """
    age = np.asarray(age, dtype=float)
    return plateau * np.minimum(1.0, age / ramp)


def refi_s_curve(incentive, k, center):
    """借換インセンティブに対する反応（ロジスティックS字, 0〜1）。

    k が大きいほど急峻（=金利感応度が高い）、center はS字の変曲点。
    """
    incentive = np.asarray(incentive, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (incentive - center)))


def cpr_japan(age, incentive):
    """日本型プリペイメント（年率CPR）。

    構成: 団信の床 + 季節性 ×(住み替え等のターンオーバー + 弱い借換反応)。
    借換振幅を小さく、S字を緩やか（k 小）にし、金利感応度を低く設定する。
    """
    season = seasoning_ramp(age, ramp=30.0, plateau=1.0)
    turnover = 0.045                       # 生活イベント主導のベース
    refi = 0.030 * refi_s_curve(incentive, k=150.0, center=0.006)  # 弱い借換
    dansin_floor = 0.006                   # 団信等の非自発的償還の床
    return np.clip(dansin_floor + season * (turnover + refi), 0.0, 0.40)


def cpr_us(age, incentive):
    """米国型プリペイメント（年率CPR）。

    借換振幅が大きく、S字が急峻（k 大）で、インセンティブに強く反応する。
    米国の借換行動を代表するパラメータで固定する。
    """
    season = seasoning_ramp(age, ramp=30.0, plateau=1.0)
    turnover = 0.060
    refi = 0.450 * refi_s_curve(incentive, k=500.0, center=0.005)  # 強い借換
    return np.clip(season * (turnover + refi), 0.0, 0.80)


def fit_psa_multiple(ages, observed_cpr):
    """実績CPRへ PSA 曲線を最小二乗で当てたときの PSA 倍率を返す。

    観測 ≈ m × PSA100(age) を満たす倍率 m を、原点を通る回帰で解く。
    m = Σ(base·obs) / Σ(base·base)。米国式の age ドリブンな枠組みを日本
    データへ当てはめる操作に対応する。
    """
    ages = np.asarray(ages, dtype=float)
    base = psa_cpr(ages, 100.0)
    m = float(np.sum(base * observed_cpr) / np.sum(base * base))
    return m * 100.0  # PSA 表記（100 = PSA100）


def price_pool(cf, curve):
    """月次キャッシュフローを割引カーブで現在価値化する。"""
    t = cf["month"] / 12.0
    df = curve.discount(t)
    return float(np.sum(cf["cashflow"] * df))


# %% [markdown]
# ### 合成ディスクロージャー系列の生成
#
# 単一プールを120ヶ月観測した月次公表を模します。プールのWACは低金利下の
# 固定として $1.8\%$、市場の住宅ローン金利 $r_{\text{mkt}}$ が金利低下→反発と
# 動く経路を与え、インセンティブ $I=\text{WAC}-r_{\text{mkt}}$ を作ります。
# 観測CPRは日本型モデル＋小さな観測ノイズです。

# %%
n_obs = 120
age_path = np.arange(1, n_obs + 1)          # 経過月 = プール月齢
wac_obs = 0.018

# 市場金利: 低下してから反発する経路（インセンティブが正へ振れて戻る）
t_grid = np.linspace(0, 1, n_obs)
mkt_rate = 0.018 - 0.011 * np.sin(np.pi * t_grid) + 0.001 * rng.standard_normal(n_obs)
incentive_path = wac_obs - mkt_rate         # 借換インセンティブ（正=借換有利）

cpr_true = cpr_japan(age_path, incentive_path)
cpr_obs = np.clip(cpr_true + 0.003 * rng.standard_normal(n_obs), 0.0, None)

disc = pd.DataFrame(
    {
        "month": age_path,
        "market_rate": mkt_rate,
        "incentive": incentive_path,
        "cpr_observed": cpr_obs,
    }
)
display(disc.head())
print(f"\n観測CPRの平均: {cpr_obs.mean():.4f} / 最大: {cpr_obs.max():.4f}")
print(f"インセンティブ範囲: [{incentive_path.min():+.4f}, {incentive_path.max():+.4f}]")

# %% [markdown]
# 合成ディスクロージャーを可視化します。インセンティブが正に振れる局面で
# CPRが上がりますが、上がり方は緩やかです（日本型の低感応度）。

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(disc["month"], disc["incentive"] * 100, color="C1")
ax[0].axhline(0, color="gray", lw=0.8)
ax[0].set_title("借換インセンティブ I = WAC - 市場金利")
ax[0].set_xlabel("経過月")
ax[0].set_ylabel("インセンティブ (bp)")

ax[1].plot(disc["month"], disc["cpr_observed"] * 100, color="C0", label="観測CPR")
ax[1].set_title("月次ディスクロージャー（合成）: 実績CPR")
ax[1].set_xlabel("経過月")
ax[1].set_ylabel("CPR (%)")
ax[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 日本型と米国型のS字の対比
#
# 同じインセンティブ軸に対して、日本型と米国型のCPRを重ねます。米国型は
# 急峻に跳ね上がり、日本型は緩やかに立ち上がることが、感応度差の本質です。


# %% [markdown]
# **数値例**：経過月数 $\text{age}=60$・インセンティブ $I=+80\text{bp}$ のとき、日本型 $\text{CPR}=6.8\%$ に対し米国型は $42.8\%$ です。同じ借換誘因でも米国型が数倍強く反応することが数値で確認できます。
# %%
inc_axis = np.linspace(-0.005, 0.015, 200)
age_fixed = 60.0  # 十分にシーズニングしたプール
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(inc_axis * 100, cpr_us(age_fixed, inc_axis) * 100, label="米国型 CPR", color="C3")
ax.plot(inc_axis * 100, cpr_japan(age_fixed, inc_axis) * 100, label="日本型 CPR", color="C0")
ax.set_xlabel("借換インセンティブ (bp)")
ax.set_ylabel("CPR (%)  ※月齢60ヶ月")
ax.set_title("プリペイメントS字曲線: 日米の金利感応度の差")
ax.legend()
fig.tight_layout()
plt.show()

# 感応度（S字の最大傾き, %CPR / 100bp）を数値化
d_inc = inc_axis[1] - inc_axis[0]
slope_us = np.max(np.diff(cpr_us(age_fixed, inc_axis))) / d_inc * 0.01
slope_jp = np.max(np.diff(cpr_japan(age_fixed, inc_axis))) / d_inc * 0.01
print(f"最大感応度  米国型: {slope_us*100:6.2f} %CPR / 100bp")
print(f"最大感応度  日本型: {slope_jp*100:6.2f} %CPR / 100bp")
print(f"感応度比（米/日）: {slope_us/slope_jp:5.1f} 倍")

# %% [markdown]
# ### bondlab.mbs による日本型プールの評価
#
# WAC $2.0\%$・WAM $360$ヶ月のプールを、日本型CPRの下で評価します。将来の
# インセンティブは、持続的に借換有利（$I=+80$bp）というシナリオを置きます
# （金利が下がった局面を想定）。まずCPR→SMMへ変換し、`mbs_cashflows` に
# 渡します。

# %%
POOL_BAL = 100.0
POOL_WAC = 0.020
POOL_WAM = 360
ages_life = np.arange(1, POOL_WAM + 1)
incentive_scn = np.full(POOL_WAM, 0.008)   # 持続的に +80bp（借換有利）

cpr_jp_life = cpr_japan(ages_life, incentive_scn)
smm_jp_life = cpr_to_smm(cpr_jp_life)
cf_jp = mbs_cashflows(POOL_BAL, POOL_WAC, POOL_WAM, smm_jp_life)
wal_jp = weighted_average_life(cf_jp)

print(f"日本型プール  平均CPR: {cpr_jp_life.mean():.4f}")
print(f"日本型プール  WAL   : {wal_jp:.2f} 年")

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(cf_jp["month"], cf_jp["interest"], width=1.0, label="利息", color="C0")
ax.bar(cf_jp["month"], cf_jp["scheduled_principal"], width=1.0,
       bottom=cf_jp["interest"], label="予定元本", color="C2")
ax.bar(cf_jp["month"], cf_jp["prepayment"], width=1.0,
       bottom=cf_jp["interest"] + cf_jp["scheduled_principal"],
       label="期限前償還", color="C1")
ax.set_xlabel("経過月")
ax.set_ylabel("キャッシュフロー")
ax.set_title(f"日本型プールの月次キャッシュフロー（WAL={wal_jp:.1f}年）")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## QuantLib検証
#
# **本回の「検証」の位置づけ**：日本のRMBSに対応するプリペイメント・モデルは
# QuantLibに標準搭載されていません（QuantLibのMBS機能は米国式のPSA/計量が前提
# です）。そこで本回では、QuantLibを**日本型評価のベンチマークとしては使えない**
# ことを逆手にとり、検証を **「米国モデル（S8-1のPSA/計量）を日本データに当てはめた
# ときのフィットの悪さ（ミスフィット）を定量化する」** ことと位置づけます。これは
# 日米プリペイメント行動差の実証そのものです。
#
# なお、キャッシュフローの割引計算そのものは商品普遍なので、QuantLibを
# **割引エンジンの答え合わせ**には使えます。まずその一点だけ確認し、その後で
# ミスフィットの定量化に入ります。
#
# ### (a) 割引計算の QuantLib クロスチェック
#
# 日本型プールのキャッシュフローを、フラットな連続複利 $1.2\%$ で割引いた
# 現在価値を、`bondlab` の割引カーブと QuantLib の `FlatForward` の双方で
# 計算し、一致することを確認します。

# %%
import QuantLib as ql

flat_rate = 0.012

# bondlab: 定数割引レートの par から近似せず、直接フラットカーブを組む
tenors = np.arange(1, POOL_WAM + 1) / 12.0
dfs_flat = np.exp(-flat_rate * tenors)
from bondlab.curve import DiscountCurve
curve_flat = DiscountCurve(tenors, dfs_flat, interp="log_linear")
pv_bondlab = price_pool(cf_jp, curve_flat)

# QuantLib: 同じフラット連続複利カーブで割引
today = ql.Date(7, 7, 2026)
ql.Settings.instance().evaluationDate = today
ql_curve = ql.FlatForward(today, flat_rate, ql.Actual365Fixed(), ql.Continuous)
pv_ql = 0.0
for m, cf in zip(cf_jp["month"], cf_jp["cashflow"]):
    d = today + ql.Period(int(m), ql.Months)
    pv_ql += cf * ql_curve.discount(d)

print(f"bondlab 割引 PV : {pv_bondlab:.6f}")
print(f"QuantLib 割引 PV: {pv_ql:.6f}")
print(f"相対差         : {abs(pv_bondlab - pv_ql) / pv_bondlab:.2e}")
assert abs(pv_bondlab - pv_ql) / pv_bondlab < 5e-3
print("割引計算の整合を確認しました（差はカレンダー月換算の微差のみ）")

# %% [markdown]
# ### (b) 米国式PSAモデルの日本データへのミスフィット定量化
#
# 米国のツールキットで日本の実績を説明しようとする2通りを試します。
#
# 1. **PSA倍率フィット（age ドリブン）**: 観測CPRに PSA 曲線を最小二乗で当てる。
#    PSAは月齢だけの単調関数なので、インセンティブ主導の上下動を表現できず、
#    残差に構造が残ります。
# 2. **米国S字モデルの移植（高感応度固定）**: 米国の借換パラメータで固定した
#    `cpr_us` を同じインセンティブ経路に適用する。日本の実績を大きく過大予測
#    します。
#
# それぞれ RMSE・決定係数 $R^2$ で定量化し、参考として日本型モデルを当てた
# ときの誤差も並べます。

# %%
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(obs, pred):
    obs = np.asarray(obs)
    ss_res = np.sum((obs - np.asarray(pred)) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


# 1) PSA 倍率フィット
psa_hat = fit_psa_multiple(age_path, cpr_obs)
cpr_psa_pred = (psa_hat / 100.0) * psa_cpr(age_path, 100.0)

# 2) 米国S字モデルの移植
cpr_us_pred = cpr_us(age_path, incentive_path)

# 参考) 正しい枠組み（日本型モデル）
cpr_jp_pred = cpr_japan(age_path, incentive_path)

rows = []
for name, pred in [
    ("米国式 PSA倍率フィット", cpr_psa_pred),
    ("米国S字モデル移植", cpr_us_pred),
    ("（参考）日本型モデル", cpr_jp_pred),
]:
    rows.append(
        {
            "モデル": name,
            "RMSE(CPR)": rmse(cpr_obs, pred),
            "R^2": r2(cpr_obs, pred),
            "平均予測CPR": float(np.mean(pred)),
        }
    )
fit_table = pd.DataFrame(rows)
print(f"実績CPRの平均: {cpr_obs.mean():.4f}")
print(f"当てはめたPSA倍率: {psa_hat:.0f} (PSA表記)")
display(fit_table)

# %% [markdown]
# 残差と予測を可視化します。米国式は、インセンティブが正に振れる局面で
# 実績を大きく上回る（過大予測）か、あるいは age だけでは山谷を追えず残差が
# うねります。

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(age_path, cpr_obs * 100, "o", ms=3, color="C0", label="実績CPR")
ax[0].plot(age_path, cpr_psa_pred * 100, color="C4", label="米国式 PSA倍率フィット")
ax[0].plot(age_path, cpr_us_pred * 100, color="C3", label="米国S字モデル移植")
ax[0].plot(age_path, cpr_jp_pred * 100, color="C2", lw=1.2, ls="--", label="日本型モデル")
ax[0].set_xlabel("経過月")
ax[0].set_ylabel("CPR (%)")
ax[0].set_title("実績 vs 各モデル予測")
ax[0].legend(fontsize=8)

ax[1].plot(age_path, (cpr_psa_pred - cpr_obs) * 100, color="C4", label="PSA倍率フィット残差")
ax[1].plot(age_path, (cpr_us_pred - cpr_obs) * 100, color="C3", label="米国S字移植残差")
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_xlabel("経過月")
ax[1].set_ylabel("残差 (予測 - 実績, %CPR)")
ax[1].set_title("米国モデルの残差構造")
ax[1].legend(fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### (c) ミスフィットが投資家キャッシュフローに波及する大きさ
#
# モデル誤差は最終的に**評価誤差**として効きます。同じ将来インセンティブ
# シナリオ（持続的 $+80$bp）に対し、日本型CPRと米国型CPRでプールを走らせ、
# WALと価格の差を見ます。米国型モデルで日本のプールを評価すると、償還を
# 過大に見積もり、WALを大きく短く読み違えます。

# %%
smm_us_life = cpr_to_smm(cpr_us(ages_life, incentive_scn))
cf_us = mbs_cashflows(POOL_BAL, POOL_WAC, POOL_WAM, smm_us_life)
wal_us = weighted_average_life(cf_us)

# 割引カーブ（JGB風のパー利回りからブートストラップ）
par_tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30]
par_rates = [0.001, 0.002, 0.003, 0.005, 0.008, 0.011, 0.015, 0.018, 0.021]
# 月次グリッドへ内挿してからブートストラップ（等間隔の年次で簡便に）
jgb_curve = bootstrap_par(par_tenors, par_rates, frequency=1)

price_jp = price_pool(cf_jp, jgb_curve)
price_us = price_pool(cf_us, jgb_curve)

print(f"WAL  日本型: {wal_jp:6.2f} 年   米国型: {wal_us:6.2f} 年   差: {wal_jp - wal_us:+.2f} 年")
print(f"価格 日本型: {price_jp:8.4f}   米国型: {price_us:8.4f}   差: {price_jp - price_us:+.4f}")
print(f"価格差（額面100あたり）: {abs(price_jp - price_us):.4f} 円")
print("→ 米国モデルで日本プールを評価すると、償還を過大予測しWALを短く誤読し、"
      "価格を誤って評価します。これがミスフィットの実務コストです。")

# %% [markdown]
# ## 実データ適用
#
# 住宅金融支援機構のディスクロージャーは、月次で残高・期限前償還率・延滞率を
# 公表しています。ただし本教材では再配布可否が不明確なため、実データの
# 転載は行わず、**日本の低い金利感応度を反映した合成データ**で代替しています
# （前節までの `disc` がその合成公表系列です）。
#
# ここでは、日米の実務的な差を制度要因から整理し、合成データ上でそれが
# どう現れるかを対応づけます。
#
# | 論点 | 米国 | 日本 | データ上の現れ方 |
# |---|---|---|---|
# | 借換の金利感応度 | 高い（急峻なS字） | 低い（緩やかなS字） | インセンティブ正の局面でのCPR上昇幅が小さい |
# | 非自発的償還 | 相対的に小 | 団信・住み替えで床が厚い | 低インセンティブ局面でもCPRが0に落ちない |
# | 投資家 | 短期売買中心 | バイ・アンド・ホールド中心 | 予測可能性（低感応度）が価値として評価される |
# | 負の凸性 | 強い | 弱い | 金利低下時のWAL短縮が限定的 |
#
# 実データに移す際の実務注意：機構の公表CPRはシリーズ・発行年度で水準が
# 異なるため、単一パラメータで全体を当てるのではなく、発行ヴィンテージ別に
# シーズニングと感応度を推定するのが定石です。

# %%
# 合成データで「金利低下局面のWAL短縮の弱さ（負の凸性の弱さ）」を確認する。
inc_levels = np.linspace(-0.004, 0.012, 9)
wal_curve_jp, wal_curve_us = [], []
for inc in inc_levels:
    scn = np.full(POOL_WAM, inc)
    wal_curve_jp.append(weighted_average_life(
        mbs_cashflows(POOL_BAL, POOL_WAC, POOL_WAM, cpr_to_smm(cpr_japan(ages_life, scn)))))
    wal_curve_us.append(weighted_average_life(
        mbs_cashflows(POOL_BAL, POOL_WAC, POOL_WAM, cpr_to_smm(cpr_us(ages_life, scn)))))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(inc_levels * 100, wal_curve_us, "-o", color="C3", label="米国型")
ax.plot(inc_levels * 100, wal_curve_jp, "-o", color="C0", label="日本型")
ax.set_xlabel("借換インセンティブ (bp)")
ax.set_ylabel("WAL (年)")
ax.set_title("インセンティブに対するWALの反応（負の凸性の強弱）")
ax.legend()
fig.tight_layout()
plt.show()

print(f"WAL短縮幅（-40bp→+120bp）  米国型: {wal_curve_us[0]-wal_curve_us[-1]:.2f} 年")
print(f"WAL短縮幅（-40bp→+120bp）  日本型: {wal_curve_jp[0]-wal_curve_jp[-1]:.2f} 年")

# %% [markdown]
# ## 演習
#
# 1. **日本のプリペイメントが米国より金利感応度が低い理由**について、仮説を
#    3つ挙げてください。そのうち、本回の合成データで検証できるものを1つ選び、
#    定量的に検証してください。
#    - ヒント: 「借換コストが高いと、インセンティブがしきい値を超えるまで
#      反応しない」→ S字の変曲点 `center` を右にずらすと、低インセンティブ域で
#      CPRが立ち上がりにくくなる、という形で検証できます。
# 2. **団信・超過担保が投資家キャッシュフローに与える影響**を数値で示して
#    ください。団信については、団信の床 `dansin_floor` を $0$ と $0.006$ で
#    切り替えたときのWAL差を出し、超過担保については、信用損失が発生しても
#    OC部分が先に吸収する構造を、簡単な損失シナリオで元本毀損の有無として
#    示してください。
#
# 解答例は `solutions/S8/sol_0803.py` に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/08_securitization.md`。ここでは初出語の一行要約のみ
# 示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | 機構MBS | JHF MBS / Agency RMBS | 住宅金融支援機構が住宅ローン債権を裏付けに発行する日本最大の証券化商品 |
# | 超過担保 | over-collateralization | 裏付けプール残高を発行債券残高より多く置き、信用損失を先に吸収させる仕組み |
# | 団体信用生命保険 | group credit life insurance | 借入人の死亡等でローンが一括弁済される保険。金利に鈍感な非自発的償還を生む |
# | ディスクロージャー | disclosure | 機構が月次で公表する残高・期限前償還率・延滞率等の情報開示 |