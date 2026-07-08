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
# # S2-4 マルチカーブフレームワーク（OIS割引・SOFR先物）
#
# ## 学習目標
#
# - LIBOR改革後に「割引カーブ」と「推計（フォワード）カーブ」を分ける理由を、担保付取引と担保金利から説明できる
# - OIS割引（OIS discounting）が担保付デリバティブの評価で標準になった背景を、CSAと担保金利の関係で述べられる
# - 後決め複利RFR（backward-looking compounded RFR）の後決めスワップを、フォワードカーブと割引カーブに分けて価格式に落とせる
# - OISカーブと推計カーブのデュアルブートストラップ（dual bootstrap）をスクラッチで実装し、割引はOIS・フォワードは推計カーブ、というマルチカーブ評価ができる
# - シングルカーブ評価とマルチカーブ評価で、同じスワップのNPVがどれだけ食い違うかを数値で示せる
# - SOFR先物（SOFR futures）を短期ゾーンに使うときの凸性調整（convexity adjustment）の起源と、Hull-White下の解析近似を説明できる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# マルチカーブは、担保付デリバティブを扱う金利スワップデスクとXVAデスクにとっての必須の評価枠組みです。LIBOR改革後は、割引に使うカーブ（担保金利＝OIS/RFR）と、変動金利のフォワードを推計するカーブを分けるのが標準になりました。CSA付きの取引では受払いする担保に担保金利が付くため、割引金利は担保金利と一致していなければ理論的に整合しません。ここを単一カーブのまま評価すると、スワップのNPVが体系的にずれ、値付け・担保授受・ヘッジのすべてに誤差が乗ります。
#
# 収益とリスク管理への繋がりは明確です。マーケットメイクのデスクは、正しい割引・推計カーブから理論値を出して初めて、適正なビッド・オファーを提示できます。割引カーブを取り違えれば、日々の値洗いとP&Lがずれ、担保管理でも過不足が出ます。短期ゾーンでは、SOFR先物を使うときに凸性調整を入れないとフォワードを系統的に読み違え、これは先物ベーシスや短期RVの損益に直接効きます。デュアルブートストラップで割引はOIS・フォワードは推計カーブと役割を分けることが、これらの誤差を断つ前提になります。
#
# 具体的には、担保付スワップのポートフォリオを毎日評価し、シングルカーブ評価との差を把握したうえで、割引カーブの変化に対するリスク（OIS DV01）も別建てで管理します。マルチカーブを組めないと、担保付取引を正しく価格付けできず、XVAやファンディングの評価にも進めません。この notebook は、その標準枠組みをスクラッチで組み、シングルカーブとの乖離を数値で確かめることを主題にしています。
# %% [markdown]
# ## 理論
#
# ### なぜ割引カーブと推計カーブを分けるのか
#
# 2008年の金融危機まで、金利デリバティブの評価は1本のLIBORスワップカーブで
# 完結していました。将来のLIBORフォワードもこのカーブから読み、キャッシュ
# フローの割引もこのカーブで行う、という単一カーブ（single curve）の世界です。
#
# 危機で二つの前提が崩れました。第一に、LIBOR（無担保銀行間金利）とOIS
# （翌日物金利の複利）の差、いわゆるLIBOR-OISスプレッドが数十bpから百bp超へ
# 拡大し、テナーの違うLIBOR（3か月物と6か月物など）の間でも無視できない
# ベーシスが開きました。「銀行間で無リスクに貸せる1本の金利」という理想は
# 成り立たなくなりました。第二に、デリバティブ取引の大半がCSA（信用補完
# 契約, Credit Support Annex）のもとで担保付取引になりました。値洗いで生じた
# エクスポージャーを日々担保でやり取りし、その担保には担保金利（多くは翌日物
# のRFR）が付きます。
#
# ### 担保金利で割り引く — OIS割引の論理
#
# 担保付取引の評価は、無担保の割引率ではなく担保の資金調達コストで割り引く、
# というのが要点です。直感的には次のように考えます。担保を差し入れる側は、
# その担保に対して担保金利しか受け取れません。裏を返すと、担保付デリバティブ
# のヘッジ・資金繰りは担保金利で回るので、将来キャッシュフローの現在価値も
# 担保金利で割り引くのが整合的です。担保金利が翌日物RFRであれば、割引カーブは
# そのRFRを複利で積んだOISカーブになります。これがOIS割引（OIS discounting）です。
#
# ここで割引と推計の役割が分離します。
#
# | 役割 | 使うカーブ | 何を読むか |
# |---|---|---|
# | 割引（discounting） | OISカーブ（担保金利） | 将来CFの現在価値 $DF_{ois}(t)$ |
# | 推計（projection/forwarding） | 各インデックスのカーブ | 将来のフォワード金利 $L(t_1,t_2)$ |
#
# フォワード金利は「そのインデックスで将来借りるといくらか」を表すので、
# インデックス固有のカーブ（3か月LIBOR、6か月LIBOR、あるいはRFR）から読みます。
# 割引はどのインデックスの取引であっても共通の担保金利、すなわちOISカーブで
# 行います。1本だったカーブが、割引用（1本）＋推計用（インデックスごと）へ
# 分かれた、というのがマルチカーブ（multi-curve）の姿です。
#
# ### RFRへの移行 — SOFR / TONA / SONIA
#
# LIBORはパネル銀行の呈示に依存し操作リスクを抱えていたため、各通貨は
# 後決めの取引ベースRFR（リスクフリーレート, risk-free rate）へ移行しました。
#
# | 通貨 | RFR | 対象 |
# |---|---|---|
# | USD | SOFR | 国債レポ（担保付翌日物） |
# | JPY | TONA | 無担保コール翌日物 |
# | GBP | SONIA | 無担保翌日物 |
#
# RFRは翌日物（overnight）なので、LIBORのような「3か月先を1点で決める」
# フォワードタームレートを直接は持ちません。実務では観測期間の翌日物RFRを
# 複利で積み上げた後決め複利（backward-looking compounded）レートを使います。
#
# ### 後決め複利RFRと価格式
#
# 観測期間 $[t_1, t_2]$ の後決め複利RFRは、期間中の翌日物金利 $r_i$（適用日数
# $\delta_i$）を複利で積んで年率化したものです。
#
# $$
# R(t_1, t_2) = \frac{1}{\tau}\left(\prod_i \left(1 + r_i \delta_i\right) - 1\right),
# \qquad \tau = \sum_i \delta_i
# $$
#
# 割引カーブ（＝推計に使う同じRFRカーブ）で期待値を取ると、この複利の
# 積は望遠鏡のように連なり、期待複利成長は割引係数の比に一致します。
#
# $$
# \mathbb{E}\!\left[\prod_i (1 + r_i \delta_i)\right]
# = \frac{DF_{\text{proj}}(t_1)}{DF_{\text{proj}}(t_2)}
# $$
#
# したがって $[t_1, t_2]$ の期待複利RFRから求まる単純フォワード金利は、
# 推計カーブの割引係数の比だけで書けます。
#
# $$
# L(t_1, t_2) = \frac{1}{\tau}\left(\frac{DF_{\text{proj}}(t_1)}{DF_{\text{proj}}(t_2)} - 1\right)
# $$
#
# これはLIBOR時代のフォワード式と同じ形ですが、割引に使うカーブとは別の
# 推計カーブ $DF_{\text{proj}}$ から読む点が違います。金利スワップのフェアレートは、
# フロートレッグPVを固定レッグの年金価値（アニュイティ）で割ったものです。
# 割引はすべてOISカーブで行います。
#
# $$
# S = \frac{\displaystyle\sum_j DF_{ois}(t_j)\, L(t_{j-1}, t_j)\, \tau_j}
#          {\displaystyle\sum_i DF_{ois}(T_i)\, \alpha_i}
# $$
#
# 分子（フロート）と分母（固定アニュイティ）で割引はOIS、フォワードだけ推計
# カーブ、というのがマルチカーブ評価の実体です。
#
# ### デュアルブートストラップの考え方
#
# カーブが2本になると構築も2段になります。
#
# 1. OISカーブの構築：OIS（固定 vs 後決め複利RFR）のパーレートから $DF_{ois}$ を
#    剥ぎ取る。OISは割引も推計も同じRFRカーブなので、フロートレッグPVは
#    $1 - DF_{ois}(T_n)$ に畳めて、単一カーブのブートストラップで解けます。
# 2. 推計カーブの構築：手順1で固めた $DF_{ois}$ を割引に使いながら、スワップの
#    パーレートから $DF_{\text{proj}}$ を剥ぎ取る。各テナーで「そのスワップが
#    パー（NPV=0）になる」条件から末端ノードを1つずつ解きます。
#
# 手順2が割引カーブ（OIS）と推計カーブ（スワップ）を同時に使うため、
# デュアルブートストラップ（dual bootstrap）と呼びます。
#
# ### SOFR先物と凸性調整
#
# 短期ゾーンの推計カーブには、流動性の高いSOFR先物を入力に使えます。ただし
# 先物価格から読めるのは先物レート（futures rate）で、カーブが必要とする
# フォワードレート（FRAレート）とは一致しません。両者の差が凸性調整
# （convexity adjustment）です。
#
# 違いは日々の値洗い（マージン）にあります。先物は毎日値洗いされ、金利が
# 上がると先物ショート側に即日で損益が付きます。金利とマージンの資金繰りコスト
# は正に相関するので、金利が上がった（＝先物利益が出た）ときほど高い金利で
# 再運用でき、下がったときほど低い金利で調達すればよい、という非対称が
# 先物保有者に有利に働きます。この有利さの分だけ先物レートはフォワード
# （FRA）レートより高く付きます。
#
# $$
# \text{先物レート} = \text{フォワードレート} + \text{凸性調整}, \qquad \text{凸性調整} \ge 0
# $$
#
# Hull-Whiteモデル（平均回帰 $a$、ボラ $\sigma$）のもとで、$[t_1, t_2]$ を対象と
# する先物レートの凸性調整は次の解析近似で書けます（$B(u,v) = (1-e^{-a(v-u)})/a$）。
#
# $$
# \text{CA}(t_1, t_2)
# = \frac{B(t_1, t_2)}{t_2 - t_1}
#   \left[ B(t_1, t_2)\left(1 - e^{-2 a t_1}\right) + 2 a\, B(0, t_1)^2 \right]
#   \frac{\sigma^2}{4a}
# $$
#
# 平均回帰を消す極限 $a \to 0$ ではHo-Leeモデルの単純式へ落ちます。
#
# $$
# \text{CA}(t_1, t_2) \xrightarrow{a \to 0} \tfrac{1}{2}\,\sigma^2\, t_1\, t_2
# $$
#
# 満期 $t_1$ が遠いほど、ボラ $\sigma$ が大きいほど調整は二乗で効きます。数値は
# スクラッチ実装と演習2で確認します。



# %% [markdown]
# **数値例**：$\sigma=1\%$、$t_1=1,\ t_2=1.25$ 年で Ho-Lee 極限（$a\to0$）の凸性調整は $\text{CA}=\tfrac12\times0.01^2\times1\times1.25=6.25\times10^{-5}=0.63\,\mathrm{bp}$ です。平均回帰 $a=0.03$ の Hull-White 式では $0.60\,\mathrm{bp}$ とやや小さくなり、先物レートはこの分だけフォワード（FRA）レートより高く付きます。
# %% [markdown]
# **数値例**：3か月期間 $[1,\,1.25]$ 年（$\tau=0.25$）で推計カーブの割引係数が $DF_{\text{proj}}(1)=0.965605$、$DF_{\text{proj}}(1.25)=0.955997$ なら、単純フォワード金利は $L=\dfrac{1}{0.25}\left(\dfrac{0.965605}{0.955997}-1\right)=4.020\%$ です。割引に使う OIS カーブは分子・分母の $DF_{ois}$ 側にだけ現れ、フォワードはこの推計カーブの比だけで決まります。
# %% [markdown]
# ## スクラッチ実装
#
# OISカーブと推計カーブのデュアルブートストラップを notebook 内で実装します。
# 割引係数の器には `bondlab.curve.DiscountCurve` を2本使い、1本を割引（OIS）、
# もう1本を推計（スワップ）に割り当てます。単純化のため、OISもスワップも
# 年1回払い・整数年テナーの理想グリッドとし、フォワードは連続する年ノードの
# 割引係数比から読みます。この単純化により、各ブートストラップ段が閉形式で
# 解け、マルチカーブの骨格だけを取り出せます。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `df(curve, t)` | カーブ, 年数 | 割引係数(float) | `DiscountCurve.discount` をスカラで安全に呼ぶ薄いラッパ |
# | `bootstrap_ois(tenors, rates)` | テナー, OISパーレート | `DiscountCurve` | OISのパーから $DF_{ois}$ を剥ぎ取る（単一カーブ） |
# | `bootstrap_dual(tenors, swap_rates, disc)` | テナー, スワップパー, 割引カーブ | `DiscountCurve` | OIS割引下でスワップから $DF_{proj}$ を剥ぎ取る（デュアル） |
# | `swap_fair_rate(tenor, disc, proj)` | 満期, 割引カーブ, 推計カーブ | フェアレート | マルチカーブのパースワップレート |
# | `swap_npv(fixed, tenor, disc, proj)` | 固定レート, 満期, 割引, 推計 | NPV | 固定払い（フロート受け）から見たスワップNPV |
# | `swap_npv_single(fixed, tenor, curve)` | 固定レート, 満期, カーブ | NPV | 割引＝推計を同一カーブで行うシングルカーブ評価 |
# | `hw_convexity(t1, t2, a, sigma)` | 開始, 終了, 平均回帰, ボラ | 凸性調整 | Hull-White下の先物-フォワード凸性調整 |
# | `holee_convexity(t1, t2, sigma)` | 開始, 終了, ボラ | 凸性調整 | $a\to0$ 極限のHo-Lee単純式 |

# %%
import numpy as np

import bondlab
from bondlab.curve import DiscountCurve, bootstrap_par

np.random.seed(0)
print("bondlab version:", bondlab.__version__)


def df(curve: DiscountCurve, t) -> float:
    """DiscountCurve.discount をスカラ float で返す。

    discount はベクトル対応のため長さ1配列を返すことがある。np.ravel で
    1次元化して先頭要素を取り、numpy の 0 次元配列変換警告を避ける。
    """
    return float(np.ravel(curve.discount(t))[0])


def bootstrap_ois(tenors, rates) -> DiscountCurve:
    """OISのパーレートから割引係数を逐次に剥ぎ取る（年1回払い・整数年）。

    OISは割引も推計も同じRFRカーブなので、フロートレッグPVは望遠鏡和で
    $1 - DF(T_n)$ に畳める。固定レッグ年金 $\\sum DF$ と釣り合う条件

        K * (Σ_{k<n} DF_k + DF_n) = 1 - DF_n

    から $DF_n = (1 - K\\,\\Sigma_{k<n} DF_k)/(1 + K)$ を解く。
    """
    tenors = np.asarray(tenors, dtype=float)
    rates = np.asarray(rates, dtype=float)
    dfs = np.empty_like(tenors)
    annuity = 0.0  # Σ DF over previous nodes
    for i, (_T, K) in enumerate(zip(tenors, rates)):
        dfs[i] = (1.0 - K * annuity) / (1.0 + K)
        annuity += dfs[i]
    return DiscountCurve(tenors, dfs)


def bootstrap_dual(tenors, swap_rates, disc: DiscountCurve) -> DiscountCurve:
    """OIS割引カーブ disc を固定したまま、スワップから推計カーブを剥ぎ取る。

    年1回払いのスワップで、割引はすべて disc（OIS）、フォワードは推計カーブの
    連続年ノードの比 $DF_{proj}(i-1)/DF_{proj}(i) - 1$ から読む。末端ノード
    $DF_{proj}(n)$ 以外は既知なので、パー条件 フロートPV = S * アニュイティ を
    末端について解くと閉形式になる。

        DF_proj(n) = DF_proj(n-1) / (1 + RHS / DF_ois(n)),
        RHS = S_n * Σ_{i≤n} DF_ois(i) − Σ_{i<n} DF_ois(i)(DF_proj(i-1)/DF_proj(i) − 1)
    """
    tenors = np.asarray(tenors, dtype=float)
    swap_rates = np.asarray(swap_rates, dtype=float)
    proj_nodes = [1.0]        # DF_proj(0)=1 と各年ノードを積む
    dfs = np.empty_like(tenors)
    annuity = 0.0             # Σ DF_ois(i)
    prev_float = 0.0          # 既存ノードまでのフロートPV
    for i, (T, S) in enumerate(zip(tenors, swap_rates)):
        dfo = df(disc, T)
        annuity += dfo
        rhs = S * annuity - prev_float
        dfp_n = proj_nodes[-1] / (1.0 + rhs / dfo)
        proj_nodes.append(dfp_n)
        dfs[i] = dfp_n
        prev_float += dfo * (proj_nodes[-2] / dfp_n - 1.0)
    return DiscountCurve(tenors, dfs)


# %% [markdown]
# 評価関数を用意します。フォワードは推計カーブ、割引はOISカーブという分担を
# コードでそのまま書き下します。比較用に、割引も推計も同一カーブで行う
# シングルカーブ評価も置きます。

# %%
def swap_fair_rate(tenor: int, disc: DiscountCurve, proj: DiscountCurve) -> float:
    """マルチカーブのパースワップレート（年1回払い・整数年）。"""
    years = range(1, tenor + 1)
    annuity = sum(df(disc, t) for t in years)
    floatpv = sum(df(disc, t) * (df(proj, t - 1) / df(proj, t) - 1.0) for t in years)
    return floatpv / annuity


def swap_npv(fixed: float, tenor: int, disc: DiscountCurve, proj: DiscountCurve) -> float:
    """固定払い（フロート受け）から見たマルチカーブのスワップNPV。"""
    years = range(1, tenor + 1)
    annuity = sum(df(disc, t) for t in years)
    floatpv = sum(df(disc, t) * (df(proj, t - 1) / df(proj, t) - 1.0) for t in years)
    return floatpv - fixed * annuity


def swap_npv_single(fixed: float, tenor: int, curve: DiscountCurve) -> float:
    """割引＝推計を同一カーブで行うシングルカーブのスワップNPV。"""
    years = range(1, tenor + 1)
    annuity = sum(df(curve, t) for t in years)
    floatpv = sum(df(curve, t) * (df(curve, t - 1) / df(curve, t) - 1.0) for t in years)
    return floatpv - fixed * annuity


# %% [markdown]
# 合成の市場データを置きます。OISレートはスワップレートより一段低く、その差が
# LIBOR-OISスプレッド（テナーベーシス）に相当します。まずOISカーブを組み、
# それを割引に使ってスワップから推計カーブをデュアルブートストラップします。

# %%
tenors = list(range(1, 11))  # 1〜10年の整数グリッド
ois_rates = [0.0300, 0.0310, 0.0320, 0.0326, 0.0330, 0.0333, 0.0335, 0.0337, 0.0339, 0.0340]
swap_rates = [0.0320, 0.0330, 0.0338, 0.0344, 0.0349, 0.0353, 0.0356, 0.0359, 0.0361, 0.0363]

ois_curve = bootstrap_ois(tenors, ois_rates)
proj_curve = bootstrap_dual(tenors, swap_rates, ois_curve)

print(f"{'テナー':>6} {'OIS DF':>10} {'推計 DF':>10} {'OISゼロ%':>10} {'推計ゼロ%':>10}")
for t in tenors:
    print(f"{t:>6} {df(ois_curve, t):>10.6f} {df(proj_curve, t):>10.6f}"
          f" {float(ois_curve.zero_rate(t)) * 100:>10.4f} {float(proj_curve.zero_rate(t)) * 100:>10.4f}")

# %% [markdown]
# ### 整合性チェック：パースワップが 0 で評価されるか
#
# デュアルブートストラップが正しければ、構築に使った各テナーのスワップは
# パー（NPV=0）で再評価されるはずです。OIS割引でフロート・固定を評価し、
# NPVが機械精度でゼロに戻ることを `assert` で守ります。

# %%
max_abs_npv = 0.0
print(f"{'テナー':>6} {'パーNPV':>14} {'再計算フェア%':>14} {'入力スワップ%':>14}")
for t, s in zip(tenors, swap_rates):
    npv = swap_npv(s, t, ois_curve, proj_curve)
    fair = swap_fair_rate(t, ois_curve, proj_curve)
    max_abs_npv = max(max_abs_npv, abs(npv))
    print(f"{t:>6} {npv:>14.2e} {fair * 100:>14.6f} {s * 100:>14.6f}")

print(f"\nパーNPVの最大絶対値: {max_abs_npv:.2e}")
assert max_abs_npv < 1e-10  # 構築テナーはパーで戻る
# 5年フェアレートが入力に一致する。
assert abs(swap_fair_rate(5, ois_curve, proj_curve) - swap_rates[4]) < 1e-12

# %% [markdown]
# ### SOFR先物の凸性調整を実装する
#
# Hull-White下の凸性調整と、その $a\to0$ 極限のHo-Lee式を実装し、両者が極限で
# 一致することを確認します。3か月物SOFR先物（$t_2 - t_1 = 0.25$）を想定します。

# %%
def hw_convexity(t1: float, t2: float, a: float, sigma: float) -> float:
    """Hull-White下の先物-フォワード凸性調整。"""
    def bfun(u, v):
        return (1.0 - np.exp(-a * (v - u))) / a
    b = bfun(t1, t2)
    return b / (t2 - t1) * (b * (1.0 - np.exp(-2.0 * a * t1))
                            + 2.0 * a * bfun(0.0, t1) ** 2) * sigma ** 2 / (4.0 * a)


def holee_convexity(t1: float, t2: float, sigma: float) -> float:
    """Ho-Lee（平均回帰ゼロ）の単純式 ½σ²t₁t₂。"""
    return 0.5 * sigma ** 2 * t1 * t2


sigma = 0.010   # 年率1.0%の絶対ボラ
tau = 0.25      # 3か月

print(f"{'先物満期t1':>10} {'HL(a→0)%':>12} {'HW(a=0.03)%':>14} {'差(bp)':>10}")
for t1 in [0.5, 1.0, 2.0, 3.0, 5.0]:
    hl = holee_convexity(t1, t1 + tau, sigma)
    hw = hw_convexity(t1, t1 + tau, 0.03, sigma)
    print(f"{t1:>10} {hl * 100:>12.5f} {hw * 100:>14.5f} {(hl - hw) * 1e4:>10.3f}")

# a→0 でHWがHo-Leeへ収束する。
assert abs(hw_convexity(3.0, 3.25, 1e-8, sigma) - holee_convexity(3.0, 3.25, sigma)) < 1e-9

# %% [markdown]
# 平均回帰 $a>0$ は遠い満期での不確実性の伸びを抑えるので、凸性調整はHo-Lee
# より小さく出ます。満期が延びるほど両者の差（bp）が開きます。

# %% [markdown]
# ## QuantLib検証
#
# 何を検証するかを先に明示します。
#
# 1. **自作カーブの内部整合**（厳密）：デュアルブートストラップで組んだカーブ上で、
#    構築テナーのパースワップが NPV≈0 に戻ることを上のスクラッチ実装で機械精度
#    （$10^{-10}$ 未満）まで確認済みです。これがマルチカーブ構築の正しさの中心的な
#    根拠です。
# 2. **QuantLib による独立ビルド**（ばらつき許容）：QuantLib の `OISRateHelper` /
#    `SwapRateHelper` / `PiecewiseLogLinearDiscount` で、外生のOIS割引カーブ上に
#    スワップ推計カーブをデュアルに組み、5年スワップのフェアレートとパーNPVを
#    確認します。QuantLib は実際のカレンダー・日数計算・2営業日決済・半年払いを
#    使うので、自作の理想グリッド（年1回・整数年）とは規約が違い、数値は一致
#    しません。ここで見るのは「同じ枠組みが独立実装でも動き、パーで NPV=0 に
#    なる」という枠組みレベルの整合です。

# %%
import QuantLib as ql

today = ql.Date(15, 5, 2026)
ql.Settings.instance().evaluationDate = today
cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

# --- OISカーブ（担保金利, SOFR）---
sofr = ql.OvernightIndex("SOFR", 1, ql.USDCurrency(), cal, ql.Actual360())
ois_helpers = [
    ql.OISRateHelper(2, ql.Period(t, ql.Years),
                     ql.QuoteHandle(ql.SimpleQuote(r)), sofr)
    for t, r in zip(tenors, ois_rates)
]
ql_ois = ql.PiecewiseLogLinearDiscount(0, cal, ois_helpers, ql.Actual365Fixed())
ql_ois.enableExtrapolation()
ois_handle = ql.YieldTermStructureHandle(ql_ois)

print("QuantLib OIS カーブ（割引係数）:")
print(f"{'テナー':>6} {'QL OIS DF':>12} {'自作 OIS DF':>12}")
for t in [1, 2, 3, 5, 7, 10]:
    d = today + ql.Period(t, ql.Years)
    print(f"{t:>6} {ql_ois.discount(d):>12.6f} {df(ois_curve, t):>12.6f}")

# %% [markdown]
# QuantLib のOIS割引係数を外生で渡し、スワップのパーレートから推計カーブを
# デュアルに組みます。`SwapRateHelper` の最終引数に割引カーブハンドルを渡すと、
# QuantLib が内部でOIS割引・スワップ推計の分離ブートストラップを行います。

# %%
libor = ql.USDLibor(ql.Period(3, ql.Months))
sw_tenors = [2, 3, 5, 7, 10]
sw_rates = [swap_rates[t - 1] for t in sw_tenors]
swap_helpers = [
    ql.SwapRateHelper(
        ql.QuoteHandle(ql.SimpleQuote(r)), ql.Period(t, ql.Years),
        cal, ql.Semiannual, ql.ModifiedFollowing,
        ql.Thirty360(ql.Thirty360.BondBasis), libor,
        ql.QuoteHandle(), ql.Period(0, ql.Days), ois_handle,  # 外生OIS割引
    )
    for t, r in zip(sw_tenors, sw_rates)
]
ql_proj = ql.PiecewiseLogLinearDiscount(0, cal, swap_helpers, ql.Actual365Fixed())
ql_proj.enableExtrapolation()
proj_handle = ql.YieldTermStructureHandle(ql_proj)

# 推計カーブにインデックスを紐付け、OIS割引エンジンで5年スワップを組む。
libor_proj = ql.USDLibor(ql.Period(3, ql.Months), proj_handle)
swap5 = ql.MakeVanillaSwap(ql.Period(5, ql.Years), libor_proj, 0.0,
                           ql.Period(0, ql.Days),
                           discountingTermStructure=ois_handle)
ql_fair = swap5.fairRate()
print(f"QuantLib 5年スワップ フェアレート : {ql_fair * 100:.4f}%")
print(f"入力した5年スワップレート          : {swap_rates[4] * 100:.4f}%")
print(f"自作マルチカーブ 5年フェアレート   : {swap_fair_rate(5, ois_curve, proj_curve) * 100:.4f}%")

# パーで組んだスワップは NPV=0 に戻る（枠組みの整合）。
par_swap5 = ql.MakeVanillaSwap(ql.Period(5, ql.Years), libor_proj, ql_fair,
                               ql.Period(0, ql.Days),
                               discountingTermStructure=ois_handle)
print(f"QuantLib パースワップNPV           : {par_swap5.NPV():.2e}")
assert abs(par_swap5.NPV()) < 1e-6

# %% [markdown]
# QuantLib のフェアレートは規約差（半年払い・実カレンダー・2営業日決済・
# ログ線形補間の刻み方）のため入力とも自作とも数bpずれますが、パースワップは
# NPV≈0 に戻り、割引はOIS・推計は別カーブという枠組みが独立実装でも成立する
# ことを確認できます。厳密な整合は手順1の自作パーNPV（$10^{-10}$ 未満）で担保
# しています。

# %% [markdown]
# ## 実データ適用
#
# 合成データのまま、マルチカーブとシングルカーブで同じスワップのNPVを比べ、
# 差を定量化します。シングルカーブ側は、割引も推計もスワップカーブ1本で行う
# 危機前の評価に相当します。ここでは `bondlab.curve.bootstrap_par` でスワップ
# パーから1本の割引カーブを組み、それを割引にも推計にも使います。

# %%
single_curve = bootstrap_par(tenors, swap_rates, frequency=1)

# シングルカーブでも構築テナーはパーで戻る（1本カーブの整合）。
single_par_max = max(abs(swap_npv_single(s, t, single_curve))
                     for t, s in zip(tenors, swap_rates))
print(f"シングルカーブ パーNPV最大絶対値: {single_par_max:.2e}")
assert single_par_max < 1e-10

# %% [markdown]
# 固定レートを市場水準から外した「オフマーケット」スワップを両評価で価格付け
# します。5年・固定3.0%（受けフロート）のスワップで、割引カーブの違いが
# NPVにどれだけ効くかを見ます。

# %%
import pandas as pd

fixed_off = 0.030
rows = []
for tenor in [2, 5, 10]:
    npv_multi = swap_npv(fixed_off, tenor, ois_curve, proj_curve)
    npv_single = swap_npv_single(fixed_off, tenor, single_curve)
    rows.append({
        "満期(年)": tenor,
        "固定レート%": fixed_off * 100,
        "マルチNPV": round(npv_multi, 6),
        "シングルNPV": round(npv_single, 6),
        "差(想定元本比bp)": round((npv_multi - npv_single) * 1e4, 3),
    })

diff_table = pd.DataFrame(rows)
display(diff_table)

# %% [markdown]
# 割引カーブがOIS（低め）かスワップ（高め）かで、同じキャッシュフローの現在
# 価値が変わり、NPVに差が出ます。満期が延びるほど割引の効きが積み上がり、
# 差は拡大します。値は想定元本1あたりで、想定元本比のbpに直しています。

# %% [markdown]
# ### 金利水準を振って差の感応度を見る
#
# OISとスワップの水準差（テナーベーシス）を一律に広げると、割引カーブの分離が
# NPV差にどう効くかが見えます。ベーシスを 0→60bp まで振り、5年・固定3.0%
# スワップのマルチ／シングルNPV差をプロットします。

# %%
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
base_ois = np.array(ois_rates)
base_swap = np.array(swap_rates)
extra_basis = np.linspace(0.0, 0.0060, 25)  # 追加ベーシス 0〜60bp

diffs = []
for db in extra_basis:
    # スワップだけを持ち上げ、OIS-スワップ差（テナーベーシス）を広げる。
    oc = bootstrap_ois(tenors, base_ois)
    pc = bootstrap_dual(tenors, base_swap + db, oc)
    sc = bootstrap_par(tenors, base_swap + db, frequency=1)
    diffs.append((swap_npv(fixed_off, 5, oc, pc)
                  - swap_npv_single(fixed_off, 5, sc)) * 1e4)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot((extra_basis + (base_swap[4] - base_ois[4])) * 1e4, diffs, marker="o")
ax.set_xlabel("5年 スワップ-OIS ベーシス (bp)")
ax.set_ylabel("マルチ − シングル NPV 差 (想定元本比 bp)")
ax.set_title("割引カーブ分離によるNPV差 — テナーベーシス依存")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ベーシスが広いほどOIS割引とスワップ割引の乖離が大きくなり、マルチ／シングルの
# NPV差も広がります。危機前の低ベーシス環境では単一カーブでも実害が小さく、
# ベーシス拡大局面で割引カーブ分離が効いてくる、という歴史的経緯と整合します。

# %% [markdown]
# ## 演習
#
# 1. **シングル vs マルチのNPV差はどの金利水準で拡大するか**：本編の5年・固定3.0%
#    スワップについて、(a) OISとスワップの水準差（テナーベーシス）を 0〜80bp、
#    (b) 金利全体の水準（OISとスワップを同じ幅だけ平行シフト）を −100〜+100bp、
#    の2軸で振り、マルチ／シングルのNPV差を表にせよ。差を主に動かすのは
#    ベーシスか水準かを述べ、その理由を割引カーブの役割から説明せよ。
# 2. **SOFR先物の凸性調整の大きさ**：3か月物SOFR先物の凸性調整を、先物満期
#    $t_1 \in \{0.5, 1, 2, 3, 5, 7, 10\}$ 年、ボラ $\sigma \in \{0.5\%, 1.0\%, 1.5\%\}$
#    について Hull-White（$a=0.03$）とHo-Lee（$a\to0$）で計算し、bpで表にせよ。
#    調整が満期・ボラのそれぞれに対しどの次数で増えるか、平均回帰があると
#    どちら向きにずれるかを、式と数値の両面から説明せよ。
#
# 解答例は `solutions/S2/sol_0204.py` に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/02_curves.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | OIS | overnight index swap | 固定と翌日物金利の複利を交換するスワップ。担保金利の代理でカーブを組む |
# | SOFR/TONA | secured overnight financing rate / Tokyo overnight average rate | 米ドル・円の後決め翌日物RFR。LIBORの後継 |
# | CSA | credit support annex | 担保のやり取りを定める付属契約。割引に使う担保金利を規定する |
# | RFR | risk-free rate | 取引ベースの翌日物リスクフリー金利。SOFR/TONA/SONIA等 |
# | デュアルブートストラップ | dual bootstrap | OIS割引を固定し推計カーブをスワップから剥ぎ取る2段のカーブ構築 |
# | 凸性調整 | convexity adjustment | 先物レートとフォワード（FRA）レートの差。日々の値洗いに由来する |
