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
# # S7-3 Merton構造モデル
#
# ## 学習目標
#
# - 株式を「企業価値を原資産とするコールオプション」と捉える構造アプローチ（structural approach）の発想を説明できる
# - 距離to default（distance to default, DD）とデフォルト確率（probability of default, PD）を、企業価値・資産ボラ・負債から自力で計算できる
# - Merton理論とKMV流の実務化（default point の設定と実証マッピング）を区別し、両者の間の飛躍がどこにあるかを言える
# - 観測できる株価・株式ボラから、観測できない企業価値・資産ボラを反復解法で逆算できる
# - 構造モデル（structural model）と誘導型モデル（reduced-form model）の思想の違いを対比できる
#
# 本シリーズの `bondlab.credit` は、S7-1〜2 で強度（ハザード）モデルを組みました。
# S7-3 はそれと対をなす構造モデルを扱います。強度モデルがデフォルトを「外から
# 降ってくるジャンプ」として扱うのに対し、構造モデルは企業のバランスシートから
# デフォルトを内生的に説明します。


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# Merton の構造モデルは、株式市場の情報を信用の値付けに持ち込むための橋渡しです。株式を企業価値に対するコールオプションと読めば、株価・株式ボラという流動性の高い観測量から、社債やCDSが織り込むべきデフォルト確率を導けます。クレジットデスクやクレジットヘッジファンドは、この構造モデルが出す理論スプレッドと、市場で観測されるCDS・社債スプレッドを突き合わせ、乖離を相対価値の機会として捉えます。代表例が**キャピタルストラクチャー・アービトラージ**で、同一発行体の株式（またはエクイティ・オプション）とCDSの間の割高・割安に賭け、資本構成をまたいでヘッジしたポジションを組みます。距離to default（DD）が縮んでいるのに株が楽観的、といった不整合が仕掛けどころになります。
#
# 運用会社の信用アナリストやリスク部門は、KMV流に実務化したモデル（default point の設定とDDからEDFへの実証マッピング）を、発行体スクリーニングと格下げの早期警戒に使います。財務諸表ベースの格付が遅行的なのに対し、株価から毎日更新されるDD／EDFは信用悪化を先取りしやすく、ポートフォリオのアンダーウェイト判断や与信枠の見直しに効きます。docs でいえば、運用会社（債券）のアクティブ運用がクレジットの見立てでアルファを狙う局面の、判断材料の一つです。
#
# 構造モデルは、誘導型（ハザード）モデルが答えない「なぜこの発行体はデフォルトするのか」という因果を、レバレッジと資産ボラで説明します。両者は対立ではなく役割分担で、日々の値付け・ブートストラップは誘導型（S7-1〜2）が担い、構造モデルは発行体の信用力の水準観とストレス時の振る舞い（ウィングof default に近づくとスプレッドが非線形に跳ねる）を与えます。XVAやカウンターパーティ信用の文脈では、株価と信用が連動するライトウェイ／ラップウェイリスクの直感を得るのにも使われます。
# %% [markdown]
# ## 理論
#
# ### 構造モデルの出発点：株式はコールオプション
#
# Merton (1974) は、企業の資本構成をオプションの言葉で読み替えました。企業価値
# （資産価値）を $V_t$、満期 $T$ に一括返済される負債の額面を $D$ とします。満期
# 時点で株主が手にするのは、資産で負債を返した残りです。
#
# $$ E_T = \max(V_T - D,\ 0). $$
#
# これは行使価格 $D$、原資産 $V_T$ のコールオプションのペイオフそのものです。
# 株主は有限責任なので、$V_T < D$ なら会社を債権者に引き渡してゼロで降りられます。
# つまり **株式 = 企業価値に対するコールオプション**、**デフォルト = そのコールが
# アウト・オブ・ザ・マネーで満期を迎えること** と定義されます。
#
# 一方、債権者が受け取るのは $\min(V_T,\ D) = D - \max(D - V_T,\ 0)$ です。これは
# 「安全な債券をロング＋企業価値に対するプットをショート」と読めます。売った
# プットの価値が、信用リスクによる利回り上乗せ（クレジットスプレッド）の源泉です。
#
# ### 企業価値の想定と株式・負債の評価式
#
# 企業価値がリスク中立測度のもとで幾何ブラウン運動に従うとします。
#
# $$ dV_t = r\,V_t\,dt + \sigma_V\,V_t\,dW_t. $$
#
# すると株式はコールなので、Black–Scholes 式がそのまま使えます。
#
# $$ E_0 = V_0\,N(d_1) - D\,e^{-rT}\,N(d_2), $$
#
# $$ d_1 = \frac{\ln(V_0/D) + (r + \tfrac12 \sigma_V^2)T}{\sigma_V \sqrt{T}}, \qquad
# d_2 = d_1 - \sigma_V \sqrt{T}. $$
#
# ここで $r$ は無リスク金利、$\sigma_V$ は資産ボラティリティ（asset volatility）です。
# 負債（リスクのある社債）の現在価値は $B_0 = V_0 - E_0$ で、額面を現在価値に
# 割り引いた $D e^{-rT}$ との差が信用リスクぶんの割引です。
#
# ### 距離to default（DD）とデフォルト確率（PD）
#
# デフォルトは $V_T < D$ で起こります。リスク中立測度のもとで
#
# $$ \Pr(V_T < D) = N(-d_2). $$
#
# この $d_2$ を **距離to default（DD）** と呼びます。式を並べ替えると、DD は
# 「現在の対数レバレッジ $\ln(V_0/D)$ が、資産ボラ1単位ぶんで測って何個ぶん
# デフォルト点から離れているか」を表す標準化された距離だと読めます。
#
# $$ \mathrm{DD} = d_2 = \frac{\ln(V_0/D) + (r - \tfrac12 \sigma_V^2)T}{\sigma_V \sqrt{T}}, \qquad
# \mathrm{PD} = N(-\mathrm{DD}). $$
#
# DD が大きいほど、資産がデフォルト点まで落ちるには何標準偏差もの下落が必要で、
# PD は小さくなります。DD はレバレッジ（$V_0/D$）とボラ（$\sigma_V$）の二つを
# 一つの尺度に束ねているのが要点です。
#
# なお本ノートで計算する PD はリスク中立測度のもの（$N(-d_2)$）です。実世界の
# 期待デフォルト頻度はドリフトを $r$ から資産の期待収益 $\mu$ に置き換えた別物で、
# 一般に本ノートの値より小さくなります。両者の混同は構造モデルの典型的な誤りです。
#
# ### 観測できない企業価値の逆算
#
# 実務上の壁は、$V_0$ と $\sigma_V$ が市場で直接観測できないことです。観測できるのは
# 株式時価総額 $E_0$ と株式ボラ $\sigma_E$、そして会計上の負債 $D$ です。そこで
# 未知数2つ（$V_0, \sigma_V$）に対し式を2本立てて解きます。
#
# 1. 株式の評価式：$E_0 = V_0 N(d_1) - D e^{-rT} N(d_2)$
# 2. ボラの関係式：$\sigma_E\,E_0 = \dfrac{\partial E_0}{\partial V_0}\,\sigma_V\,V_0 = N(d_1)\,\sigma_V\,V_0$
#
# 2本目は伊藤の補題から出ます。株式ボラは、コールのデルタ $N(d_1)$ を通じて資産
# ボラと結びつきます。この連立を反復で解くのが、後述のスクラッチ実装の中心です。
#
# ### Merton理論からKMVへの飛躍
#
# ここまでが Merton の理論です。実務（KMV / Moody's Analytics の EDF）は、理論を
# そのまま使わず、いくつかの「飛躍」を挟んで実証モデルへ作り替えました。理論と
# 実務の境目を意識することが、この節の目的です。
#
# | 論点 | Merton理論 | KMV流の実務化 |
# |---|---|---|
# | 負債の扱い | 満期 $T$ の額面 $D$ を一括返済する単一のゼロクーポン債 | 短期負債の全額＋長期負債の半分などで **default point** を設定する |
# | デフォルト定義 | 満期時点で $V_T < D$ | 資産が default point を割った時点（実務的なしきい値） |
# | DD→PDの写像 | $N(-\mathrm{DD})$（正規分布を仮定） | DD を **実証デフォルト頻度の履歴** に写像した EDF テーブルを使う |
# | 分布の仮定 | 資産は対数正規 | 正規性は仮定せず、DD と実際のデフォルト率の対応を実データで作る |
#
# 最大の飛躍は DD→PD の写像です。Merton は $N(-\mathrm{DD})$ という理論分布を使い
# ますが、実際のデフォルトは正規分布の裾より頻繁に起こります。KMV は「DD が
# 同じ企業群を過去に追いかけ、1年後に何％がデフォルトしたか」を集計した実証
# テーブルで PD を与えます。default point の置き方（短期負債＋長期負債の半分）も
# 理論から演繹されたものではなく、実データへの当てはまりで選ばれた経験則です。
# 本ノートのスクラッチ実装は Merton 理論の側（$N(-\mathrm{DD})$）を実装し、KMV の
# 実証テーブルには踏み込みません。この線引きを意識してください。
#
# ### 構造モデルと誘導型モデルの思想の違い
#
# S7-1〜2 の強度モデルは **誘導型（reduced-form）** に分類されます。両者は同じ
# 信用リスクを別の哲学で捉えます。
#
# | 観点 | 構造モデル（Merton/KMV） | 誘導型モデル（ハザード） |
# |---|---|---|
# | デフォルトの原因 | 企業価値が負債を下回る（内生的・説明的） | 強度 $\lambda$ で到来するジャンプ（外生的・記述的） |
# | 主な入力 | 株価・株式ボラ・負債（バランスシート） | CDS スプレッド・社債価格（市場価格） |
# | デフォルトの予測可能性 | 資産が連続に動くため直前に近づく様子が見える | 突然到来し、直前でも予測できない |
# | 短期スプレッド | ゼロに収束しがち（連続過程では直前の急死が起きない） | $\lambda(1-R)$ に収束し、非ゼロを説明できる |
# | 得意な用途 | 未上場を含む企業の PD 推定、資本構成の分析 | 市場価格へのキャリブレーション、デリバティブ評価 |
#
# 構造モデルは「なぜデフォルトするか」を語れる代わりに、株価から出した短期
# スプレッドが実測より小さくなりがちです。誘導型は「なぜ」は語らない代わりに、
# 市場価格に素直に合わせられます。実務では両者を用途で使い分けます。


# %% [markdown]
# **数値例**：$V_0=120$、$D=80$、$\sigma_V=0.25$、$T=1$、$r=0.02$ のとき $d_1\approx 1.827$、$\mathrm{DD}=d_2\approx 1.577$ で、リスク中立デフォルト確率は $\mathrm{PD}=N(-\mathrm{DD})=N(-1.577)\approx 5.74\%$ です（資産がデフォルト点まで約 1.58 標準偏差ぶん離れている）。
#
# **数値例**：同じ数値で株式（コール）は $E_0=120\,N(1.827)-80\,e^{-0.02}\,N(1.577)\approx 42.0$、負債は残余 $V_0-E_0\approx 77.98$ です。額面の現在価値 $80\,e^{-0.02}\approx 78.42$ よりわずかに低く、その差 $\approx 0.44$ が信用リスクぶんの割引にあたります。
# %% [markdown]
# ## スクラッチ実装
#
# Merton の株式評価・DD・PD、そして株価から企業価値を逆算する反復解法を numpy と
# scipy だけで実装し、`bondlab.credit` と一致することを確認します。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `equity_call(V, sV, D, T, r)` | 企業価値, 資産ボラ, 負債, 満期, 金利 | 株式価値 | コールとしての株式評価 |
# | `debt_value(V, sV, D, T, r)` | 同上 | 負債価値 | $V - E$ でリスク社債の現在価値 |
# | `dd_pd(V, sV, D, T, r)` | 同上 | (DD, PD) | 距離to default と $N(-\mathrm{DD})$ |
# | `back_out_asset(E, sE, D, T, r)` | 株式価値, 株式ボラ, 負債, 満期, 金利 | (V, sV) | 反復で企業価値・資産ボラを逆算 |

# %%
import numpy as np
from scipy import stats

import bondlab
from bondlab import credit

print("bondlab version:", bondlab.__version__)

np.random.seed(0)


def equity_call(V, sV, D, T, r):
    """企業価値 V を原資産、負債額面 D を行使価格とするコールとしての株式価値。"""
    d1 = (np.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
    d2 = d1 - sV * np.sqrt(T)
    return V * stats.norm.cdf(d1) - D * np.exp(-r * T) * stats.norm.cdf(d2)


def debt_value(V, sV, D, T, r):
    """リスクのある社債の現在価値。株式がコールなら負債は残余（V - E）。"""
    return V - equity_call(V, sV, D, T, r)


def dd_pd(V, sV, D, T, r):
    """距離to default（d2）と、そのリスク中立デフォルト確率 N(-DD) を返す。"""
    dd = (np.log(V / D) + (r - 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
    pd = stats.norm.cdf(-dd)
    return float(dd), float(pd)


def back_out_asset(E, sE, D, T, r, tol=1e-10, max_iter=200):
    """観測できる株式価値 E・株式ボラ sE から、企業価値 V・資産ボラ sV を逆算する。

    連立2式（株式=コール、sE*E = N(d1)*sV*V）を不動点反復で解く。初期値は
    「企業価値 ≒ 株式時価総額 + 負債現在価値」「資産ボラ ≒ 株式ボラ」から始める。
    """
    V = E + D * np.exp(-r * T)   # 資産 ≈ 株式 + 負債の現在価値
    sV = sE                       # 資産ボラの初期値は株式ボラ
    for _ in range(max_iter):
        d1 = (np.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
        delta = stats.norm.cdf(d1)               # コールのデルタ = ∂E/∂V
        E_model = equity_call(V, sV, D, T, r)
        sV_new = sE * E / (delta * V)            # ボラ関係式を sV について解く
        V_new = V + (E - E_model)                # 株式評価式の残差で V を補正
        if abs(V_new - V) < tol and abs(sV_new - sV) < tol:
            V, sV = V_new, sV_new
            break
        V, sV = V_new, sV_new
    return float(V), float(sV)


# %% [markdown]
# ### bondlab との一致確認
#
# 適当な企業価値・資産ボラを置き、自作関数と `bondlab.credit` の各関数が一致する
# ことを `assert` で守ります。`bondlab.credit.merton_pd` は $N(-d_2)$ を返す約束です。

# %%
V0, sV0, D0, T0, r0 = 120.0, 0.25, 80.0, 1.0, 0.02

E_mine = equity_call(V0, sV0, D0, T0, r0)
E_lib = credit.merton_equity(V0, sV0, D0, T0, r0)
dd_mine, pd_mine = dd_pd(V0, sV0, D0, T0, r0)
dd_lib = credit.distance_to_default(V0, sV0, D0, T0, r0)
pd_lib = credit.merton_pd(V0, sV0, D0, T0, r0)

print(f"株式価値   自作={E_mine:.8f}  bondlab={E_lib:.8f}")
print(f"距離to def 自作={dd_mine:.8f}  bondlab={dd_lib:.8f}")
print(f"PD         自作={pd_mine:.8f}  bondlab={pd_lib:.8f}")
print(f"負債価値   V-E ={debt_value(V0, sV0, D0, T0, r0):.8f}"
      f"  額面現価={D0 * np.exp(-r0 * T0):.8f}")

assert abs(E_mine - E_lib) < 1e-12
assert abs(dd_mine - dd_lib) < 1e-12
assert abs(pd_mine - pd_lib) < 1e-12
print("スクラッチ実装は bondlab.credit と一致しました")

# %% [markdown]
# ## QuantLib検証
#
# QuantLib には Merton 構造モデルの部品がありません（信用は主にハザード／CDS 側で
# 提供されます）。そこで本ノートの検証は、外部ベンチマークではなく **往復テスト**
# と **自作実装 == `bondlab`** の二本立てで代替します。この位置づけを明記しておきます。
#
# - 往復テスト：企業価値 $V,\sigma_V$ から理論株式 $E,\sigma_E$ を作り、それを入力に
#   `back_out_asset` で $V,\sigma_V$ を逆算して、元に戻るかを確認する（内部整合性）。
# - 一致確認：`back_out_asset` と `bondlab.credit.solve_asset` の収束先が一致するかを
#   確認する（実装の独立検証）。
#
# ### 往復テスト：企業価値 → 株式 → 逆算企業価値
#
# 資産ボラの関係式 $\sigma_E E_0 = N(d_1)\sigma_V V_0$ から理論株式ボラを作ります。

# %%
# 1) 真の企業価値・資産ボラから、観測されるはずの株式価値・株式ボラを生成
d1_true = (np.log(V0 / D0) + (r0 + 0.5 * sV0 ** 2) * T0) / (sV0 * np.sqrt(T0))
delta_true = stats.norm.cdf(d1_true)
E_obs = equity_call(V0, sV0, D0, T0, r0)
sE_obs = delta_true * V0 / E_obs * sV0        # sE = N(d1) V sV / E

# 2) 観測量だけから企業価値・資産ボラを逆算
V_rec, sV_rec = back_out_asset(E_obs, sE_obs, D0, T0, r0)

print(f"真の企業価値   V ={V0:.8f}  逆算={V_rec:.8f}")
print(f"真の資産ボラ  sV ={sV0:.8f}  逆算={sV_rec:.8f}")
assert abs(V_rec - V0) < 1e-6
assert abs(sV_rec - sV0) < 1e-6
print("往復テスト（企業価値→株式→逆算）を通過しました")

# %% [markdown]
# ### bondlab.credit.solve_asset の収束確認
#
# 同じ観測量を `bondlab.credit.solve_asset` に渡し、自作の `back_out_asset` と同じ
# 企業価値・資産ボラへ収束することを確認します。

# %%
sol = credit.solve_asset(E_obs, sE_obs, D0, T0, r0)
print(f"solve_asset -> asset={sol['asset']:.8f}  asset_vol={sol['asset_vol']:.8f}")
print(f"back_out_asset -> V ={V_rec:.8f}  sV       ={sV_rec:.8f}")

assert abs(sol["asset"] - V_rec) < 1e-6
assert abs(sol["asset_vol"] - sV_rec) < 1e-6
assert abs(sol["asset"] - V0) < 1e-6
assert abs(sol["asset_vol"] - sV0) < 1e-6
print("solve_asset は自作実装・真値の双方と一致して収束しました")

# %% [markdown]
# ## 実データ適用
#
# 合成した実在風の2社に構造モデルを当てます。ネットワークは使わず、財務データは
# コード内に固定します。観測量は **株式時価総額・株式ボラ・負債（default point）**
# の3つで、これらから企業価値・資産ボラを逆算し、DD と PD を求めます。
#
# - A社：高格付を想定。低レバレッジ・低い株式ボラ。
# - B社：低格付を想定。高レバレッジ・高い株式ボラ。
#
# 単位は「億円」とし、負債は KMV 流に短期負債＋長期負債の一部を束ねた default point
# として与えます（値は例示。実証テーブルへの写像はここでは行いません）。

# %%
import pandas as pd

r_mkt, T_mkt = 0.01, 1.0

# 観測量: (株式時価総額, 株式ボラ, 負債=default point) 単位は億円
firms = {
    "A社（高格付・低レバレッジ）": dict(equity=3200.0, equity_vol=0.24, debt=2600.0),
    "B社（低格付・高レバレッジ）": dict(equity=500.0, equity_vol=0.50, debt=1800.0),
}

rows = []
for name, f in firms.items():
    sol = credit.solve_asset(f["equity"], f["equity_vol"], f["debt"], T_mkt, r_mkt)
    V, sV = sol["asset"], sol["asset_vol"]
    dd = credit.distance_to_default(V, sV, f["debt"], T_mkt, r_mkt)
    pd_val = credit.merton_pd(V, sV, f["debt"], T_mkt, r_mkt)
    rows.append({
        "企業": name,
        "株式時価総額": f["equity"],
        "株式ボラ": f["equity_vol"],
        "負債(DP)": f["debt"],
        "企業価値V": round(V, 1),
        "資産ボラσV": round(sV, 4),
        "レバレッジV/D": round(V / f["debt"], 3),
        "DD": round(dd, 3),
        "PD(%)": round(pd_val * 100, 4),
    })

df = pd.DataFrame(rows).set_index("企業")
display(df.T)

# 格付との整合: 高格付ほど DD が大きく PD が小さいはず
dd_a = df.loc[df.index[0], "DD"]
dd_b = df.loc[df.index[1], "DD"]
assert dd_a > dd_b, "高格付社の DD は低格付社より大きいはず"
print(f"\nDD: A社={dd_a} > B社={dd_b} — 高格付ほど DD が大きく、格付と整合します")

# %% [markdown]
# A社は DD が大きくデフォルト点から遠いため PD はほぼゼロ、B社は DD が小さく
# 1年 PD が有意に立ちます。DD の大小が想定した格付順序と一致しており、構造モデル
# が格付と整合するリスク序列を再現できていることが読めます。ただし A社の
# 「PD ≒ 0」は、Merton 理論が短期のデフォルトを過小評価しがちな性質（前掲の対比表）
# の表れでもあります。KMV の実証テーブルなら、同じ DD でもわずかに非ゼロの EDF を
# 与えます。

# %%
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
names = ["A社", "B社"]
axes[0].bar(names, df["DD"].values, color=["#2f6f4f", "#a13d3d"])
axes[0].set_title("距離to default（DD）")
axes[0].set_ylabel("DD（標準偏差の数）")
axes[1].bar(names, df["PD(%)"].values, color=["#2f6f4f", "#a13d3d"])
axes[1].set_title("1年デフォルト確率 PD")
axes[1].set_ylabel("PD (%)")
for ax in axes:
    ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 演習
#
# ### 演習1：2社のDD/PD計算と格付整合
#
# 次の3社目 C社（中格付を想定）を加え、A・B・C の3社について企業価値・資産ボラを
# 逆算し、DD と PD を求めよ。DD の順序が想定した格付順序（A > C > B）と整合するかを
# 確認し、整合しない場合はどの入力がそれを崩しているかを述べよ。
#
# - C社：株式時価総額 1200、株式ボラ 0.35、負債（default point）2000、$r=0.01$, $T=1$。
#
# ### 演習2：資産ボラ・負債水準を振ってDDの感応度
#
# A社の逆算後の企業価値 $V$ を固定したまま、(a) 資産ボラ $\sigma_V$ を 0.10〜0.40、
# (b) 負債 $D$ を $0.5V$〜$0.95V$ で振り、それぞれ DD がどう動くかをプロットせよ。
# DD はボラの上昇と負債の増加のどちらに強く反応するか、また PD への効き方の違いを
# 一言で述べよ。
#
# 解答例は `solutions/S7/sol_0703.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/07_credit.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | 構造モデル | structural model | 企業価値と負債の関係からデフォルトを内生的に説明するモデル |
# | 距離to default | distance to default (DD) | 企業価値がデフォルト点まで何標準偏差離れているかを測る指標（$d_2$） |
# | デフォルト確率 | probability of default (PD) | 一定期間内にデフォルトする確率。構造モデルでは $N(-\mathrm{DD})$ |
# | KMV | KMV | Merton を実務化し、default point と実証マッピングで EDF を与える枠組み |