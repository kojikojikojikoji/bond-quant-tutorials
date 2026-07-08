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
# # S6-3 スワップション
#
# ## 学習目標
#
# - スワップション（swaption）を「フォワードスワップレートのオプション」として定義し、payer / receiver を区別できる
# - アニュイティ（annuity）をニュメレールに選ぶと、フォワードスワップレートがマルチンゲールになることを、S4-5 のニュメレール変換の応用として説明できる
# - Black'76 によるスワップション評価「価格 = アニュイティ × Black」を自分で実装し、`bondlab` と一致することを確認できる
# - Jamshidian 分解（Jamshidian decomposition）が 1 ファクターモデルの下でのみ成立する理由を説明できる
# - キャッシュ決済（cash settlement）と物理決済（physical settlement）の違いが価値差を生むことを、アニュイティの取り違えとして理解できる
# - ATM ストラドル（straddle）のベガ・ガンマを数値計算し、スワップションが「ボラティリティを売買する道具」であることを説明できる
#
# 本 notebook は S4-5「リスク中立評価とニュメレール変換」で入口だけ触れたアニュイティ測度を、
# スワップションの評価として具体化します。割引・フォワードは S2 のカーブ、ボラは市場から
# 与え、`bondlab` と QuantLib で二重に答え合わせをします。


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# スワップションは金利ボラティリティを売買するための主力商品で、レートデスクのオプション帳の中心にあります。デスクは顧客（コーラブル債・仕組み預金の発行体、期限前償還オプション付きローンを持つ銀行、ヘッジ目的の年金など）にスワップションを売り買いし、原資産であるフォワードスワップレートのデルタをスワップで、ボラのリスクを他のスワップションで相殺します。「価格 = アニュイティ × Black」という評価式は、原資産をフォワードスワップレート、割引ニュメレールをアニュイティに取り替えたときの標準表示で、ここが正確でないとベガ・ガンマの符号と大きさを取り違え、ヘッジ後に意図しないボラのポジションが残ります。
#
# 収益の源泉は、ATM ストラドルのベガ・ガンマの数値で見たとおり、インプライドボラと実現ボラの差です。市場が織り込むボラが割高と見ればストラドルを売り、フォワードスワップレートのデルタを日々ヘッジしながらタイムディケイを取り、割安と見れば買ってガンマを取ります。これはマーケットメイクのスプレッド収益であると同時に、ボラそのものへの相対価値・マクロのポジションにもなります。さらにスワップションのボラ水準はコーラブル債・MBS のオプション評価（次の notebook）とつながっており、キャッシュ商品のオプション性を割安・割高に評価する RV の入力にもなります。
#
# キャッシュ決済と物理決済でアニュイティの取り方が変わり価値差が生じる点、Jamshidian 分解が1ファクターモデルでしか成立しない点は、実装とモデル選択の実務論点です。決済方式を取り違えれば約定価格とブックの評価がずれ、モデルバリデーション（S5-5）はこうした前提のずれと近似の適用範囲を独立に検証します。スワップションはさらに、複数のヨーロピアンに分解して評価するバミューダ商品（コーラブル）の構成部品でもあり、ここで固めた評価がデスクのエキゾチック帳全体の土台になります。
# %% [markdown]
# ## 理論
#
# ### 1. フォワードスワップとパースワップレート
#
# 満期 $T_e$（オプション満期）から始まり、支払日 $T_e=T_0<T_1<\dots<T_n$ で固定金利を
# 受け払いする**フォワードスタートの金利スワップ**を考えます。固定レッグの現在価値は、
# 各期の割引係数 $P(0,T_i)$ と日数 $\tau_i=T_i-T_{i-1}$ を使って
#
# $$
# \mathrm{PV}_{\text{fixed}} = K \sum_{i=1}^{n} \tau_i\, P(0,T_i) = K\cdot A(0),
# \qquad
# A(0) = \sum_{i=1}^{n} \tau_i\, P(0,T_i)
# $$
#
# と書けます。ここで $A(0)$ を**アニュイティ**（または PVBP、レベル）と呼びます。固定レッグ 1bp
# あたりの価値であり、以降の主役になります。
#
# 変動レッグの現在価値は、シングルカーブでは望遠鏡和で $P(0,T_e)-P(0,T_n)$ に畳み込まれます。
# 両レッグを等しくする固定金利が**フォワードパースワップレート**です。
#
# $$
# S(0) = \frac{P(0,T_e) - P(0,T_n)}{A(0)}.
# $$
#
# これは「フォワードスワップを今このレートで約定すれば価値ゼロ」というレートで、スワップションの
# 原資産にあたります。`bondlab` では `par_swap_rate` と `swap_annuity` が対応します（後述のとおり、
# これらは**スポットスタート**のスワップを前提とするので、フォワードスワップにはアニュイティの
# 差分を取って対応します）。
#
# ### 2. スワップションとペイオフ
#
# **スワップション**は、満期 $T_e$ に「固定金利 $K$ のスワップに入る権利」です。
#
# | 種別 | 権利 | 満期ペイオフ |
# |---|---|---|
# | ペイヤー（payer） | 固定を払い変動を受けるスワップに入る | $A(T_e)\,\big(S(T_e)-K\big)^+$ |
# | レシーバー（receiver） | 固定を受け変動を払うスワップに入る | $A(T_e)\,\big(K-S(T_e)\big)^+$ |
#
# 満期にスワップレート $S(T_e)$ が確定すると、権利行使で得られるスワップの価値は
# $A(T_e)\,(S(T_e)-K)$（ペイヤー）です。これが正のときだけ行使するので、ペイオフに
# アニュイティ $A(T_e)$ が掛かります。ペイヤーは $S$ のコール、レシーバーは $S$ のプットです。
#
# ### 3. アニュイティ測度とマルチンゲール性（S4-5 の応用）
#
# ここで S4-5 のニュメレール変換を使います。アニュイティ $A(t)=\sum_i \tau_i P(t,T_i)$ は
# 割引債の正係数和なので、それ自体が正価格の資産です。これを**ニュメレール**に選ぶと、
# 対応する**アニュイティ測度 $\mathbb{Q}^A$** の下で、任意の資産価格をアニュイティで割った量が
# マルチンゲールになります。
#
# フォワードスワップレートは、変動レッグと固定 1bp の価値の比
#
# $$
# S(t) = \frac{P(t,T_e)-P(t,T_n)}{A(t)}
# $$
#
# であり、分子 $P(t,T_e)-P(t,T_n)$ は**トレード可能な資産の価格**（フォワードスタートの
# 変動レッグ）です。よってその「アニュイティ建て価格」である $S(t)$ は $\mathbb{Q}^A$ の下で
# マルチンゲールになります。
#
# $$
# S(t) = \mathbb{E}^{\mathbb{Q}^A}\!\big[\,S(T_e)\,\big|\,\mathcal{F}_t\big].
# $$
#
# これがスワップション評価の鍵です。ニュメレールをマネーマーケット口座からアニュイティに
# 取り替えたことで、割引因子（本来は確率的）がニュメレール $A$ に吸収され、**スワップレート
# だけの 1 変数問題**に落ちます。
#
# ### 4. Black'76 による評価
#
# $\mathbb{Q}^A$ の下でペイヤースワップションの価格は、ニュメレール $A(t)$ を掛けた条件付き期待値です。
#
# $$
# V^{\text{pay}}(0) = A(0)\,\mathbb{E}^{\mathbb{Q}^A}\!\big[(S(T_e)-K)^+\big].
# $$
#
# ここで $S(T_e)$ が対数正規（$\mathbb{Q}^A$ 下でドリフトなし、ボラ $\sigma$）と仮定すると、
# 期待値は **Black'76** そのものになります。
#
# $$
# V^{\text{pay}}(0) = A(0)\,\big[\,S(0)\,\Phi(d_1) - K\,\Phi(d_2)\,\big],\qquad
# d_{1,2} = \frac{\ln(S(0)/K) \pm \tfrac12\sigma^2 T_e}{\sigma\sqrt{T_e}}.
# $$
#
# レシーバーは同じ $A(0)$ を掛けた Black プットです。つまり
#
# $$
# \boxed{\ \text{スワップション価格} = \text{アニュイティ} \times \text{Black}\ }
# $$
#
# が本 notebook で一致確認する中心式です。負金利域では対数正規が使えないので、$S$ を正規と
# 仮定する **Bachelier** 版に差し替えます（`bondlab` の `model="bachelier"`）。
#
# ### 5. Jamshidian 分解と 1 ファクターの制約
#
# スワップションは満期に $n$ 本の割引債（クーポン債）を一度に受け渡すオプションと見なせます。
# 一般に「複数資産のバスケットのオプション」は分解できませんが、**Jamshidian 分解**は
# 次を主張します。
#
# > 短期金利 $r$ が**単一のファクター** $x$ の単調関数であるモデル（1 ファクター）では、
# > すべての割引債価格 $P(T_e,T_i)$ が同じ $x$ の単調関数になる。したがって「バスケットが
# > 行使境界を超える」事象は、$x$ が単一の閾値 $x^\*$ を超える事象と**同値**になり、
# > バスケットのオプションが**各割引債オプションの和**にちょうど分解できる。
#
# 閾値 $x^\*$ はバスケット価値が行使価格に一致する点として一意に決まり、そこでの各割引債価格を
# 個別行使価格 $K_i=P(x^\*,T_i)$ に取れば、スワップション $=\sum_i c_i\times(\text{割引債オプション})$
# と書けます。QuantLib の `JamshidianSwaptionEngine` はこの分解を Hull-White で実装したものです
# （S5-4 で使いました）。
#
# **なぜ 1 ファクターでしか成立しないか。** 2 ファクター以上では、満期の割引債価格 $P(T_e,T_i)$ が
# 複数の状態変数 $(x,y)$ に依存します。すると「バスケットが行使境界を超える」領域は $(x,y)$ 平面上の
# 曲線で仕切られ、$x$ 単独の閾値では表せません。行使境界が 1 次元の閾値に潰れないため、各割引債
# オプションの和への厳密分解ができないのです。1 ファクターは、すべての年限の金利が完全相関
# （同じ $x$ で動く）という強い仮定であり、その代償として厳密解が得られます。
#
# ### 6. キャッシュ決済と物理決済
#
# 満期の受け渡し方法に 2 種類あります。
#
# | 決済方式 | 満期に起きること | 使うアニュイティ |
# |---|---|---|
# | 物理決済（physical） | 実際にスワップを開始する | 割引カーブ由来の $A(T_e)=\sum_i \tau_i P(T_e,T_i)$ |
# | キャッシュ決済（cash） | スワップの現在価値を現金で受け取る | 決済レート $S(T_e)$ の内部収益率で割った $A_{\text{cash}}(S)=\sum_i \dfrac{\tau_i}{(1+\tau S)^{i}}$ |
#
# 物理決済では割引カーブ全体で固定レッグを割り引くので、評価は前述の $A(0)\times$Black で厳密です。
# キャッシュ決済では、市場慣行として**単一の決済スワップレート $S(T_e)$ を内部収益率**として固定
# レッグを割り引きます。$A_{\text{cash}}$ は $S$ の関数であり、割引カーブの形とは切り離されます。
#
# 価値差が出る理由は、$\mathbb{Q}^A$ 下でのペイオフに、物理決済では現れない**アニュイティ比
# $A_{\text{cash}}(S(T_e))/A(T_e)$** が余分に掛かるからです。この比は定数ではなく $S(T_e)$ に依存
# するので、期待値の中で行使確率と相関し、**コンベクシティ調整**を生みます。一次近似（調整前）では
#
# $$
# V_{\text{cash}} \approx A_{\text{cash}}(S(0))\times\text{Black},\qquad
# V_{\text{phys}} = A(0)\times\text{Black}
# $$
#
# となり、差はおおむね $\big(A_{\text{cash}}(S(0))-A(0)\big)\times\text{Black}$ です。実データ適用で
# この差を数値で見ます。


# %% [markdown]
# **数値例**：ATM ペイヤースワップション（$S(0)=K=3.0\%,\ T_e=5,\ \sigma=25\%,\ A(0)=4.0$）では、Black 部分が $S\,\Phi(d_1)-K\,\Phi(d_2)=0.006604$ なので価格は $A(0)\times\text{Black}=4.0\times0.006604=0.026417$ です。
#
# **数値例**：キャッシュ決済アニュイティ $A_{\text{cash}}(S)=\sum_{i=1}^{n}\dfrac{\tau}{(1+\tau S)^{i}}$ に $S=3.0\%,\ \tau=1,\ n=5$ を代入すると $A_{\text{cash}}=\dfrac{1}{1.03}+\cdots+\dfrac{1}{1.03^5}=4.5797$ となります。
# %% [markdown]
# ## スクラッチ実装
#
# 「価格 = アニュイティ × Black」を自作し、`bondlab.pricing.swaption_black` と一致させます。
# 続いて ATM ストラドルのベガ・ガンマを数値計算し、解析式と突き合わせます。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `black76_scratch(F, K, T, vol, option)` | フォワード, 行使, 満期, ボラ, 種別 | Black 価格（df=1） | 対数正規オプションの素の値 |
# | `swaption_scratch(F, K, T, vol, annuity, option)` | 上記 + アニュイティ | スワップション価格 | アニュイティ × Black |
# | `forward_swap(curve, expiry, tenor, freq)` | カーブ, 満期, テナー, 頻度 | (S0, annuity) | フォワードスワップレートと前方アニュイティ |
# | `num_vega(price_fn, vol, h)` | 価格関数, ボラ, 幅 | ベガ | ボラ中心差分 $\partial V/\partial\sigma$ |
# | `num_gamma(price_fn, F, h)` | 価格関数, フォワード, 幅 | ガンマ | フォワード 2 階差分 $\partial^2 V/\partial F^2$ |

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.stats import norm

import bondlab
from bondlab.pricing import par_swap_rate, swap_annuity, swaption_black, black76, bachelier
from bondlab.curve import bootstrap_par

print("bondlab version:", bondlab.__version__)

SEED = 20260707
rng = np.random.default_rng(SEED)
plt.rcParams["axes.unicode_minus"] = False


def black76_scratch(F, K, T, vol, option="call"):
    """Black'76（対数正規、df=1）でのコール/プット価格を自作する。"""
    if T <= 0 or vol <= 0:
        payoff = max(F - K, 0.0) if option == "call" else max(K - F, 0.0)
        return payoff
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if option == "call":
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def swaption_scratch(F, K, T, vol, annuity, option="payer"):
    """スワップション価格 = アニュイティ × Black。payer=コール、receiver=プット。"""
    opt = "call" if option == "payer" else "put"
    return annuity * black76_scratch(F, K, T, vol, opt)


def forward_swap(curve, expiry, tenor, freq=1):
    """割引カーブからフォワードスワップレート S0 と前方アニュイティ A を返す。

    T_e から始まり T_e+tenor まで、年 freq 回払いのフォワードスワップ。
    bondlab.swap_annuity はスポットスタート前提（先頭 tau を t=0 から測る）なので、
    「満期までの全アニュイティ」から「満期前アニュイティ」を差し引いて前方アニュイティを得る。
    S0 = (P(T_e) - P(T_n)) / A。
    """
    tau = 1.0 / freq
    n_pre = int(round(expiry * freq))
    n_all = int(round((expiry + tenor) * freq))
    times_all = tau * np.arange(1, n_all + 1)
    times_pre = tau * np.arange(1, n_pre + 1)
    ann = swap_annuity(curve, times_all) - swap_annuity(curve, times_pre)
    s0 = (curve.discount(expiry) - curve.discount(expiry + tenor)) / ann
    return float(s0), float(ann)


def num_vega(price_fn, vol, h=1e-4):
    """ボラに関する中心差分ベガ。price_fn(vol) が価格を返す。"""
    return (price_fn(vol + h) - price_fn(vol - h)) / (2 * h)


def num_gamma(price_fn, F, h=1e-4):
    """フォワードに関する 2 階差分ガンマ。price_fn(F) が価格を返す。"""
    return (price_fn(F + h) - 2 * price_fn(F) + price_fn(F - h)) / h ** 2


# %% [markdown]
# ### 自作と bondlab の一致確認
#
# 適当なフォワードスワップレート・アニュイティ・ボラで、自作 `swaption_scratch` と
# `bondlab.pricing.swaption_black` が一致することを確認します。

# %%
F0, K, expiry, vol, annuity = 0.030, 0.030, 5.0, 0.25, 4.0

for opt in ("payer", "receiver"):
    mine = swaption_scratch(F0, K, expiry, vol, annuity, opt)
    lib = swaption_black(F0, K, expiry, vol, annuity, option=opt)
    lib_manual = annuity * black76(F0, K, expiry, vol, df=1.0,
                                   option="call" if opt == "payer" else "put")
    print(f"{opt:9s}: 自作={mine:.8f}  bondlab={lib:.8f}  手計算={lib_manual:.8f}")
    assert abs(mine - lib) < 1e-12
    assert abs(mine - lib_manual) < 1e-12

# ITM/OTM もまとめて確認する。
for k in (0.020, 0.025, 0.030, 0.035, 0.040):
    for opt in ("payer", "receiver"):
        assert abs(swaption_scratch(F0, k, expiry, vol, annuity, opt)
                   - swaption_black(F0, k, expiry, vol, annuity, option=opt)) < 1e-12
print("→ 自作と bondlab.swaption_black がすべて一致しました")

# %% [markdown]
# ### ペイヤー・レシーバー・パリティ
#
# Black のプット・コール・パリティより、$C-P = S(0)-K$ です。両辺にアニュイティを掛けると
#
# $$
# V^{\text{pay}} - V^{\text{rec}} = A(0)\,(S(0)-K)
# $$
#
# となり、右辺は**フォワードスワップの現在価値**そのものです。つまり「ペイヤー買い＋レシーバー
# 売り」は、行使価格 $K$ のフォワードペイヤースワップと同じ持ち高になります。

# %%
for k in (0.020, 0.030, 0.045):
    vpay = swaption_black(F0, k, expiry, vol, annuity, option="payer")
    vrec = swaption_black(F0, k, expiry, vol, annuity, option="receiver")
    fwd_swap_pv = annuity * (F0 - k)
    print(f"K={k:.3f}: payer-receiver={vpay - vrec:+.8f}  A*(S0-K)={fwd_swap_pv:+.8f}")
    assert abs((vpay - vrec) - fwd_swap_pv) < 1e-12
print("→ payer − receiver = アニュイティ × (S0 − K) = フォワードスワップ PV")

# %% [markdown]
# ### ATM ストラドルのベガ・ガンマ
#
# **ストラドル**は同一行使・同一満期のペイヤーとレシーバーを同時に買う持ち高です。ATM
# （$K=S(0)$）では方向感（デルタ）がほぼ消え、ボラティリティへの感応度＝**ベガ**が最大化
# されます。ストラドルは「ボラを買う」代表的な組成です。
#
# ベガとガンマを数値微分で求め、解析式と突き合わせます。単一 Black オプションのベガ・ガンマは
#
# $$
# \frac{\partial V}{\partial\sigma} = A\,S\,\varphi(d_1)\sqrt{T},\qquad
# \frac{\partial^2 V}{\partial S^2} = A\,\frac{\varphi(d_1)}{S\,\sigma\sqrt{T}}
# $$
#
# で、コール・プットで共通です。したがってストラドル（2 本）ではこの 2 倍になります。


# %% [markdown]
# **数値例**：上の ATM ストラドル（$d_1=\tfrac12\sigma\sqrt{T_e}=0.2795$）のベガは $2AS\,\varphi(d_1)\sqrt{T_e}=2\times4.0\times0.030\times\varphi(0.2795)\times\sqrt5=0.2059$。ボラ $+1\%\text{pt}$ では $0.2059\times0.01=0.00206$ だけ価格が動きます。
# %%
def straddle_price(F, K, T, vol, annuity):
    """ATM 近傍のストラドル価格 = payer + receiver。"""
    return (swaption_black(F, K, T, vol, annuity, option="payer")
            + swaption_black(F, K, T, vol, annuity, option="receiver"))


K_atm = F0  # ATM
# 数値ベガ：ボラを動かす（F, K, T, annuity は固定）
vega_num = num_vega(lambda v: straddle_price(F0, K_atm, expiry, v, annuity), vol)
# 数値ガンマ：フォワードを動かす（K は固定のまま）
gamma_num = num_gamma(lambda F: straddle_price(F, K_atm, expiry, vol, annuity), F0)

# 解析式（単一オプションを 2 倍）
d1 = (np.log(F0 / K_atm) + 0.5 * vol ** 2 * expiry) / (vol * np.sqrt(expiry))
vega_ana = 2 * annuity * F0 * norm.pdf(d1) * np.sqrt(expiry)
gamma_ana = 2 * annuity * norm.pdf(d1) / (F0 * vol * np.sqrt(expiry))

print(f"ストラドル価格              = {straddle_price(F0, K_atm, expiry, vol, annuity):.8f}")
print(f"ベガ  : 数値={vega_num:.6f}  解析={vega_ana:.6f}  差={abs(vega_num - vega_ana):.2e}")
print(f"ガンマ: 数値={gamma_num:.6f}  解析={gamma_ana:.6f}  差={abs(gamma_num - gamma_ana):.2e}")
assert abs(vega_num - vega_ana) < 1e-3
assert abs(gamma_num - gamma_ana) < 1e-2
print("→ 数値微分と解析式が一致（ストラドルはベガ・ガンマの塊）")

# %% [markdown]
# ### ボラを 1 %pt 動かすと価格はいくら動くか
#
# ベガは「ボラ 1.0（＝100%pt）あたりの価格変化」です。実務で使う「1 %pt（＝0.01）あたり」は
# ベガ $\times 0.01$ です。ストラドルの損益がほぼボラ変化だけで決まる様子を図示します。

# %%
vols = np.linspace(0.10, 0.40, 61)
prices = np.array([straddle_price(F0, K_atm, expiry, v, annuity) for v in vols])
tangent = (straddle_price(F0, K_atm, expiry, vol, annuity)
           + vega_ana * (vols - vol))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(vols * 100, prices, color="steelblue", lw=2, label="straddle price")
ax.plot(vols * 100, tangent, color="crimson", ls="--", lw=1.5,
        label=f"vega tangent at {vol*100:.0f}%")
ax.axvline(vol * 100, color="gray", ls=":", lw=1)
ax.set_xlabel("implied vol (%)")
ax.set_ylabel("ATM straddle price")
ax.set_title("ATM straddle is (almost) linear in vol near the money")
ax.legend()
plt.tight_layout()
plt.show()

print(f"ボラ +1%pt でのストラドル価格変化（近似）= {vega_ana * 0.01:.6f}")

# %% [markdown]
# ## QuantLib 検証
#
# QuantLib の `BlackSwaptionEngine` と突合します。**検証の位置づけ**を明記すると、QuantLib は
# フォワードスワップレートとアニュイティを（Act/365 の日数計算・スケジュールから）独立に構築
# します。したがって、
#
# 1. QuantLib のスワップション NPV が「（QuantLib 自身の）アニュイティ × Black」に一致すること、
# 2. その値が `bondlab` のフォワードスワップレート・アニュイティで組んだ評価と（日数計算差の範囲で）
#    一致すること、
#
# を確かめます。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

TODAY = ql.Date(1, 1, 2024)
ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()

# 合成カーブ（後段の実データ適用と共通に作る）。
par_tenors = np.arange(1, 31, dtype=float)
par_rates = 0.03 + 0.01 * (1.0 - np.exp(-par_tenors / 8.0))
curve = bootstrap_par(par_tenors, par_rates, frequency=1)


def bondlab_to_ql_curve(curve, max_t=30.0):
    """bondlab の割引カーブを QuantLib の DiscountCurve ハンドルへ写す。"""
    node_t = np.concatenate([[0.0], np.arange(1, int(max_t) + 1, dtype=float)])
    dates = [TODAY + ql.Period(int(round(t * 365)), ql.Days) for t in node_t]
    dfs = [1.0] + [float(curve.discount(t)) for t in node_t[1:]]
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, DC))


ts = bondlab_to_ql_curve(curve)
ql_index = ql.IborIndex("Synthetic", ql.Period(1, ql.Years), 0, ql.USDCurrency(),
                        ql.NullCalendar(), ql.Unadjusted, False, DC, ts)


def ql_swaption(expiry, tenor, strike, vol, kind="payer"):
    """QuantLib で単一スワップションを組み、NPV と自身のフォワード・アニュイティを返す。"""
    ex_date = TODAY + ql.Period(int(expiry), ql.Years)
    schedule = ql.Schedule(
        ex_date, ex_date + ql.Period(int(tenor), ql.Years),
        ql.Period(1, ql.Years), ql.NullCalendar(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
    )
    side = ql.VanillaSwap.Payer if kind == "payer" else ql.VanillaSwap.Receiver
    swap = ql.VanillaSwap(side, 1.0, schedule, strike, DC, schedule, ql_index, 0.0, DC)
    swap.setPricingEngine(ql.DiscountingSwapEngine(ts))
    swaption = ql.Swaption(swap, ql.EuropeanExercise(ex_date))
    swaption.setPricingEngine(ql.BlackSwaptionEngine(ts, ql.QuoteHandle(ql.SimpleQuote(vol))))
    fair = swap.fairRate()
    ann = abs(swap.fixedLegBPS()) * 1e4  # 固定 1bp の価値 → アニュイティ
    return float(swaption.NPV()), float(fair), float(ann)


# ATM ペイヤー 5y×5y を突合する。
expiry, tenor, vol = 5, 5, 0.20
s0_ql_probe = ql_swaption(expiry, tenor, 0.03, vol, "payer")[1]  # 一旦 fair を取得
npv_ql, fair_ql, ann_ql = ql_swaption(expiry, tenor, s0_ql_probe, vol, "payer")
T = DC.yearFraction(TODAY, TODAY + ql.Period(expiry, ql.Years))

manual_ql = ann_ql * black76(fair_ql, fair_ql, T, vol, df=1.0, option="call")
print(f"QuantLib fair rate     = {fair_ql:.6f}")
print(f"QuantLib annuity       = {ann_ql:.6f}")
print(f"QuantLib swaption NPV  = {npv_ql:.8f}")
print(f"annuity × black76      = {manual_ql:.8f}")
assert abs(npv_ql - manual_ql) < 1e-8
print("→ QuantLib NPV = アニュイティ × Black を厳密確認")

# %% [markdown]
# ### bondlab のフォワードスワップと突合
#
# `bondlab` 側で作ったフォワードスワップレート・前方アニュイティで同じ ATM ペイヤーを評価し、
# QuantLib と（日数計算の差の範囲で）一致することを確認します。

# %%
s0_bl, ann_bl = forward_swap(curve, expiry, tenor, freq=1)
v_bl = swaption_black(s0_bl, s0_bl, T, vol, ann_bl, option="payer")

print(f"{'':12s}{'fwd swap':>12s}{'annuity':>12s}{'ATM payer':>14s}")
print(f"{'bondlab':12s}{s0_bl:12.6f}{ann_bl:12.6f}{v_bl:14.8f}")
print(f"{'QuantLib':12s}{fair_ql:12.6f}{ann_ql:12.6f}{npv_ql:14.8f}")
assert abs(s0_bl - fair_ql) < 5e-4
assert abs(ann_bl - ann_ql) < 5e-3
print("→ フォワードスワップレート・アニュイティ・価格が一致（差は Act/365 の日数計算由来）")

# %% [markdown]
# ### ボラ取引としてのスワップション
#
# スワップションはトレーダーにとって「金利ボラティリティのポジション」です。ATM ストラドルを
# 買うと、金利が上下どちらに大きく動いても利益が出て、動かなければベガの分だけ時間価値を失います。
# これは「実現ボラ vs インプライドボラ」の賭けです。QuantLib で ATM ストラドルの NPV を組み、
# 自作の解析ベガと突き合わせます。

# %%
npv_pay = ql_swaption(expiry, tenor, fair_ql, vol, "payer")[0]
npv_rec = ql_swaption(expiry, tenor, fair_ql, vol, "receiver")[0]
straddle_ql = npv_pay + npv_rec

# 数値ベガ（QuantLib のボラを揺らす）。
h = 1e-4
up = (ql_swaption(expiry, tenor, fair_ql, vol + h, "payer")[0]
      + ql_swaption(expiry, tenor, fair_ql, vol + h, "receiver")[0])
dn = (ql_swaption(expiry, tenor, fair_ql, vol - h, "payer")[0]
      + ql_swaption(expiry, tenor, fair_ql, vol - h, "receiver")[0])
vega_ql = (up - dn) / (2 * h)
vega_bl = 2 * ann_ql * fair_ql * norm.pdf(0.5 * vol * np.sqrt(T)) * np.sqrt(T)

print(f"QuantLib ATM ストラドル NPV = {straddle_ql:.8f}")
print(f"QuantLib 数値ベガ           = {vega_ql:.6f}")
print(f"解析ベガ（2×A·S·φ(d1)·√T） = {vega_bl:.6f}")
assert abs(vega_ql - vega_bl) < 1e-2
print("→ ストラドルのベガが QuantLib と一致（ボラ 1.0 あたりの価格感応度）")

# %% [markdown]
# ## 実データ適用
#
# 実データは合成データ（seed 固定）で代用します。前段で作った `bootstrap_par` の合成カーブと、
# 満期・テナー依存の**合成ボラ面**を用いて、複数の payer / receiver スワップションを評価します。
# あわせてキャッシュ決済と物理決済の価値差を数値で見ます。

# %% [markdown]
# ### 合成ボラ面とフォワードスワップグリッド
#
# 満期 $T_e\in\{1,2,3,5,7\}$、テナー $\{1,2,3,5,7\}$ のグリッドで、各セルのフォワードスワップ
# レートと前方アニュイティを `bondlab` で計算します。合成ボラは「短満期でやや高く、長テナーで
# 減衰」する市場風の形を与えます。

# %%
EXPIRIES = [1, 2, 3, 5, 7]
TENORS = [1, 2, 3, 5, 7]


def synthetic_vol(expiry, tenor):
    """満期・テナー依存の合成インプライドボラ（対数正規、市場風のコブ）。"""
    base = 0.22
    hump = 0.03 * np.exp(-((np.log(expiry) - np.log(2.0)) ** 2) / 0.8)
    tenor_decay = -0.008 * np.log(tenor)
    return base + hump + tenor_decay


print(f"{'exp×ten':>8s}{'fwd swap':>12s}{'annuity':>12s}{'vol':>8s}"
      f"{'ATM payer':>12s}{'ATM recv':>12s}")
grid = []
for e in EXPIRIES:
    for te in TENORS:
        s0, ann = forward_swap(curve, e, te, freq=1)
        vol_cell = synthetic_vol(e, te)
        vpay = swaption_black(s0, s0, float(e), vol_cell, ann, option="payer")
        vrec = swaption_black(s0, s0, float(e), vol_cell, ann, option="receiver")
        grid.append((e, te, s0, ann, vol_cell, vpay, vrec))
        print(f"{e}y×{te}y{s0:12.5f}{ann:12.5f}{vol_cell:8.4f}{vpay:12.6f}{vrec:12.6f}")

# ATM では payer と receiver が一致するはず（S0=K）。
for (e, te, s0, ann, vc, vpay, vrec) in grid:
    assert abs(vpay - vrec) < 1e-12
print("→ ATM では payer = receiver（S0=K のため）")

# %% [markdown]
# ### ペイヤー／レシーバーのストライク依存
#
# 満期 5y・テナー 5y のセルで、行使価格 $K$ を振って payer / receiver の価格を描きます。
# payer は $K$ の減少関数、receiver は増加関数で、ATM で交差します。

# %%
e, te = 5, 5
s0, ann = forward_swap(curve, e, te, freq=1)
vol_cell = synthetic_vol(e, te)
strikes = np.linspace(s0 - 0.015, s0 + 0.015, 61)
pay = np.array([swaption_black(s0, k, float(e), vol_cell, ann, option="payer") for k in strikes])
rec = np.array([swaption_black(s0, k, float(e), vol_cell, ann, option="receiver") for k in strikes])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(strikes * 100, pay, color="crimson", lw=2, label="payer")
ax.plot(strikes * 100, rec, color="steelblue", lw=2, label="receiver")
ax.axvline(s0 * 100, color="gray", ls=":", lw=1, label=f"ATM = {s0*100:.2f}%")
ax.set_xlabel("strike (%)")
ax.set_ylabel("swaption price")
ax.set_title(f"{e}y×{te}y payer / receiver vs strike")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### キャッシュ決済と物理決済の価値差
#
# 理論 6 節の一次近似で、キャッシュ決済アニュイティ $A_{\text{cash}}(S_0)=\sum_i \tau/(1+\tau S_0)^i$ を
# 使った価格と、物理決済（割引カーブ由来のアニュイティ）価格を比べます。ATM ペイヤーで、満期・
# テナーごとの差を表にします。金利水準とカーブの形により、両アニュイティがずれる分だけ価値が
# 変わります。

# %%
def cash_annuity(s0, tenor, freq=1):
    """内部収益率 s0 で固定レッグを割ったキャッシュ決済アニュイティ。"""
    tau = 1.0 / freq
    n = int(round(tenor * freq))
    return float(np.sum([tau / (1.0 + tau * s0) ** i for i in range(1, n + 1)]))


print(f"{'exp×ten':>8s}{'A_phys':>10s}{'A_cash':>10s}"
      f"{'V_phys':>12s}{'V_cash':>12s}{'差(bp相当)':>12s}")
for e in EXPIRIES:
    for te in TENORS:
        s0, ann = forward_swap(curve, e, te, freq=1)
        vol_cell = synthetic_vol(e, te)
        a_cash = cash_annuity(s0, te, freq=1)
        black_val = black76(s0, s0, float(e), vol_cell, df=1.0, option="call")
        v_phys = ann * black_val
        v_cash = a_cash * black_val
        print(f"{e}y×{te}y{ann:10.4f}{a_cash:10.4f}"
              f"{v_phys:12.6f}{v_cash:12.6f}{(v_cash - v_phys) * 1e4:12.3f}")

print("\n→ A_cash は決済レートの内部収益率で割る単純化のため A_phys とずれ、価値差を生む")
print("  （一次近似。厳密にはアニュイティ比のコンベクシティ調整が加わる）")

# %% [markdown]
# ## 演習
#
# 1. **ATM ストラドルのベガ・ガンマとボラ取引**
#    満期 3y・テナー 5y の ATM ストラドルについて、`bondlab` のフォワードスワップレート・
#    アニュイティと合成ボラ（`synthetic_vol`）を使い、ベガ（ボラの中心差分）とガンマ
#    （フォワードの 2 階差分）を数値計算せよ。解析式
#    $\partial V/\partial\sigma=2A S\varphi(d_1)\sqrt T$、
#    $\partial^2 V/\partial S^2=2A\varphi(d_1)/(S\sigma\sqrt T)$ と一致することを確認し、
#    「実現ボラがインプライドを上回れば買い手が儲かる」ことをベガ・ガンマの語で説明せよ。
#
# 2. **ペイヤー・レシーバー・パリティ（フォワードスワップ）**
#    満期 5y・テナー 5y で行使価格 $K$ を $\{S_0-0.01,\,S_0,\,S_0+0.01\}$ と振り、
#    各 $K$ で $V^{\text{pay}}-V^{\text{rec}}$ を計算せよ。これが $A(0)(S_0-K)$（フォワード
#    スワップ PV）に一致することを数値で確認し、行使価格が ATM を挟んで符号反転する理由を述べよ。
#
# 解答例は `solutions/S6/sol_0603.py` に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/06_derivatives.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | スワップション | swaption | 将来スワップに入る権利。フォワードスワップレートのオプション |
# | アニュイティ測度 | annuity measure | アニュイティをニュメレールに選んだ測度。スワップレートがマルチンゲール |
# | Jamshidian 分解 | Jamshidian decomposition | 1 ファクター下でスワップションを割引債オプションの和に分解する手法 |
# | ストラドル | straddle | 同一行使・満期の payer と receiver を同時保有する、ボラ取引の代表形 |
