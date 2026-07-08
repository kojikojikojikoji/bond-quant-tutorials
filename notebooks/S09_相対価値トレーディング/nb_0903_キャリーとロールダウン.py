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
# # S9-3 キャリーとロールダウン
#
# ## 学習目標
#
# - キャリー（carry）を、レポ・ファンディングを差し引いた**厳密版**で定義し、
#   ロールダウン（rolldown）と明確に切り分けられる
# - 「カーブ不変仮定」のもとで、キャリー＋ロールがフォワードレート
#   （forward rate）とどう結び付くかを式で示せる
# - 恒等式 **キャリー＋ロール ≒ スポット − フォワード** をゼロクーポンの枠組みで
#   導出し、割引カーブ上で数値検証できる
# - 任意ポジションのキャリー＋ロールを計算する関数を自作し、
#   `bondlab.curve` のゼロ／フォワードを使って年限×保有期間のヒートマップを描ける
# - ブレークイーブンインフレ（BEI, break-even inflation）の初歩を押さえ、
#   名目と実質の相対価値を一段掘り下げて評価できる

# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# キャリーとロールダウンは、docs/債券ファンドの業務.md の収益源でいう「キャリー戦略」そのものの計算基盤です。国債中心の RV ファンドは、カーブが動かなくても時間の経過だけで確定する損益——保有債券の利回りとレポ調達コストの差（キャリー）、および順イールドのカーブを満期に向かって滑り降りる価格上昇（ロールダウン）——を積み上げて稼ぎます。方向性の当て物ではなく、「順イールドで正のキャリーが乗る年限を、資金調達コストより高い利回りで持ち続ける」ことが収益の柱で、収束を待つ RV ポジションの多くはこのキャリーで保有期間中の時間損益を賄います。
#
# 実務でキャリーをクーポンではなくレポ差し引きの厳密版で測るのは、ここを間違えると符号を取り違えるからです。国債はレバレッジで持つのが前提で、買った債券はレポ市場で担保に出して資金を調達します。したがって稼げるかどうかは「ランニング利回り − レポレート」で決まり、順イールドかつレポが短期金利に近ければ正のキャリー、逆イールドや対象銘柄が品薄でスペシャルレポ（調達金利が極端に低い）になると符号も大きさも変わります。狙った年限のキャリーが厚くても、その銘柄がスペシャル化していればレポの取り分が削られる——この資金調達コストとの差こそが、キャリー戦略の実際の取り分です。
#
# 年限 × 保有期間のキャリー＋ロールのヒートマップは、ファンドが「どこに資金を置くか」を決めるための地図です。カーブの傾きが急な区間ほどロールダウンが厚く、単位リスクあたりの時間損益が大きくなるので、同じデュレーションを取るなら最もキャリー＋ロールが乗る年限を選びます。恒等式「キャリー＋ロール ≒ スポット − フォワード」は、この時間損益がフォワードレートに市場が織り込んだ期待と表裏一体であることを示します。フォワードが実現スポットを上回っている（＝市場が織り込みすぎている）なら、カーブ不変を仮定したキャリー＋ロールを取りにいく価値があり、逆ならフォワードの当たり方に賭ける取引に反転します。
#
# ブレークイーブンインフレ（BEI）まで踏み込むのは、名目と実質のキャリーを相対比較して一段深い相対価値を取るためです。名目債と物価連動債のキャリー＋ロールを同じ土俵で評価すれば、「インフレ調整後でどちらの年限が割に合うか」を判断でき、単一カーブ上の rich/cheap だけでは見えない名目 vs 実質の歪みに資本を配分できます。
#

# %% [markdown]
# ## 理論
#
# ### 1. 総リターンの3分解
#
# ある債券ポジションを保有期間 $h$ だけ持ったときの、ファンディングを差し引いた
# 総損益は、次の3つに分解できます。
#
# $$
# \underbrace{\text{総リターン}}_{\text{実現}}
# \;=\;
# \underbrace{\text{キャリー}}_{\text{時間経過}}
# \;+\;
# \underbrace{\text{ロールダウン}}_{\text{カーブを滑り降りる}}
# \;+\;
# \underbrace{\text{カーブ変化項}}_{\text{金利が動いた分}}
# $$
#
# キャリーとロールダウンは**カーブが不変**（今日の利回り曲線が形を変えない）と
# 仮定したときに時間の経過だけで確定する部分で、これが「時間で稼ぐ分」です。
# 第3項はカーブが実際に動いたときの評価損益で、キャリー・ロールの外側にあります。
# 本ノートでは主に前2項を扱います。
#
# ### 2. キャリーの厳密な定義（レポ・ファンディング考慮）
#
# キャリー（carry）とは「保有しているだけで得る利息 − そのポジションを賄う
# ファンディング費用」です。国債をレバレッジで持つ実務では、買った債券を
# レポ市場（repo, 現先取引）で担保に出して資金を調達します。したがって
# ファンディング費用はレポレート $r_{\text{repo}}$ で決まります。
#
# 概念式で書くと、
#
# $$
# \text{キャリー} \;\approx\; \big(\text{ランニング利回り} - r_{\text{repo}}\big)\times h .
# $$
#
# 順イールド（利回りが右上がり）かつレポが短期金利に近いとき、保有債券の
# 利回りがレポより高いので**正のキャリー**になります。逆イールドや、対象銘柄が
# 品薄で**スペシャル（special）**なレポ（$r_{\text{repo}}$ が極端に低い）になると、
# キャリーは大きく変わります。ここが「厳密版」で外せない点で、
# キャリーはクーポンだけでは決まらず**資金調達コストとの差**で決まります。
#
# ### 3. ロールダウン
#
# ロールダウン（rolldown）は、カーブが不変でも**時間が経つと満期が近づき、
# 曲線上のより短い（順イールドなら低い）利回り点に移る**ことで生じる価格上昇です。
# 残存 $T$ 年・利回り $y(T)$ の債券は、$h$ 年後には残存 $T-h$ 年になり、
# カーブ不変なら利回りは $y(T-h)$ に「滑り降り」ます。価格変化は
#
# $$
# \text{ロールダウン} \;\approx\; D \cdot \big(y(T) - y(T-h)\big)
# $$
#
# （$D$ は修正デュレーション）で近似できます。曲線が急な（スティープな）区間ほど
# $y(T)-y(T-h)$ が大きく、ロールダウンも大きくなります。
#
# ### 4. カーブ不変仮定とフォワードレート：キャリー＋ロール ≒ スポット − フォワード
#
# ここが本ノートの核心です。ゼロクーポン債で厳密に導出します。満期 $T$ の
# ゼロを今日 $DF(T)$ で買い、保有期間 $h$ をレポで賄うとします（レポ＝カーブの
# 短期金利と仮定）。
#
# **フォワード価格**（無裁定で決まる先渡価格）は
#
# $$
# F \;=\; \frac{DF(T)}{DF(h)} \;=\; e^{-f(h,T)\,(T-h)},
# $$
#
# ここで $f(h,T)$ は区間 $[h,T]$ の連続複利フォワードレートです（`forward_rate`）。
# 一方、**カーブ不変**のもとで $h$ 年後にこのゼロ（残存 $T-h$）が付ける価格は、
# 同じスポットカーブを使って
#
# $$
# V_h \;=\; DF(T-h) \;=\; e^{-z(T-h)\,(T-h)}
# $$
#
# です（$z$ はゼロレート、`zero_rate`）。買値をレポで転がした費用がちょうど
# フォワード価格 $F$ に一致するので、ファンディングを差し引いた総損益は
#
# $$
# \boxed{\;\text{キャリー} + \text{ロール} \;=\; V_h - F \;=\; DF(T-h) - \frac{DF(T)}{DF(h)}\;}
# $$
#
# となります。$V_h$ は「スポットカーブを滑り降りた将来価値」、$F$ は
# 「フォワード価格」ですから、これがそのまま **キャリー＋ロール ≒ スポット − フォワード**
# の正体です。利回りで言えば、
#
# $$
# \text{損益} > 0 \iff z(T-h) < f(h,T),
# $$
#
# すなわち**カーブ不変で滑り降りたスポット利回りが、フォワード利回りより低い**とき
# に儲かります。フォワードは「市場が織り込んだブレークイーブン」であり、
# キャリー＋ロールで稼ぐとは**フォワードが実現しない方に賭ける**ことに等しい、
# というのがこの恒等式のメッセージです。
#
# ### 5. キャリー・ロールの厳密分解（クーポン債・任意キャッシュフロー）
#
# 複数キャッシュフロー $\{(t_i, c_i)\}$ を持つポジションでも、同じ論法で
# 曖昧さなく分解できます。今日の価格 $P_0=\sum_i c_i\,DF(t_i)$、レポレート
# $r_{\text{repo}}$ で $h$ だけ賄うとして、
#
# $$
# F = P_0\,e^{r_{\text{repo}}\,h}, \qquad
# V^{\text{noroll}} = \sum_i c_i\,DF(t_i)\,e^{z(t_i)\,h}, \qquad
# V_h = \sum_i c_i\,DF(t_i - h),
# $$
#
# と置くと、
#
# $$
# \text{キャリー} = V^{\text{noroll}} - F,\qquad
# \text{ロール} = V_h - V^{\text{noroll}},\qquad
# \text{合計} = V_h - F .
# $$
#
# $V^{\text{noroll}}$ は「各キャッシュフローが**自分の元の年限の利回りのまま**時間だけ
# 進んだ」仮想値で、これとフォワードの差がキャリー（各年限の利回りとレポの差）、
# 元の利回りから短い年限の利回りへ滑り降りる差がロールです。ゼロクーポンなら
# 第4節の式に一致します。この分解を後ほどそのまま実装します。
#
# ### 6. ブレークイーブンインフレ（BEI）の初歩
#
# 名目国債の利回りには、実質金利と期待インフレの両方が含まれます。物価連動国債
# （インフレ連動債）は元本・クーポンが物価指数で**インデクセーション**
# （indexation, 物価スライド）されるため、投資家は実質利回りを受け取ります。
# 同一年限の名目利回り $y^{\text{nom}}$ と実質利回り $y^{\text{real}}$ の差が
# **ブレークイーブンインフレ**です。
#
# $$
# \text{BEI}(T) \;=\; y^{\text{nom}}(T) - y^{\text{real}}(T)
# \;\approx\; \underbrace{\mathbb{E}[\pi]}_{\text{期待インフレ}}
# \;+\; \underbrace{\text{インフレリスクプレミアム}}_{\text{不確実性の対価}} .
# $$
#
# BEI が「期待インフレそのもの」ではなくリスクプレミアムを含む点が実務上重要です。
# 「名目を買ってレポで実質を売る（BEI ロング）」というポジションにも当然
# キャリー＋ロールがあり、名目カーブと実質カーブそれぞれのキャリー・ロールの
# 差として評価できます。本ノートでは実質金利を合成して（名目 − 実質 = BEI の関係を
# 作って）、名目と実質の相対価値を一段だけ掘り下げます。

# %% [markdown]
# **数値例**：10年の名目利回りが $y^{\text{nom}}=1.0\%$、実質（物価連動）利回りが $y^{\text{real}}=-1.2\%$ なら $\text{BEI}=1.0-(-1.2)=2.2\%$。この 2.2% が期待インフレとインフレリスクプレミアムの合計に相当します。
#

# %% [markdown]
# **数値例**：連続複利ゼロで $z(1)=0.3\%,\ z(9)=0.9\%,\ z(10)=1.0\%$（$T=10,\ h=1$）とすると $DF(9)=0.9222,\ DF(10)=0.9048,\ DF(1)=0.9970$ です。キャリー＋ロール $=DF(9)-DF(10)/DF(1)=0.9222-0.9076=+0.0146$（額面1あたり）。利回りで見ると滑り降りたスポット $z(9)=0.90\%$ がフォワード $f(1,10)=1.08\%$ を下回るため、約 18 bp のピックアップになります。
#

# %% [markdown]
# ## スクラッチ実装
#
# 第5節の厳密分解を、割引カーブ上でそのまま実装します。まず今日のパー利回りから
# ゼロカーブをブートストラップし、`DiscountCurve.zero_rate` / `forward_rate` /
# `discount` を土台に使います。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `make_cashflows(coupon, maturity, freq)` | クーポン率, 満期(年), 年間回数 | `(times, amounts)` | 額面1・満期一括償還の CF 列を生成 |
# | `carry_roll(curve, times, amounts, horizon, repo)` | カーブ, CF時点, CF額, 保有期間, レポ率(連続) | `dict(carry, roll, total, ...)` | 第5節の厳密分解を計算 |
# | `carry_roll_bp(curve, tenor, horizon, coupon, freq, repo)` | カーブ, 年限, 保有期間, クーポン, 回数, レポ | float | 総キャリー＋ロールを**年率bp**で返す（ヒートマップ用） |
# | `zero_forward_check(curve, tenor, horizon)` | カーブ, 年限, 保有期間 | `dict` | ゼロの「スポット−フォワード」恒等式を検算 |

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
for _f in ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Noto Sans JP", "TakaoPGothic", "IPAPGothic"]:
    if any(_f == _n.name for _n in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd

from bondlab.curve import bootstrap_par, DiscountCurve

np.random.seed(0)

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def make_cashflows(coupon, maturity, freq=1):
    """額面1・満期一括償還のクーポン債キャッシュフローを返す。

    coupon は年率（小数）、maturity は年、freq は年間クーポン回数。
    返り値は (times, amounts)。最終期にクーポン＋額面1を載せる。
    """
    n = int(round(maturity * freq))
    times = np.array([(k + 1) / freq for k in range(n)], dtype=float)
    amounts = np.full(n, coupon / freq, dtype=float)
    amounts[-1] += 1.0
    return times, amounts


def carry_roll(curve, times, amounts, horizon, repo=None):
    """第5節の厳密分解でキャリー・ロール・合計を計算する。

    curve   : DiscountCurve（今日のスポットカーブ）
    times   : キャッシュフロー時点（年、horizon より後を前提）
    amounts : 各時点のキャッシュフロー額
    horizon : 保有期間 h（年）
    repo    : 連続複利のレポレート。None ならカーブの短期ゼロ z(h) を使う。

    返り値の price 系は額面1あたりの金額、yield 系は概算の年率利回り（bp）。
    """
    times = np.asarray(times, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    df = curve.discount(times)                     # 今日の割引係数
    z = curve.zero_rate(times)                     # 各 CF のゼロレート
    p0 = float(np.sum(amounts * df))               # 今日の価格

    if repo is None:
        repo = curve.zero_rate(horizon)            # レポ＝カーブ短期金利
    fwd_price = p0 * np.exp(repo * horizon)         # F = P0 * e^{repo*h}

    v_noroll = float(np.sum(amounts * df * np.exp(z * horizon)))   # 利回り据置
    v_roll = float(np.sum(amounts * curve.discount(times - horizon)))  # 滑り降り

    carry = v_noroll - fwd_price
    roll = v_roll - v_noroll
    total = v_roll - fwd_price

    # 参考：年率換算した利回りベースの寄与（価格 → 利回りは -total/(P0*Dur) 相当）。
    dur = float(np.sum(times * amounts * df) / p0)  # マコーレー年数（近似デュレーション）
    return dict(
        price0=p0, forward_price=fwd_price, value_horizon=v_roll,
        carry=carry, roll=roll, total=total,
        carry_bp=carry / p0 / horizon * 1e4,
        roll_bp=roll / p0 / horizon * 1e4,
        total_bp=total / p0 / horizon * 1e4,
        duration=dur, repo=repo,
    )


def carry_roll_bp(curve, tenor, horizon, coupon=None, freq=1, repo=None):
    """年限 tenor・保有期間 horizon の総キャリー＋ロールを年率 bp で返す。

    coupon 未指定ならパー債（クーポン＝その年限のパー利回りに近い値）とみなし、
    カーブのゼロレートを近似クーポンに使う。
    """
    if coupon is None:
        coupon = curve.zero_rate(tenor)
    times, amounts = make_cashflows(coupon, tenor, freq)
    return carry_roll(curve, times, amounts, horizon, repo)["total_bp"]


def zero_forward_check(curve, tenor, horizon):
    """ゼロクーポンで「キャリー＋ロール ＝ スポット − フォワード」を検算する。

    価格恒等式 total = DF(T-h) - DF(T)/DF(h) と、
    利回り恒等式（rolled spot z(T-h) と forward f(h,T)）の両方を返す。
    """
    T, h = float(tenor), float(horizon)
    df_T = curve.discount(T)
    df_h = curve.discount(h)
    df_Tmh = curve.discount(T - h)

    total_price = df_Tmh - df_T / df_h        # スクラッチ式
    v_h = df_Tmh                              # スポット（滑り降り）価値
    fwd_price = df_T / df_h                   # フォワード価格

    z_rolled = curve.zero_rate(T - h)         # 滑り降りたスポット利回り
    f_ht = curve.forward_rate(h, T)           # 区間 [h,T] のフォワード利回り

    # carry_roll 関数の合計とも一致するはず（ゼロ＝満期に額面1のみ）。
    cr = carry_roll(curve, [T], [1.0], h, repo=curve.zero_rate(h))
    return dict(
        total_price=total_price, value_horizon=v_h, forward_price=fwd_price,
        z_rolled=z_rolled, forward_yield=f_ht, yield_pickup_bp=(f_ht - z_rolled) * 1e4,
        carry_roll_total=cr["total"],
    )


# %% [markdown]
# ### 今日のカーブを組む
#
# サンプルの UST パー利回りパネルから最新営業日を取り出します。
# `bootstrap_par` は**等間隔テナー**を前提とするため、与えられた粗いテナー
# （0.5, 1, 2, 3, 5, 7, 10, 20, 30 年）を年刻みグリッド 1〜30 年に線形補間してから
# ブートストラップします（この前提を破ると年金部分の重みがずれます。詳細は最終節）。

# %%
panel = pd.read_csv("data/samples/synthetic_ust_par_panel.csv")
dates = sorted(panel["date"].unique())
today = dates[-1]
snap = panel[panel["date"] == today].sort_values("tenor")
print(f"評価日: {today}  ノード数: {len(snap)}")

grid = np.arange(1.0, 31.0)                                 # 年刻み 1..30
par_on_grid = np.interp(grid, snap["tenor"].values, snap["par_yield"].values)
curve = bootstrap_par(grid, par_on_grid, frequency=1)

for t in [1, 2, 5, 10, 30]:
    print(f"  {t:2d}年  ゼロ {curve.zero_rate(float(t))*100:5.2f}%   "
          f"DF {curve.discount(float(t)):.4f}")

# %% [markdown]
# ### 1銘柄のキャリー・ロール分解
#
# 10年パー債（クーポン＝10年ゼロ、額面1、年1回払い）を保有期間3か月で分解します。
# レポはカーブの短期金利を使います。

# %%
cpn10 = curve.zero_rate(10.0)
t10, a10 = make_cashflows(cpn10, 10.0, freq=1)
res = carry_roll(curve, t10, a10, horizon=0.25)

print(f"10年債 クーポン {cpn10*100:.2f}%  保有0.25年  レポ {res['repo']*100:.2f}%")
print(f"  今日の価格 P0        = {res['price0']:.5f}")
print(f"  フォワード価格 F     = {res['forward_price']:.5f}")
print(f"  カーブ不変の3M後価値 = {res['value_horizon']:.5f}")
print(f"  キャリー   = {res['carry']:+.5f}  ({res['carry_bp']:+6.1f} bp/年)")
print(f"  ロール     = {res['roll']:+.5f}  ({res['roll_bp']:+6.1f} bp/年)")
print(f"  合計       = {res['total']:+.5f}  ({res['total_bp']:+6.1f} bp/年)")
assert abs((res["carry"] + res["roll"]) - res["total"]) < 1e-12
print("  分解の整合（carry + roll = total）を確認しました")

# %% [markdown]
# ### 年限 × 保有期間のヒートマップ
#
# パー債を仮定し、年限（縦）× 保有期間（横）で総キャリー＋ロールを**年率bp**にして
# 並べます。順イールドかつ曲線が急な中期ゾーンで大きくなる傾向が読み取れます。

# %%
tenors_hm = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
horizons_hm = np.array([0.25, 0.5, 1.0, 2.0])

hm = np.array([[carry_roll_bp(curve, T, h) for h in horizons_hm]
               for T in tenors_hm])

fig, ax = plt.subplots(figsize=(8, 5.5))
im = ax.imshow(hm, aspect="auto", cmap="RdYlGn", origin="lower")
ax.set_xticks(range(len(horizons_hm)))
ax.set_xticklabels([f"{h:g}年" for h in horizons_hm])
ax.set_yticks(range(len(tenors_hm)))
ax.set_yticklabels([f"{int(T)}年" for T in tenors_hm])
ax.set_xlabel("保有期間 h")
ax.set_ylabel("債券の年限 T")
ax.set_title("パー債のキャリー＋ロール（年率 bp）")
for i in range(len(tenors_hm)):
    for j in range(len(horizons_hm)):
        ax.text(j, i, f"{hm[i, j]:.0f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="年率 bp")
fig.tight_layout()
plt.show()

best_idx = np.unravel_index(np.argmax(hm), hm.shape)
print(f"最大: 年限 {int(tenors_hm[best_idx[0]])}年 × 保有 "
      f"{horizons_hm[best_idx[1]]:g}年 で {hm[best_idx]:.1f} bp/年")

# %% [markdown]
# ## QuantLib検証
#
# QuantLib そのものではなく、本ノートの核心である恒等式
# **キャリー＋ロール ≒ スポット − フォワード** の**数値整合**をここで検証します
# （この節は「自作分解が無裁定のフォワード価格と一致するか」の答え合わせに
# 充てる位置づけです）。加えて、`DiscountCurve.forward_rate` が与える
# フォワード割引係数と、割引係数の比 $DF(T)/DF(h)$ が一致することも確認します。

# %%
print(f"{'T年':>4} {'h年':>5} {'total(価格式)':>14} {'V_h - F':>12} "
      f"{'z(T-h)%':>9} {'f(h,T)%':>9} {'ピックアップbp':>13}")
for T, h in [(2, 0.25), (5, 0.5), (10, 1.0), (30, 1.0), (10, 2.0)]:
    chk = zero_forward_check(curve, T, h)
    # 価格恒等式：スクラッチ total と V_h - F が一致
    assert abs(chk["total_price"] - (chk["value_horizon"] - chk["forward_price"])) < 1e-12
    # carry_roll 関数の合計とも一致
    assert abs(chk["total_price"] - chk["carry_roll_total"]) < 1e-10
    # forward_rate と割引係数比の整合
    fwd_from_rate = np.exp(-chk["forward_yield"] * (T - h))
    assert abs(fwd_from_rate - chk["forward_price"]) < 1e-12
    print(f"{T:>4} {h:>5} {chk['total_price']:>14.6f} "
          f"{chk['value_horizon'] - chk['forward_price']:>12.6f} "
          f"{chk['z_rolled']*100:>9.3f} {chk['forward_yield']*100:>9.3f} "
          f"{chk['yield_pickup_bp']:>13.1f}")

print("\nすべての年限・保有期間で恒等式が数値一致しました "
      "（キャリー＋ロール ＝ スポット − フォワード）。")

# %% [markdown]
# 利回りで見ると、キャリー＋ロールが正になるのは滑り降りたスポット利回り
# $z(T-h)$ がフォワード利回り $f(h,T)$ を下回るときです。上の「ピックアップbp」
# は $f(h,T)-z(T-h)$ で、順イールドかつ曲線が急なほど大きくなります。

# %% [markdown]
# ## 実データ適用
#
# サンプルパネル（合成 UST パー利回り、60営業日）を使い、
# (a) キャリー＋ロールが最大の年限を特定し、(b) そのカーブ変化への脆弱性を測り、
# (c) 合成 BEI を作って名目と実質の相対価値を一段見ます。ネットワークは使いません。

# %% [markdown]
# ### (a) キャリー＋ロール最大の年限
#
# 保有期間 $h=0.25$ 年（3か月）で、各年限のパー債のキャリー＋ロールを年率bpで比較します。

# %%
tenor_axis = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
H = 0.25
cr_by_tenor = np.array([carry_roll_bp(curve, T, H) for T in tenor_axis])

best_t = tenor_axis[int(np.argmax(cr_by_tenor))]
print(f"保有期間 {H:g}年でのキャリー＋ロール（年率bp）")
for T, v in zip(tenor_axis, cr_by_tenor):
    mark = "  ← 最大" if T == best_t else ""
    print(f"  {int(T):2d}年: {v:6.1f} bp{mark}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar([str(int(T)) for T in tenor_axis], cr_by_tenor, color="#2c7fb8")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("年限（年）")
ax.set_ylabel("キャリー＋ロール（年率 bp）")
ax.set_title(f"年限別キャリー＋ロール（保有 {H:g}年, 評価日 {today}）")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### (b) カーブ変化への脆弱性（ブレークイーブン金利変化）
#
# キャリー＋ロールは「カーブ不変」で稼ぐ分です。実際にはカーブが動くと消えます。
# 最大年限のポジションについて、**総キャリー＋ロールをちょうど打ち消す平行金利上昇幅
# （ブレークイーブン）** を、価格の一次近似
# $\Delta P \approx -P_0 \cdot D \cdot \Delta y$ から求めます。この幅が小さいほど脆弱です。

# %%
cpn_b = curve.zero_rate(best_t)
tb, ab = make_cashflows(cpn_b, best_t, freq=1)
rb = carry_roll(curve, tb, ab, horizon=H)
# total（価格・額面1あたり） = P0 * D * Δy_breakeven を解く
breakeven_dy_bp = rb["total"] / (rb["price0"] * rb["duration"]) * 1e4

print(f"{int(best_t)}年ポジション（保有 {H:g}年）")
print(f"  総キャリー＋ロール = {rb['total']:+.5f}（額面1あたり, {rb['total_bp']:+.1f} bp/年）")
print(f"  近似デュレーション = {rb['duration']:.2f} 年")
print(f"  ブレークイーブン平行金利上昇 ≒ {breakeven_dy_bp:.1f} bp")
print(f"  → 保有期間中にカーブが約 {breakeven_dy_bp:.1f} bp 以上ベアスティープ／"
      f"平行上昇すると、稼いだキャリー＋ロールは消えます。")

# %% [markdown]
# ### (b') カーブ不変仮定のヒストリカル検証
#
# パネル全期間で、この年限のスポット利回りが実際にどれだけ動いたかを見ます。
# 日次変化の標準偏差と、ブレークイーブン幅を突き合わせて脆弱性を評価します。

# %%
def spot_series(panel, tenor):
    """各営業日のカーブを組み、指定年限のゼロレートの時系列を返す。"""
    out = []
    for d in sorted(panel["date"].unique()):
        s = panel[panel["date"] == d].sort_values("tenor")
        g = np.arange(1.0, 31.0)
        py = np.interp(g, s["tenor"].values, s["par_yield"].values)
        c = bootstrap_par(g, py, frequency=1)
        out.append(c.zero_rate(float(tenor)))
    return np.array(out)


z_hist = spot_series(panel, best_t)
dz_bp = np.diff(z_hist) * 1e4
horizon_days = int(round(H * 252))
# 保有期間（≈63営業日）に対応する累積変化の標準偏差（ランダムウォーク近似）
sd_over_h = np.std(dz_bp, ddof=1) * np.sqrt(horizon_days)

print(f"{int(best_t)}年ゼロレートのヒストリカル（{len(z_hist)}営業日）")
print(f"  期間の利回りレンジ: {z_hist.min()*100:.2f}% 〜 {z_hist.max()*100:.2f}%")
print(f"  日次変化の標準偏差: {np.std(dz_bp, ddof=1):.2f} bp")
print(f"  保有 {H:g}年（≈{horizon_days}営業日）換算の標準偏差: {sd_over_h:.1f} bp")
print(f"  ブレークイーブン {breakeven_dy_bp:.1f} bp と比較 → "
      f"{'カーブ変動がブレークイーブンを上回りやすく脆弱' if sd_over_h > breakeven_dy_bp else 'ブレークイーブン内に収まりやすい'}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(z_hist)), z_hist * 100, color="#d95f0e")
ax.set_xlabel("営業日インデックス")
ax.set_ylabel(f"{int(best_t)}年ゼロレート（%）")
ax.set_title(f"{int(best_t)}年ゼロレートの推移（カーブ不変仮定の妥当性チェック）")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### (c) 合成 BEI と名目 vs 実質の相対価値
#
# 実質金利データが無いので、名目 − 実質 = BEI の関係を保ったまま合成します。
# BEI に緩やかな期間構造（短期やや低め・長期やや高め）を与え、実質パー利回りを
# `名目 − BEI` で作ります。名目カーブと実質カーブそれぞれのキャリー＋ロールを比べます。

# %%
def synth_bei(tenors):
    """合成 BEI 期間構造（％）。短期 1.8% → 長期 2.5% へ緩やかに上昇。"""
    tenors = np.asarray(tenors, dtype=float)
    return 0.018 + 0.007 * (1.0 - np.exp(-tenors / 8.0))


bei_grid = synth_bei(grid)
real_par_on_grid = par_on_grid - bei_grid                # 実質パー利回り
real_curve = bootstrap_par(grid, real_par_on_grid, frequency=1)

print(f"{'年限':>4} {'名目CR(bp)':>11} {'実質CR(bp)':>11} {'BEI(%)':>8} {'差(bp)':>8}")
rows = []
for T in [2, 5, 10, 30]:
    nom = carry_roll_bp(curve, float(T), H)
    rea = carry_roll_bp(real_curve, float(T), H, coupon=real_curve.zero_rate(float(T)))
    b = float(synth_bei(T)) * 100
    rows.append((T, nom, rea, b, nom - rea))
    print(f"{T:>4} {nom:>11.1f} {rea:>11.1f} {b:>8.2f} {nom - rea:>8.1f}")

# %% [markdown]
# ### 名目 vs 実質の考察
#
# 名目カーブの方がキャリー＋ロールが大きく出るのは、名目利回りが実質＋BEI の分だけ
# 高く、順イールドの傾きも立ちやすいためです。ただし名目のキャリーには**インフレの
# 実現**というリスクが乗っています。BEI が期待インフレ＋インフレリスクプレミアムで
# ある以上、名目キャリーの一部は「インフレ連動しないことの対価」に過ぎません。
# 実質債（TIPS 類似）はキャリー＋ロールこそ小さいものの、インフレが上振れしても
# 元本がインデクセーションで守られるため、名目との差（＝BEI ロングのキャリー）は
# **将来インフレの見方**そのものを取りに行くポジションになります。相対価値としては、
# 「名目の高いキャリーが、負う BEI 変動リスクに見合うか」を、(b) と同じくブレークイーブン
# 幅で測るのが筋の良い評価になります。

# %% [markdown]
# ## 演習
#
# 1. **キャリー＋ロール最大年限の特定とリスク評価**：保有期間を $h=0.5$ 年に変え、
#    パー債のキャリー＋ロールが最大となる年限を特定せよ。さらにその年限について、
#    総キャリー＋ロールを打ち消すブレークイーブン平行金利上昇幅（bp）を求め、
#    (b) と同じ枠組みで脆弱性を一言で述べよ。
# 2. **カーブ不変仮定のヒストリカル検証**：パネル全期間について、各営業日に
#    「$h=1$ か月後の予測キャリー＋ロール」と「1か月後に実際に実現したトータルリターン」を
#    比較し、カーブ不変仮定がどの程度成り立ったか（予測と実現の差の分布）を評価せよ。
#    予測が実現を系統的に外す局面（カーブが動いた時期）があるかを述べよ。
#
# 解答例は `solutions/S9/sol_0903.py` にあります。

# %% [markdown]
# ## bootstrap_par の前提に関する注意
#
# `bondlab.curve.bootstrap_par` は docstring どおり**等間隔テナー（1/frequency 刻み）**を
# 前提とします。年金項の累積和 `running` が「1期＝1/frequency」を暗黙に仮定するため、
# サンプルパネルの粗いテナー（0.5, 1, 2, 3, 5, 7, 10, 20, 30 年）をそのまま渡すと、
# たとえば 10→20 年の10年ギャップを1期分として扱ってしまい、カーブが歪みます。
# 本ノートでは年刻みグリッドへ線形補間してから渡すことで前提を満たしています。
# 関数側で等間隔をチェックしていない点は、利用時の落とし穴として意識してください。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/09_trading.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | [キャリー](../../glossary/09_trading.md#carry) | carry | 利息収入からファンディング費用（レポ）を差し引いた、時間で稼ぐ分 |
# | [ロールダウン](../../glossary/09_trading.md#rolldown) | rolldown | カーブ不変でも満期が近づき曲線を滑り降りて生じる価格変化 |
# | [フォワードレート](../../glossary/09_trading.md#forward-rate) | forward rate | 割引係数比から定まる将来区間の金利。キャリー＋ロールのブレークイーブン |
# | [レポ](../../glossary/09_trading.md#repo) | repo | 債券を担保に短期資金を調達する現先取引。ファンディング費用を決める |
# | [ブレークイーブンインフレ](../../glossary/09_trading.md#break-even-inflation-bei) | break-even inflation (BEI) | 名目利回り − 実質利回り。期待インフレ＋インフレリスクプレミアム |
