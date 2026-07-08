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
# # S1-4 クリーン/ダーティ価格と市場慣行の国際比較
#
# ## 学習目標
#
# - 経過利子（accrued interest）を規約別に導出し、クリーン価格とダーティ価格を相互変換できる
# - 決済慣行（T+1 / T+2）と受渡日の定義を、主要国債市場ごとに区別できる
# - UST・JGB・Gilt・Bund の利回り慣行（半年複利・単利・年複利）の違いを説明できる
# - ACT/ACT の ISDA 変種と ISMA（Bond）変種の違いを式で示し、なぜ債券の経過利子は ISMA を使うかを数値で説明できる
# - 経過利子とクリーン⇔ダーティ変換をスクラッチ実装し、`bondlab.bond.FixedRateBond.accrued` および QuantLib と突合できる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# クリーン価格とダーティ価格の区別は、気配と実際の受渡金額をつなぐ実務の要です。市場で表示・比較されるのはクリーン価格ですが、買い手が実際に払うのは経過利子を足したダーティ価格です。マーケットメイクのデスクは気配（クリーン）で値付けしつつ、約定・決済・担保評価はダーティで行うため、この変換を一度でも誤ると、受渡金額が合わず決済が失敗します。日々の在庫評価損益も、経過利子の増減を正しく計上して初めて正確になります。
#
# 決済慣行（T+1・T+2）と受渡日の違いは、キャリー戦略の損益計算に直結します。保有しているだけで積み上がる経過利子は受渡日基準で計算されるため、米国債（T+1）と日本国債（T+2）では同じ約定日でも経過利子が変わります。レポで資金を調達してポジションを持つRVファンドは、この経過利子とレポ金利（ファンディングコスト）の差でキャリーを稼ぐので、受渡日を1日ずれて計算すると利ざやの符号すら狂います。
#
# 利回り慣行と ACT/ACT の変種（ISDA と ISMA）の違いは、国際的な相対価値取引で特に効きます。米国債・日本国債・英国債・独国債は半年複利・単利・年複利と利回りの定義が異なり、経過利子は債券では ISMA を使うのが慣行です。ここを揃えずに各国の国債を比較すると、市場慣行の違いによる見かけの差を割安・割高と誤認し、収束しないトレードに賭けてしまいます。裏を返せば、この慣行差を正しく吸収できることが、クロスマーケットの裁定で稼ぐ前提になります。この経過利子計算が無ければ、正しい受渡金額も、正確な国際比較もできません。
# %% [markdown]
# ## 理論
#
# ### 経過利子・クリーン価格・ダーティ価格
#
# クーポン債は利払日にまとめてクーポンを支払うため、利払日の間に売買すると、
# 直前利払日から受渡日までの利息は売り手に帰属する。この未払い期間分の利息を
# 経過利子（accrued interest）と呼ぶ。買い手が実際に支払う代金（ダーティ価格,
# dirty price）は、市場で気配表示される価格（クリーン価格, clean price）に経過
# 利子を足したものになる。
#
# $$
# P_{\text{dirty}} = P_{\text{clean}} + AI
# $$
#
# クリーン価格を気配に使う理由は、経過利子が受渡日とともに鋸歯状に増減し、利払
# 日で不連続に落ちるためである。この季節変動を取り除いたクリーン価格のほうが、
# 銘柄間・時点間の比較に向く。実際に支払うのはダーティ価格であり、評価・約定・
# 会計はダーティ価格で行う。
#
# ### 決済慣行（T+1 / T+2）と受渡日
#
# 経過利子は約定日ではなく受渡日（settlement date, 実際に資金と証券が交換される
# 日）を基準に計算する。受渡日は約定日から数営業日後に設定され、その日数は市場
# ごとに異なる。
#
# | 市場 | 標準決済 | 受渡日の目安 |
# |---|---|---|
# | 米国債（UST） | T+1 | 約定の翌営業日 |
# | 日本国債（JGB） | T+2 | 約定の2営業日後 |
# | 英国債（Gilt） | T+1 | 約定の翌営業日 |
# | 独国債（Bund） | T+2 | 約定の2営業日後 |
#
# 同じ約定日でも決済慣行が違えば受渡日が変わり、経過利子も変わる。以降の計算は
# すべて受渡日を入力に取る。

# %% [markdown]
# ### 利回り慣行の国際比較
#
# 同じキャッシュフローでも、利回りの表示規約は市場ごとに異なる。価格から利回り
# を逆算する式が違うため、同一銘柄でも表示される利回り数値がずれる。
#
# | 市場 | クーポン頻度 | 利回り慣行 | 割引・換算の型 |
# |---|---|---|---|
# | 米国債（UST） | 半年 | 半年複利（street convention） | $(1 + y/2)^{-(w+j)}$ |
# | 日本国債（JGB） | 半年 | 単利（最終利回り単利） | 線形の単利式 |
# | 英国債（Gilt） | 半年 | 半年複利 | $(1 + y/2)^{-(w+j)}$ |
# | 独国債（Bund） | 年1回 | 年複利 | $(1 + y)^{-(w+j)}$ |
#
# 半年複利のダーティ価格は、次クーポンまでの残存割合を $w$、$j$ 番目の将来
# キャッシュフロー（$j=0$ が次クーポン）を $c_j$ として、
#
# $$
# P_{\text{dirty}} = \sum_{j \ge 0} \frac{c_j}{(1 + y/f)^{\,w + j}}
# $$
#
# で表す（$f$ は年間利払回数）。JGB の単利最終利回りは複利で割り引かず、購入
# 価格 $P$・年クーポン $C$・残存年数 $n$ から線形式で定義する。
#
# $$
# y_{\text{simple}} = \frac{\,C + (100 - P)/n\,}{P}
# $$
#
# 分子はクーポン収入と償還差益を年率に均した額、分母は投下元本である。複利効果
# を無視するぶん、クーポンが高いほど・年限が長いほど複利利回りとの差が開く。

# %% [markdown]
# ### ACT/ACT の ISDA 変種と ISMA（Bond）変種
#
# 経過利子は「直前利払日から受渡日までの期間割合」に比例する。その期間割合を
# ACT/ACT で測るとき、分母の取り方が2通りあり、これが債券価格計算で最も混同
# されやすい論点である。
#
# **ISMA（Bond）変種**は、クーポン期間そのものを分母に取る。直前クーポン日を
# $t_{\text{prev}}$、次クーポン日を $t_{\text{next}}$、受渡日を $t_s$、期中の
# 実日数を $d(\cdot,\cdot)$ とすると、
#
# $$
# AI_{\text{ISMA}} = \frac{C}{f}\cdot\frac{d(t_{\text{prev}},\,t_s)}{d(t_{\text{prev}},\,t_{\text{next}})}
# $$
#
# つまり1回分のクーポン $C/f$ を、クーポン期間の実日数で日割りする。分母は
# 「そのクーポン期間の長さ」であり、暦年とは無関係である。
#
# **ISDA 変種**は、分母を暦年の実日数（365 または閏年 366）に取り、年をまたぐ
# 日数を各暦年の日数で按分する。年クーポン率を $c$、額面を $F$ とすると、
#
# $$
# AI_{\text{ISDA}} = F\,c\cdot \mathrm{yf}_{\text{ISDA}}(t_{\text{prev}},\,t_s),
# \qquad
# \mathrm{yf}_{\text{ISDA}} = \sum_{\text{暦年}} \frac{\text{その年に属する日数}}{365\ \text{または}\ 366}
# $$
#
# 両者は利払日ちょうど（期間割合が 0 または 1）では一致するが、期中（mid-period）
# では一致しない。ISMA が「クーポン期間の何割経過したか」を測るのに対し、ISDA は
# 「暦年ベースで何年分か」を測るためである。半年債であればクーポン期間は約 0.5 年
# だが、その 0.5 年の実日数（181〜184 日）は期によって揺れる。ISMA はこの揺れを
# 期間長として正しく織り込むのに対し、ISDA は暦年 365/366 に固定するため、経過
# 利子が期の長短を反映しない。
#
# **債券の経過利子・クリーン/ダーティ価格は ISMA（Bond）変種が標準**である。
# クーポンは「その期間に対して」支払われる約定なので、経過分は期間長で日割りする
# のが利払いの実態に合う。ISDA 変種はデリバティブの想定元本にかかる金利計算
# （スワップの変動側など）で用いられ、債券の経過利子には使わない。`bondlab` の
# ACT/ACT はこの ISMA（Bond）準拠であり、以降で QuantLib の
# `ActualActual.Bond` と機械精度で一致することを確認する。



# %% [markdown]
# **数値例**：同じ期間（2024-01-15→2024-04-15、実日数91日、2024年は閏年で分母366）を ISDA 変種で測ると $\mathrm{yf}_{\text{ISDA}}=91/366=0.248634$、$AI_{\text{ISDA}}=100\times 0.03\times 0.248634=0.745902$ です。ISMA の $0.75$ と期中では一致しません。
# %% [markdown]
# **数値例**：直前利払日 2024-01-15・次回 2024-07-15・受渡 2024-04-15、年クーポン $c=3\%$・半年払い（$f=2$）・額面100なら、1回分のクーポンは $C/f=1.5$、クーポン期間182日のうち91日経過なので $AI_{\text{ISMA}}=1.5\times\dfrac{91}{182}=0.75$ です。
# %% [markdown]
# ## スクラッチ実装
#
# 経過利子（ISMA / ISDA 両変種）とクリーン⇔ダーティ変換を素朴に実装する。
# クーポン日の列は満期から逆算して刻む（`bondlab` と同じ月数ベース）。ACT/ACT
# の暦年按分は `bondlab.daycount.year_fraction` に委譲し、ISDA 変種の実装に使う。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `coupon_dates(issue, maturity, freq)` | 発行日, 満期, 年利払回数 | `date` のリスト | 満期から逆算した利払日の列 |
# | `surrounding(dates, issue, settle)` | 利払日列, 発行日, 受渡日 | `(prev, next)` | 受渡日を挟む直前・次の利払日 |
# | `accrued_isma(prev, nxt, settle, coupon, freq, face)` | 前後利払日, 受渡日, 年クーポン率, 頻度, 額面 | 経過利子 | ISMA（Bond）変種の経過利子 |
# | `accrued_isda(prev, settle, coupon, face)` | 直前利払日, 受渡日, 年クーポン率, 額面 | 経過利子 | ISDA 変種の経過利子 |
# | `dirty_from_clean(clean, ai)` | クリーン価格, 経過利子 | ダーティ価格 | クリーン→ダーティ |
# | `clean_from_dirty(dirty, ai)` | ダーティ価格, 経過利子 | クリーン価格 | ダーティ→クリーン |

# %%
import datetime as dt

import numpy as np

import bondlab
from bondlab import daycount
from bondlab.bond import FixedRateBond

np.random.seed(0)
print("bondlab version:", bondlab.__version__)


def _add_months(d: dt.date, months: int) -> dt.date:
    """月加算（月末は各月の末日に丸める）。"""
    m0 = d.month - 1 + months
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    last = 31 if m == 12 else (dt.date(y, m + 1, 1) - dt.timedelta(days=1)).day
    return dt.date(y, m, min(d.day, last))


def coupon_dates(issue: dt.date, maturity: dt.date, freq: int):
    """満期から逆算してクーポン日の列を作る（issue より後のみ残す）。"""
    step = 12 // freq
    dates = [maturity]
    d = maturity
    while d > issue:
        d = _add_months(d, -step)
        dates.append(d)
    return [x for x in sorted(set(dates)) if x > issue]


def surrounding(dates, issue: dt.date, settle: dt.date):
    """受渡日を挟む (直前利払日, 次利払日) を返す。発行日も候補に含める。"""
    prev, nxt = None, None
    for d in [issue] + dates:
        if d <= settle:
            prev = d
        elif nxt is None:
            nxt = d
    return prev, nxt


def accrued_isma(prev, nxt, settle, coupon, freq, face=100.0):
    """ISMA（Bond）変種。1回分クーポンをクーポン期間の実日数で日割りする。"""
    d_full = (nxt - prev).days
    d_part = (settle - prev).days
    periodic = face * coupon / freq
    return periodic * d_part / d_full if d_full > 0 else 0.0


def accrued_isda(prev, settle, coupon, face=100.0):
    """ISDA 変種。年クーポンに ACT/ACT(ISDA) の年数を掛ける（暦年 365/366 按分）。"""
    yf = daycount.year_fraction(prev, settle, "ACT/ACT")
    return face * coupon * yf


def dirty_from_clean(clean, ai):
    """クリーン価格に経過利子を足してダーティ価格にする。"""
    return clean + ai


def clean_from_dirty(dirty, ai):
    """ダーティ価格から経過利子を引いてクリーン価格にする。"""
    return dirty - ai


# %% [markdown]
# 標準的な半年債を1本組み、受渡日を期中に置いて経過利子を計算する。同じ受渡日で
# ISMA と ISDA を並べ、差を確認する。

# %%
issue = dt.date(2024, 6, 15)
maturity = dt.date(2029, 6, 15)
coupon = 0.02          # 年 2%
freq = 2               # 半年利払い
settle = dt.date(2026, 9, 10)

dates = coupon_dates(issue, maturity, freq)
prev, nxt = surrounding(dates, issue, settle)
print(f"直前利払日: {prev}   次利払日: {nxt}   受渡日: {settle}")
print(f"クーポン期間の実日数: {(nxt - prev).days} 日   経過日数: {(settle - prev).days} 日")

ai_isma = accrued_isma(prev, nxt, settle, coupon, freq)
ai_isda = accrued_isda(prev, settle, coupon)
print(f"経過利子 ISMA(Bond) = {ai_isma:.15f}")
print(f"経過利子 ISDA       = {ai_isda:.15f}")
print(f"mid-period のズレ    = {ai_isda - ai_isma:.2e}")

# %% [markdown]
# `bondlab.bond.FixedRateBond` の経過利子はスクラッチの ISMA と一致するはずである。
# クリーン⇔ダーティ変換も、価格から経過利子を分離・合成できることを確認する。

# %%
bond = FixedRateBond(issue, maturity, coupon, frequency=freq, convention="ACT/ACT")
ai_lib = bond.accrued(settle)
print(f"bondlab.accrued        = {ai_lib:.15f}")
print(f"スクラッチ ISMA との差 = {abs(ai_lib - ai_isma):.2e}")
assert abs(ai_lib - ai_isma) < 1e-12

# クリーン→ダーティ→クリーンの往復
ytm = 0.03
clean = bond.clean_price(ytm, settle)
dirty = dirty_from_clean(clean, ai_lib)
clean_back = clean_from_dirty(dirty, ai_lib)
print(f"クリーン価格 = {clean:.8f}")
print(f"ダーティ価格 = {dirty:.8f}   (bondlab: {bond.dirty_price(ytm, settle):.8f})")
assert abs(dirty - bond.dirty_price(ytm, settle)) < 1e-10
assert abs(clean_back - clean) < 1e-12
print("クリーン⇔ダーティ変換のチェックを通過しました")

# %% [markdown]
# ## QuantLib検証
#
# QuantLib の `FixedRateBond` に `ActualActual.Bond`（= ISMA）を渡すと、経過利子は
# `bondlab` およびスクラッチ ISMA と機械精度で一致する。対して `ActualActual.ISDA`
# を渡すと、同じ受渡日でも経過利子がずれる。これが「債券は ISMA を使う」ことの
# 数値的な裏付けになる。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

sched = ql.Schedule(
    ql.Date(15, 6, 2024),
    ql.Date(15, 6, 2029),
    ql.Period(ql.Semiannual),
    ql.NullCalendar(),
    ql.Unadjusted,
    ql.Unadjusted,
    ql.DateGeneration.Backward,
    False,
)
dc_bond = ql.ActualActual(ql.ActualActual.Bond, sched)
qlb_isma = ql.FixedRateBond(0, 100.0, sched, [coupon], dc_bond)
qlb_isda = ql.FixedRateBond(0, 100.0, sched, [coupon], ql.ActualActual(ql.ActualActual.ISDA))

d = ql.Date(10, 9, 2026)
ql_ai_isma = qlb_isma.accruedAmount(d)
ql_ai_isda = qlb_isda.accruedAmount(d)
print(f"QuantLib accrued ISMA(Bond) = {ql_ai_isma:.15f}")
print(f"QuantLib accrued ISDA       = {ql_ai_isda:.15f}")
print(f"bondlab − QL(ISMA) の差      = {abs(ai_lib - ql_ai_isma):.2e}")
print(f"QL(ISDA) − QL(ISMA) の差     = {ql_ai_isda - ql_ai_isma:.2e}")

# ISMA 変種は機械精度で一致する。
assert abs(ai_lib - ql_ai_isma) < 1e-12
assert abs(ai_isma - ql_ai_isma) < 1e-12
# ISDA 変種は mid-period で有意にずれる。
assert abs(ql_ai_isda - ql_ai_isma) > 1e-4

# %% [markdown]
# クリーン価格も突合する。半年複利の street convention どうしなら、`bondlab` の
# ダーティ/クリーン価格は QuantLib と一致する。

# %%
dc_price = ql.ActualActual(ql.ActualActual.Bond, sched)
qlb_px = ql.FixedRateBond(0, 100.0, sched, [coupon], dc_price)
ql_clean = qlb_px.cleanPrice(ytm, dc_price, ql.Compounded, ql.Semiannual, d)
ql_dirty = qlb_px.dirtyPrice(ytm, dc_price, ql.Compounded, ql.Semiannual, d)
print(f"クリーン価格  bondlab = {clean:.10f}   QuantLib = {ql_clean:.10f}")
print(f"ダーティ価格  bondlab = {dirty:.10f}   QuantLib = {ql_dirty:.10f}")
assert abs(clean - ql_clean) < 1e-8
assert abs(dirty - ql_dirty) < 1e-8
print("価格の突合を通過しました")

# %% [markdown]
# ### ISDA が期中でずれる様子を複数受渡日で見る
#
# 受渡日を期首から期末まで動かし、ISMA と ISDA の経過利子を並べる。受渡日を
# 直前利払日に取ったときだけ両者ゼロで一致し、期が進むほど差は開く。ISMA は
# 期末で 1 回分クーポン（1.0）に達するのに対し、ISDA は暦年 365 を基準にする
# ため期末で 1.0 をわずかに超え、期中も一貫して ISMA より大きい。

# %%
sample_days = [prev + dt.timedelta(days=k) for k in range(0, (nxt - prev).days + 1, 20)]
print(f"{'受渡日':>12} {'ISMA':>12} {'ISDA':>12} {'ISDA-ISMA':>12}")
for s in sample_days:
    a_i = accrued_isma(prev, nxt, s, coupon, freq)
    a_d = accrued_isda(prev, s, coupon)
    print(f"{str(s):>12} {a_i:>12.8f} {a_d:>12.8f} {a_d - a_i:>12.2e}")

# %% [markdown]
# ## 実データ適用
#
# ネットワークは使わず、実在の年限・クーポンに寄せた合成銘柄を組む。同一銘柄
# （同一キャッシュフロー）の利回りを UST 慣行（半年複利）と JGB 慣行（単利）で
# 表示し、差を定量化する。ここでは受渡日を利払日ちょうどに取り、経過利子ゼロ・
# クリーン=ダーティのクリーンな状態で利回り差だけを取り出す。

# %%
def simple_yield(price, annual_coupon, years):
    """JGB 単利最終利回り。年率に均した収益を投下元本で割る。"""
    return (annual_coupon + (100.0 - price) / years) / price


# 合成 UST 風: 5年・クーポン 4%
ust = FixedRateBond(dt.date(2021, 6, 15), dt.date(2031, 6, 15), 0.04, frequency=2, convention="ACT/ACT")
ust_settle = dt.date(2026, 6, 15)     # 利払日ちょうど（残存 5 年）
ust_years = 5.0
ust_annual_coupon = 100.0 * 0.04

# 合成 JGB 風: 10年・クーポン 0.6%
jgb = FixedRateBond(dt.date(2019, 6, 20), dt.date(2029, 6, 20), 0.006, frequency=2, convention="ACT/365F")
jgb_settle = dt.date(2026, 6, 20)     # 利払日ちょうど（残存 3 年）
jgb_years = 3.0
jgb_annual_coupon = 100.0 * 0.006

print(f"{'銘柄':<10}{'クリーン価格':>12}{'半年複利YTM(%)':>16}{'単利利回り(%)':>16}{'差(bp)':>10}")
for name, bond_i, settle_i, years_i, ac_i in [
    ("UST風", ust, ust_settle, ust_years, ust_annual_coupon),
    ("JGB風", jgb, jgb_settle, jgb_years, jgb_annual_coupon),
]:
    # まず半年複利で価格を作り、その価格から両慣行の利回りを表示する。
    y_compound = 0.045 if name == "UST風" else 0.009
    px = bond_i.clean_price(y_compound, settle_i)
    y_comp_back = bond_i.yield_from_price(px, settle_i)   # UST 慣行（半年複利）
    y_simple = simple_yield(px, ac_i, years_i)            # JGB 慣行（単利）
    diff_bp = (y_comp_back - y_simple) * 1e4
    print(f"{name:<10}{px:>12.5f}{y_comp_back * 100:>16.5f}{y_simple * 100:>16.5f}{diff_bp:>10.3f}")

# %% [markdown]
# クーポンが高く年限が長い UST 風のほうが、複利と単利の差が大きい。単利は複利
# 効果（クーポンの再投資と割引の複利化）を無視するため、同じ価格でも表示利回りが
# ずれる。市場をまたいで利回りを比較するときは、まず慣行を揃える必要がある。

# %% [markdown]
# ## 演習
#
# 1. **同一 CF の利回りを UST 慣行 / JGB 慣行で比較する。**
#    クーポン 3%・残存 7 年・半年利払いの合成債を、クリーン価格 96.0 で受渡日を
#    利払日ちょうどに取って組め。UST 慣行（半年複利 YTM, `yield_from_price`）と
#    JGB 慣行（単利最終利回り, `simple_yield`）の利回りを求め、差を bp で示せ。
#    クーポンを 1% に下げると差がどう変わるかも述べよ。
#
# 2. **ISDA vs ISMA の経過利子差を複数受渡日で比較する。**
#    本文の 2% 半年債について、直前利払日から次利払日までを 10 日刻みで受渡日を
#    動かし、`accrued_isma` と `accrued_isda` の差を配列に集めよ。差が最大になる
#    受渡日と、その値を報告せよ。利払日ちょうどで差がゼロになることも確認せよ。
#
# 解答例は `solutions/S1/sol_0104.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/01_bond_basics.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | 経過利子 | accrued interest | 直前利払日から受渡日までの、売り手に帰属する未払い利息 |
# | クリーン価格 | clean price | 経過利子を除いた気配価格。銘柄・時点間の比較に使う |
# | ダーティ価格 | dirty price | クリーン価格に経過利子を足した実際の受渡代金 |
# | T+1決済 | T+1 settlement | 約定の翌営業日に受け渡す慣行。UST・Gilt が採用 |
# | 受渡日 | settlement date | 資金と証券が実際に交換される日。経過利子計算の基準日 |
