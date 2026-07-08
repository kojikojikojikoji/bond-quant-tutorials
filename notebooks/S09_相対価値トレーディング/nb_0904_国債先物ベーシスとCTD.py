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
# # S9-4 国債先物ベーシスとCTD
#
# ## 学習目標
#
# - 先物と現物の裁定関係を、受渡し（デリバリー）の仕組みから説明できる
# - コンバージョンファクター（conversion factor）の計算式と、6%基準の由来を理解する
# - グロスベーシス・ネットベーシス・インプライドレポ（implied repo）を自分で計算できる
# - 受渡適格銘柄バスケットから最割安銘柄（CTD）を特定し、金利シフトで CTD が
#   入れ替わる条件（CTDスイッチのオプション性）を分析できる
#
# 国債先物は「標準物（架空の6%クーポン債）」を対象に取引され、決済時には
# 売り方が受渡適格な実在銘柄の中から1つを選んで渡します。どの銘柄を渡すのが
# 最も得か、その銘柄が最割安銘柄（CTD, cheapest-to-deliver）です。CTD の特定と、
# 金利変動での入れ替わり（スイッチ）は、先物のヘッジ比率・相対価値の土台になります。


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# 先物ベーシスと CTD は、docs/債券ファンドの業務.md でいう「先物ベーシス・CTD（先物と現物の裁定）」の実装で、RV ファンドとセルサイドの双方が使う土台です。ネットベーシスは受渡しの実質コストそのものなので、これが理論水準から乖離したときにファンドはベーシストレードを建てます。ネットベーシスが割高（先物が現物に対して安い）なら現物を買って先物を売り（キャッシュ・アンド・キャリー）、割安なら逆を建てて、受渡日に向けてベーシスが理論値へ収束する方に賭けます。方向性の金利観ではなく、同じキャッシュフローを持つ先物と現物の値付けの歪みが受渡しで必ず解消するという裁定構造に賭ける点が旨味です。
#
# インプライドレポ（implied repo）は、この裁定の採算を測る中心指標です。現物を買って先物で売り渡すと決めたときに、その取引が生む含意の調達利回りがインプライドレポで、これが実際に市場でファンディングできるレポレートを上回っていれば、現物をレポで調達して先物で渡すだけでリスクなく利ざやが抜けます。ファンドはこの「インプライドレポ − 実効レポ」の差を、レバレッジをかけて数多く積むことで薄い裁定益を意味のある収益に育てます。ここでも稼ぎの符号を決めるのはレポ市場の状態で、CTD 銘柄がスペシャル化すると調達コストが変わり、ベーシスの水準自体が動きます。
#
# CTD の特定と、金利シフトで CTD が入れ替わる条件（CTD スイッチ）の分析は、ヘッジ比率と裁定の安全性を同時に左右します。先物は CTD の値動きに連動するので、先物で現物ポートフォリオをヘッジするデスクは CTD を基準にヘッジ比率（CF 調整後の DV01 マッチ）を組みます。CTD が金利変動で別銘柄に入れ替わると、この連動関係が飛ぶため、CTD スイッチはヘッジ担当にとってのベーシスリスクそのものです。一方で RV ファンドにとって、CTD スイッチはオプション性として収益機会になります。先物の売り手は最も安い銘柄を選んで渡せる権利（デリバリーオプション）を持ち、その価値を過小評価したベーシスは買い、過大評価したベーシスは売る——スイッチ確率とオプション価値の見立てが、そのままアルファの源泉になります。
#
# これらはセルサイドのマーケットメイク（RFQ 値付けと在庫管理）とも直結します。先物・現物の両面クォートを出すデスクは、CTD とインプライドレポを常時把握して値付けの基準に落とし込み、在庫リスクとスプレッド収益の綱引きを管理します。ベーシスの計算が正確でなければ、裁定機会を逃すか、割高な在庫を掴んで損益を毀損するかのどちらかになります。
# %% [markdown]
# ## 理論
#
# ### 標準物と受渡しの仕組み
#
# 国債先物（日本国債先物）は、残存10年・クーポン6%の**標準物（notional bond）**という
# 架空債券を対象に価格が決まります。実在の債券は年限もクーポンもばらばらなので、
# そのままでは1つの先物で受渡しできません。そこで、受渡適格の各銘柄を標準物に
# 換算する係数として**コンバージョンファクター（conversion factor, CF）**を使います。
#
# 決済日に売り方が銘柄 $i$ を渡すと、買い方から受け取る金額（請求金額, invoice）は
#
# $$ \text{請求金額}_i = F \cdot \mathrm{CF}_i + \text{経過利子}_i $$
#
# です。$F$ は先物価格、$\mathrm{CF}_i$ は銘柄 $i$ の CF です。売り方は市場で現物を
# 買って渡すので、渡す銘柄ごとに損益が変わります。**渡すのが最も得な（正味コストが
# 最小の）銘柄が CTD** です。
#
# ### コンバージョンファクターの定義と6%基準の由来
#
# CF は「その受渡適格銘柄を、標準物の利回り（＝6%）で割り引いた価格を100で割った値」
# として定義されます。額面100の銘柄 $i$（年クーポン $c_i$、残存 $T_i$、半年払い）に対し、
#
# $$ \mathrm{CF}_i = \frac{1}{100}\sum_{k} \frac{c_i/2}{(1+0.06/2)^{\,n_k}} + \frac{100}{(1+0.06/2)^{\,n_N}} \Big/ 100 $$
#
# すなわち $\mathrm{CF}_i = P_i(6\%)/100$（受渡日基準の6%利回りでのクリーン価格を100で割る）です。
#
# 6%という基準は、標準物のクーポンそのものです。歴史的に主要国の国債先物は
# 標準物クーポンを6%（かつては8%）に固定してきました。基準が固定なので、CF は
# 金利水準では変わらず、銘柄のクーポンと残存だけで一意に決まります。含意は次の2点です。
#
# - **クーポン $c_i < 6\%$ の銘柄は $\mathrm{CF}_i < 1$**、$c_i > 6\%$ なら $\mathrm{CF}_i > 1$。
#   クーポンが標準物6%と等しく残存が整数なら $\mathrm{CF}_i \approx 1$ です。
# - CF は金利に依存しないため、$F \cdot \mathrm{CF}_i$ という換算は、金利が動くと銘柄間で
#   「ずれ」を生みます。このずれが CTD の入れ替わりを生む源です。
#
# ### グロスベーシス・キャリー・ネットベーシス
#
# 現物と先物の価格差を**ベーシス（basis）**と呼びます。銘柄 $i$ のクリーン価格を $P_i$ として、
#
# $$ \text{グロスベーシス}_i = P_i - F\cdot \mathrm{CF}_i $$
#
# グロスベーシスには「現物を持ち越す間のキャリー（クーポン収入 − レポ調達コスト）」が
# 混ざっています。キャリーを差し引いた純粋な割高・割安が**ネットベーシス（net basis）**です。
#
# $$ \text{ネットベーシス}_i = \underbrace{P_i^{\,\text{fwd}}}_{\text{受渡日フォワード価格}} - F\cdot \mathrm{CF}_i $$
#
# ここでフォワード価格は、現物のダーティ価格をレポ金利で受渡日まで持ち越し、途中の
# クーポンを差し引いて、受渡日の経過利子を除いたものです。ネットベーシスは受渡しの
# 実質コストそのものであり、**CTD はネットベーシスが最小の銘柄**です。
#
# ### インプライドレポ
#
# **インプライドレポ（implied repo rate, IRR）**は、「現物を買って先物を売り、受渡日に
# 渡す」という裁定を組んだときの実現利回りです。銘柄 $i$ について、
#
# $$ \mathrm{IRR}_i = \frac{\big(F\cdot \mathrm{CF}_i + \text{受渡日経過利子}_i + \text{期中クーポン}_i\big) - \big(P_i + \text{経過利子}_i\big)}{\big(P_i + \text{経過利子}_i\big)\cdot t} $$
#
# 分子は「受渡しで受け取る総額 − 現物を買う総額」、$t$ は受渡しまでの年数です。
# 実現利回りが最も高い銘柄を渡すのが最も得なので、**CTD はインプライドレポが最大の銘柄**です。
# ネットベーシス最小とインプライドレポ最大は同じ銘柄を指し、これは相互チェックに使えます。
#
# ### CTDスイッチのオプション性
#
# CF は金利に依存しないのに、現物価格 $P_i$ は金利に依存します。したがって金利が動くと
# ネットベーシスの銘柄間順位が入れ替わり、CTD が別銘柄に移ります。これが**CTDスイッチ**です。
# 経験則として、
#
# - 市場利回りが標準物クーポン（6%）**より低い**とき、CTD は**低デュレーション（短い・高クーポン）**銘柄
# - 市場利回りが6%**より高い**とき、CTD は**高デュレーション（長い・低クーポン）**銘柄
#
# になります。先物の売り方は「最も安い銘柄を選んで渡せる」オプションを持っており、これを
# **CTDスイッチのオプション（delivery option）**と呼びます。金利が6%から大きく離れているほど
# スイッチは起きにくく、オプションは深いアウト・オブ・ザ・マネーになります。本 notebook では
# 現在の低金利下（利回り $\ll 6\%$）で、$\pm 50\text{bp}$ 程度ではスイッチが起きないことを確認します。



# %% [markdown]
# **数値例**：現物クリーン $P_i=101.50$、先物 $F=100.20$、$\mathrm{CF}_i=1.02$ なら グロスベーシス $=101.50-100.20\times1.02=-0.70$（円/額面100）。キャリー調整後のフォワードクリーンが $101.35$ ならネットベーシス $=101.35-100.20\times1.02=-0.85$ で、キャリーぶんだけ実質コストが小さく評価されます。
# %% [markdown]
# **数値例**：クーポン2%・残存10年・半年払いの銘柄を標準物利回り6%で評価すると $P_i(6\%)=70.25$ 円、よって $\mathrm{CF}=P_i(6\%)/100\approx0.702$ です。クーポンが標準物6%を下回るため CF は1を大きく割り込みます。
# %% [markdown]
# ## スクラッチ実装
#
# CF・グロス／ネットベーシス・インプライドレポを自作します。CF の部品と検証には
# `bondlab.bt.conversion_factor`（簡易版）を使い、notebook では端数期間を含む厳密版へ
# 拡張します。厳密版は `bondlab.bond.FixedRateBond` の street convention 価格計算を
# 再利用し、受渡日基準で6%利回りのクリーン価格を求めます。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `exact_conversion_factor(coupon, maturity, delivery, ...)` | クーポン, 満期日, 受渡日 | CF | 端数期間込みの厳密版CF（6%利回りクリーン価格/100） |
# | `forward_clean_price(clean, ai_s, ai_d, coupons, repo, t)` | 現物クリーン, 経過利子(現在/受渡日), 期中クーポン, レポ, 年数 | 受渡日フォワードクリーン価格 | 現物を受渡日まで持ち越した価格 |
# | `gross_basis(clean, futures, cf)` | クリーン価格, 先物価格, CF | グロスベーシス | 現物と先物換算値の単純差 |
# | `net_basis(fwd_clean, futures, cf)` | フォワード価格, 先物価格, CF | ネットベーシス | キャリー調整後のベーシス |
# | `implied_repo(clean, ai_s, futures, cf, ai_d, coupons, t)` | 上記一式 | インプライドレポ | 現物買い／先物売りの裁定実現利回り |
# | `build_bond(coupon, maturity, freq)` | クーポン, 満期日, 頻度 | FixedRateBond | 受渡適格銘柄の債券オブジェクト生成 |

# %%
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
import bondlab
from bondlab.bt import conversion_factor as simple_conversion_factor
from bondlab.bond import FixedRateBond
from bondlab.curve import bootstrap_par

np.random.seed(0)
print("bondlab version:", bondlab.__version__)

# 発行日は十分過去に固定する（受渡適格帯の残存だけを扱うので起点の厳密性は不要）。
ISSUE = dt.date(2005, 1, 1)


def build_bond(coupon: float, maturity: dt.date, freq: int = 2) -> FixedRateBond:
    """受渡適格銘柄の債券オブジェクトを作る。額面100・半年払いを既定にする。"""
    return FixedRateBond(issue=ISSUE, maturity=maturity, coupon=coupon, frequency=freq)


def exact_conversion_factor(coupon: float, maturity: dt.date, delivery: dt.date,
                            notional_coupon: float = 0.06, freq: int = 2) -> float:
    """端数期間を含む厳密版コンバージョンファクター。

    受渡日 `delivery` を決済日として、標準物利回り（6%）でのクリーン価格を100で割る。
    次クーポンまでの端数期間は FixedRateBond の street convention（指数 w+j）で厳密に
    織り込まれる。簡易版 `bondlab.bt.conversion_factor` は残存を半年グリッドに丸める点が違う。
    """
    bond = build_bond(coupon, maturity, freq)
    return bond.clean_price(notional_coupon, delivery) / 100.0


def forward_clean_price(clean: float, ai_settle: float, ai_deliv: float,
                        coupons: float, repo: float, t: float) -> float:
    """現物を受渡日まで持ち越したフォワードクリーン価格。

    ダーティ価格をレポ金利で持ち越し、期中クーポンを差し引き、受渡日の経過利子を除く。
    """
    dirty = clean + ai_settle
    fwd_dirty = dirty * (1.0 + repo * t) - coupons
    return fwd_dirty - ai_deliv


def gross_basis(clean: float, futures: float, cf: float) -> float:
    """グロスベーシス = 現物クリーン価格 − 先物価格 × CF。"""
    return clean - futures * cf


def net_basis(fwd_clean: float, futures: float, cf: float) -> float:
    """ネットベーシス = 受渡日フォワード価格 − 先物価格 × CF（キャリー調整後）。"""
    return fwd_clean - futures * cf


def implied_repo(clean: float, ai_settle: float, futures: float, cf: float,
                 ai_deliv: float, coupons: float, t: float) -> float:
    """インプライドレポ（現物買い／先物売り裁定の実現利回り、年率）。"""
    invoice = futures * cf + ai_deliv + coupons
    cost = clean + ai_settle
    return (invoice - cost) / (cost * t)


# %% [markdown]
# ### 厳密版CFの動作確認
#
# クーポンが標準物と同じ6%の銘柄では、厳密版 CF がほぼ1になることを確かめます。端数期間の
# ぶんだけ厳密に1からわずかにずれます。低クーポンの実在 JGB では CF が1を大きく下回ります。

# %%
today = dt.date(2026, 6, 22)      # 評価日（決済日）
delivery = dt.date(2026, 9, 20)   # 先物の受渡日
horizon = (delivery - today).days / 365.0

for ytm_years in [7, 9, 11]:
    mat = today + dt.timedelta(days=round(ytm_years * 365.25))
    cf6 = exact_conversion_factor(0.06, mat, delivery)
    print(f"クーポン6%・残存約{ytm_years}年  厳密版CF = {cf6:.6f}")

# %% [markdown]
# ## QuantLib検証
#
# 本シリーズは自作実装を QuantLib で答え合わせしますが、CF と CTD の判定は取引所固有の
# 規約に依存し、QuantLib に単純な突合相手がありません。そこで本節は QuantLib の代わりに
# **(1) 簡易版CF＝クーポン6%で CF≈1 の確認**と、**(2) 自作 CTD ロジックの内部整合**
# （ネットベーシス最小 = インプライドレポ最大 = 同一銘柄）で検証とします。その旨を明記します。
#
# ### 検証1：簡易版CFはクーポン6%でCF≈1
#
# `bondlab.bt.conversion_factor` は残存を半年グリッドに丸めた簡易版です。クーポン6%・整数年で
# ちょうど1になり、これが厳密版の妥当性チェックの基準になります。厳密版との差は端数期間に由来します。

# %%
print(f"{'残存(年)':>8} {'簡易版CF(6%)':>14} {'厳密版CF(6%)':>14} {'差':>12}")
for yy in [7.0, 8.5, 10.0, 10.9]:
    mat = today + dt.timedelta(days=round(yy * 365.25))
    cf_simple = simple_conversion_factor(0.06, yy)          # 簡易版（半年グリッド）
    cf_exact = exact_conversion_factor(0.06, mat, delivery)  # 厳密版（端数期間込み）
    print(f"{yy:>8.1f} {cf_simple:>14.6f} {cf_exact:>14.6f} {cf_exact - cf_simple:>12.6f}")

# 整数年・クーポン6%では簡易版はちょうど1。
assert abs(simple_conversion_factor(0.06, 10) - 1.0) < 1e-9
# 厳密版も端数期間ぶんの微小差を除けば1近傍。
assert abs(exact_conversion_factor(0.06, today + dt.timedelta(days=round(10 * 365.25)), delivery) - 1.0) < 5e-3
print("検証1 OK: クーポン6%で簡易版CF=1、厳密版CF≈1")

# %% [markdown]
# ## 実データ適用
#
# 合成の受渡適格銘柄ユニバース `data/samples/synthetic_jgb_universe.csv` を、日本国債先物の
# 受渡適格帯（残存7〜11年）に絞ってバスケットを作ります。パー利回りから割引カーブを
# ブートストラップし、各銘柄の適正利回りに `rich_cheap_bp` を上乗せして市場クリーン価格を作ります。
# 先物価格 $F$ は、フォワード価格 ÷ CF の最小値（＝最割安銘柄のネットベーシスが0になる水準）として
# 無裁定で決めます。

# %%
universe = pd.read_csv("data/samples/synthetic_jgb_universe.csv")
basket = universe[(universe.maturity_years >= 7.0) & (universe.maturity_years <= 11.0)].reset_index(drop=True)
print("受渡適格バスケット（残存7〜11年）:")
display(basket)

# パー利回りカーブ（低金利の右肩上がり）を組み、ブートストラップで割引カーブを作る。
tenors = np.arange(1, 21).astype(float)
par_rates = np.clip(0.002 + 0.0016 * tenors, None, 0.020)
curve = bootstrap_par(tenors, par_rates, frequency=1)
print(f"\nゼロレート例  7年={curve.zero_rate(7.0):.4%}  10年={curve.zero_rate(10.0):.4%}")

REPO = 0.001  # 短期レポ金利（0.10%）を裁定の資金調達コストに使う。


# %%
def basis_table(shift_bp: float, futures: float | None = None) -> tuple[pd.DataFrame, float]:
    """金利シフト（bp）を与え、バスケットのベーシス指標一式を計算する。

    futures を渡さない場合は、最割安銘柄のネットベーシスが0になる無裁定先物価格を内生的に決める。
    """
    rows = []
    for _, r in basket.iterrows():
        mat = today + dt.timedelta(days=round(r.maturity_years * 365.25))
        bond = build_bond(r.coupon, mat)
        fair = np.interp(r.maturity_years, tenors, par_rates)          # 適正利回り
        y = fair + shift_bp * 1e-4 + r.rich_cheap_bp * 1e-4           # 市場利回り
        clean = bond.clean_price(y, today)
        ai_s = bond.accrued(today)
        ai_d = bond.accrued(delivery)
        coupons = sum(c for d, c in bond.cashflows() if today < d <= delivery)  # 期中クーポン
        cf = exact_conversion_factor(r.coupon, mat, delivery)
        fwd = forward_clean_price(clean, ai_s, ai_d, coupons, REPO, horizon)
        rows.append(dict(bond_id=r.bond_id, coupon=r.coupon, maturity_years=r.maturity_years,
                         cf=cf, clean=clean, ai_settle=ai_s, ai_deliv=ai_d, coupons=coupons,
                         fwd_clean=fwd, market_yield=y))
    df = pd.DataFrame(rows)
    if futures is None:
        futures = float((df.fwd_clean / df.cf).min())
    df["gross_basis"] = [gross_basis(c, futures, cf) for c, cf in zip(df.clean, df.cf)]
    df["net_basis"] = [net_basis(f, futures, cf) for f, cf in zip(df.fwd_clean, df.cf)]
    df["implied_repo"] = [
        implied_repo(c, ai_s, futures, cf, ai_d, cp, horizon)
        for c, ai_s, cf, ai_d, cp in zip(df.clean, df.ai_settle, df.cf, df.ai_deliv, df.coupons)
    ]
    return df, futures


base, F = basis_table(0.0)
print(f"\n無裁定先物価格 F = {F:.4f}")
print(base[["bond_id", "coupon", "maturity_years", "cf", "clean",
            "gross_basis", "net_basis", "implied_repo"]].round(5).to_string(index=False))

ctd_by_net = base.loc[base.net_basis.idxmin(), "bond_id"]
ctd_by_irr = base.loc[base.implied_repo.idxmax(), "bond_id"]
print(f"\nCTD（ネットベーシス最小）= {ctd_by_net}")
print(f"CTD（インプライドレポ最大）= {ctd_by_irr}")

# %% [markdown]
# ### 検証2：CTDロジックの内部整合
#
# ネットベーシス最小の銘柄と、インプライドレポ最大の銘柄が一致することを確かめます。理論上
# 両者は同じ CTD を指すため、一致しなければ実装のどこかにバグがあることになります。

# %%
assert ctd_by_net == ctd_by_irr, "ネットベーシス最小とインプライドレポ最大が食い違う（実装を疑う）"
# CTD のネットベーシスは（無裁定Fの決め方から）ほぼ0。
assert abs(base.net_basis.min()) < 1e-6
# ネットベーシスとインプライドレポは逆順（順位相関 -1）になるはず。
order_net = base.sort_values("net_basis").bond_id.tolist()
order_irr = base.sort_values("implied_repo", ascending=False).bond_id.tolist()
assert order_net == order_irr
print(f"検証2 OK: CTD = {ctd_by_net}（ネットベーシス最小 = インプライドレポ最大で一致）")

# %% [markdown]
# ### CFとネットベーシスの可視化
#
# CF はクーポンが低いほど1を大きく下回ります。ネットベーシスが最小（≈0）の銘柄が CTD です。

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
colors = ["#d62728" if b == ctd_by_net else "#1f77b4" for b in base.bond_id]
axes[0].bar(base.bond_id, base.cf, color=colors)
axes[0].axhline(1.0, color="gray", ls="--", lw=1)
axes[0].set_title("コンバージョンファクター（赤=CTD）")
axes[0].set_ylabel("CF")
axes[1].bar(base.bond_id, base.net_basis, color=colors)
axes[1].set_title("ネットベーシス（赤=CTD, 最小≈0）")
axes[1].set_ylabel("ネットベーシス（円/額面100）")
for ax in axes:
    ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 金利±50bpでのCTDスイッチ分析
#
# 先物価格 $F$ を現状水準に固定したまま、カーブを平行に $\pm 50\text{bp}$ 動かして、CTD が
# 入れ替わるかを見ます。あわせて、どこまで金利が動けばスイッチが起きるかを広いレンジで探ります。

# %%
print(f"{'シフト(bp)':>9} {'CTD':>8} {'CTDネットベーシス':>16}")
for bp in [-50, -25, 0, 25, 50]:
    df_bp, _ = basis_table(bp, futures=F)   # F は現状で固定
    ctd = df_bp.loc[df_bp.net_basis.idxmin(), "bond_id"]
    print(f"{bp:>9d} {ctd:>8} {df_bp.net_basis.min():>16.5f}")

# 広いレンジでスイッチ閾値を探す。
shifts = np.arange(-200, 601, 25)
ctd_path = []
for bp in shifts:
    df_bp, _ = basis_table(float(bp), futures=F)
    ctd_path.append(df_bp.loc[df_bp.net_basis.idxmin(), "bond_id"])
switch_points = [(int(shifts[i]), ctd_path[i - 1], ctd_path[i])
                 for i in range(1, len(shifts)) if ctd_path[i] != ctd_path[i - 1]]
print("\nCTDが入れ替わるシフト:", switch_points if switch_points else "±50bpの範囲外")

fig, ax = plt.subplots(figsize=(9, 3.6))
ids = sorted(set(ctd_path))
ymap = {b: k for k, b in enumerate(ids)}
ax.step(shifts, [ymap[b] for b in ctd_path], where="post", color="#1f77b4")
ax.axvspan(-50, 50, color="orange", alpha=0.15, label="±50bp")
ax.set_yticks(range(len(ids)))
ax.set_yticklabels(ids)
ax.set_xlabel("平行シフト（bp）")
ax.set_title("CTD銘柄と金利シフト（±50bp帯は色付き）")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# $\pm 50\text{bp}$ の範囲では CTD は動きません。市場利回り（約1%台）が標準物クーポン6%から
# 大きく下にあるため、CTD は低デュレーションの短い銘柄に張り付き、CTDスイッチのオプションは
# 深いアウト・オブ・ザ・マネーです。スイッチが起きるのは利回りが6%に近づく（+200bp超）
# 局面で、そこでは高デュレーションの長い銘柄へ CTD が移ります。この非対称性が、先物の
# 売り方が持つデリバリーオプションの価値の源泉です。

# %% [markdown]
# ## 演習
#
# 1. **CTD特定と金利シフトでのスイッチ分析**：受渡適格帯を残存6〜11年に広げて
#    バスケットを組み直し、現状の CTD を特定せよ。次に、カーブの平行シフトを
#    $-100\text{bp}$ から $+400\text{bp}$ まで動かし、CTD が最初に入れ替わるシフト幅を求めよ。
#    スイッチ前後で CTD のデュレーション（残存年数で代用）がどう変わるかを述べよ。
# 2. **インプライドセレクションとネットベーシスの関係**：現状バスケットの各銘柄について、
#    インプライドレポとネットベーシスを散布図にし、両者が逆相関（順位が反転）することを
#    確認せよ。さらに、レポ金利 `REPO` を 0.10% から 0.50% に上げたとき、ネットベーシスと
#    CTD がどう変わるかを説明せよ（キャリーの符号に注目）。
#
# 解答例は `solutions/S9/sol_0904.py` に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/09_trading.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | [コンバージョンファクター](../../glossary/09_trading.md#conversion-factor) | conversion factor | 受渡適格銘柄を標準物6%利回りで割り引いた価格/100。金利に依存しない換算係数 |
# | [CTD（最割安銘柄）](../../glossary/09_trading.md#cheapest-to-deliver) | cheapest-to-deliver | 先物の受渡しで渡すのが最も得な銘柄。ネットベーシス最小＝インプライドレポ最大 |
# | [インプライドレポ](../../glossary/09_trading.md#implied-repo-rate) | implied repo rate | 現物買い／先物売りの裁定を組んだときの実現利回り。最大の銘柄が CTD |
# | [ネットベーシス](../../glossary/09_trading.md#net-basis) | net basis | グロスベーシスからキャリーを除いた受渡しの実質コスト。最小の銘柄が CTD |
