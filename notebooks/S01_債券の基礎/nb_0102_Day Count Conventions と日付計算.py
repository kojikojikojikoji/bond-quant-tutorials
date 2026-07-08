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
# # S1-2 Day Count Conventions と日付計算
#
# ## 学習目標
#
# - 主要な日数計算規約（day count convention）4種（ACT/360・ACT/365F・30/360・
#   ACT/ACT）の定義を数式で書き下し、どの市場で使われるかを対応づけられる
# - 各規約が生まれた市場慣行の経緯、月末ルール、うるう年の扱いの違いを説明できる
# - 4種の年数計算と、営業日調整つきクーポン日スケジュール生成をスクラッチで実装し、
#   `bondlab.daycount` と一致させられる
# - QuantLib の `DayCounter` と `Schedule` を基準に、自作実装を100ケース規模で突合できる
# - 休日調整（翌営業日・修正翌営業日・前営業日）が利払日列に与える効果を、実物大の
#   スケジュールで確認できる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# 日数計算規約は地味ですが、これを取り違えると受け取るクーポン額・経過利子・割引係数がすべて数銭〜数bpずれます。マーケットメイクの現場では、セルサイドのトレーダーがビッド・オファースプレッドで稼ぐとき、そのスプレッドはしばしば1銘柄あたり数bpしかありません。値付けに使う年数計算が相手と1日でも食い違えば、約定価格の突合が合わず、利益だと思っていたスプレッドが計算ミスで消えます。ACT/360 と ACT/365F を混同したまま数千枚のポジションを評価すれば、損益の符号すら誤りかねません。
#
# 相対価値（RV）ファンドの収益は「割安なキャッシュフローを買い、割高を売る」ことで生まれますが、その割高・割安の判定は規約を市場ごとに正しく合わせて初めて意味を持ちます。米国債は ACT/ACT、日本国債は ACT/365F、マネーマーケットの変動金利は ACT/360 というように、同じ「利回り」でも土台の年数が違います。ここを揃えずにカーブを引くと、実在しない歪みを裁定機会と勘違いし、収束しないトレードに賭けてしまいます。
#
# 具体的には、スワップの固定側と変動側で規約が異なる（固定 30/360、変動 ACT/360 など）ため、キャリー（利ざや）の日次計上額を出すにも規約別の年数が要ります。休日調整（翌営業日・修正翌営業日・前営業日）まで含めて利払日を正しく打てないと、キャッシュフローの日付そのものがずれ、決済・担保管理が破綻します。この部品が無ければ、以降のあらゆる価格計算・リスク計算が始められません。
# %% [markdown]
# ## 理論
#
# ### 年数（year fraction）という共通部品
#
# 債券のクーポン、経過利子、割引に使う年数――これらはすべて「2つの日付の間が
# 何年ぶんか」という1つの量に帰着します。開始日を $d_1$、終了日を $d_2$ とすると、
# 年数 $\tau(d_1, d_2)$ は次の形をとります。
#
# $$ \tau(d_1, d_2) = \frac{\text{日数}(d_1, d_2)}{\text{基準日数}} $$
#
# 分子（日数の数え方）と分母（1年を何日とみなすか）の取り決めが day count
# convention です。同じ2日付でも規約が違えば $\tau$ は変わり、クーポン額・経過利子・
# 割引係数がすべてずれます。分子・分母のどちらを実日数（actual）にし、どちらを
# 固定値にするかで規約が枝分かれします。
#
# | 規約 | 分子 | 分母 | 主な市場 |
# |---|---|---|---|
# | ACT/360 | 実日数 | 360 | マネーマーケット（LIBOR/SOFR 系短期金利、USD/EUR 預金・スワップ変動側） |
# | ACT/365F | 実日数 | 365（固定） | 日本国債（JGB）、GBP マネーマーケット、多くのアジア市場 |
# | 30/360 | 30日/月換算 | 360 | 米国社債・地方債の慣行、円建て社債の一部 |
# | ACT/ACT | 実日数 | その年の実日数 | 米国債（Treasury）、ソブリン債の経過利子（ISMA 版） |
#
# ### ACT/360（実日数 / 360）
#
# 分子は実日数、分母は 360 固定です。
#
# $$ \tau_{\mathrm{ACT/360}} = \frac{d_2 - d_1}{360} $$
#
# 分母が実際の年長（365 または 366）より小さいため、$\tau$ は実際の経過年数より
# **大きく**出ます。1年（365日）保有しても年数は $365/360 \approx 1.0139$ となり、
# 日歩ベースの利息がわずかに嵩みます。起源は電卓もない時代の手計算にあります。
# 1か月を30日、1年を360日とみなすと、月利・日割りが割り切れて暗算しやすい。
# ACT/360 はそのうち分子だけを実日数に戻した折衷で、短期の資金取引に定着しました。
# 現在も USD・EUR のマネーマーケットと変動金利の計算はこの規約が標準です。
#
# ### ACT/365F（実日数 / 365 固定）
#
# 分子は実日数、分母は常に 365 です。末尾の F は Fixed（固定）で、うるう年でも
# 分母を 366 にしない点を強調しています。
#
# $$ \tau_{\mathrm{ACT/365F}} = \frac{d_2 - d_1}{365} $$
#
# 1年ちょうど保有すれば平年は $\tau = 1$、うるう年は $\tau = 366/365 \approx 1.0027$
# となります。日本国債の利含み計算、GBP のマネーマーケット、豪ドル等で使われます。
# ACT/360 に比べ分母が大きいぶん、同じ実日数でも年数は小さく出ます。
#
# ### 30/360（30日/月・360日/年）
#
# 分子・分母をともに「1か月30日・1年360日」という理想暦に載せ替えます。米国社債で
# 使われる Bond Basis（US 版）の定義は次のとおりです。
#
# $$ \tau_{30/360} = \frac{360\,(y_2 - y_1) + 30\,(m_2 - m_1) + (d_2' - d_1')}{360} $$
#
# ここで日 $d_1, d_2$ には月末調整を施します。
#
# - $d_1 = 31$ なら $d_1' = 30$
# - $d_2 = 31$ かつ $d_1' = 30$ なら $d_2' = 30$（それ以外は $d_2' = d_2$）
#
# 30/360 の利点は、半年ごと・四半期ごとのクーポンがどの期間も**同額**になること
# です。実日数だと2〜8月期は181〜184日で期ごとに変動しますが、30/360 では常に
# 180/360 = 0.5 年ぶんになり、クーポン額が一定します。事務処理を単純化したい
# 社債市場の要請から生まれた慣行です。なお US Bond Basis は2月末に特段の調整を
# しません（2月末を30日扱いにするのは 30E/360 ISDA という別変種で、S1-4 で触れます）。
#
# ### ACT/ACT（実日数 / 実日数、ISDA 版）
#
# 分子・分母をともに実日数にします。ISDA 版は期間が年をまたぐとき、各暦年に属する
# 日数を、その年の実長（平年365・うるう年366）で割って足し合わせます。
#
# $$ \tau_{\mathrm{ACT/ACT\,ISDA}}
#   = \frac{\text{開始年に属する日数}}{D_{y_1}}
#   + (y_2 - y_1 - 1)
#   + \frac{\text{終了年に属する日数}}{D_{y_2}} $$
#
# ここで $D_y$ は年 $y$ の実日数（$366$ if うるう年 else $365$）です。うるう年を
# 厳密に按分するため、4規約のうち経過年数の近似としては最も素直で、米国債の利回り
# 換算に使われます。ただし債券の経過利子には、クーポン期間の実日数を分母にする
# ISMA/Bond 版（ACT/ACT ICMA）が使われ、ISDA 版とは値が異なります。本 notebook は
# 4規約の year fraction に集中し、ISMA 版の経過利子は S1-4 で扱います。
#
# ### うるう年と月末ルールの要点
#
# - ACT/360・ACT/365F は分母が固定なので、うるう年でも分母は変わりません。差は分子
#   （実日数）にだけ現れます。
# - ACT/ACT ISDA だけが分母をうるう年で 366 に切り替えます。
# - 30/360 は実日数を一切見ず、$d=31$ の丸めだけを行います。2月の日数（28/29）は
#   結果に影響しません（US Bond Basis の場合）。
#
# ### 休日調整（business day adjustment）
#
# 規約が定めるのは「年数の数え方」ですが、そもそも利払日が休日に当たると資金決済が
# できません。そこで理論上の利払日（unadjusted date）を、営業日カレンダー上の
# 実際の支払日（adjusted date）へずらします。主な規約は次の3つです。
#
# | 調整規約 | 英語 | 動作 |
# |---|---|---|
# | 翌営業日 | Following | 休日なら後ろ（未来）方向の直近営業日へ |
# | 修正翌営業日 | Modified Following | 翌営業日へ。ただし月をまたぐ場合は前営業日へ戻す |
# | 前営業日 | Preceding | 休日なら前（過去）方向の直近営業日へ |
#
# 実務で最も使われるのが修正翌営業日（modified following）です。単純な翌営業日だと、
# 月末の利払日が翌月にはみ出して利息期間の月数がずれることがあります。修正版は
# 「翌営業日にすると月が変わってしまうときだけ前営業日に戻す」ことで、利払日を同じ
# 月内に留め、期間構造を保ちます。休日の集合は市場（国）ごとに異なり、これを
# 休日カレンダー（holiday calendar）と呼びます。日本なら土日に加えて国民の祝日、
# 年末年始などが非営業日です。




# %% [markdown]
# **数値例**：2023-11-01→2024-02-01 を ACT/ACT（ISDA）で測ると、2023年に属する分が $61/365=0.167123$、2024年に属する分が $31/366=0.084699$（2024年はうるう年なので分母366）で、合計 $\tau=0.251823$ です。
# %% [markdown]
# **数値例**：2024-01-31→2024-07-31 を 30/360 で測ると、月末調整で $d_1'=30$、さらに $d_1'=30$ かつ $d_2=31$ より $d_2'=30$ となり、$\tau=\dfrac{30\times(7-1)+(30-30)}{360}=\dfrac{180}{360}=0.5$ です。実日数（181日）ではなく、常に半年ちょうどの $0.5$ 年になります。
# %% [markdown]
# **数値例**：うるう年の1年（2024-01-01→2025-01-01、実日数366日）を ACT/365F で測ると $\tau=366/365=1.002740$ です。分母が365固定なので、平年の1年（$\tau=1$）よりわずかに大きく出ます。
# %% [markdown]
# ## スクラッチ実装
#
# 4規約の year fraction と、営業日調整つきのクーポン日スケジュール生成を自分で
# 書きます。標準ライブラリ `datetime` だけを使い、外部の金融ライブラリには依存
# しません。まず year fraction の補助関数から実装します。
#
# ### 自作関数（year fraction）
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `is_leap(y)` | 西暦年 `y:int` | `bool` | うるう年判定（4で割れ、100で割れず、または400で割れる） |
# | `thirty_360(s, e)` | 開始日・終了日 `date` | `float` | 30/360（US Bond Basis）の年数。月末31日を丸める |
# | `act_act_isda(s, e)` | 開始日・終了日 `date` | `float` | ACT/ACT（ISDA）の年数。暦年ごとに実日数で按分 |
# | `year_fraction(s, e, conv)` | 開始日・終了日 `date`, 規約名 `str` | `float` | 規約名で4種の年数計算を振り分ける入口 |

# %%
import datetime as dt

import numpy as np

np.random.seed(0)


def is_leap(y: int) -> bool:
    """西暦年 y がうるう年なら True。"""
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def thirty_360(s: dt.date, e: dt.date) -> float:
    """30/360 (US Bond Basis)。31日の月末調整つきで年数を返す。"""
    d1, d2 = s.day, e.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    days = 360 * (e.year - s.year) + 30 * (e.month - s.month) + (d2 - d1)
    return days / 360.0


def act_act_isda(s: dt.date, e: dt.date) -> float:
    """ACT/ACT (ISDA)。年をまたぐ日数を各年の実日数で按分する。"""
    if s == e:
        return 0.0
    if s.year == e.year:
        denom = 366.0 if is_leap(s.year) else 365.0
        return (e - s).days / denom
    # 開始年の残り日数を、開始年の実日数で割る
    next_year_start = dt.date(s.year + 1, 1, 1)
    denom_s = 366.0 if is_leap(s.year) else 365.0
    frac = (next_year_start - s).days / denom_s
    # 間に挟まる満年の数をそのまま足す
    frac += e.year - s.year - 1
    # 終了年の頭からの日数を、終了年の実日数で割る
    end_year_start = dt.date(e.year, 1, 1)
    denom_e = 366.0 if is_leap(e.year) else 365.0
    frac += (e - end_year_start).days / denom_e
    return frac


def year_fraction(s: dt.date, e: dt.date, conv: str) -> float:
    """規約名 conv に従って s→e の年数を返す。"""
    if conv == "ACT/360":
        return (e - s).days / 360.0
    if conv == "ACT/365F":
        return (e - s).days / 365.0
    if conv == "30/360":
        return thirty_360(s, e)
    if conv == "ACT/ACT":
        return act_act_isda(s, e)
    raise ValueError(f"未知の規約: {conv!r}")


# 同じ2日付を4規約で計算し、値が食い違うことを確認する。
s = dt.date(2024, 1, 31)  # うるう年をまたぐ半年
e = dt.date(2024, 7, 31)
for conv in ["ACT/360", "ACT/365F", "30/360", "ACT/ACT"]:
    print(f"{conv:9s}: 実日数 {(e - s).days:3d} 日 → 年数 {year_fraction(s, e, conv):.8f}")

# %% [markdown]
# 分子が実日数の3規約（ACT/360・ACT/365F・ACT/ACT）は実日数182日を反映して値が
# 動く一方、30/360 は31日を30日に丸めるためちょうど 0.5 になります。ACT/360 が
# 最も大きく、ACT/365F がそれより小さく出るのは分母の大小（360 < 365）の帰結です。

# %% [markdown]
# ### 自作関数（クーポン日スケジュール）
#
# 満期から利払間隔ぶんだけ遡って利払日を刻み（backward 生成）、各日を営業日調整
# します。カレンダーは土日のみを非営業日とする単純版で実装し、後で QuantLib の
# `WeekendsOnly` と突合します（実在の祝日カレンダーは「実データ適用」で使います）。
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `add_months(d, n)` | 基準日 `date`, 月数 `n:int` | `date` | n か月ずらす。月末日が無い場合はその月の末日に丸める |
# | `is_weekend(d)` | 日付 `date` | `bool` | 土曜・日曜なら True |
# | `adjust(d, conv)` | 日付 `date`, 調整規約 `str` | `date` | Following / Preceding / ModifiedFollowing で営業日へ寄せる |
# | `coupon_schedule(eff, mat, months, conv)` | 発行日・満期・利払間隔月数・調整規約 | `list[date]` | 満期から逆算した利払日列（営業日調整後） |

# %%
import calendar as _cal


def add_months(d: dt.date, n: int) -> dt.date:
    """d を n か月ずらす。日が無ければその月の末日に丸める。"""
    total = d.month - 1 + n
    y = d.year + total // 12
    m = total % 12 + 1
    last = _cal.monthrange(y, m)[1]
    return dt.date(y, m, min(d.day, last))


def is_weekend(d: dt.date) -> bool:
    """土曜(5)・日曜(6)なら True。"""
    return d.weekday() >= 5


def adjust(d: dt.date, conv: str) -> dt.date:
    """営業日調整。conv は 'Following' / 'Preceding' / 'ModifiedFollowing'。"""
    if conv == "Following":
        while is_weekend(d):
            d += dt.timedelta(days=1)
        return d
    if conv == "Preceding":
        while is_weekend(d):
            d -= dt.timedelta(days=1)
        return d
    if conv == "ModifiedFollowing":
        forward = adjust(d, "Following")
        if forward.month != d.month:
            return adjust(d, "Preceding")  # 月をまたぐなら前営業日へ戻す
        return forward
    raise ValueError(f"未知の調整規約: {conv!r}")


def coupon_schedule(eff: dt.date, mat: dt.date, months: int, conv: str) -> list:
    """満期 mat から months か月ずつ遡って利払日を刻み、営業日調整して返す。"""
    unadjusted = [mat]
    k = 1
    while True:
        d = add_months(mat, -months * k)
        if d <= eff:
            break
        unadjusted.append(d)
        k += 1
    unadjusted.append(eff)
    unadjusted = sorted(set(unadjusted))
    return [adjust(d, conv) for d in unadjusted]


# 2026-05-15 発行・2031-05-15 満期・半年利払いを修正翌営業日で生成する。
eff = dt.date(2026, 5, 15)
mat = dt.date(2031, 5, 15)
sched = coupon_schedule(eff, mat, 6, "ModifiedFollowing")
for d in sched:
    tag = "（週末調整あり）" if d.day != 15 else ""
    print(d.isoformat(), ["月", "火", "水", "木", "金", "土", "日"][d.weekday()] + "曜", tag)

# %% [markdown]
# 理論上の利払日はすべて15日ですが、週末に当たった日（例：土曜の15日）は修正翌
# 営業日で翌月曜へ寄っています。月をまたがない限り前方向にずらすのが modified
# following の挙動です。

# %% [markdown]
# ## QuantLib検証
#
# 自作の year fraction を、QuantLib の `DayCounter` および `bondlab.daycount` と
# 突合します。4規約それぞれ、QuantLib の対応する day counter は次のとおりです。
#
# | 規約 | QuantLib DayCounter |
# |---|---|
# | ACT/360 | `Actual360()` |
# | ACT/365F | `Actual365Fixed()` |
# | 30/360 | `Thirty360(Thirty360.BondBasis)` |
# | ACT/ACT | `ActualActual(ActualActual.ISDA)` |

# %%
import QuantLib as ql

from bondlab import daycount as bl_dc

print("QuantLib version:", ql.__version__)

ql_counters = {
    "ACT/360": ql.Actual360(),
    "ACT/365F": ql.Actual365Fixed(),
    "30/360": ql.Thirty360(ql.Thirty360.BondBasis),
    "ACT/ACT": ql.ActualActual(ql.ActualActual.ISDA),
}


def to_ql(d: dt.date) -> "ql.Date":
    return ql.Date(d.day, d.month, d.year)


# 2018〜2032 年の範囲でランダムに100ペアを引き、3実装を突合する。
def random_date(rng) -> dt.date:
    y = int(rng.integers(2018, 2033))
    m = int(rng.integers(1, 13))
    day = int(rng.integers(1, _cal.monthrange(y, m)[1] + 1))
    return dt.date(y, m, day)


rng = np.random.default_rng(0)
max_diff = {c: 0.0 for c in ql_counters}
n_cases = 100
for _ in range(n_cases):
    a = random_date(rng)
    b = random_date(rng)
    if b < a:
        a, b = b, a
    for conv, counter in ql_counters.items():
        v_scratch = year_fraction(a, b, conv)
        v_bondlab = bl_dc.year_fraction(a, b, conv)
        v_ql = counter.yearFraction(to_ql(a), to_ql(b))
        max_diff[conv] = max(
            max_diff[conv], abs(v_scratch - v_ql), abs(v_scratch - v_bondlab)
        )

print(f"{n_cases} ケースでの最大絶対誤差（自作 vs QuantLib / vs bondlab）:")
for conv, d in max_diff.items():
    print(f"  {conv:9s}: {d:.2e}")

assert max(max_diff.values()) < 1e-9, "day count が QuantLib と一致しません"
print("4規約すべてが QuantLib・bondlab と一致しました")

# %% [markdown]
# 4規約とも最大誤差は浮動小数点の丸め（$10^{-16}$ 程度）に収まり、自作実装・
# `bondlab`・QuantLib の3者が一致します。次にスケジュール生成を突合します。QuantLib
# の `Schedule` を土日のみ非営業日（`WeekendsOnly`）で作り、自作 `coupon_schedule`
# と日付列が完全一致するかを見ます。

# %%
ql_conv = {
    "Following": ql.Following,
    "Preceding": ql.Preceding,
    "ModifiedFollowing": ql.ModifiedFollowing,
}
freq_of_months = {3: ql.Quarterly, 6: ql.Semiannual, 12: ql.Annual}


def ql_schedule_dates(eff, mat, months, conv):
    sch = ql.Schedule(
        to_ql(eff),
        to_ql(mat),
        ql.Period(freq_of_months[months]),
        ql.WeekendsOnly(),
        ql_conv[conv],
        ql_conv[conv],
        ql.DateGeneration.Backward,
        False,
    )
    return [dt.date(d.year(), d.month(), d.dayOfMonth()) for d in sch]


schedule_cases = [
    (dt.date(2026, 5, 15), dt.date(2031, 5, 15), 6, "ModifiedFollowing"),
    (dt.date(2024, 2, 29), dt.date(2029, 2, 28), 6, "ModifiedFollowing"),
    (dt.date(2025, 3, 30), dt.date(2030, 3, 30), 12, "Following"),
    (dt.date(2026, 1, 31), dt.date(2028, 1, 31), 3, "Preceding"),
]
for eff_, mat_, mo_, cv_ in schedule_cases:
    mine = coupon_schedule(eff_, mat_, mo_, cv_)
    theirs = ql_schedule_dates(eff_, mat_, mo_, cv_)
    status = "一致" if mine == theirs else "不一致"
    print(f"{eff_}→{mat_} {mo_}か月 {cv_:18s}: {len(mine):2d}日付 {status}")
    assert mine == theirs

print("すべてのスケジュールが QuantLib と一致しました")

# %% [markdown]
# ## 実データ適用
#
# 合成した日本国債（JGB）を想定し、実在の祝日カレンダー（`ql.Japan()`）で利払日を
# 生成します。JGB は ACT/365F・半年利払いが標準です。土日だけでなく国民の祝日でも
# 利払日がずれるため、修正翌営業日の効果が単純版より多く現れます。10年債の想定で、
# 発行 2020-06-20・満期 2030-06-20 の利払日列を作ります。

# %%
import pandas as pd

jgb_eff = ql.Date(20, 6, 2020)
jgb_mat = ql.Date(20, 6, 2030)
jgb_sched = ql.Schedule(
    jgb_eff,
    jgb_mat,
    ql.Period(ql.Semiannual),
    ql.Japan(),
    ql.ModifiedFollowing,
    ql.ModifiedFollowing,
    ql.DateGeneration.Backward,
    False,
)

weekday_ja = ["月", "火", "水", "木", "金", "土", "日"]
rows = []
prev = None
for d in jgb_sched:
    py = dt.date(d.year(), d.month(), d.dayOfMonth())
    # 理論日（毎年6/20・12/20）からのズレを見る
    nominal = dt.date(py.year, py.month, 20)
    shifted = "" if py.day == 20 else f"20日→{py.day}日"
    yf = year_fraction(prev, py, "ACT/365F") if prev else np.nan
    rows.append(
        {
            "利払日": py.isoformat(),
            "曜日": weekday_ja[py.weekday()],
            "調整": shifted,
            "前回からの年数(ACT/365F)": round(yf, 6) if prev else None,
        }
    )
    prev = py

jgb_df = pd.DataFrame(rows)
display(jgb_df)

# %% [markdown]
# 「調整」列が空でない行は、理論上の20日が休日（祝日・土日）に当たり、営業日へ
# ずれた利払日です。年数列を見ると、ACT/365F では期間の実日数がそのまま分子に
# 出るため、調整のあった期は前後の期と年数がわずかに増減します。半年ぶんは
# ほぼ $0.5$ 前後（実日数181〜184日 ÷ 365）で、期ごとに数日ぶん揺れます。
#
# 次に、同じ利払日列に対して休日調整規約だけを差し替え、生成される日付がどれだけ
# 変わるかを比較します。

# %%
adjustments = {
    "Following": ql.Following,
    "ModifiedFollowing": ql.ModifiedFollowing,
    "Preceding": ql.Preceding,
    "Unadjusted": ql.Unadjusted,
}
compare = {}
for name, conv in adjustments.items():
    sch = ql.Schedule(
        jgb_eff,
        jgb_mat,
        ql.Period(ql.Semiannual),
        ql.Japan(),
        conv,
        conv,
        ql.DateGeneration.Backward,
        False,
    )
    compare[name] = [f"{d.year()}-{d.month():02d}-{d.dayOfMonth():02d}" for d in sch]

compare_df = pd.DataFrame(compare)
# 調整規約の間で日付が食い違う行だけを表示する。
differ = compare_df[compare_df.nunique(axis=1) > 1]
print("休日調整規約によって利払日が変わる期のみ抜粋:")
display(differ)

# %% [markdown]
# `Unadjusted` は理論日（20日）のまま、`Following` は後ろ、`Preceding` は前へ寄せ、
# `ModifiedFollowing` は基本 `Following` と同じですが月をまたぐ場合のみ `Preceding`
# と一致します。この差が経過利子・クーポン期間の日数に効いてくるため、規約は
# 商品ごとに正しく合わせる必要があります。

# %% [markdown]
# ## 演習
#
# 1. **うるう年・月末ルールのエッジケース**：以下の10ケースについて、自作
#    `year_fraction` と QuantLib の対応 day counter が一致することを検証する
#    ユニットテスト（`assert` の集合）を書け。少なくとも「うるう年2/29をまたぐ期間」
#    「1/31→2/28」「1/31→3/31（30/360 の月末丸め）」「年跨ぎの ACT/ACT」を含めること。
#    どのケースで4規約の値が最も割れるかを一言で述べよ。
# 2. **修正翌営業日でのスケジュール生成**：発行 2025-04-10・満期 2035-04-10・
#    半年利払いの円建て社債を想定し、`ql.Japan()` カレンダー・修正翌営業日で利払日列を
#    生成せよ。次に、生成した各利払期について ACT/365F の年数を求め、期ごとの年数の
#    最大値と最小値の差（何日ぶんか）を示せ。
#
# 解答例は `solutions/S1/sol_0102.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/01_bond_basics.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | [日数計算規約](../../glossary/01_bond_basics.md#day-count-convention) | day count convention | 2日付間の年数の数え方の取り決め。分子・分母の選び方で分類 |
# | [ACT/360](../../glossary/01_bond_basics.md#actual-360) | actual/360 | 実日数を360で割る。マネーマーケットの標準 |
# | [ACT/365F](../../glossary/01_bond_basics.md#actual-365-fixed) | actual/365 fixed | 実日数を365（固定）で割る。JGB 等で使用 |
# | [30/360](../../glossary/01_bond_basics.md#30-360-us-bond-basis) | 30/360 | 1か月30日・1年360日換算。31日を丸める。米国社債の慣行 |
# | [ACT/ACT](../../glossary/01_bond_basics.md#actual-actual-isda) | actual/actual (ISDA) | 実日数を各暦年の実日数で按分。米国債で使用 |
# | [修正翌営業日](../../glossary/01_bond_basics.md#modified-following) | modified following | 翌営業日へ。ただし月をまたぐ場合は前営業日へ戻す |
# | [利払スケジュール](../../glossary/01_bond_basics.md#coupon-schedule) | coupon schedule | 満期から刻んで営業日調整した利払日の列 |
# | [休日カレンダー](../../glossary/01_bond_basics.md#holiday-calendar) | holiday calendar | 市場ごとの非営業日（土日・祝日）の集合 |
