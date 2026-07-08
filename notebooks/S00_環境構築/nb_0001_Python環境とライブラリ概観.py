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
# # S0-1 Python環境とライブラリ概観
#
# ## 学習目標
#
# - 債券クオンツで使う Python ライブラリを、守備範囲で切り分けて説明できる
# - 自作ライブラリ `bondlab` のレイヤー構成と、その分け方の理由を理解する
# - 「自作関数には説明とテストを添える」開発フローを、実際のコードで一往復する
# - QuantLib の `Date` / `Calendar` / `Schedule` を操作し、営業日計算を実行できる
#
# このシリーズを通じて `bondlab` を育てる。S0-1 はその最初の1本として、
# 環境の確認と開発フローの型作りに充てる。


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# 一見すると地味な環境構築ですが、ここで作る「自作して QuantLib で答え合わせをする」開発フローと、営業日・カレンダー・スケジュールの正しい取り扱いは、債券デスクの収益とリスク管理の土台になります。銀行の金利デスククオンツはカーブ構築やデリバティブ評価のモデルを組み、リスク管理クオンツは DV01 や VaR を計算しますが、そのどれもが「日付を正しく刻めること」を前提にしています。利払日・受渡日・経過利子の計算は営業日規約（Modified Following など）と各国の祝日カレンダーに依存し、ここが一日ずれるだけで価格・キャッシュフロー・レポの資金決済が狂います。マーケットメイクで両面クォートを出す場面でも、受渡日の想定違いはそのまま損益の取り違えになります。
#
# `bondlab` を層構造（rates → bond → curve → analytics → …）で積む設計は、変更の影響範囲を局所化するための実務的な作法です。フロントのプライシングモデルはモデルバリデーション部門が独立に検証・承認してから稼働に乗る（米 FRB の SR 11-7 が代表例）ため、「自作関数に仕様表とテストを添える」習慣は、そのまま検証可能なコードを書く訓練になります。本文で `assert` による往復チェックや規約変換の不変性を確認しているのは、後からライブラリを直したときに壊れていないことを機械的に保証する回帰テストの下地です。
#
# 具体的には、東京・ニューヨーク・ロンドンで営業日数が食い違うことを実際に数える演習が、クロスマーケットの取引で効いてきます。日米金利スプレッドや先物ベーシスのように複数市場をまたぐ戦略では、片方が休場の日をどう揃えるかで指標が変わります。この土台が無いと、そもそも国債一本の価格すら正しく出せず、値付けも相対価値分析もリスク計測も始まりません。
# %% [markdown]
# ## 理論：ライブラリの守備範囲と bondlab の設計
#
# ### 使用ライブラリの分担
#
# 債券クオンツのコードは、少数の土台ライブラリの上に自作ロジックを積む。
# それぞれの守備範囲は重ならないように選ぶ。
#
# | ライブラリ | 主な役割 | 本シリーズでの使いどころ |
# |---|---|---|
# | numpy | 数値配列・線形代数 | すべてのスクラッチ実装の基盤 |
# | scipy | 最適化・求根・補間・統計 | YTM 求解、カーブフィット、キャリブレーション |
# | pandas | 時系列・表データ | 市場データの整形と可視化 |
# | matplotlib | 作図 | カーブ・残差・PnL の可視化 |
# | QuantLib | 金融商品の評価基盤（C++実装） | 自作実装の検証用ベンチマーク |
# | statsmodels | 回帰・時系列分析 | 平均回帰の推定（S9） |
# | cvxpy | 凸最適化 | ポートフォリオ複製・最適化（S10） |
#
# 方針は「まず numpy/scipy で自作し、QuantLib で答え合わせをする」。面接で
# 問われるのはモデルを導出・実装できるかであって、ライブラリを呼べるかでは
# ないため、QuantLib は検証役に徹する。
#
# ### bondlab のレイヤー構成
#
# 依存の向きが下から上へ一方向になるよう層を切る。上位層は下位層だけに依存し、
# 逆流させない。
#
# ```
# data    市場データ取得・キャッシュ          （S0）
# rates   複利規約・割引の基本                （S1）
# bond    債券価格・YTM・経過利子             （S1）
# curve   イールドカーブ構築                  （S2）
# analytics  デュレーション・DV01・KRD・PCA   （S3）
# risk    VaR / ES とバックテスト検定         （S3）
# sim     確率過程シミュレーション            （S4）
# models  短期金利モデル                      （S5）
# pricing デリバティブ評価                    （S6）
# credit  ハザード・CDS・Merton・CVA          （S7）
# bt      戦略バックテスト基盤                （S9）
# ```
#
# この分け方には二つの狙いがある。第一に、各シリーズが対応する層だけを触るので
# 変更の影響範囲が読みやすい。第二に、最終的に層ごとのテストを持つ小さな
# ライブラリが手元に残り、それ自体がポートフォリオになる。

# %% [markdown]
# ## スクラッチ実装：最初の自作関数とテスト
#
# `bondlab.rates` に複利規約の変換関数を実装してある（詳細は S1-1 で扱う）。
# ここでは開発フローの型として、「関数の仕様を表で示し、使い、テストで守る」
# 流れを一往復する。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `discount_factor(rate, t, convention)` | レート, 年数, 複利規約 | 割引係数 | レート→割引係数 |
# | `rate_from_discount(df, t, convention)` | 割引係数, 年数, 複利規約 | レート | 割引係数→レート（逆関数） |
# | `to_continuous(rate, convention)` | レート, 複利規約 | 連続複利レート | 離散→連続（t 非依存） |


# %% [markdown]
# **数値例**：半年複利 5%・3年の割引係数は $DF = \left(1 + \dfrac{0.05}{2}\right)^{-2 \times 3} = 0.86229687$ です。
#
# 等価な連続複利レートは $r_c = 2 \ln\left(1 + \dfrac{0.05}{2}\right) = 0.049385$（約 4.9385%）となり、これを使うと $e^{-0.049385 \times 3} = 0.86229687$ で、同じ割引係数に一致します。
# %%
import numpy as np

import bondlab
from bondlab import rates

print("bondlab version:", bondlab.__version__)

# 半年複利 5% を割引係数へ落とし、逆算で元に戻るかを確認する。
r_semi = 0.05
t = 3.0
df = rates.discount_factor(r_semi, t, "semiannual")
r_back = rates.rate_from_discount(df, t, "semiannual")
print(f"DF(半年複利5%, 3年)      = {df:.8f}")
print(f"逆算した半年複利レート   = {r_back:.8f}")

# 半年複利 5% と等価な連続複利レート。
r_cont = rates.to_continuous(r_semi, "semiannual")
print(f"等価な連続複利レート     = {r_cont:.8f}")

# %% [markdown]
# 逆算が元の値に戻り、規約変換も割引係数を保つことを、`assert` で明示的に
# 守る。notebook 内のこうしたチェックが、後でライブラリを変更したときの
# 回帰テストの下地になる。

# %%
# 往復誤差はゼロに近いはず。
assert abs(r_back - r_semi) < 1e-12

# 連続複利へ変換しても、同じ t での割引係数は一致する。
df_cont = rates.discount_factor(r_cont, t, "continuous")
assert abs(df_cont - df) < 1e-12

# ベクトル化：複数レート・複数年数をまとめて処理できる。
rs = np.array([0.01, 0.02, 0.05])
ts = np.array([1.0, 5.0, 10.0])
dfs = rates.discount_factor(rs, ts, "annual")
print("ベクトル入力の割引係数:", np.round(dfs, 6))
print("チェックを通過しました")

# %% [markdown]
# ## QuantLib 検証：日付と営業日計算
#
# QuantLib は債券評価の基盤を C++ で実装したライブラリで、本シリーズでは
# 自作コードの答え合わせに使う。まずは日付・カレンダー・スケジュールという
# 最も基本的な部品を操作する。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

# 日付の生成と加算。日数・週・月・年の単位で進められる。
d0 = ql.Date(15, 5, 2026)  # 2026-05-15（日, 月, 年 の順に注意）
print("基準日:", d0)
print("3か月後:", d0 + ql.Period(3, ql.Months))
print("1年後  :", d0 + ql.Period(1, ql.Years))

# %% [markdown]
# ### 各国のカレンダーと営業日
#
# 国ごとに祝日が異なるため、営業日数も変わる。東京・ニューヨーク・ロンドンの
# 2026年の営業日数を数えて比較する。

# %%
calendars = {
    "東京 (Japan)": ql.Japan(),
    "ニューヨーク (US settlement)": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    "ロンドン (UK)": ql.UnitedKingdom(),
}

start = ql.Date(1, 1, 2026)
end = ql.Date(31, 12, 2026)

print(f"{'市場':32s} {'2026年の営業日数':>16s}")
for label, cal in calendars.items():
    business_days = sum(
        1
        for n in range(start.serialNumber(), end.serialNumber() + 1)
        if cal.isBusinessDay(ql.Date(n))
    )
    print(f"{label:32s} {business_days:>16d}")

# %% [markdown]
# ### スケジュール生成
#
# 半年利払いの債券を想定し、利払日の列を生成する。カーブ構築や債券評価では、
# この「日付の列」を正しく作れることが出発点になる。

# %%
schedule = ql.Schedule(
    ql.Date(15, 5, 2026),                 # 開始日
    ql.Date(15, 5, 2031),                 # 満期
    ql.Period(ql.Semiannual),             # 利払頻度
    ql.Japan(),                           # カレンダー
    ql.ModifiedFollowing,                 # 休日調整
    ql.ModifiedFollowing,
    ql.DateGeneration.Backward,           # 満期から逆算して日付を刻む
    False,
)
print("生成された利払日:")
for d in schedule:
    print("  ", d, "曜日番号", d.weekday())

# %% [markdown]
# ## 演習
#
# 1. 東京・ニューヨーク・ロンドンについて、2026年の**月ごと**の営業日数を
#    数え、`pandas.DataFrame`（行=月, 列=市場）にまとめて棒グラフにせよ。
#    どの月に差が大きいか、その理由（各国の祝日）を一言で説明せよ。
# 2. `bondlab.rates.convert_rate` を使い、年1回複利 4%・年数 1〜30 年を
#    半年複利・連続複利へ変換し、年数とともに規約間の差がどう変わるかを
#    プロットせよ。差が年数に対して増えるか減るかを述べよ。
#
# 解答例は `solutions/S0/` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/00_tooling.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | 仮想環境 | virtual environment | プロジェクト単位で依存を隔離する Python の実行環境 |
# | パッケージ | package | 再利用可能なモジュールの集合。ここでは `bondlab` |
# | ユニットテスト | unit test | 関数単位で期待挙動を自動検証する仕組み |
# | 割引係数 | discount factor | 将来1単位の現在価値。レートと年数から定まる |
# | 複利規約 | compounding convention | 利息を年に何回複利計算するかの取り決め |
