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
# # S1-2 演習 解答例

# %%
import datetime as dt
import calendar as _cal

import numpy as np
import pandas as pd
import QuantLib as ql

from bondlab import daycount as bl_dc

np.random.seed(0)


def to_ql(d: dt.date) -> "ql.Date":
    return ql.Date(d.day, d.month, d.year)


ql_counters = {
    "ACT/360": ql.Actual360(),
    "ACT/365F": ql.Actual365Fixed(),
    "30/360": ql.Thirty360(ql.Thirty360.BondBasis),
    "ACT/ACT": ql.ActualActual(ql.ActualActual.ISDA),
}

# %% [markdown]
# ## 演習1：うるう年・月末ルールのエッジケース10件
#
# 各ケースで、`bondlab.daycount.year_fraction`（=本文のスクラッチ実装と一致）と
# QuantLib の対応 day counter が全規約で一致することを `assert` で検証する。

# %%
edge_cases = [
    # (開始, 終了, 説明)
    (dt.date(2024, 2, 29), dt.date(2025, 2, 28), "うるう年2/29→翌年2/28"),
    (dt.date(2024, 1, 31), dt.date(2024, 2, 29), "1/31→うるう年2/29"),
    (dt.date(2023, 1, 31), dt.date(2023, 2, 28), "1/31→平年2/28"),
    (dt.date(2025, 1, 31), dt.date(2025, 3, 31), "1/31→3/31（30/360の月末丸め）"),
    (dt.date(2025, 3, 31), dt.date(2025, 4, 30), "3/31→4/30"),
    (dt.date(2024, 12, 31), dt.date(2025, 1, 1), "年跨ぎ1日（ACT/ACT）"),
    (dt.date(2023, 6, 15), dt.date(2027, 6, 15), "うるう年を含む複数年（4年）"),
    (dt.date(2024, 2, 29), dt.date(2028, 2, 29), "うるう年→うるう年（4年）"),
    (dt.date(2025, 8, 31), dt.date(2026, 2, 28), "8/31→2/28（半年・月末）"),
    (dt.date(2020, 12, 31), dt.date(2021, 12, 31), "年末→翌年末（丸1年跨ぎ）"),
]

for s, e, desc in edge_cases:
    for conv, counter in ql_counters.items():
        v_bl = bl_dc.year_fraction(s, e, conv)
        v_ql = counter.yearFraction(to_ql(s), to_ql(e))
        assert abs(v_bl - v_ql) < 1e-9, (desc, conv, v_bl, v_ql)

print(f"{len(edge_cases)} ケース × 4規約すべてが QuantLib と一致しました")

# %% [markdown]
# 各ケースを4規約で並べ、値がどう割れるかを表で確認する。

# %%
rows = []
for s, e, desc in edge_cases:
    row = {"説明": desc, "実日数": (e - s).days}
    for conv in ql_counters:
        row[conv] = round(bl_dc.year_fraction(s, e, conv), 6)
    row["最大−最小"] = round(
        max(bl_dc.year_fraction(s, e, c) for c in ql_counters)
        - min(bl_dc.year_fraction(s, e, c) for c in ql_counters),
        6,
    )
    rows.append(row)

edge_df = pd.DataFrame(rows)
print(edge_df.to_string(index=False))

worst = edge_df.loc[edge_df["最大−最小"].idxmax()]
print(f"\n規約間の差が最大なのは「{worst['説明']}」（差 {worst['最大−最小']}）")

# %% [markdown]
# 規約間の差は、期間が長いほど分母の違い（360 vs 365 vs 実日数）が累積して開く。
# 10ケース中では丸1年以上の期間で差が大きく、ACT/360 が最も年数を大きく、
# ACT/365F が最も小さく見積もる。短期（年跨ぎ1日など）では差は小さい。

# %% [markdown]
# ## 演習2：修正翌営業日でのスケジュール生成と期別年数
#
# 発行 2025-04-10・満期 2035-04-10・半年利払いの円建て社債を、`ql.Japan()`
# カレンダー・修正翌営業日で生成し、各利払期の ACT/365F 年数を求める。

# %%
eff = ql.Date(10, 4, 2025)
mat = ql.Date(10, 4, 2035)
sched = ql.Schedule(
    eff,
    mat,
    ql.Period(ql.Semiannual),
    ql.Japan(),
    ql.ModifiedFollowing,
    ql.ModifiedFollowing,
    ql.DateGeneration.Backward,
    False,
)

dates = [dt.date(d.year(), d.month(), d.dayOfMonth()) for d in sched]
weekday_ja = ["月", "火", "水", "木", "金", "土", "日"]

rows = []
for prev, cur in zip(dates[:-1], dates[1:]):
    yf = bl_dc.year_fraction(prev, cur, "ACT/365F")
    rows.append(
        {
            "期首": prev.isoformat(),
            "期末": cur.isoformat(),
            "期末曜日": weekday_ja[cur.weekday()],
            "実日数": (cur - prev).days,
            "年数(ACT/365F)": round(yf, 6),
        }
    )

sched_df = pd.DataFrame(rows)
print(sched_df.to_string(index=False))

yfs = sched_df["年数(ACT/365F)"]
days = sched_df["実日数"]
span_days = int(days.max() - days.min())
print(
    f"\n期別年数の最大 {yfs.max():.6f} / 最小 {yfs.min():.6f} / "
    f"差 {yfs.max() - yfs.min():.6f}（実日数で {span_days} 日ぶん）"
)

# %% [markdown]
# 半年利払いでも、期の実日数は181〜184日で一定しない（暦の非対称・うるう年・
# 休日調整による）。ACT/365F では実日数がそのまま分子に出るため、期ごとの年数は
# 数日ぶん揺れる。この揺れがクーポンの日割り・経過利子に効く。年数を毎期きっちり
# 0.5 にしたい設計では 30/360 が選ばれる。
