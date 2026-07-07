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
# # S2-5 演習 解答例
#
# 実データ日次カーブ構築パイプライン（`nb_0205_pipeline`）の演習解答。
# 本文のパイプライン関数を最小構成で再掲し、その上で2問に答える。

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from bondlab.curve import bootstrap_par

np.random.seed(0)

PAR_BAND = (-0.01, 0.20)


def load_par_panel(path) -> pd.DataFrame:
    """縦持ちパネルを (行=日付, 列=テナー) へ整形する。"""
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"])
    return raw.pivot(index="date", columns="tenor", values="par_yield").sort_index().sort_index(axis=1)


def check_curve_quality(curve, tenors):
    """DF 単調減少・フォワード非負を検査し警告リストを返す。"""
    t = np.asarray(tenors, dtype=float)
    w = []
    dfs = curve.discount(t)
    if np.any(np.diff(dfs) >= 0) or np.any(dfs <= 0) or np.any(dfs > 1.0 + 1e-12):
        w.append("df_not_monotone")
    fwd = (np.log(dfs[:-1]) - np.log(dfs[1:])) / (t[1:] - t[:-1])
    if np.any(fwd < 0):
        w.append("negative_forward")
    return w


def par_reproduction_error(curve, tenors, par_rates, frequency=1):
    """パー債価格が 1 に戻る最大絶対誤差。"""
    t = np.asarray(tenors, dtype=float)
    c = np.asarray(par_rates, dtype=float)
    dfs = curve.discount(t)
    cpn = c / frequency
    prices = np.array([cpn[i] * dfs[: i + 1].sum() + dfs[i] for i in range(len(t))])
    return float(np.max(np.abs(prices - 1.0)))


def run_pipeline(panel, frequency=1, band=PAR_BAND):
    """全日を無停止で処理し (ゼロレートDF, ログDF) を返す（本文の縮約版）。"""
    tenors = np.asarray(panel.columns, dtype=float)
    zero_rows, logs = {}, []
    last_good, prev = None, None
    lo, hi = band
    for d in panel.index:
        vals = panel.loc[d].to_numpy(dtype=float)
        fatal = np.any(np.isnan(vals)) or np.any((vals < lo) | (vals > hi))
        status, reason, curve = "ok", None, None
        if fatal:
            status, reason = "fallback", "入力検証で致命的警告"
        else:
            curve = bootstrap_par(list(tenors), vals, frequency=frequency)
            if check_curve_quality(curve, tenors) or par_reproduction_error(curve, tenors, vals, frequency) > 1e-8:
                status, reason = "fallback", "カーブ品質検査で不合格"
        if status == "fallback":
            zero_rows[d] = last_good.zero_rate(tenors) if last_good is not None else np.full(len(tenors), np.nan)
        else:
            last_good = curve
            zero_rows[d] = curve.zero_rate(tenors)
        logs.append({"date": d, "status": status, "fallback_reason": reason})
        prev = d
    zero_df = pd.DataFrame(zero_rows).T
    zero_df.columns = tenors
    zero_df.index.name = "date"
    return zero_df, pd.DataFrame(logs).set_index("date")


DATA_PATH = Path("data") / "samples" / "synthetic_ust_par_panel.csv"
panel = load_par_panel(DATA_PATH)
TENORS = np.asarray(panel.columns, dtype=float)

# %% [markdown]
# ## 演習1：日単位の異常度で大変動日をランク付け
#
# テナー別 $z$ スコアを二乗和平方根 $\sqrt{\sum_i z_i^2}$ で1日1値に集約する。
# テナー単位の警告が「どの点が飛んだか」を見るのに対し、日単位の集約は
# 「全体としてどの日が荒れたか」を見る。1テナーだけ大きく飛んだ日より、
# 多数のテナーが同時に中程度に動いた日の方が、日単位では上位に来やすい。

# %%
zero_df, log_df = run_pipeline(panel, frequency=1)

dz = zero_df.diff()                                   # 前日比（初日 NaN）
sigma = dz.std(axis=0)
zscore = dz.divide(sigma.replace(0, np.nan), axis=1)  # テナー別 z スコア

day_anomaly = np.sqrt((zscore ** 2).sum(axis=1))      # 日単位の異常度
day_anomaly = day_anomaly.dropna()

top5 = day_anomaly.sort_values(ascending=False).head(5)
print("日単位の異常度（√Σz²）上位5営業日:")
for d, v in top5.items():
    n_tenor_flag = int((zscore.loc[d].abs() > 2.0).sum())
    print(f"  {d.date()}  異常度 = {v:6.3f}  （テナー単位2σ超 {n_tenor_flag} 本）")

# テナー単位の警告日と日単位の上位日は一致するとは限らない。
tenor_flag_days = set(zscore.index[(zscore.abs() > 2.0).any(axis=1)])
day_top_days = set(top5.index)
print("\nテナー単位2σ超が出た日数 :", len(tenor_flag_days))
print("日単位 上位5 と重なる日数 :", len(day_top_days & tenor_flag_days))
print("→ 日単位集約は、単テナーの突出より複数テナー同時変動を上位に押し上げる。")

# %% [markdown]
# ## 演習2：外れ値注入とフォールバックの発火確認
#
# 中ほどの1営業日・1テナー（30年）に定義域外の $50\%$ を差し込む。その日は
# 入力検証で致命的警告となり `fallback` に落ち、ゼロレートは前日を持ち越す。
# 翌日は入力が正常なので通常構築へ復帰する。

# %%
inject_date = panel.index[30]
inject_tenor = 30.0
prev_date = panel.index[29]
next_date = panel.index[31]

panel_bad = panel.copy()
panel_bad.loc[inject_date, inject_tenor] = 0.50   # 50% = 定義域外の外れ値

zero_clean, log_clean = run_pipeline(panel, frequency=1)
zero_bad, log_bad = run_pipeline(panel_bad, frequency=1)

print(f"注入日 : {inject_date.date()}  テナー {inject_tenor:g}Y に 50% を注入")
print(f"注入日のステータス      : {log_bad.loc[inject_date, 'status']}"
      f"（理由: {log_bad.loc[inject_date, 'fallback_reason']}）")
print(f"翌営業日のステータス    : {log_bad.loc[next_date, 'status']}")

# 注入日のゼロレートが前日の値を持ち越していることを確認する。
carried = np.allclose(zero_bad.loc[inject_date].to_numpy(),
                      zero_bad.loc[prev_date].to_numpy())
print(f"\n注入日ゼロレート == 前日ゼロレート : {carried}（フォールバックで持ち越し）")

# 翌日以降は正常な構築へ復帰し、汚染前と一致することを確認する。
recovered = np.allclose(zero_bad.loc[next_date].to_numpy(),
                        zero_clean.loc[next_date].to_numpy())
print(f"翌日ゼロレート == 汚染前の翌日      : {recovered}（正常構築に復帰）")

assert log_bad.loc[inject_date, "status"] == "fallback"
assert log_bad.loc[next_date, "status"] == "ok"
assert carried and recovered

# 参考：注入日を bp で前日と比較（差はゼロ、= 持ち越しの証拠）。
diff_bp = (zero_bad.loc[inject_date] - zero_bad.loc[prev_date]) * 1e4
print(f"\n注入日 − 前日 の最大差: {diff_bp.abs().max():.3e} bp（≈0 なら完全持ち越し）")
print("フォールバックが1日で局所化し、パイプラインは無停止で継続した。")
