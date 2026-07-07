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
# # S11-3 演習 解答例
#
# 日次カーブモニタリングダッシュボードの演習2問の解答例です。本文
# `notebooks/S11_capstone/nb_1103_dashboard.py` と同じ計算関数を再構築し、
# (1) ショック規模を変えた誤報率×検出率のトレードオフ、(2) 今朝の画面からの
# RV 発注メモ、を示します。

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bondlab.curve import bootstrap_par, fit_nss, nss, DiscountCurve
from bondlab.analytics import bump_curve

np.random.seed(0)

BP = 1e4
BENCH_TENORS = (0.5, 2.0, 5.0, 10.0, 30.0)
FLY_LEGS = (2.0, 5.0, 10.0)
KEY_TENORS = (2.0, 5.0, 10.0, 30.0)
HOLD_HORIZON = 0.25


# %% [markdown]
# ## 本文と共通の計算関数（再掲）
#
# 本文の `## スクラッチ実装` と同一ロジックです。演習に必要な最小限を再構築
# します。

# %%
def load_par_panel(path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"])
    return raw.pivot(index="date", columns="tenor", values="par_yield").sort_index().sort_index(axis=1)


def _curve_from_row(tenors, par_rates, freq=1) -> DiscountCurve:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bootstrap_par(np.asarray(tenors, float), np.asarray(par_rates, float), frequency=freq)


def butterfly_series(panel, legs=FLY_LEGS):
    w_lo, belly, w_hi = legs
    return ((2.0 * panel[belly] - panel[w_lo] - panel[w_hi]) * BP).rename("butterfly_bp")


def richcheap_residuals(panel, bench=BENCH_TENORS, lam_grid=None):
    if lam_grid is None:
        lam_grid = np.linspace(0.8, 8.0, 8)
    tenors = np.asarray(panel.columns, dtype=float)
    bidx = [list(tenors).index(b) for b in bench]
    rows = {}
    for d, row in panel.iterrows():
        y = row.to_numpy(float)
        fit = fit_nss(tenors[bidx], y[bidx], lam_grid=lam_grid)
        rows[d] = (y - nss(tenors, **fit)) * BP
    out = pd.DataFrame(rows).T
    out.columns = tenors
    return out


def carry_roll(curve, tenors, horizon=HOLD_HORIZON):
    tenors = np.asarray(tenors, dtype=float)
    out = {}
    for t in tenors:
        if t <= horizon:
            continue
        out[t] = (curve.zero_rate(float(t)) - curve.forward_rate(float(horizon), float(t))) * BP
    return pd.Series(out, name="carry_roll_bp")


def price_bond_on_curve(curve, coupon, maturity, freq=1):
    n = int(round(maturity * freq))
    times = np.arange(1, n + 1) / freq
    cfs = np.full(n, coupon / freq)
    cfs[-1] += 1.0
    return float(np.sum(cfs * curve.discount(times)))


def key_rate_durations(curve, coupon, maturity, keys=KEY_TENORS, freq=1, bump=1e-4, width=None):
    keys = np.asarray(keys, dtype=float)
    if width is None:
        width = float(np.median(np.diff(keys)))
    p0 = price_bond_on_curve(curve, coupon, maturity, freq)
    krd = {}
    for k in keys:
        p_up = price_bond_on_curve(bump_curve(curve, k, bump, width=width), coupon, maturity, freq)
        p_dn = price_bond_on_curve(bump_curve(curve, k, -bump, width=width), coupon, maturity, freq)
        krd[k] = -(p_up - p_dn) / (2.0 * bump * p0)
    return pd.Series(krd, name="KRD")


def inject_shocks(z_series, n_shocks=8, size=2.5, seed=0):
    rng = np.random.default_rng(seed)
    z = z_series.to_numpy(float).copy()
    idx = rng.choice(len(z), size=n_shocks, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_shocks)
    z[idx] = z[idx] + size * signs
    mask = np.zeros(len(z), dtype=bool)
    mask[idx] = True
    return pd.Series(z, index=z_series.index), mask


def detection_tradeoff(z_series, mask, thresholds):
    z = z_series.to_numpy(float)
    rows = []
    for thr in thresholds:
        fired = np.abs(z) > thr
        rows.append({"threshold": float(thr),
                     "detection": float(fired[mask].mean()) if mask.any() else np.nan,
                     "false_alarm": float(fired[~mask].mean()) if (~mask).any() else np.nan})
    return pd.DataFrame(rows)


DATA_PATH = Path("data") / "samples" / "synthetic_ust_par_panel.csv"
panel = load_par_panel(DATA_PATH)
TENORS = np.asarray(panel.columns, dtype=float)

# %% [markdown]
# ## 演習1：ショック規模を変えた誤報率×検出率のトレードオフ
#
# バタフライ $z$ の平常系列に、$1.5\sigma$・$2.5\sigma$・$4.0\sigma$ の3規模で
# ショックを注入し、それぞれの検出率×誤報率曲線を描きます。さらに「誤報率
# 5% 以下」制約のもとでの最大検出率を表にまとめます。

# %%
fly = butterfly_series(panel, FLY_LEGS)
fly_z = (fly - fly.mean()) / fly.std(ddof=1)

thr_grid = np.linspace(0.5, 4.0, 22)
sizes = [1.5, 2.5, 4.0]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
summary = []
for size in sizes:
    z_inj, mask = inject_shocks(fly_z, n_shocks=8, size=size, seed=0)
    tr = detection_tradeoff(z_inj, mask, thr_grid)
    ax.plot(tr["false_alarm"], tr["detection"], marker="o", ms=3, label=f"ショック {size:g}σ")
    # 誤報率5%以下の制約下での最大検出率。
    ok = tr[tr["false_alarm"] <= 0.05]
    best = ok.loc[ok["detection"].idxmax()] if not ok.empty else None
    summary.append({
        "ショック規模": f"{size:g}σ",
        "誤報率5%以下での最大検出率": round(float(best["detection"]), 3) if best is not None else np.nan,
        "そのときの閾値|z|": round(float(best["threshold"]), 2) if best is not None else np.nan,
    })

ax.set_xlabel("誤報率（非注入日の発火割合）")
ax.set_ylabel("検出率（注入日の捕捉割合）")
ax.set_title("ショック規模別の検出率×誤報率トレードオフ")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

summary_df = pd.DataFrame(summary)
print("誤報率5%以下の制約下での最大検出率:")
print(summary_df.to_string(index=False))

# %% [markdown]
# **考察**：閾値を上げて誤報を嫌うほど、小さなショックほど先に見逃されます。
# $4.0\sigma$ のような大きな逸脱は高い閾値でも捕捉できますが、$1.5\sigma$ の
# 弱い逸脱は誤報率 5% 制約下では検出率が大きく落ちます。閾値は「見逃したくない
# 最小のショック規模」と「許容できる誤報率」の綱引きで決めるべきで、単一の
# 固定値（$z>2$ 等）では規模の異なる逸脱を同じ感度で扱えません。

# %%
# 検証：誤報を嫌う（閾値が高い）ほど、小さいショックの検出率は大きいショック以下になる。
det_by_size = {}
for size in sizes:
    z_inj, mask = inject_shocks(fly_z, n_shocks=8, size=size, seed=0)
    tr = detection_tradeoff(z_inj, mask, thr_grid)
    ok = tr[tr["false_alarm"] <= 0.05]
    det_by_size[size] = float(ok["detection"].max()) if not ok.empty else 0.0
assert det_by_size[1.5] <= det_by_size[4.0] + 1e-9, "小ショックが大ショックより検出しやすいのは不自然"
print("\n検証通過：誤報率5%制約下で、小ショックの検出率 <= 大ショックの検出率")

# %% [markdown]
# ## 演習2：今朝の画面からの RV 発注メモ
#
# 最新営業日のダッシュボード数値を計算し、KRD・キャリー／ロール・rich/cheap・
# バタフライ $z$ を並べて、1つの注文を根拠つきで選びます。

# %%
asof = panel.index[-1]
asof_row = panel.loc[asof]
curve = _curve_from_row(TENORS, asof_row.to_numpy(float), 1)

rc_hist = richcheap_residuals(panel, BENCH_TENORS)
rc_z_today = ((rc_hist - rc_hist.mean()) / rc_hist.std(ddof=1)).loc[asof]
cr = carry_roll(curve, TENORS)
coupon = float(asof_row[10.0])
krd = key_rate_durations(curve, coupon, 10.0, KEY_TENORS, 1)
fly_z_today = float(fly_z.loc[asof])

print(f"as-of: {asof.date()}")
print(f"バタフライ 2-5-10 z（当日）: {fly_z_today:+.2f}（正=ベリー割安）")
print("\nrich/cheap z（正=cheap, 負=rich）:")
print(rc_z_today.round(2).to_dict())
print("\nキャリー＋ロール（3か月, bp）:")
print(cr.round(2).to_dict())
print("\nKRD（10年債, 年）:")
print(krd.round(3).to_dict())

# 発注ロジック：最も cheap なテナー（rich/cheap z が最大）を、キャリーの符号で裏取り。
cheapest = rc_z_today.astype(float).idxmax()
richest = rc_z_today.astype(float).idxmin()
print(f"\n最も cheap（z最大）なテナー: {cheapest:g}Y (z={rc_z_today[cheapest]:+.2f})")
print(f"最も rich（z最小）なテナー: {richest:g}Y (z={rc_z_today[richest]:+.2f})")

# %% [markdown]
# ### 発注メモ（解答例）
#
# 合成データは滑らかで rich/cheap 残差は 0.1bp オーダー（$z$ は正規化された
# 相対値）ですが、画面の読み方の型として次のように書きます。
#
# > **注文**：バタフライ 2-5-10 のベリー（5年）に対する方向取引を1本。
# > 当日のバタフライ $z$ の符号に従い、$z>0$（ベリー割安）ならベリー買い×
# > ウイング売りの DV01 中立バタフライ、$z<0$ なら反対を組む。
# >
# > **根拠**：(1) バタフライパネルの $z$ が当日この水準にあること、(2) rich/cheap
# > パネルで 5年が cheap/rich どちら寄りかが $z$ の符号と整合するか、(3) キャリー
# > ／ロールパネルで 5年保有のキャリーが逆風でないこと、を確認したうえで発注。
# > KRD パネルで既存持ち高の 5年感応度が過大でないことも点検する。
# >
# > **保有期間**：3〜4週間（バタフライ $z$ の平均回帰を想定, S9-1 の半減期観点）。
# >
# > **手仕舞い**：バタフライ $z$ が $0$ 近傍（$|z|<0.5$）へ回帰したら利確。
# > 逆に $z$ が同符号にさらに 1σ 拡大したら、想定シナリオ崩れとして損切り。
#
# メモの要点は「どのパネルの、どの数値を根拠にしたか」を明示し、入りと出口の
# 条件を数値で書くことです。画面はそのための共通言語になります。

# %%
# 解答の一貫性チェック：発注ロジックが決定論的に1つのテナーを選ぶこと。
assert cheapest in list(TENORS) and richest in list(TENORS)
assert np.isfinite(fly_z_today)
print("解答例の計算が決定論的に完了しました")
