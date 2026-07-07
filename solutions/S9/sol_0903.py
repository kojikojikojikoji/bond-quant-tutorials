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
# # S9-3 演習 解答例
#
# キャリーとロールダウンの演習2問の解答例です。本文 `nb_0903_carry_roll` と同じ
# `bondlab.curve` を使い、キャリー＋ロールの厳密分解を再実装します。

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bondlab.curve import bootstrap_par

np.random.seed(0)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def make_cashflows(coupon, maturity, freq=1):
    """額面1・満期一括償還のクーポン債 CF を返す（本文と同じ）。"""
    n = int(round(maturity * freq))
    times = np.array([(k + 1) / freq for k in range(n)], dtype=float)
    amounts = np.full(n, coupon / freq, dtype=float)
    amounts[-1] += 1.0
    return times, amounts


def carry_roll(curve, times, amounts, horizon, repo=None):
    """本文と同じ厳密分解（total = V_h - F）。"""
    times = np.asarray(times, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    df = curve.discount(times)
    z = curve.zero_rate(times)
    p0 = float(np.sum(amounts * df))
    if repo is None:
        repo = curve.zero_rate(horizon)
    fwd_price = p0 * np.exp(repo * horizon)
    v_noroll = float(np.sum(amounts * df * np.exp(z * horizon)))
    v_roll = float(np.sum(amounts * curve.discount(times - horizon)))
    dur = float(np.sum(times * amounts * df) / p0)
    return dict(price0=p0, total=v_roll - fwd_price, carry=v_noroll - fwd_price,
                roll=v_roll - v_noroll, duration=dur,
                total_bp=(v_roll - fwd_price) / p0 / horizon * 1e4)


def build_curve(panel, date):
    """指定営業日のパー利回りを年刻みグリッドに補間してブートストラップ。"""
    s = panel[panel["date"] == date].sort_values("tenor")
    g = np.arange(1.0, 31.0)
    py = np.interp(g, s["tenor"].values, s["par_yield"].values)
    return bootstrap_par(g, py, frequency=1)


panel = pd.read_csv("data/samples/synthetic_ust_par_panel.csv")
dates = sorted(panel["date"].unique())
today = dates[-1]
curve = build_curve(panel, today)

# %% [markdown]
# ## 演習 1：保有0.5年でのキャリー＋ロール最大年限とリスク
#
# 保有期間を $h=0.5$ 年にして、年限別のキャリー＋ロールを比較し、最大年限の
# ブレークイーブン平行金利上昇幅を求めます。

# %%
H = 0.5
tenors = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)

cr_bp = []
for T in tenors:
    cpn = curve.zero_rate(float(T))
    t, a = make_cashflows(cpn, float(T), freq=1)
    cr_bp.append(carry_roll(curve, t, a, H)["total_bp"])
cr_bp = np.array(cr_bp)

best_t = tenors[int(np.argmax(cr_bp))]
print(f"保有 {H:g}年 のキャリー＋ロール（年率bp）")
for T, v in zip(tenors, cr_bp):
    print(f"  {int(T):2d}年: {v:6.1f} bp" + ("  ← 最大" if T == best_t else ""))

# 最大年限のブレークイーブン金利上昇幅
cpn_b = curve.zero_rate(float(best_t))
tb, ab = make_cashflows(cpn_b, float(best_t), freq=1)
rb = carry_roll(curve, tb, ab, H)
breakeven_bp = rb["total"] / (rb["price0"] * rb["duration"]) * 1e4
print(f"\n最大年限 {int(best_t)}年:")
print(f"  総キャリー＋ロール = {rb['total']:+.5f}（額面1あたり）")
print(f"  デュレーション     = {rb['duration']:.2f} 年")
print(f"  ブレークイーブン平行金利上昇 ≒ {breakeven_bp:.1f} bp")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar([str(int(T)) for T in tenors], cr_bp, color="#2c7fb8")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("年限（年）")
ax.set_ylabel("キャリー＋ロール（年率 bp）")
ax.set_title(f"年限別キャリー＋ロール（保有 {H:g}年）")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察（演習1）
#
# 保有0.5年でも、キャリー＋ロールは順イールドかつ曲線が急な中期ゾーンで大きくなります。
# 最大年限のブレークイーブン金利上昇幅は保有期間中に許容できるカーブ変化の目安で、
# この幅を超えて金利が上がるとキャリー＋ロールは消えます。デュレーションが長い年限ほど
# 同じ損益でもブレークイーブン幅は狭くなり、金利変動に脆弱です。したがって「キャリー＋ロールが
# 最大」＝「最良のポジション」ではなく、ブレークイーブン幅で測ったリスク調整後で選ぶべきです。

# %%
assert cr_bp[int(np.argmax(cr_bp))] == cr_bp.max()
assert breakeven_bp > 0                       # 正のキャリー＋ロールなら正のブレークイーブン
assert rb["duration"] > 0
print("演習1 チェックを通過しました")

# %% [markdown]
# ## 演習 2：カーブ不変仮定のヒストリカル検証
#
# パネル全期間で、各営業日 $d$ の「1か月後の予測キャリー＋ロール」と、実際に $d+21$
# 営業日後に実現したトータルリターンを比較します。予測はカーブ不変を仮定し、実現は
# 実際に動いたカーブで同一ポジションを再評価して測ります。

# %%
HORIZON_DAYS = 21                       # ≈1か月
h_years = HORIZON_DAYS / 252.0
TENOR = 10.0                            # 検証対象の年限

pred_bp, real_bp = [], []
for i, d0 in enumerate(dates[:-HORIZON_DAYS]):
    c0 = build_curve(panel, d0)
    c1 = build_curve(panel, dates[i + HORIZON_DAYS])
    cpn = c0.zero_rate(TENOR)
    t, a = make_cashflows(cpn, TENOR, freq=1)

    # 予測：カーブ不変のキャリー＋ロール（総額、額面1あたり）
    pr = carry_roll(c0, t, a, h_years)
    pred = pr["total"]

    # 実現：h年後に実際のカーブ c1 で残存 (t - h) を再評価し、ファンディングを引く
    p0 = float(np.sum(a * c0.discount(t)))
    fwd_price = p0 * np.exp(c0.zero_rate(h_years) * h_years)
    v_real = float(np.sum(a * c1.discount(t - h_years)))
    real = v_real - fwd_price

    pred_bp.append(pred / p0 * 1e4)
    real_bp.append(real / p0 * 1e4)

pred_bp = np.array(pred_bp)
real_bp = np.array(real_bp)
gap_bp = real_bp - pred_bp

print(f"{int(TENOR)}年債・保有{HORIZON_DAYS}営業日（予測 vs 実現、bp / 額面1）")
print(f"  予測キャリー＋ロール 平均: {pred_bp.mean():6.1f} bp（ほぼ一定）")
print(f"  実現トータルリターン 平均: {real_bp.mean():6.1f} bp")
print(f"  差（実現 − 予測）    平均: {gap_bp.mean():6.1f} bp  標準偏差: {gap_bp.std(ddof=1):5.1f} bp")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(pred_bp, label="予測（カーブ不変）", color="#2c7fb8")
axes[0].plot(real_bp, label="実現", color="#d95f0e", alpha=0.8)
axes[0].set_xlabel("開始営業日インデックス")
axes[0].set_ylabel("リターン（bp）")
axes[0].set_title("予測キャリー＋ロール vs 実現")
axes[0].legend()
axes[1].hist(gap_bp, bins=15, color="#756bb1", edgecolor="white")
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_xlabel("実現 − 予測（bp）")
axes[1].set_ylabel("頻度")
axes[1].set_title("カーブ変化による誤差の分布")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 考察（演習2）
#
# 予測キャリー＋ロールは各営業日でほぼ一定（カーブ形状がゆっくりしか変わらないため）ですが、
# 実現リターンは大きくばらつきます。差（実現 − 予測）はカーブ変化項そのもので、平均は
# ゼロ近傍でも標準偏差が予測の大きさを上回る局面があります。金利が上昇した時期には実現が
# 予測を大きく下回り（キャリー＋ロールが食われる）、低下した時期には上回ります。つまり
# カーブ不変仮定は「平均的な期待値」としては使えても、短期の実現損益はカーブ変化が支配的で、
# キャリー＋ロールだけを根拠にポジションを取るのは危ういことが確認できます。

# %%
assert len(pred_bp) == len(dates) - HORIZON_DAYS
assert np.std(pred_bp, ddof=1) < np.std(real_bp, ddof=1)   # 予測は実現よりばらつかない
assert np.isfinite(gap_bp).all()
print("演習2 チェックを通過しました")
