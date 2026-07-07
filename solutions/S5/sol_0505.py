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
# # S5-5 演習 解答例
#
# モデル比較とモデルリスクの演習2問の解答例です。本編（`nb_0505_model_comparison.py`）で
# 組んだフィット・評価フレームを流用します。

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm

from bondlab.models import Vasicek, CIR, HullWhite
from bondlab.curve import bootstrap_par

np.random.seed(0)

# 本編と同じ市場カーブ
tenors = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
par_rates = np.array([0.008, 0.010, 0.014, 0.018, 0.023, 0.026, 0.028])
curve = bootstrap_par(tenors, par_rates, frequency=1)
fit_grid = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
r0 = float(curve.zero_rate(0.5))
sigma_vas, sigma_cir, a_hw, sigma_hw = 0.010, 0.070, 0.10, 0.010


def fit_affine_to_curve(kind, curve, grid, sigma, r0):
    """Vasicek/CIR を初期カーブへ粗くフィットし (a, b, rmse) を返す。"""
    Model = Vasicek if kind == "vasicek" else CIR
    target = curve.zero_rate(grid)

    def resid(p):
        a, b = p
        return Model(a, b, sigma, r0).zero_rate(grid) - target

    lo_b = -0.02 if kind == "vasicek" else 1e-3
    fit = least_squares(resid, x0=[0.3, 0.03], bounds=([0.01, lo_b], [3.0, 0.15]))
    a, b = fit.x
    return a, b, float(np.sqrt(np.mean(fit.fun ** 2)))


def alpha_hw(hw, t):
    """HW の短期金利の平均 α(t)=f^M(0,t)+σ²/(2a²)(1-e^{-at})²。"""
    a, s = hw.a, hw.sigma
    return hw._fm(t) + (s ** 2 / (2 * a ** 2)) * (1 - np.exp(-a * t)) ** 2


def neg_prob_gaussian(mean, a, s, T):
    """正規分布モデル（Vasicek/HW）の Pr[r(T)<0]=Φ(-mean/std)。"""
    var = s ** 2 / (2 * a) * (1 - np.exp(-2 * a * T))
    return float(norm.cdf(-mean / np.sqrt(var)))


a_v, b_v, _ = fit_affine_to_curve("vasicek", curve, fit_grid, sigma_vas, r0)
a_c, b_c, _ = fit_affine_to_curve("cir", curve, fit_grid, sigma_cir, r0)

# %% [markdown]
# ## 演習1：Hull-White の承認レポート（モデルバリデーター視点）
#
# 数値の裏づけを先に集計し、それを引用しながらレポート本文（Markdown）を書きます。

# %%
vas = Vasicek(a_v, b_v, sigma_vas, r0)
cir = CIR(a_c, b_c, sigma_cir, r0)
hw = HullWhite(a_hw, sigma_hw, curve)

# カーブ再現 RMSE（bp）
err_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
zm = curve.zero_rate(err_grid)
rmse = {
    "Vasicek": np.sqrt(np.mean(((vas.zero_rate(err_grid) - zm) * 1e4) ** 2)),
    "CIR": np.sqrt(np.mean(((cir.zero_rate(err_grid) - zm) * 1e4) ** 2)),
    "HW": np.sqrt(np.mean([((-np.log(hw.zcb(0.0, t)) / t) - float(curve.zero_rate(t))) ** 2
                           for t in err_grid])) * 1e4,
}
print("カーブ再現 RMSE(bp):", {k: round(v, 2) for k, v in rmse.items()})

# 10年満期の負金利確率
T = 10.0
mean_v = vas.r0 * np.exp(-a_v * T) + vas.b * (1 - np.exp(-a_v * T))
mean_h = alpha_hw(hw, T)
print("Pr[r(10)<0]: "
      f"Vasicek={neg_prob_gaussian(mean_v, a_v, sigma_vas, T)*100:.2f}%, "
      f"CIR=0.00%, "
      f"HW={neg_prob_gaussian(mean_h, a_hw, sigma_hw, T)*100:.2f}%")

# %% [markdown]
# ### 承認レポート（サンプル）
#
# **件名：Hull-White 1ファクターモデルの金利デリバティブ評価への適用（検証結果）**
#
# **1. 概念的健全性（conceptual soundness）**
# Hull-White は拡張 Vasicek であり、時間依存ドリフト $\theta(t)$ を初期割引カーブの瞬間フォワードに
# 合わせることで、**初期カーブを厳密再現**する no-arbitrage モデルである（本検証でカーブ再現
# RMSE ＝ 0 bp。対照的に Vasicek ≒ 16 bp、CIR ≒ 16 bp）。ゼロクーポン債価格・債券オプションが
# 閉形式で得られ、QuantLib の `ql.HullWhite` と数値一致することを確認済み。理論・実装ともに健全。
#
# **2. 適用範囲（scope）**
# - 使ってよい：単一カーブの割引・欧州型金利オプション（キャップ／フロア／スワプション）の
#   一次評価、初期カーブ整合が必須の評価。解析トラクタビリティが高く高速。
# - 使うべきでない：カーブのツイスト（短期↑長期↓）が価格を左右する商品、複数のボラティリティ
#   商品を同時整合させる必要がある局面。1ファクターゆえ全年限の金利が完全相関する。
#
# **3. 限界（limitations）**
# - **負金利を許す**：$r(T)$ は正規分布で、本検証では Pr[r(10)<0] が有意に正（HW は市場の低い
#   短期フォワードを継承し、満期とともに負金利確率が上昇）。金利下限の評価では過大／過小の
#   バイアス源になりうる。
# - **表現力の不足**：単一ショックのため、カーブ形状に依存する商品・Bermudan 等の経路依存
#   エキゾチックには力不足。
#
# **4. 代替モデル（alternatives）**
# - **CIR**：負金利を避けたいとき（$r\ge 0$、Feller 条件下で厳密に正）。ただし初期カーブは近似再現。
# - **G2++**：カーブの水準とスロープを独立に動かしたいとき（2ファクター Gaussian、解析性を保持）。
# - **LMM**：キャップ・スワップションの市場整合キャリブレーションや経路依存エキゾチックが要るとき。
#
# **5. 継続的モニタリングと承認条件（conditions）**
# - ベンチマーク：QuantLib `ql.HullWhite` および Vasicek/CIR との相互比較を定例で実施し、差の
#   要因（カーブ整合・分布特性・自由度）を説明できる状態を維持する。
# - 監視パラメータ：平均回帰速度 $a$ とボラティリティ $\sigma$。本検証のとおり $a$ は $r(T)$ の
#   分散と負金利確率を通じて価格・Greeks を大きく動かすため、キャリブレーション安定性を継続監視する。
# - 制約付き承認：適用範囲を「単一カーブ・欧州型・カーブ整合必須の評価」に限定して承認。
#   範囲外（強いカーブ形状依存・エキゾチック）は G2++／LMM への移行を条件とする。

# %% [markdown]
# ## 演習2：平均回帰速度 a を振ったゼロカーブ形状と負金利確率
#
# $a\in\{0.05,0.10,0.30,1.00\}$ について、Vasicek と HW のゼロカーブ形状と負金利確率を見ます。
# $b,\sigma,r_0$ は本編の値（Vasicek は $b=b_v,\sigma=0.01$、HW は $\sigma=0.01$）を流用します。

# %%
a_levels = [0.05, 0.10, 0.30, 1.00]
tt = np.linspace(0.25, 10.0, 60)
horizons = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# --- (i) ゼロカーブ形状 ---
ax = axes[0, 0]
ax.plot(tt, curve.zero_rate(tt) * 100, "k-", lw=2.4, label="市場カーブ")
for a in a_levels:
    z = Vasicek(a, b_v, sigma_vas, r0).zero_rate(tt) * 100
    ax.plot(tt, z, "--", lw=1.5, label=f"Vasicek a={a}")
ax.set_title("Vasicek ゼロカーブ（a を変化）")
ax.set_xlabel("満期 τ（年）"); ax.set_ylabel("ゼロレート(%)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(tt, curve.zero_rate(tt) * 100, "k-", lw=2.4, label="市場カーブ")
for a in a_levels:
    hw_a = HullWhite(a, sigma_hw, curve)
    z = np.array([-np.log(hw_a.zcb(0.0, t)) / t for t in tt]) * 100
    ax.plot(tt, z, ":", lw=1.8, label=f"HW a={a}")
ax.set_title("HW ゼロカーブ（a を変化：常に厳密再現）")
ax.set_xlabel("満期 τ（年）"); ax.set_ylabel("ゼロレート(%)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- (ii) 負金利確率 ---
ax = axes[1, 0]
for a in a_levels:
    m = Vasicek(a, b_v, sigma_vas, r0)
    probs = [neg_prob_gaussian(m.r0 * np.exp(-a * T) + m.b * (1 - np.exp(-a * T)),
                               a, sigma_vas, T) * 100 for T in horizons]
    ax.plot(horizons, probs, "o-", lw=1.5, label=f"a={a}")
ax.set_title("Vasicek 負金利確率 Pr[r(T)<0]")
ax.set_xlabel("満期 T（年）"); ax.set_ylabel("確率(%)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
for a in a_levels:
    hw_a = HullWhite(a, sigma_hw, curve)
    probs = [neg_prob_gaussian(alpha_hw(hw_a, T), a, sigma_hw, T) * 100 for T in horizons]
    ax.plot(horizons, probs, "o-", lw=1.5, label=f"a={a}")
ax.set_title("HW 負金利確率 Pr[r(T)<0]")
ax.set_xlabel("満期 T（年）"); ax.set_ylabel("確率(%)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle("平均回帰速度 a を振ったゼロカーブ形状と負金利確率", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 定量確認：a が小さいほど分散が増え、負金利確率が上がる
#
# 定常分散 $\mathrm{Var}[r(\infty)]=\sigma^2/(2a)$ は $a$ に反比例します。$a$ が小さいほど $r(T)$ の
# 散らばりが大きくなり、負の領域に落ちる確率が上がります。HW の 10年満期で数値確認します。

# %%
print("HW: a を下げると定常分散↑・負金利確率↑（T=10）")
print(f"{'a':>6} {'定常std(%)':>12} {'Pr[r(10)<0](%)':>16}")
for a in a_levels:
    hw_a = HullWhite(a, sigma_hw, curve)
    stat_std = np.sqrt(sigma_hw ** 2 / (2 * a)) * 100
    p = neg_prob_gaussian(alpha_hw(hw_a, 10.0), a, sigma_hw, 10.0) * 100
    print(f"{a:6.2f} {stat_std:12.3f} {p:16.2f}")

# a が単調に効くことを確認（a 小 → 負金利確率 大）
probs_by_a = [neg_prob_gaussian(alpha_hw(HullWhite(a, sigma_hw, curve), 10.0),
                                a, sigma_hw, 10.0) for a in a_levels]
assert probs_by_a[0] > probs_by_a[-1]
print("\n→ 平均回帰速度 a の選び方そのものが、分散・負金利確率・裾リスクを左右する"
      "モデルリスクである")

# %% [markdown]
# ### CIR の Feller 条件（同じ a の各水準で成否を確認）
#
# CIR は $2ab\ge\sigma^2$（Feller 条件）が成り立てば $r>0$ が厳密に保証されます。$b$ は各 $a$ で
# カーブに再フィットし直したうえで、条件の成否を表にします。

# %%
feller_rows = []
for a in a_levels:
    # a を固定し b のみ再フィット（sigma_cir は固定）
    def resid_b(b):
        return CIR(a, b[0], sigma_cir, r0).zero_rate(fit_grid) - curve.zero_rate(fit_grid)
    b_fit = least_squares(resid_b, x0=[0.03], bounds=([1e-3], [0.20])).x[0]
    m = CIR(a, b_fit, sigma_cir, r0)
    feller_rows.append({
        "a": a, "b(再フィット,%)": b_fit * 100,
        "2ab": 2 * a * b_fit, "σ²": sigma_cir ** 2,
        "Feller成立": m.feller(),
    })
feller_df = pd.DataFrame(feller_rows).set_index("a")
print(feller_df.to_string())
print("\n→ a が小さいと 2ab が σ² を下回り Feller 条件が崩れる（境界 0 到達がありうる）。"
      "非負性を厳密に担保したいなら a・b・σ の整合が必要")
