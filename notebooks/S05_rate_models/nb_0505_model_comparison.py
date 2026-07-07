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
# # S5-5 モデル比較とモデルリスク
#
# ## 学習目標
#
# - Vasicek・CIR・Hull-White を、分布特性・負金利の扱い・カーブ整合性という比較軸で切り分けて説明できる
# - 同一の市場データに3モデルをフィットし、同一商品を3通りで評価する比較フレームを自分で組める
# - モデル選択が価格・Greeks に与える影響を、一覧表として定量評価できる
# - モデルバリデーション（model validation）の実務枠組み（FRB SR 11-7）を理解し、Hull-White の承認レポートを書ける
# - 1ファクターモデルの限界と、多因子拡張（G2++・HJM・LMM）の位置づけ・使い分けを俯瞰できる
#
# S5 の締めくくりとして、S5-1〜S5-4 で個別に見た3モデルを横並びにし、「どのモデルを選ぶか」
# という意思決定そのものがリスク源になること（＝**モデルリスク**）を数値で体験します。

# %% [markdown]
# ## 理論
#
# ### 1. 3モデルの比較軸
#
# S5 で扱った3つの1ファクター短期金利モデルを、実務で問われる4つの軸で並べます。
#
# | 軸 | Vasicek | CIR | Hull-White（拡張 Vasicek） |
# |---|---|---|---|
# | SDE | $dr=a(b-r)dt+\sigma\,dW$ | $dr=a(b-r)dt+\sigma\sqrt{r}\,dW$ | $dr=(\theta(t)-a r)dt+\sigma\,dW$ |
# | $r(T)$ の分布 | 正規（Gaussian） | 非心カイ二乗（スケール） | 正規（Gaussian） |
# | 負金利 | 起こりうる | 起きない（$r\ge 0$） | 起こりうる |
# | 初期カーブ整合 | 近似（フィット誤差が残る） | 近似（フィット誤差が残る） | **厳密**（$\theta(t)$ で一致させる） |
# | パラメータ数 | 4（$a,b,\sigma,r_0$） | 4（$a,b,\sigma,r_0$） | 2＋カーブ（$a,\sigma$ ＋ $\theta(t)$） |
# | 解析トラクタビリティ | 高（ZCB・債券オプション閉形式） | 高（ZCB 閉形式、ZCB オプションも） | 高（ZCB 閉形式、カーブ整合） |
#
# ここで最も効くのが**カーブ整合性**です。Vasicek・CIR は時間一様（$a,b,\sigma$ が定数）なので、
# 観測される初期割引カーブを一般には厳密には再現できません。Hull-White は時間依存の
# ドリフト $\theta(t)$ を初期フォワードに合わせることで、初期カーブを厳密に再現します。
# 「今日の債券価格すら合わないモデルで、明日のデリバティブ価格を語れるか」という問いが、
# no-arbitrage モデル（Hull-White）を評価系の標準にしている理由です。
#
# ### 2. 価格差はどこから来るか
#
# 同じ商品を3モデルで評価すると価格が食い違います。その源泉は3つに分解できます。
#
# 1. **分布特性の違い**：$r(T)$ が正規（Vasicek・HW）か非心カイ二乗（CIR）かで、裾の厚さと
#    下限の有無が変わります。金利上限・下限のようなオプション性を持つ商品は、裾の形が
#    そのまま価格に効きます。とくに低金利局面では「負金利を許すか否か」が価格を大きく分けます。
# 2. **自由度（パラメータの数と種類）の違い**：Hull-White は $\theta(t)$ という関数自由度で
#    初期カーブを吸収するため、残る $a,\sigma$ がボラティリティ構造だけを担います。Vasicek・CIR は
#    同じ $a,b,\sigma$ でカーブの水準・傾き・曲率とボラティリティを同時に説明せねばならず、
#    どこかに歪みが出ます。
# 3. **カーブ整合の有無**：フォワードカーブがずれれば、割引そのものがずれます。オプション性が
#    無い割引債ですら、Vasicek・CIR は初期カーブ再現誤差の分だけ HW と食い違います。
#
# この notebook では、これら3源泉が「価格差」と「Greeks 差」として現れる様子を一覧表にします。
#
# ### 3. モデルバリデーションの実務（FRB SR 11-7）
#
# 米連邦準備制度（FRB）と OCC が2011年に出した監督指針 **SR 11-7 “Guidance on Model Risk
# Management”** は、モデルリスク管理の事実上の標準です。中核は「モデルは必ず間違っており
# （all models are wrong）、その誤りが損失・誤判断を生む可能性そのものがリスクである」という
# 立場に立ち、検証を3本柱で求めます。
#
# | 柱 | 内容 | この notebook での対応 |
# |---|---|---|
# | 概念的健全性の評価 | 理論・仮定・数式導出・適用範囲が妥当か | 比較軸の表・分布特性の議論 |
# | 継続的モニタリング | 前提が市場環境で崩れていないか、パラメータが安定か | 感応度分析（パラメータを振って価格変化を見る） |
# | アウトカム分析（ベンチマーク比較・バックテスト） | 独立な基準（**ベンチマークモデル**）や実現値と突き合わせる | QuantLib 検証・3モデル相互比較・カーブ再現誤差 |
#
# 実務の要点は3つです。第一に、**独立検証（independent validation）**：開発者と別の担当が、
# 前提を疑いながら再現・反証する。第二に、**ベンチマークモデル比較**：唯一の正解が無い評価では、
# 別実装・別モデルと突き合わせて差を説明できる状態を保つ。第三に、**感応度分析（sensitivity
# analysis）**：入力（カーブ・ボラ・平均回帰速度）を動かして、出力がどれだけ動くか、その動きが
# 経済的に説明できるかを確認する。差が大きい入力は、それ自体がモデルリスクの所在です。
#
# ### 4. 1ファクターの限界と多因子への拡張（実装はしない）
#
# 1ファクターモデルは、すべての年限の金利が単一のショック $dW$ で動きます。したがって
# 「短期は上がり長期は下がる」ようなツイストを表現できず、異なる年限の金利が完全相関します。
# キャップとスワップションの整合的な同時評価や、カーブ形状に依存する商品には力不足です。
# その受け皿として、多因子・カーブ全体モデルへ拡張します。
#
# | モデル | 状態変数 | 表現できるようになること | 主な用途・使い分け |
# |---|---|---|---|
# | **G2++**（2ファクター Gaussian） | 相関する2つの Gaussian ファクター | カーブの水準とスロープの独立な動き、より豊かなスワップション表面 | HW の自然な拡張。解析トラクタビリティを保ちつつ表現力を上げたいとき |
# | **HJM**（Heath-Jarrow-Morton） | 瞬間フォワードカーブ全体 $f(t,T)$ | フォワードカーブの任意形状、ボラティリティ構造を直接指定 | 理論の枠組み（ドリフト条件が無裁定を保証）。多くのモデルの母体 |
# | **LMM**（LIBOR Market Model／BGM） | 観測可能な離散フォワード金利群 | キャップ・スワップションの市場整合的キャリブレーション、Bermudan | エキゾチック金利デリバの実務標準。市場クオートと直結 |
#
# 使い分けの筋は「必要な表現力の最小限を選ぶ」です。単一銘柄の割引・単純なオプションなら
# HW で十分、カーブ形状に効く商品や複数のボラ商品を同時に扱うなら G2++ 以上、経路依存の
# エキゾチックなら LMM、というのが標準的な段階です。本 notebook の実装は1ファクター3種に
# 留め、多因子は位置づけの俯瞰に留めます。

# %% [markdown]
# ## スクラッチ実装
#
# 同一の市場カーブに Vasicek・CIR・Hull-White の3モデルをフィットし、同一商品を3通りで
# 評価する比較フレームを組みます。商品は次の2つです。
#
# - **割引債** $P(0,T)$：オプション性の無い素の商品。カーブ整合性の差がそのまま出ます。
# - **金利キャップレット（interest rate caplet）**：満期 $T$ で1回、期間 $\delta$ ・想定元本1の
#   変動金利のうち上限 $K$ を超えた分 $\delta\cdot\max(r(T)-K,\,0)$ を受け取り、経路割引します。
#   これはコーラブル債・キャップ付ローンに埋め込まれる金利オプション性の最小単位で、
#   $r(T)$ の分布形が価格に直に効きます。
#
# フィット方針は S5 の設計に従います。Hull-White は $\theta(t)$ で初期カーブを厳密再現するので
# フィット不要。Vasicek・CIR は**時間一様**なので厳密には合わせられず、`zero_rate(tau)` を
# カーブの `zero_rate(tau)` にスカラー最小二乗で**粗くフィット**します。ボラティリティ $\sigma$ は
# カーブ形状からはほぼ決まらない（オプション市場が決める量）ため、現実的な水準に固定し、
# 平均回帰速度 $a$ と長期水準 $b$ のみをフィットします。

# %% [markdown]
# ### 使用する自作関数
#
# 短期金利パスの生成と商品評価は本 notebook で自作します（QuantLib は後段の検証役）。
# Vasicek・CIR は `bondlab.models` の厳密サンプリングを使い、Hull-White のパス生成のみ
# $r(t)=x(t)+\alpha(t)$ 分解で notebook 内に書きます（`bondlab` の HW オブジェクトを利用）。
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `fit_affine_to_curve(kind, curve, grid, sigma, r0)` | モデル種別, カーブ, テナー格子, $\sigma$, $r_0$ | `(a, b, rmse)` | Vasicek/CIR を初期カーブへ粗くフィット（ゼロレート最小二乗） |
# | `alpha_hw(hw, t)` | HW モデル, 時点 | $\alpha(t)$ | HW の短期金利の平均関数 $\alpha(t)=f^M(0,t)+\frac{\sigma^2}{2a^2}(1-e^{-at})^2$ |
# | `simulate_hw(hw, T, n_steps, n_paths, seed)` | HW モデル, 満期, ステップ数, パス数, 乱数種 | `(times, r_paths)` | HW 短期金利パス（$x$ を厳密 OU サンプリングし $\alpha(t)$ を足す） |
# | `short_rate_paths(spec, T, n_steps, n_paths, seed)` | モデル指定, 満期, ステップ数, パス数, 乱数種 | `(times, r_paths)` | 3モデル共通のパス生成ディスパッチャ |
# | `price_bond_mc(times, r_paths)` | 時間格子, 短期金利パス | `(price, se)` | 経路割引による割引債 $P(0,T)$ の MC 価格 |
# | `price_caplet_mc(times, r_paths, K, delta)` | 時間格子, 短期金利パス, 上限 $K$, 期間 $\delta$ | `(price, se)` | 金利キャップレットの MC 価格 |
# | `neg_rate_prob(spec, T)` | モデル指定, 満期 | $\Pr[r(T)<0]$ | 満期時点の負金利確率（解析） |

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm

import bondlab
from bondlab.models import Vasicek, CIR, HullWhite
from bondlab.curve import bootstrap_par

print("bondlab version:", bondlab.__version__)

np.random.seed(0)  # notebook 全体の再現性（各シミュレーションには seed を別途渡す）


def fit_affine_to_curve(kind, curve, grid, sigma, r0):
    """Vasicek/CIR を初期カーブへ粗くフィットし (a, b, rmse) を返す。

    モデルの zero_rate(tau) をカーブの zero_rate(tau) にスカラー最小二乗で
    合わせる。sigma・r0 は固定し、平均回帰速度 a と長期水準 b のみ推定する
    （sigma はカーブ形状からほぼ決まらないため、外生的に与える）。
    """
    Model = Vasicek if kind == "vasicek" else CIR
    target = curve.zero_rate(grid)

    def resid(p):
        a, b = p
        return Model(a, b, sigma, r0).zero_rate(grid) - target

    lo_b = -0.02 if kind == "vasicek" else 1e-3   # CIR は b>0（長期水準は正）
    fit = least_squares(resid, x0=[0.3, 0.03], bounds=([0.01, lo_b], [3.0, 0.15]))
    a, b = fit.x
    rmse = float(np.sqrt(np.mean(fit.fun ** 2)))
    return a, b, rmse


def alpha_hw(hw, t):
    """Hull-White の短期金利 r(t) の平均 α(t)。

    r(t) = x(t) + α(t)、x は OU（平均0）。α(t)=f^M(0,t)+σ²/(2a²)(1-e^{-at})²。
    f^M は初期カーブの瞬間フォワードで、HW オブジェクトの _fm を用いる。
    """
    a, s = hw.a, hw.sigma
    return hw._fm(t) + (s ** 2 / (2 * a ** 2)) * (1 - np.exp(-a * t)) ** 2


def simulate_hw(hw, T, n_steps, n_paths, seed):
    """Hull-White 短期金利パスを厳密サンプリングで生成する。

    x は OU（dx=-a x dt+σ dW, x0=0）を遷移分布で厳密に進め、各時点で α(t) を
    足して r(t) を得る。初期カーブを再現するので E[P(0,T)] はカーブ DF に一致。
    """
    a, s = hw.a, hw.sigma
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, T, n_steps + 1)
    x = np.zeros(n_paths)
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = alpha_hw(hw, ts[0])
    mean_rev = np.exp(-a * dt)
    var = s ** 2 / (2 * a) * (1 - mean_rev ** 2)
    for i in range(n_steps):
        x = x * mean_rev + np.sqrt(var) * rng.standard_normal(n_paths)
        out[:, i + 1] = x + alpha_hw(hw, ts[i + 1])
    return ts, out


def short_rate_paths(spec, T, n_steps, n_paths, seed):
    """3モデル共通の短期金利パス生成ディスパッチャ。

    spec は ("vasicek", model) / ("cir", model) / ("hw", model) のタプル。
    Vasicek・CIR は bondlab の厳密サンプリング、HW は simulate_hw を使う。
    """
    kind, model = spec
    if kind == "vasicek":
        return model.simulate(T, n_steps, n_paths, seed=seed)
    if kind == "cir":
        return model.simulate_exact(T, n_steps, n_paths, seed=seed)
    return simulate_hw(model, T, n_steps, n_paths, seed)


def _path_discount(times, r_paths):
    """各パスの割引係数 D(T)=exp(-∫_0^T r dt) を台形近似で返す。"""
    dt = np.diff(times)
    integ = np.sum(0.5 * (r_paths[:, :-1] + r_paths[:, 1:]) * dt, axis=1)
    return np.exp(-integ)


def price_bond_mc(times, r_paths):
    """経路割引による割引債 P(0,T) の MC 価格と標準誤差。"""
    D = _path_discount(times, r_paths)
    n = D.size
    return float(D.mean()), float(D.std(ddof=1) / np.sqrt(n))


def price_caplet_mc(times, r_paths, K, delta):
    """金利キャップレット δ·max(r(T)-K,0) の経路割引 MC 価格と標準誤差。"""
    D = _path_discount(times, r_paths)
    payoff = delta * np.maximum(r_paths[:, -1] - K, 0.0)
    disc_payoff = D * payoff
    n = disc_payoff.size
    return float(disc_payoff.mean()), float(disc_payoff.std(ddof=1) / np.sqrt(n))


def neg_rate_prob(spec, T):
    """満期時点の負金利確率 Pr[r(T)<0]（解析）。

    Vasicek・HW は r(T) が正規なので Φ(-mean/std)。CIR は r≥0 なので 0。
    """
    kind, model = spec
    if kind == "cir":
        return 0.0
    a, s = model.a, model.sigma
    var = s ** 2 / (2 * a) * (1 - np.exp(-2 * a * T))
    if kind == "vasicek":
        mean = model.r0 * np.exp(-a * T) + model.b * (1 - np.exp(-a * T))
    else:  # hw
        mean = alpha_hw(model, T)
    return float(norm.cdf(-mean / np.sqrt(var)))


# %% [markdown]
# ### 同一市場データへの3モデルのフィット
#
# 合成のパー利回りカーブ（右上がり）をブートストラップして割引カーブを作り、これを共通の
# 市場データとして3モデルをフィットします。$r_0$ は最短年限のゼロレート、$\sigma$ は各モデルで
# 現実的な水準に固定します（CIR は $\sigma\sqrt{r}$ が瞬間ボラなので、水準を合わせるため
# 大きめの値を取ります）。

# %%
tenors = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
par_rates = np.array([0.008, 0.010, 0.014, 0.018, 0.023, 0.026, 0.028])
curve = bootstrap_par(tenors, par_rates, frequency=1)

fit_grid = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
r0 = float(curve.zero_rate(0.5))

sigma_vas, sigma_cir, a_hw, sigma_hw = 0.010, 0.070, 0.10, 0.010

a_v, b_v, rmse_v = fit_affine_to_curve("vasicek", curve, fit_grid, sigma_vas, r0)
a_c, b_c, rmse_c = fit_affine_to_curve("cir", curve, fit_grid, sigma_cir, r0)

vas = Vasicek(a_v, b_v, sigma_vas, r0)
cir = CIR(a_c, b_c, sigma_cir, r0)
hw = HullWhite(a_hw, sigma_hw, curve)

specs = {"Vasicek": ("vasicek", vas), "CIR": ("cir", cir), "HW": ("hw", hw)}

print(f"共通 r0 = {r0*100:.3f}%")
print(f"Vasicek: a={a_v:.4f}, b={b_v*100:.3f}%, σ={sigma_vas}  "
      f"→ カーブ再現 RMSE = {rmse_v*1e4:.2f} bp")
print(f"CIR    : a={a_c:.4f}, b={b_c*100:.3f}%, σ={sigma_cir}  "
      f"→ カーブ再現 RMSE = {rmse_c*1e4:.2f} bp（Feller {cir.feller()}）")
print(f"HW     : a={a_hw:.4f}, σ={sigma_hw}  → カーブ再現は厳密（RMSE = 0 bp）")

# %% [markdown]
# Vasicek・CIR は時間一様のため、右上がりカーブを十数 bp の誤差でしか追えません。HW は
# 定義から厳密再現です。この「フィット誤差の有無」が、以降のすべての価格差の一次的な源泉に
# なります。ゼロカーブを重ね描きして視覚的に確認します。

# %%
tt = np.linspace(0.25, 10.0, 60)
z_mkt = curve.zero_rate(tt) * 100
z_vas = vas.zero_rate(tt) * 100
z_cir = cir.zero_rate(tt) * 100
z_hw = np.array([-np.log(hw.zcb(0.0, t)) / t for t in tt]) * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tt, z_mkt, "k-", lw=2.4, label="市場カーブ（bootstrap_par）")
ax.plot(tt, z_vas, "--", color="steelblue", lw=1.8, label=f"Vasicek（RMSE {rmse_v*1e4:.1f}bp）")
ax.plot(tt, z_cir, "--", color="seagreen", lw=1.8, label=f"CIR（RMSE {rmse_c*1e4:.1f}bp）")
ax.plot(tt, z_hw, ":", color="crimson", lw=2.6, label="HW（厳密再現）")
ax.set_xlabel("満期 τ（年）")
ax.set_ylabel("連続複利ゼロレート（%）")
ax.set_title("同一市場カーブへの3モデルフィット")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 同一商品を3通りで評価する（価格差・Greeks 差の一覧）
#
# 割引債 $P(0,5)$ と、満期 $T=2$・期間 $\delta=0.5$・上限 $K=2.0\%$ の金利キャップレットを、
# 3モデルで評価します。Greeks は**共通乱数（common random numbers）**による bump-and-revalue で
# 求めます。同じ乱数種を使うことで、パス由来のノイズが差分でほぼ相殺され、少ないパス数でも
# 安定した感応度が得られます。
#
# - **金利感応度 $\partial V/\partial r$**（+1bp あたり）：Vasicek・CIR は $r_0$、HW は初期カーブ全体を
#   +25bp／−25bp 平行移動して中心差分（HW は $r_0$ という素の状態変数を持たないため、
#   カーブの平行移動で代替する。パラメータ化が異なる点は解釈上の注意点）。
# - **ベガ $\partial V/\partial \sigma$**（$\sigma$ の +10%相対 あたり）：各モデルの $\sigma$ を ±10% 振って中心差分。

# %%
T_cap, delta_cap, K_cap = 2.0, 0.5, 0.020
N_STEPS, N_PATHS, SEED = 48, 40000, 20260707


def build_spec(name, dr=0.0, vol_mult=1.0):
    """Greeks 用に r0（またはカーブ）と σ を bump したモデル spec を作る。"""
    if name == "Vasicek":
        return ("vasicek", Vasicek(a_v, b_v, sigma_vas * vol_mult, r0 + dr))
    if name == "CIR":
        return ("cir", CIR(a_c, b_c, sigma_cir * vol_mult, r0 + dr))
    bumped_curve = curve if dr == 0.0 else bootstrap_par(tenors, par_rates + dr, frequency=1)
    return ("hw", HullWhite(a_hw, sigma_hw * vol_mult, bumped_curve))


def caplet_value(name, dr=0.0, vol_mult=1.0):
    """共通乱数で金利キャップレットを評価（bump-and-revalue 用）。"""
    spec = build_spec(name, dr=dr, vol_mult=vol_mult)
    times, r_paths = short_rate_paths(spec, T_cap, N_STEPS, N_PATHS, SEED)
    price, _ = price_caplet_mc(times, r_paths, K_cap, delta_cap)
    return price


bump_r = 25e-4       # 平行移動 ±25bp（中心差分、ノイズ低減のため 1bp より大きめに取る）
vol_bump = 0.10      # σ の ±10% 相対

rows = []
for name in ["Vasicek", "CIR", "HW"]:
    spec = specs[name]
    # 割引債 P(0,5)：解析値と MC 値
    zcb_analytic = float(vas.zcb(5.0) if name == "Vasicek"
                         else cir.zcb(5.0) if name == "CIR"
                         else hw.zcb(0.0, 5.0))
    times5, r5 = short_rate_paths(spec, 5.0, 60, N_PATHS, SEED)
    zcb_mc, zcb_se = price_bond_mc(times5, r5)

    # 金利キャップレット価格
    cap_price = caplet_value(name)
    # Greeks（共通乱数の中心差分）
    dV_dr = (caplet_value(name, dr=+bump_r) - caplet_value(name, dr=-bump_r)) / (2 * bump_r) * 1e-4
    vega = (caplet_value(name, vol_mult=1 + vol_bump)
            - caplet_value(name, vol_mult=1 - vol_bump)) / 2.0

    rows.append({
        "モデル": name,
        "P(0,5) 解析": zcb_analytic,
        "P(0,5) MC": zcb_mc,
        "キャップレット(bp)": cap_price * 1e4,
        "∂V/∂r (bp/1bp)": dV_dr * 1e4,
        "ベガ (bp/+10%σ)": vega * 1e4,
    })

curve_df5 = float(curve.discount(5.0))
compare = pd.DataFrame(rows).set_index("モデル")
compare["P(0,5) 再現誤差(bp)"] = (-np.log(compare["P(0,5) 解析"] / curve_df5)) / 5.0 * 1e4

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print(f"市場カーブ P(0,5) = {curve_df5:.6f}\n")
print(compare.to_string())

# %% [markdown]
# 表から読み取れる**モデルリスク**の実像は次の通りです。
#
# - **割引債ですら差が出る**：Vasicek・CIR の $P(0,5)$ は市場カーブから十数 bp ずれます（再現誤差列）。
#   HW は誤差ゼロ。オプション性ゼロの商品でも、カーブ整合の有無だけで価格が動きます。
# - **キャップレットの差は桁違いに拡大**：HW のキャップレット価格は Vasicek・CIR の約2倍です。
#   HW はフォワードが市場に一致し、かつ平均回帰が緩く（$a=0.1$）$r(T)$ の分散が大きいため、
#   上限超過の確率が高くなります。Vasicek・CIR はカーブ追随のために速い平均回帰（$a\!\approx\!1.6$）が
#   選ばれ、$r(T)$ の分布が長期水準へ強く縮むので、上限を超えにくくなります。
# - **Greeks の差はさらに顕著**：金利感応度・ベガとも HW が突出します。ヘッジ量の見積もりが
#   モデル選択で数倍変わる、というのは実運用で致命的です。価格だけでなく Greeks を並べて
#   初めて、モデルリスクの全体像が見えます。

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metrics = [("キャップレット(bp)", "キャップレット価格 (bp)"),
           ("∂V/∂r (bp/1bp)", "金利感応度 ∂V/∂r (bp/1bp)"),
           ("ベガ (bp/+10%σ)", "ベガ (bp/+10%σ)")]
colors = ["steelblue", "seagreen", "crimson"]
for ax, (col, title) in zip(axes, metrics):
    ax.bar(compare.index, compare[col], color=colors)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, axis="y")
plt.suptitle("同一キャップレットの評価差とGreeks差（モデルリスク）", y=1.03)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## QuantLib 検証
#
# **検証の位置づけ**を明記します。本 notebook の3モデルの ZCB（ゼロクーポン債価格）は
# `bondlab.models` の自作実装です。ここでは、各モデルの ZCB が **QuantLib の対応クラスと
# 数値一致する**ことを確かめ、実装の正しさを独立なベンチマークで裏づけます。対応は次の通りです。
#
# | bondlab | QuantLib | 生成 |
# |---|---|---|
# | `Vasicek(a,b,σ,r0)` | `ql.Vasicek(r0,a,b,σ)` | 引数順に注意（$r_0$ が先頭） |
# | `CIR(a,b,σ,r0)` | `ql.CoxIngersollRoss(r0,b,a,σ)` | 引数順に注意（$r_0,\theta,k,\sigma$） |
# | `HullWhite(a,σ,curve)` | `ql.HullWhite(termstructure,a,σ)` | 同一の割引カーブを渡す |
#
# いずれも `discountBond(t, T, r)` でゼロクーポン債価格を取り、自作の `zcb` と突き合わせます。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

taus = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

# --- Vasicek: ql.Vasicek(r0, a, b, sigma) ---
ql_vas = ql.Vasicek(r0, a_v, b_v, sigma_vas)
vas_rows = []
for tau in taus:
    mine = float(vas.zcb(tau))
    theirs = ql_vas.discountBond(0.0, float(tau), r0)
    vas_rows.append((tau, mine, theirs, abs(mine - theirs)))
vas_chk = pd.DataFrame(vas_rows, columns=["τ", "bondlab", "QuantLib", "|差|"])
print("\n[Vasicek] bondlab vs ql.Vasicek")
print(vas_chk.to_string(index=False))
assert vas_chk["|差|"].max() < 1e-8

# --- CIR: ql.CoxIngersollRoss(r0, theta=b, k=a, sigma) ---
ql_cir = ql.CoxIngersollRoss(r0, b_c, a_c, sigma_cir)
cir_rows = []
for tau in taus:
    mine = float(cir.zcb(tau))
    theirs = ql_cir.discountBond(0.0, float(tau), r0)
    cir_rows.append((tau, mine, theirs, abs(mine - theirs)))
cir_chk = pd.DataFrame(cir_rows, columns=["τ", "bondlab", "QuantLib", "|差|"])
print("\n[CIR] bondlab vs ql.CoxIngersollRoss")
print(cir_chk.to_string(index=False))
assert cir_chk["|差|"].max() < 1e-8

# %% [markdown]
# ### Hull-White の検証
#
# Hull-White は初期カーブに依存するので、まず bondlab の割引カーブと同一の期間構造を
# QuantLib 側に構築します。$t=0$ では両者とも初期カーブを厳密再現するはずなので、
# `discountBond(0, T, r0)` が市場カーブ $P(0,T)$ に一致することをまず確認し、次に $t>0$ の
# 状態依存 ZCB $P(t,T;r_t)$ を突き合わせます。
#
# なお $t>0$ の比較点は、カーブの**ノード間**（節点でない年）に取ります。ログ線形補間の
# 割引カーブは瞬間フォワードがノードで折れ曲がる（微分が不連続）ため、HW が数値微分で
# 求める $f^M(0,t)$ とノード上で突き合わせると補間規約の差で $10^{-4}$ 程度の食い違いが
# 出ます。ノード間ではこの折れ曲がりが無く、両者は倍精度の丸め誤差の範囲で一致します。

# %%
today = ql.Date(15, 5, 2026)
ql.Settings.instance().evaluationDate = today
day_count = ql.Actual365Fixed()

node_t = curve.times[1:]  # t=0 の暗黙ノードを除く
node_dates = [today] + [today + ql.Period(int(round(t * 365)), ql.Days) for t in node_t]
node_dfs = [1.0] + [float(curve.discount(t)) for t in node_t]
ql_curve = ql.DiscountCurve(node_dates, node_dfs, day_count)
ql_handle = ql.YieldTermStructureHandle(ql_curve)
ql_hw = ql.HullWhite(ql_handle, a_hw, sigma_hw)

# t=0：初期カーブの厳密再現
hw_rows = []
for T in [1.0, 3.0, 5.0, 7.0, 10.0]:
    mine = float(hw.zcb(0.0, T))
    mkt = float(curve.discount(T))
    hw_rows.append((0.0, T, mine, mkt, abs(mine - mkt)))
# t>0：状態依存 ZCB を QuantLib と突合（比較点はノード間に取る）
r_t = 0.03
for (t, T) in [(1.5, 4.0), (2.5, 6.0), (4.0, 8.0)]:
    mine = float(hw.zcb(t, T, r_t))
    theirs = ql_hw.discountBond(t, T, r_t)
    hw_rows.append((t, T, mine, theirs, abs(mine - theirs)))

hw_chk = pd.DataFrame(hw_rows, columns=["t", "T", "bondlab", "QuantLib/市場", "|差|"])
print("[Hull-White] t=0 は市場カーブ、t>0（ノード間）は ql.HullWhite と突合")
print(hw_chk.to_string(index=False))
assert hw_chk["|差|"].max() < 1e-8
print("\n→ 3モデルとも QuantLib（および HW は市場カーブ）と一致。実装の正しさを確認")

# %% [markdown]
# ## 実データ適用
#
# 実データは合成カーブ（seed 固定の `bootstrap_par`）で代用します。ここでは
# **(1) 3モデルのゼロカーブ再現誤差** と **(2) 同一商品の評価差** を、表と図で総括します。
#
# ### (1) ゼロカーブ再現誤差
#
# 各テナーで、モデルのゼロレートと市場ゼロレートの差（bp）を並べます。HW が全テナーで
# ほぼゼロ、Vasicek・CIR が体系的にずれる様子を数値で確認します。

# %%
err_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
z_mkt_g = curve.zero_rate(err_grid)
err_rows = []
for tau in err_grid:
    zm = float(curve.zero_rate(tau))
    zv = float(vas.zero_rate(tau))
    zc = float(cir.zero_rate(tau))
    zh = -np.log(float(hw.zcb(0.0, tau))) / tau
    err_rows.append({
        "τ(年)": tau,
        "市場(%)": zm * 100,
        "Vasicek誤差(bp)": (zv - zm) * 1e4,
        "CIR誤差(bp)": (zc - zm) * 1e4,
        "HW誤差(bp)": (zh - zm) * 1e4,
    })
err_df = pd.DataFrame(err_rows).set_index("τ(年)")
print(err_df.to_string())
print("\nRMSE(bp):",
      f"Vasicek={np.sqrt(np.mean(err_df['Vasicek誤差(bp)']**2)):.2f},",
      f"CIR={np.sqrt(np.mean(err_df['CIR誤差(bp)']**2)):.2f},",
      f"HW={np.sqrt(np.mean(err_df['HW誤差(bp)']**2)):.2f}")

# %% [markdown]
# ### (2) 短期金利分布と負金利確率
#
# 満期 $T=5$ での短期金利 $r(T)$ の分布を3モデルで重ねます。Vasicek・HW は正規分布で
# 負の領域に裾を持ち、CIR は非負に張り付きます。あわせて、満期別の負金利確率
# $\Pr[r(T)<0]$（解析値）を比較します。

# %%
T_dist = 5.0
_, rv = short_rate_paths(specs["Vasicek"], T_dist, 60, N_PATHS, SEED)
_, rc = short_rate_paths(specs["CIR"], T_dist, 60, N_PATHS, SEED)
_, rh = short_rate_paths(specs["HW"], T_dist, 60, N_PATHS, SEED)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
bins = np.linspace(-0.04, 0.10, 60)
ax.hist(rv[:, -1], bins=bins, alpha=0.5, color="steelblue", density=True, label="Vasicek")
ax.hist(rc[:, -1], bins=bins, alpha=0.5, color="seagreen", density=True, label="CIR")
ax.hist(rh[:, -1], bins=bins, alpha=0.5, color="crimson", density=True, label="HW")
ax.axvline(0.0, color="black", lw=1.2, ls="--", label="r=0")
ax.set_xlabel("短期金利 r(T=5)")
ax.set_ylabel("密度")
ax.set_title("r(T=5) の分布（Vasicek/HW は負領域に裾、CIR は非負）")
ax.legend(fontsize=9)

ax = axes[1]
horizons = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
for name, color in [("Vasicek", "steelblue"), ("CIR", "seagreen"), ("HW", "crimson")]:
    probs = [neg_rate_prob(specs[name], T) * 100 for T in horizons]
    ax.plot(horizons, probs, "o-", color=color, label=name)
ax.set_xlabel("満期 T（年）")
ax.set_ylabel("負金利確率 Pr[r(T)<0]（%）")
ax.set_title("満期別の負金利確率（解析）")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

neg_tbl = pd.DataFrame({
    "T(年)": horizons,
    "Vasicek(%)": [neg_rate_prob(specs["Vasicek"], T) * 100 for T in horizons],
    "CIR(%)": [neg_rate_prob(specs["CIR"], T) * 100 for T in horizons],
    "HW(%)": [neg_rate_prob(specs["HW"], T) * 100 for T in horizons],
}).set_index("T(年)")
print(neg_tbl.to_string())

# %% [markdown]
# 同じカーブに合わせても、HW は市場の低い短期フォワードを継承し、かつ平均回帰が緩いため、
# 満期が延びるほど負金利確率がはっきり立ち上がります。CIR は定義上つねに 0 です。
# 「負金利をどう扱うか」というモデル選択が、そのまま裾リスクの見積もりを変えます。これは
# 価格・Greeks・分布のすべてに波及する、実務上もっとも重いモデルリスクの一つです。

# %% [markdown]
# ## 演習
#
# 1. **モデルバリデーターとして Hull-White の承認レポートを書く**
#    あなたはモデルバリデーターです。本 notebook のフレームを使い、Hull-White モデルを
#    金利デリバティブ評価に用いることの是非を、SR 11-7 の3本柱（概念的健全性・継続モニタリング・
#    アウトカム分析）に沿って1ページで論じよ。最低限、次を含めること。
#    (a) 適用範囲（何に使ってよく、何に使うべきでないか。1ファクター・Gaussian の帰結）、
#    (b) 限界（負金利を許すこと、単一ショックゆえカーブのツイストを表現できないこと）、
#    (c) 代替モデル（CIR＝負金利を避けたいとき、G2++／LMM＝カーブ形状や複数ボラ商品が要るとき）、
#    (d) 承認条件（どのベンチマークと突き合わせ、どのパラメータを継続監視するか）。
#    数値の裏づけとして、本 notebook のカーブ再現誤差表・価格差表・負金利確率表を引用せよ。
#
# 2. **3モデルのゼロカーブ形状と負金利確率の比較**
#    平均回帰速度 $a$ を $\{0.05, 0.10, 0.30, 1.00\}$ の各水準に固定し（$b,\sigma,r_0$ は本 notebook の
#    値を流用）、Vasicek と HW について (i) ゼロカーブ形状 $z(\tau)$ と (ii) 満期別負金利確率
#    $\Pr[r(T)<0]$ がどう変わるかを図示せよ。$a$ が小さい（平均回帰が緩い）ほど分散が増え、
#    負金利確率が上がることを確認し、「$a$ の選び方自体がモデルリスクである」ことを一言で述べよ。
#    さらに CIR について、同じ $a$ の各水準で Feller 条件 $2ab\ge\sigma^2$ の成否を表にせよ。
#
# 解答例は `solutions/S5/sol_0505.py` に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/05_rate_models.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | モデルリスク | model risk | モデルの誤りや誤用が誤った価格・リスク判断・損失を生む可能性 |
# | モデルバリデーション | model validation | 前提・実装・出力を独立に検証し、モデルの妥当性と限界を評価する営み |
# | ベンチマークモデル | benchmark model | 検証対象と突き合わせる独立な基準モデル・別実装 |
# | 感応度分析 | sensitivity analysis | 入力を振って出力の変化を測り、リスクの所在を特定する手法 |
# | G2++／LMM | G2++ / LIBOR Market Model | 多因子・カーブ全体モデル。1ファクターの表現力不足を補う拡張 |
