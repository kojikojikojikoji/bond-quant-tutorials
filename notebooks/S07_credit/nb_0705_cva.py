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
# # S7-5 CVA入門
#
# ## 学習目標
#
# - カウンターパーティリスク（counterparty risk）を「相手が破綻したときに、こちら側で正の時価だったポジションを取り losed する損失」として定義できる
# - エクスポージャープロファイル（exposure profile）を、将来価値パスから EE（expected exposure）・EPE（expected positive exposure）・ENE（expected negative exposure）として計算できる
# - CVA（credit valuation adjustment）の定義式 $\mathrm{CVA}=(1-R)\int_0^T \mathbb{E}[D(t)\,V(t)^+]\,\mathrm{d}\,\mathrm{PD}(t)$ を離散和に落とし、実装できる
# - スワップの将来価値パスを Hull-White モデルの下でモンテカルロ生成し（S5）、ハザードカーブ（S7）と結合して CVA を計算できる
# - ネッティング（netting）と担保（collateral）がエクスポージャーを削る仕組みを概観として説明できる
# - XVA デスクが何をヘッジするか（金利デルタ・ベガ、信用デルタ）を、エクスポージャーのスワップション表現から説明できる
#
# 本 notebook は S5（金利モンテカルロ）と S7（ハザードモデル）の総合演習です。フル XVA（FVA/KVA 等）には
# 踏み込まず、「CVA とは何か・エクスポージャープロファイルとは何か」に答えられる水準を目標にします。

# %% [markdown]
# ## 理論
#
# ### 1. カウンターパーティリスクとエクスポージャー
#
# 店頭（OTC）デリバティブは取引所を介さない相対契約なので、契約相手（カウンターパーティ）が
# 破綻すると、契約の履行が止まります。問題になるのは、破綻時点 $\tau$ で**こちら側が正の時価**を
# 持っていた場合です。相手に払ってもらうべき価値 $V(\tau)>0$ が回収できず、回収率 $R$ を除いた
# $(1-R)V(\tau)^+$ を失います。逆に $V(\tau)<0$（こちらが負債側）なら、破綻しても支払義務は残るので
# 損失は生じません。したがって損失は
#
# $$
# L = (1-R)\,V(\tau)^+ \,\mathbf{1}_{\{\tau \le T\}}
# $$
#
# という**片側**の形になります。この $V(t)^+=\max(V(t),0)$ を**エクスポージャー**（exposure）と呼びます。
# 将来の $V(t)$ は市場変数（ここでは金利）に依存して確率的に動くので、エクスポージャーも確率変数です。
#
# ### 2. エクスポージャープロファイル（EE / EPE / ENE）
#
# 各将来時点 $t$ でのエクスポージャーの分布を、期待値でまとめたものが**エクスポージャープロファイル**です。
#
# | 記号 | 名称 | 定義 | 意味 |
# |---|---|---|---|
# | $\mathrm{EE}(t)$ | 期待エクスポージャー（expected exposure） | $\mathbb{E}[V(t)^+]$ | その時点での平均的な「相手に対する債権」 |
# | $\mathrm{ENE}(t)$ | 期待負エクスポージャー（expected negative exposure） | $\mathbb{E}[V(t)^-]=\mathbb{E}[\min(V(t),0)]$ | 平均的な「こちらの債務」（DVA の材料） |
# | $\mathrm{EPE}$ | 期待正エクスポージャー（expected positive exposure） | $\tfrac1T\int_0^T \mathrm{EE}(t)\,\mathrm{d}t$ | プロファイルの時間平均（規制資本の入力） |
# | $\mathrm{PFE}(t)$ | 潜在的将来エクスポージャー（potential future exposure） | $V(t)^+$ の高分位点 | 限度枠管理に使うテールの指標 |
#
# 用語に幅があります。$\mathrm{EE}(t)$ の**プロファイル**そのものを EPE と呼ぶ流儀もありますが、本 notebook では
# 「$\mathrm{EE}(t)$ が時点別プロファイル、その時間平均のスカラーが EPE」と区別して使います。
#
# 金利スワップのエクスポージャーは特徴的な**山（ハンプ）**を描きます。満期直後は次のリセットまで時間があり
# 金利が動く余地が大きい一方、満期近くでは残存キャッシュフローが減ってポジションが小さくなります。この
# 「拡散効果（時間とともに不確実性が拡大）」と「アモチゼーション効果（残存が減る）」の綱引きで、プロファイルは
# 中間で山を作ります。
#
# ### 3. CVA の定義式
#
# CVA は「カウンターパーティリスクを織り込んだ、デリバティブ価値の調整額」です。無リスク価値から差し引く
# 損失の期待現在価値として、
#
# $$
# \mathrm{CVA} = (1-R)\,\mathbb{E}\!\left[\,D(\tau)\,V(\tau)^+\,\mathbf{1}_{\{\tau\le T\}}\right]
# = (1-R)\int_0^T \mathbb{E}\!\big[D(t)\,V(t)^+ \,\big|\, \tau=t\big]\;\mathrm{d}\,\mathrm{PD}(t)
# $$
#
# と書けます。$D(t)$ はマネーマーケット口座による確率的割引、$\mathrm{PD}(t)=1-S(t)$ は時点 $t$ までの累積
# デフォルト確率、$S(t)$ は S7 のハザードカーブから来る生存確率です。
#
# **独立性の仮定**（信用と市場が独立、ラップウェイ／ライトウェイリスクなし）を置くと、期待値の中の条件付けが
# 外れ、割引エクスポージャー $\mathrm{dEE}(t)=\mathbb{E}[D(t)V(t)^+]$ とデフォルト確率の積に分解できます。時間格子
# $0=t_0<t_1<\dots<t_m=T$ 上で離散化すると、
#
# $$
# \boxed{\;\mathrm{CVA}\;\approx\;(1-R)\sum_{k=1}^{m}\mathrm{dEE}(t_k)\,\big[S(t_{k-1})-S(t_k)\big]\;}
# $$
#
# となります。ここで $S(t_{k-1})-S(t_k)$ は区間 $(t_{k-1},t_k]$ の**限界デフォルト確率**です。さらに、割引を
# 確率的な $D(t)$ ではなく初期カーブの確定割引 $\mathrm{DF}(t)$ で近似し、$\mathrm{dEE}(t)\approx \mathrm{EE}(t)\,\mathrm{DF}(t)$ と
# すると、教科書的な
#
# $$
# \mathrm{CVA}\;\approx\;(1-R)\sum_{k=1}^{m}\mathrm{EE}(t_k)\,\mathrm{DF}(t_k)\,\big[S(t_{k-1})-S(t_k)\big]
# $$
#
# が得られます。これが本 notebook の実装の中心式です。確定割引で置き換える誤差は、割引因子とエクスポージャーの
# 相関に由来する二次の項で、後段で数値的に小さいことを確かめます。
#
# ### 4. エクスポージャーのスワップション表現
#
# 独立性の下で、スワップの**割引エクスポージャー**には閉じた別解釈があります。時点 $t$ に「残存スワップに
# 入る権利」を評価すると、それは満期 $t$ の**コターミナル・スワップション**（co-terminal swaption）に他なりません。
#
# $$
# \mathbb{E}\!\big[D(t)\,V(t)^+\big] \;=\; \text{（満期 } t \text{、原資産＝残存スワップ、行使価格＝当初固定金利のスワップション価格）}
# $$
#
# ペイヤースワップ（固定を払い変動を受ける）を保有していれば、その正のエクスポージャーはペイヤースワップション
# に対応します。したがって CVA は**コターミナル・スワップションのポートフォリオ**として書け、
#
# $$
# \mathrm{CVA} = (1-R)\sum_k \mathrm{Swaption}(t_k)\,\big[S(t_{k-1})-S(t_k)\big]
# $$
#
# となります。この表現は二つの意味で重要です。第一に、モンテカルロで計算した $\mathrm{dEE}(t_k)$ を、スワップション
# の解析価格と突き合わせる**検証**に使えます（後段で QuantLib の Jamshidian エンジンと突合します）。第二に、
# 「CVA はスワップションの束」という見方から、CVA が**金利ボラティリティに感応する（ベガを持つ）**ことが直ちに
# 分かります。XVA デスクがスワップションでヘッジする理由がここにあります。
#
# ### 5. ネッティングと担保（概観）
#
# エクスポージャーを削る二つの実務的な仕組みを概観します。
#
# - **ネッティング**：同じカウンターパーティとの複数取引を、ISDA マスター契約の下で**相殺**して扱う取り決めです。
#   破綻時に取引ごとの時価を合算した純額 $\big(\sum_j V_j(t)\big)^+$ でエクスポージャーを測れます。三角不等式
#   $\big(\sum_j V_j\big)^+ \le \sum_j V_j^+$ から、ネッティングセットの CVA は取引別 CVA の合計以下になります。
#   互いに符号が逆の取引ほど削減効果が大きく、演習2でこれを数値で確かめます。
# - **担保（CSA）**：時価の変動に応じて証拠金（バリエーションマージン）を日次で受け渡す取り決めです。担保が
#   完全なら残るエクスポージャーは、担保請求から実際の回収までの**マージン・ピリオド・オブ・リスク**（MPoR、
#   典型的に 10 営業日）の間に時価が動く分だけになります。エクスポージャープロファイルの山が、MPoR 幅の
#   短期エクスポージャーに置き換わって大きく縮みます。本 notebook では担保なし（無担保）の CVA を扱い、担保の
#   定量評価は扱いません。
#
# ### 6. XVA デスクの役割
#
# CVA は取引の価格に織り込まれ、多くの銀行では**XVA デスク**（あるいは CVA デスク）が集中管理します。役割は
# 大きく二つです。
#
# 1. **課金と集約**：新規取引の CVA をフロントに提示し、カウンターパーティ単位のネッティングセットで
#    エクスポージャーを集約する。
# 2. **ヘッジ**：CVA の変動リスクをヘッジする。CVA はエクスポージャー（＝スワップションの束）に依存するので、
#    金利デルタ・金利ベガを持ち、これは金利スワップ・スワップションでヘッジする。同時に、相手の信用スプレッド
#    が動くと PD が動いて CVA が動くので、**信用デルタ**を持ち、これはカウンターパーティ参照の CDS でヘッジする。
#
# CVA の対称形として、自社の破綻を織り込む DVA（debit valuation adjustment、$\mathrm{ENE}$ 側）、資金調達コストの
# FVA、資本コストの KVA などがあり、これらを総称して XVA と呼びます。本 notebook は CVA に絞ります。

# %% [markdown]
# ## スクラッチ実装
#
# 方針は次のとおりです。10 年・年 1 回払いの**ペイヤースワップ**（額面 1、固定金利＝当初のパースワップレート）を
# 保有し、Hull-White モデルの下で将来の短期金利パスをモンテカルロ生成します。各将来時点でスワップの時価
# $V(t)$ を再評価してエクスポージャープロファイルを作り、S7 のハザードカーブと結合して CVA を離散和で計算します。
#
# ### Hull-White の下でのパス生成
#
# `bondlab.models.HullWhite` の `zcb(t, T, r_t)` は、時点 $t$・短期金利 $r_t$ の下での割引債価格 $P(t,T)$ を、市場カーブに
# 整合する形（Brigo-Mercurio の式）で返します。将来のスワップ評価にはこの $P(t,T)$ を使いますが、そのためには
# 短期金利パス $r(t)$ を用意する必要があります。
#
# Hull-White は拡張 Vasicek（Vasicek 型）なので、短期金利は
#
# $$
# r(t) = f^M(0,t) + \underbrace{\frac{\sigma^2}{2a^2}\big(1-e^{-at}\big)^2}_{\text{凸性調整 }\alpha(t)} + x(t),\qquad
# \mathrm{d}x = -a\,x\,\mathrm{d}t + \sigma\,\mathrm{d}W,\;\; x(0)=0
# $$
#
# と分解できます。$f^M(0,t)$ は市場の瞬間フォワード、$x(t)$ は平均 0 の OU 過程です。この $x(t)$ は
# **`Vasicek(a, b=0, sigma, r0=0)` の短期金利そのもの**（長期水準 0・初期値 0 に回帰する OU 過程）なので、
# `Vasicek.simulate` をそのまま流用できます。得られた $x(t)$ に確定的なドリフト $f^M(0,t)+\alpha(t)$ を足し戻すと、
# パラメータが定数の Hull-White については**近似ではなく厳密に** $r(t)$ を再構成できます。この $r(t)$ を `HullWhite.zcb`
# に渡すことで、市場カーブに整合した将来割引債価格が得られます。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `simulate_hw_short_rate(hw, T, n_steps, n_paths, seed)` | HWモデル, 満期, 刻み数, パス数, seed | (時間格子, 短期金利パス) | Vasicek の OU 因子＋確定ドリフトで HW 短期金利を再構成 |
# | `swap_mtm(hw, t, r_t, K, coupon_times)` | HWモデル, 時点, 短期金利, 固定金利, 利払日列 | 時点 $t$ のペイヤースワップ時価（パス配列） | 残存キャッシュフローで $V(t)$ を再評価 |
# | `path_discount(r_full, t_grid)` | 短期金利パス, 時間格子 | 累積割引係数パス | パスに沿った $D(t)=e^{-\int_0^t r}$ を台形則で積分 |
# | `exposure_profiles(...)` | HW, パス, 割引, 格子, 固定金利, 利払日 | (EE, ENE, dEE) の各プロファイル | 時点別に $V^+$・$V^-$・$D V^+$ の平均を取る |
# | `cva_discrete(dEE, pd, recovery)` | 割引EE, 限界PD, 回収率 | CVA（スカラー） | 離散和 $(1-R)\sum \mathrm{dEE}\cdot\Delta\mathrm{PD}$ |

# %%
import numpy as np
import matplotlib.pyplot as plt

import bondlab
from bondlab.curve import bootstrap_par
from bondlab.models import HullWhite, Vasicek
from bondlab.pricing import par_swap_rate, swap_annuity
from bondlab.credit import HazardCurve, bootstrap_hazard

print("bondlab version:", bondlab.__version__)

SEED = 20260707
plt.rcParams["axes.unicode_minus"] = False

# モデルパラメータ（金利）とプロダクト定義。
A_HW, SIGMA_HW = 0.05, 0.01     # Hull-White の平均回帰速度とボラティリティ
RECOVERY = 0.40                  # 回収率 R
MATURITY = 10                    # スワップ満期（年）
FREQ = 1                         # 年 1 回払い

# 合成割引カーブ（seed 固定・ネットワーク不使用）。右肩上がりのパーカーブ。
par_tenors = np.arange(1, 31, dtype=float)
par_rates = 0.03 + 0.01 * (1.0 - np.exp(-par_tenors / 8.0))
curve = bootstrap_par(par_tenors, par_rates, frequency=1)

# 利払日（年）と、当初パースワップレート（これを固定金利にすると初期時価 0）。
coupon_times = np.arange(1, MATURITY + 1, dtype=float)
K_fixed = par_swap_rate(curve, curve, coupon_times)
print(f"10年パースワップレート（固定金利 K）= {K_fixed:.6f}")

# %%
def simulate_hw_short_rate(hw, T, n_steps, n_paths, seed):
    """Hull-White の短期金利パスを Vasicek の OU 因子から再構成して返す。

    x(t) は Vasicek(a, b=0, sigma, r0=0) の短期金利（平均 0 の OU 因子）。
    これに確定ドリフト f^M(0,t) + sigma^2/(2a^2)(1-e^{-at})^2 を足すと HW 短期金利になる。
    返り値は (時間格子, shape=(n_paths, n_steps+1) の短期金利パス)。
    """
    a, sigma = hw.a, hw.sigma
    t_grid, x = Vasicek(a, 0.0, sigma, 0.0).simulate(T, n_steps, n_paths, seed=seed)
    fwd = np.array([hw._fm(max(t, 1e-4)) for t in t_grid])          # f^M(0,t)
    conv = sigma ** 2 / (2 * a ** 2) * (1.0 - np.exp(-a * t_grid)) ** 2  # 凸性調整 α(t)
    r_full = fwd[None, :] + conv[None, :] + x
    return t_grid, r_full


def swap_mtm(hw, t, r_t, K, coupon_times):
    """時点 t・短期金利 r_t（パス配列）でのペイヤースワップ時価 V(t)。

    リセット日 t での残存スワップ [t, T_n] を単一カーブで評価する。
    変動レッグ = 1 - P(t,T_n)、固定レッグ = K Σ_{T_i>t} τ P(t,T_i)、V = 変動 - 固定。
    """
    rem = coupon_times[coupon_times > t + 1e-9]
    if rem.size == 0:
        return np.zeros_like(np.asarray(r_t, dtype=float))
    tau = 1.0 / FREQ
    p_last = hw.zcb(t, float(rem[-1]), r_t)             # P(t, T_n)
    fixed = np.zeros_like(np.asarray(r_t, dtype=float))
    for Ti in rem:
        fixed = fixed + tau * hw.zcb(t, float(Ti), r_t)  # Σ τ P(t,T_i)
    return (1.0 - p_last) - K * fixed


def path_discount(r_full, t_grid):
    """短期金利パスから累積割引係数 D(t)=exp(-∫_0^t r ds) を台形則で積分する。"""
    dt = np.diff(t_grid)
    incr = 0.5 * (r_full[:, 1:] + r_full[:, :-1]) * dt[None, :]
    cum = np.concatenate([np.zeros((r_full.shape[0], 1)), np.cumsum(incr, axis=1)], axis=1)
    return np.exp(-cum)


def exposure_profiles(hw, r_full, disc, t_grid, K, coupon_times, exposure_years):
    """各 exposure_years の時点で EE(t)=E[V^+]、ENE(t)=E[V^-]、dEE(t)=E[D·V^+] を返す。"""
    step_per_year = int(round((len(t_grid) - 1) / t_grid[-1]))
    EE, ENE, dEE = [], [], []
    for yr in exposure_years:
        if yr <= 0 or yr >= t_grid[-1]:      # 起点は時価 0、満期は残存 0
            EE.append(0.0); ENE.append(0.0); dEE.append(0.0)
            continue
        col = int(round(yr * step_per_year))
        v = swap_mtm(hw, float(yr), r_full[:, col], K, coupon_times)
        pos = np.maximum(v, 0.0)
        EE.append(float(pos.mean()))
        ENE.append(float(np.minimum(v, 0.0).mean()))
        dEE.append(float((disc[:, col] * pos).mean()))
    return np.array(EE), np.array(ENE), np.array(dEE)


def cva_discrete(exposure, pd_marginal, recovery):
    """離散 CVA = (1-R) Σ exposure(t_k)·限界PD_k。"""
    return float((1.0 - recovery) * np.sum(exposure * pd_marginal))


# %% [markdown]
# ### モンテカルロでエクスポージャープロファイルを作る
#
# 月次刻み（10 年で 120 ステップ）で短期金利パスを生成し、各年の時点でスワップ時価を再評価します。時点別に
# $V^+$・$V^-$ の平均を取ると EE・ENE のプロファイルが得られます。まず**シミュレーション自体の妥当性**を、
# 「パスに沿った割引の期待値が市場割引債を再現する」$\mathbb{E}[D(t)]\approx P^M(0,t)$ で確認します。これは
# Hull-White でマネーマーケット口座を numeraire に取ったときに厳密に成り立つ関係で、ドリフトとパスの整合を
# 一度に点検できます。

# %%
N_STEPS = 120
N_PATHS = 12000
exposure_years = np.arange(0, MATURITY + 1)

hw = HullWhite(A_HW, SIGMA_HW, curve)
t_grid, r_full = simulate_hw_short_rate(hw, float(MATURITY), N_STEPS, N_PATHS, SEED)
disc = path_discount(r_full, t_grid)

# 妥当性チェック：E[D(t)] ≈ 市場割引債 P^M(0,t)。
print(f"{'t':>4s}{'E[D(t)] (MC)':>16s}{'P^M(0,t) (curve)':>18s}{'差':>12s}")
for yr in [1, 3, 5, 7, 10]:
    col = int(round(yr * (N_STEPS / MATURITY)))
    e_d = float(disc[:, col].mean())
    pm = curve.discount(float(yr))
    print(f"{yr:>4d}{e_d:>16.6f}{pm:>18.6f}{e_d - pm:>12.2e}")

assert abs(disc[:, N_STEPS].mean() - curve.discount(float(MATURITY))) < 5e-3
print("→ パスに沿った割引の期待値が市場割引債を再現（シミュレーションのドリフト整合を確認）")

# 起点の時価が 0 であること（固定金利＝パーレート）。
v0 = swap_mtm(hw, 0.0, curve.discount(coupon_times), K_fixed, coupon_times)
print(f"初期時価 V(0) = {float(np.atleast_1d(v0).mean()):.2e}（パースワップなので 0）")

# %%
EE, ENE, dEE = exposure_profiles(hw, r_full, disc, t_grid, K_fixed, coupon_times, exposure_years)
EPE = float(np.trapz(EE, exposure_years) / MATURITY)  # プロファイルの時間平均

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(exposure_years, EE * 1e4, "o-", color="crimson", lw=2, label="EE(t) = E[V(t)+]")
ax.plot(exposure_years, ENE * 1e4, "o-", color="steelblue", lw=2, label="ENE(t) = E[V(t)-]")
ax.axhline(EPE * 1e4, color="gray", ls="--", lw=1.2, label=f"EPE (time avg) = {EPE*1e4:.1f} bp")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("time (years)")
ax.set_ylabel("exposure (bp of notional)")
ax.set_title("Exposure profile of a 10y payer swap (Hull-White MC)")
ax.legend()
plt.tight_layout()
plt.show()

print("→ ペイヤースワップの EE は中間で山を描く（拡散効果 × アモチゼーション効果）")

# %% [markdown]
# ### ハザードカーブと結合して CVA を計算する
#
# S7 のハザードモデルで、カウンターパーティの CDS パースプレッド列から生存確率カーブを剥ぎ取ります。各年の
# 限界デフォルト確率 $S(t_{k-1})-S(t_k)$ を重みにして、離散和で CVA を計算します。中心となる確定割引の式と、
# パスに沿った確率割引の式の両方を出し、後段の検証につなげます。

# %%
cds_tenors = np.array([1, 3, 5, 7, 10], dtype=float)
cds_spreads = np.array([0.008, 0.010, 0.012, 0.013, 0.015])  # 80〜150bp
hazard = bootstrap_hazard(curve, cds_tenors, cds_spreads, recovery=RECOVERY, freq=4)

S = hazard.survival(exposure_years.astype(float))
pd_marginal = S[:-1] - S[1:]                       # 各区間 (t_{k-1}, t_k] の限界 PD
DF = curve.discount(exposure_years.astype(float))

# 確定割引の式（中心）と、パス割引の式。
# 年 k のエクスポージャー EE[k] は、その年で終わる区間の限界 PD（pd_marginal[k-1]）と対にする。
# EE[1:]（年 1..10）と pd_marginal（区間 (0,1]..(9,10]、長さ 10）が要素ごとに対応する。
cva_simplified = cva_discrete(EE[1:] * DF[1:], pd_marginal, RECOVERY)
cva_discounted = cva_discrete(dEE[1:], pd_marginal, RECOVERY)

print(f"EPE（時間平均）               = {EPE*1e4:.2f} bp")
print(f"CVA（確定割引 EE×DF×PD）     = {cva_simplified*1e4:.3f} bp of notional")
print(f"CVA（パス割引 dEE×PD）        = {cva_discounted*1e4:.3f} bp of notional")
print(f"確定割引と確率割引の差        = {(cva_simplified-cva_discounted)*1e4:.3f} bp"
      f"（割引×エクスポージャー相関の二次項）")

# %% [markdown]
# ## QuantLib検証
#
# **検証の位置づけ**を最初に明記します。ここでは QuantLib のフル CVA エンジンには踏み込みません。代わりに、
# 理論 4 節の**スワップション表現**を独立な解析アンカーとして使い、次の二点を確かめます。
#
# 1. モンテカルロの割引エクスポージャー $\mathrm{dEE}(t_k)=\mathbb{E}[D(t_k)V(t_k)^+]$ が、QuantLib の Jamshidian エンジンで
#    解析評価したコターミナル・スワップション価格と、各時点で一致すること。
# 2. その結果、離散和 $\mathrm{CVA}=(1-R)\sum \mathrm{dEE}(t_k)\,\Delta\mathrm{PD}_k$ が、スワップション・ポートフォリオとして
#    組んだ CVA と一致し、モンテカルロ誤差の範囲で安定していること。
#
# 一致の程度には二つの誤差が乗ります。ひとつはモンテカルロの標本誤差、もうひとつは `bondlab`（年数ベースの
# 割引）と QuantLib（Act/365 の日数計算・実日付スケジュール）の**日数計算規約の差**です。後者は S6-3 でも見た
# 系統差で、数 % 程度残ります。したがって以下では機械精度ではなく、**モンテカルロ誤差と日数計算差の範囲での
# 整合**を確認する、という位置づけにします。あわせて、パス数を増やすと CVA 推定値が安定し、標準誤差が
# $1/\sqrt{N}$ で縮むことを確認します。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

TODAY = ql.Date(1, 1, 2024)
ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()


def bondlab_to_ql_curve(curve, max_t=30):
    """bondlab 割引カーブを QuantLib の割引カーブハンドルへ写す。"""
    node_t = np.concatenate([[0.0], np.arange(1, max_t + 1, dtype=float)])
    dates = [TODAY + ql.Period(int(round(t * 365)), ql.Days) for t in node_t]
    dfs = [1.0] + [float(curve.discount(t)) for t in node_t[1:]]
    return ql.YieldTermStructureHandle(ql.DiscountCurve(dates, dfs, DC))


ts = bondlab_to_ql_curve(curve)
ql_index = ql.IborIndex("Synthetic", ql.Period(1, ql.Years), 0, ql.USDCurrency(),
                        ql.NullCalendar(), ql.Unadjusted, False, DC, ts)
hw_model = ql.HullWhite(ts, A_HW, SIGMA_HW)
jamshidian = ql.JamshidianSwaptionEngine(hw_model, ts)


def ql_coterminal_swaption(expiry_year, strike):
    """満期 expiry_year、原資産＝残存スワップ [expiry, MATURITY] のペイヤースワップション NPV。"""
    if expiry_year >= MATURITY:   # 残存スワップが空なら価値 0
        return 0.0
    ex_date = TODAY + ql.Period(int(expiry_year), ql.Years)
    schedule = ql.Schedule(
        ex_date, TODAY + ql.Period(MATURITY, ql.Years),
        ql.Period(1, ql.Years), ql.NullCalendar(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
    )
    swap = ql.VanillaSwap(ql.VanillaSwap.Payer, 1.0, schedule, float(strike), DC,
                          schedule, ql_index, 0.0, DC)
    swap.setPricingEngine(ql.DiscountingSwapEngine(ts))
    swaption = ql.Swaption(swap, ql.EuropeanExercise(ex_date))
    swaption.setPricingEngine(jamshidian)
    return float(swaption.NPV())


# 各時点で dEE(MC) とスワップション（QL 解析）を突合する。年 1..10（10 年は残存 0）。
inner_years = exposure_years[1:]
swpt_prices = np.array([ql_coterminal_swaption(int(y), K_fixed) for y in inner_years])
dEE_inner = dEE[1:]

print(f"{'t':>4s}{'dEE(t) MC [bp]':>16s}{'swaption QL [bp]':>18s}{'差 [bp]':>12s}")
for y, d, s in zip(inner_years, dEE_inner, swpt_prices):
    print(f"{int(y):>4d}{d*1e4:>16.4f}{s*1e4:>18.4f}{(d-s)*1e4:>12.4f}")

# プロファイル全体で近い（モンテカルロ誤差＋日数計算差の範囲、数 bp）。
assert np.max(np.abs(dEE_inner - swpt_prices)) < 2e-3
print("→ 割引エクスポージャー dEE(t) がコターミナル・スワップション価格と一致（差は数 bp）")

# %% [markdown]
# ### スワップション・ポートフォリオとしての CVA
#
# 各時点のスワップション価格に限界 PD を重み付けて足すと、スワップション・ポートフォリオとしての CVA が
# 得られます。これがモンテカルロの CVA（パス割引）と一致することを確認します。

# %%
cva_swaption = cva_discrete(swpt_prices, pd_marginal, RECOVERY)

print(f"CVA（パス割引 MC）              = {cva_discounted*1e4:.4f} bp")
print(f"CVA（スワップション束・QL 解析）= {cva_swaption*1e4:.4f} bp")
print(f"相対差                          = {abs(cva_discounted-cva_swaption)/cva_swaption*100:.2f} %")
assert abs(cva_discounted - cva_swaption) / cva_swaption < 0.04
print("→ モンテカルロ CVA と解析スワップション CVA が整合（独立性の下での同値。残差は日数計算差）")

# %% [markdown]
# ### モンテカルロの収束と安定性
#
# パス数 $N$ を増やしたときの CVA 推定値と、その標準誤差を見ます。標準誤差は $1/\sqrt{N}$ に比例して縮み、
# 推定値は解析スワップション CVA（水平線）の近傍で安定します（残る数 % の差は前述の日数計算規約差）。
# CVA を「割引エクスポージャー × 限界 PD」のパス別寄与の平均とみなし、その標準偏差から標準誤差を出します。

# %%
def cva_mc_with_se(n_paths, seed):
    """パス割引 CVA の点推定と標準誤差を返す（パス別寄与の平均・標準偏差から）。"""
    tg, rf = simulate_hw_short_rate(hw, float(MATURITY), N_STEPS, n_paths, seed)
    dsc = path_discount(rf, tg)
    step_per_year = int(round(N_STEPS / MATURITY))
    contrib = np.zeros(n_paths)
    for y, w in zip(inner_years, pd_marginal):
        col = int(round(int(y) * step_per_year))
        v = swap_mtm(hw, float(y), rf[:, col], K_fixed, coupon_times)
        contrib += (1.0 - RECOVERY) * dsc[:, col] * np.maximum(v, 0.0) * w
    return float(contrib.mean()), float(contrib.std(ddof=1) / np.sqrt(n_paths))


path_counts = [1000, 2000, 4000, 8000, 16000]
means, ses = [], []
for i, n in enumerate(path_counts):
    m, se = cva_mc_with_se(n, SEED + 100 + i)
    means.append(m); ses.append(se)
    print(f"N={n:6d}: CVA = {m*1e4:.4f} ± {se*1e4:.4f} bp")

means = np.array(means); ses = np.array(ses)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.errorbar(path_counts, means * 1e4, yerr=1.96 * ses * 1e4, fmt="o-",
            color="crimson", capsize=4, label="MC CVA ± 95% CI")
ax.axhline(cva_swaption * 1e4, color="black", ls="--", lw=1.2,
           label=f"QL swaption CVA = {cva_swaption*1e4:.3f} bp")
ax.set_xscale("log")
ax.set_xlabel("number of paths N (log)")
ax.set_ylabel("CVA (bp of notional)")
ax.set_title("Monte-Carlo CVA stabilises near the analytic swaption portfolio")
ax.legend()
plt.tight_layout()
plt.show()

# 標準誤差が 1/√N で縮むこと（比の一定性）を確認。
ratio = ses * np.sqrt(path_counts)
print(f"SE × √N（ほぼ一定であるべき）= {np.round(ratio*1e4, 4)}")
print("→ 標準誤差は 1/√N で縮み、推定値は解析値の近傍で安定（離散和 CVA は安定）")

# %% [markdown]
# ## 実データ適用
#
# 実データは合成データ（seed 固定・ネットワーク不使用）で代用します。前段の合成カーブ・Hull-White
# パラメータ・ハザードカーブを土台に、**金利水準・金利ボラティリティ・信用スプレッド**を動かして CVA の感応度を
# 見ます。どのデスクが何をヘッジするかを、感応度の符号と紐づけて説明します。

# %% [markdown]
# ### 信用スプレッド感応度（再シミュレーション不要）
#
# 信用スプレッドはエクスポージャーではなく PD だけを動かすので、既に作った EE プロファイルを使い回せます。
# フラットハザード $\lambda=s/(1-R)$ を仮定し、スプレッド $s$ を振って CVA を計算します。CVA はスプレッドに
# ほぼ比例して増えます。これが XVA デスクが CDS でヘッジする**信用デルタ**の源です。

# %%
def flat_hazard(spread, recovery=RECOVERY):
    """フラットな信用スプレッド s に対応するハザードカーブ λ = s/(1-R)。"""
    return HazardCurve(np.array([float(MATURITY)]), np.array([spread / (1.0 - recovery)]))


print(f"{'spread [bp]':>12s}{'CVA [bp]':>12s}{'CVA/spread':>14s}")
base_spread = None
for s_bp in [25, 50, 100, 200, 400]:
    hz = flat_hazard(s_bp * 1e-4)
    Sf = hz.survival(exposure_years.astype(float))
    pdm = Sf[:-1] - Sf[1:]
    cva_s = cva_discrete(dEE[1:], pdm, RECOVERY)
    print(f"{s_bp:>12d}{cva_s*1e4:>12.4f}{cva_s/(s_bp*1e-4):>14.4f}")
print("→ CVA は信用スプレッドにほぼ比例（信用デルタ）。XVA デスクはカウンターパーティ CDS でヘッジ")

# %% [markdown]
# ### 金利ボラティリティ感応度（再シミュレーション）
#
# 金利ボラ $\sigma$ はエクスポージャーそのものを動かします。$\sigma$ が上がると将来価値の散らばりが増え、$V^+$ の
# 期待値（EE）が持ち上がり、CVA が増えます。CVA は**金利ベガ**を持つわけです。スワップション表現から見れば、
# CVA はスワップションの束なので、ボラに感応するのは当然です。XVA デスクはこれをスワップションでヘッジします。

# %%
print(f"{'sigma':>8s}{'EPE [bp]':>12s}{'CVA [bp]':>12s}")
for sig in [0.005, 0.010, 0.015, 0.020]:
    hw_s = HullWhite(A_HW, sig, curve)
    tg, rf = simulate_hw_short_rate(hw_s, float(MATURITY), N_STEPS, 6000, SEED + 11)
    dsc = path_discount(rf, tg)
    EEs, _, dEEs = exposure_profiles(hw_s, rf, dsc, tg, K_fixed, coupon_times, exposure_years)
    epe_s = float(np.trapz(EEs, exposure_years) / MATURITY)
    cva_v = cva_discrete(dEEs[1:], pd_marginal, RECOVERY)
    print(f"{sig:>8.3f}{epe_s*1e4:>12.3f}{cva_v*1e4:>12.4f}")
print("→ 金利ボラが上がると EE が持ち上がり CVA が増える（金利ベガ）。スワップションでヘッジ")

# %% [markdown]
# ### 金利水準感応度（再シミュレーション）
#
# パーカーブを平行シフトさせ、金利水準が CVA に与える効果を見ます。ペイヤースワップ（固定を払う）では、
# 金利が上がると受け取る変動が増えて時価がこちら有利に振れ、正のエクスポージャー（EE）が増えます。したがって
# CVA も増えます。これが CVA の**金利デルタ**で、金利スワップでヘッジします（符号はポジションの向きで反転）。

# %%
print(f"{'shift [bp]':>12s}{'par10 [%]':>12s}{'K_fwd':>10s}{'EPE [bp]':>12s}{'CVA [bp]':>12s}")
for shift_bp in [-100, -50, 0, 50, 100]:
    shifted = par_rates + shift_bp * 1e-4
    c_s = bootstrap_par(par_tenors, shifted, frequency=1)
    hw_s = HullWhite(A_HW, SIGMA_HW, c_s)
    # 固定金利は当初のまま（既存スワップの CVA を、その後の金利変化で測る）。
    tg, rf = simulate_hw_short_rate(hw_s, float(MATURITY), N_STEPS, 6000, SEED + 22)
    dsc = path_discount(rf, tg)
    EEs, _, dEEs = exposure_profiles(hw_s, rf, dsc, tg, K_fixed, coupon_times, exposure_years)
    epe_s = float(np.trapz(EEs, exposure_years) / MATURITY)
    cva_r = cva_discrete(dEEs[1:], pd_marginal, RECOVERY)
    par10 = shifted[9] * 100
    k_fwd = par_swap_rate(c_s, c_s, coupon_times)
    print(f"{shift_bp:>12d}{par10:>12.3f}{k_fwd:>10.4f}{epe_s*1e4:>12.3f}{cva_r*1e4:>12.4f}")
print("→ 金利上昇でペイヤースワップの EE が増え CVA も増える（金利デルタ）。金利スワップでヘッジ")

# %% [markdown]
# ### どのデスクが何をヘッジするか（まとめ）
#
# | 感応度 | 源泉 | 動かした変数 | ヘッジ手段 | 担当 |
# |---|---|---|---|---|
# | 信用デルタ | PD（生存確率） | 信用スプレッド | カウンターパーティ CDS | XVA デスク |
# | 金利ベガ | エクスポージャー（スワップションの束） | 金利ボラ $\sigma$ | スワップション | XVA デスク |
# | 金利デルタ | エクスポージャーの水準 | 金利水準 | 金利スワップ | XVA デスク（原資産は金利デスク） |
#
# CVA はこれら三つの感応度を同時に持ちます。XVA デスクは全カウンターパーティの CVA を集約し、信用は CDS、
# 金利デルタ・ベガはスワップ／スワップションで束ねてヘッジします。原資産の金利リスクそのものは金利デスクが
# 持ち、XVA デスクは「相手の信用がついたことによる調整分」のリスクを担う、という役割分担です。

# %% [markdown]
# ## 演習
#
# 1. **金利ボラ・信用スプレッドを振った CVA 感応度**
#    金利ボラ $\sigma\in\{0.005,0.010,0.015\}$ と信用スプレッド $s\in\{50,100,200\}$ bp の 2 次元格子で CVA を
#    計算し、表（行＝$\sigma$、列＝$s$）にまとめよ。$\sigma$ を増やすと（エクスポージャー経由で）、$s$ を増やすと
#    （PD 経由で）それぞれ CVA がどう動くかを述べ、二つの効果が近似的に「積」で効くことを確認せよ。
#
# 2. **ネッティングで CVA が減ることの確認**
#    同じカウンターパーティと、10y ペイヤースワップと 10y レシーバースワップ（固定金利を少しずらす）の 2 本を
#    持つとする。ネッティングありのエクスポージャー $\big(V_1(t)+V_2(t)\big)^+$ と、ネッティングなしの
#    $V_1(t)^+ + V_2(t)^+$ から、それぞれ CVA を計算し、$\mathrm{CVA}_{\text{netted}}\le \mathrm{CVA}_{\text{no-net}}$ を
#    数値で示せ。削減率を報告し、なぜ符号の逆な取引ほど効果が大きいかを説明せよ。
#
# 解答例は `solutions/S7/sol_0705.py` に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/07_credit.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | CVA | credit valuation adjustment | カウンターパーティの破綻損失を織り込む、デリバティブ価値の調整額 |
# | エクスポージャープロファイル | exposure profile | 将来時点ごとの期待エクスポージャー $\mathrm{EE}(t)=\mathbb{E}[V(t)^+]$ の推移 |
# | EPE | expected positive exposure | 期待エクスポージャーの時間平均。規制資本などの入力になるスカラー |
# | ネッティング | netting | 同一相手との複数取引を相殺し、純額でエクスポージャーを測る取り決め |
