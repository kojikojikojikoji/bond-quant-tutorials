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
# # S5-2 CIRモデル
#
# ## 学習目標
#
# - 平方根過程（square-root process）としての CIR モデルの特性を、Vasicek との
#   違いに引きつけて説明できる
# - フェラー条件（Feller condition）が満たされるか否かで、短期金利の分布が
#   原点近傍でどう変わるかを、数式と可視化の両方で示せる
# - CIR のゼロクーポン債価格の解析解と、遷移分布が非中心カイ二乗分布
#   （noncentral chi-squared distribution）になる理由を導出できる
# - 厳密サンプリング（exact sampling）と離散化（Euler full-truncation）を
#   自分で実装し、解析解・`bondlab.models.CIR`・QuantLib と一致確認できる
# - フェラー条件を破るパラメータで離散化がどう壊れるかを観察し、厳密
#   サンプリングとの差を定量化できる
#
# 本ノートは S5 の短期金利モデル群の2本目。S5-1 の Vasicek で扱った
# アフィン期間構造の枠組みを、金利の非負性を保証する平方根拡散へ拡張する。


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# CIR の平方根拡散は「金利は負にならない」という制約を数式に埋め込んだモデルで、金利デスクのクオンツにとっては Vasicek と対をなす選択肢です。拡散が $\sigma\sqrt{r_t}$ となるため、金利が高いほどボラティリティが高いという水準依存ボラを自然に持ちます。これは実際の短期金利の挙動と整合的で、正の金利が前提の通貨・時代や、下限ゼロを崩したくない商品（金利フロア性の強いストラクチャー、保険の最低保証など）の値付けで効きます。ここで組む厳密サンプリング（非心カイ二乗分布）と解析解、`bondlab.models.CIR`・QuantLib への突き合わせは、本番評価系でこのモデルを使う際の検算そのものです。
#
# 稼ぎ方への効き方は、分布の裾と下限の扱いに集約されます。金利の上限・下限に対するオプション性を持つ商品は、裾が正規（Vasicek）か非心カイ二乗（CIR）かで価格が変わります。デスクはこの差を理解した上でモデルを選び、マーケットメイクのスプレッド収益や相対価値の判断に落とし込みます。逆に言えば、負金利が現実に起きる市場で $r\ge 0$ を強制する CIR を使えば、フロア近傍の価値を取り違え、ヘッジが崩れて損益に跳ね返ります。「どのモデルを選ぶか」自体が収益とリスクを左右する意思決定です（S5-5 のモデルリスク）。
#
# モデルバリデーション（FRB SR 11-7）の観点では、フェラー条件 $2ab\ge\sigma^2$ が要注意点です。条件を破るパラメータでは原点が到達可能になり、素朴なオイラー離散化は負値を踏んで壊れます。本 notebook で厳密サンプリングと離散化の差を定量化するのは、この落とし穴を独立検証で炙り出す作業に対応します。検証担当は、キャリブレーション結果がフェラー条件を満たすか、離散化スキームが分布の裾を正しく再現するかを、開発担当とは別に確かめてから承認します。ここを見落とすと、原点近傍の挙動に依存する商品でモデル要因の評価誤差が積み上がります。
# %% [markdown]
# ## 理論
#
# ### CIR の確率微分方程式
#
# Cox-Ingersoll-Ross（CIR）モデルは、リスク中立測度のもとで短期金利 $r_t$ を
# 次の確率微分方程式（SDE）で表す。
#
# $$
# dr_t = a\,(b - r_t)\,dt + \sigma\sqrt{r_t}\;dW_t
# $$
#
# ここで $a>0$ は平均回帰速度、$b>0$ は長期水準、$\sigma>0$ はボラティリティ
# 係数である。Vasicek の $dr_t = a(b-r_t)dt + \sigma\,dW_t$ と比べると、拡散項が
# 定数 $\sigma$ から $\sigma\sqrt{r_t}$ に変わっただけに見える。しかしこの
# $\sqrt{r_t}$ が本モデルの性質をすべて決める。
#
# - $r_t$ が小さくなるほど拡散（ゆらぎ）が小さくなり、ドリフト項 $a(b-r_t)>0$ が
#   相対的に強く効いて $r_t$ を引き戻す。この釣り合いにより金利は非負に保たれる。
# - 分散が水準 $r_t$ に比例するため、金利が高いほどボラティリティが高くなる。
#   これは実際の短期金利の挙動（水準依存ボラティリティ）と整合的である。
#
# ### 平方根過程としての性質
#
# $r_t$ の条件付き期待値と分散は、SDE から次のように求まる。ドリフトは Vasicek
# と同形なので平均は
#
# $$
# \mathbb{E}[r_t \mid r_0] = b + (r_0 - b)e^{-a t}
# $$
#
# で、長期水準 $b$ へ指数的に回帰する。分散は $\sqrt{r_t}$ の効果で複雑になり
#
# $$
# \operatorname{Var}[r_t \mid r_0]
# = r_0\,\frac{\sigma^2}{a}\bigl(e^{-a t} - e^{-2a t}\bigr)
# + b\,\frac{\sigma^2}{2a}\bigl(1 - e^{-a t}\bigr)^2
# $$
#
# となる。$t\to\infty$ での定常分布はガンマ分布であり、これも金利を非負に
# 保つことと符合する。
#
# ### フェラー条件
#
# 原点 $r=0$ が到達可能かどうかは、パラメータで決まる。**フェラー条件**
#
# $$
# 2ab \ge \sigma^2
# $$
#
# が成り立つとき、$r_t$ は確率1で厳密に正のままで、原点に触れない。破れる
# （$2ab < \sigma^2$）と、$r_t$ は正の確率で $0$ に到達する（そこで反射して
# 再び正に戻る）。
#
# この違いは定常分布（ガンマ分布 $\text{Gamma}(\text{shape}=2ab/\sigma^2,\;
# \text{scale}=\sigma^2/2a)$）の形状に直接現れる。
#
# - $2ab/\sigma^2 \ge 1$（条件成立）：密度は原点で $0$。金利が $0$ 近傍に来にくい。
# - $2ab/\sigma^2 < 1$（条件違反）：密度は原点で発散（$r\to 0$ で $+\infty$）。
#   金利が $0$ 付近に張り付きやすい。
#
# 後半でこの形状変化を実際に描く。フェラー条件は「$0$ に触れるか」という
# 定性的な境目であると同時に、離散化スキームが壊れるかどうかの分かれ目でもある。
#
# ### ゼロクーポン債価格の解析解
#
# CIR はアフィン期間構造モデルなので、満期までの年数 $\tau$ のゼロクーポン債
# （ZCB）価格が指数アフィン形で閉じる。
#
# $$
# P(\tau, r) = A(\tau)\,e^{-B(\tau)\,r}
# $$
#
# $$
# \gamma = \sqrt{a^2 + 2\sigma^2}, \qquad
# B(\tau) = \frac{2\,(e^{\gamma\tau}-1)}{2\gamma + (a+\gamma)(e^{\gamma\tau}-1)}
# $$
#
# $$
# A(\tau) = \left(
#   \frac{2\gamma\,e^{(a+\gamma)\tau/2}}{2\gamma + (a+\gamma)(e^{\gamma\tau}-1)}
# \right)^{2ab/\sigma^2}
# $$
#
# $B, A$ はリカッチ型の常微分方程式を解いて得られる。ゼロレートは
# $R(\tau) = -\ln P(\tau, r)/\tau$ で、$P$ が閉形式なのでカーブも解析的に描ける。
#
# ### 遷移分布が非中心カイ二乗分布になる理由
#
# 平方根拡散の遷移密度は、時間 $\Delta$ 先の $r_{t+\Delta}$ が既知の $r_t$ の
# もとで**スケールした非中心カイ二乗分布**に従うことが知られている。
#
# $$
# r_{t+\Delta} \mid r_t \;=\; c \cdot \chi'^2_{\,d}(\lambda), \qquad
# c = \frac{\sigma^2\,(1 - e^{-a\Delta})}{4a}
# $$
#
# $$
# d = \frac{4ab}{\sigma^2}\ (\text{自由度}), \qquad
# \lambda = \frac{r_t\,e^{-a\Delta}}{c}\ (\text{非心度})
# $$
#
# ここで $\chi'^2_{d}(\lambda)$ は自由度 $d$・非心度 $\lambda$ の非中心カイ二乗分布。
# 直感的には、平方根過程は $d$ 本の Ornstein-Uhlenbeck 過程の2乗和として
# 構成でき（Bessel 過程との対応）、その2乗和がまさに非中心カイ二乗分布に
# なる。自由度 $d = 4ab/\sigma^2$ に注目すると、フェラー条件 $2ab \ge \sigma^2$ は
# $d \ge 2$ と同値である。$d < 2$ で原点が到達可能になるのは、カイ二乗分布の
# 自由度が2未満だと原点で密度が発散するのと同じ理屈である。
#
# この遷移分布が閉じているおかげで、離散化誤差ゼロの**厳密サンプリング**が
# 可能になる。各ステップで上式のパラメータを計算し、非中心カイ二乗乱数を
# 引くだけでよい。
#
# ### Vasicek との比較（選択基準）
#
# | 観点 | Vasicek | CIR |
# |---|---|---|
# | 拡散項 | $\sigma$（定数） | $\sigma\sqrt{r_t}$（水準依存） |
# | 金利の符号 | 負になりうる | 非負（条件付きで厳密に正） |
# | 遷移分布 | 正規分布 | 非中心カイ二乗分布 |
# | 定常分布 | 正規分布 | ガンマ分布 |
# | ボラティリティ | 水準に依存しない | 水準が高いほど大きい |
# | ZCB 解析解 | あり（アフィン） | あり（アフィン） |
# | 厳密サンプリング | OU 遷移（正規） | 非中心カイ二乗 |
# | マイナス金利環境 | 表現できる | 表現できない |
# | 実装の単純さ | 単純 | やや複雑（乱数生成が重い） |
#
# 選択基準は次のとおり。金利の非負性が本質的に重要な場面（担保・信用の
# 強度モデル、名目金利がゼロ下限に張り付く局面の回避）では CIR を選ぶ。逆に
# マイナス金利を表現したい、あるいは正規分布の解析的な扱いやすさ（アフィン
# ガウス型の多因子拡張、閉形式オプション価格）を優先する場面では Vasicek 系を
# 選ぶ。実務では両者の中間として、水準シフト付き CIR やゼロ下限を跨げる
# シフト・ガウス型も使われる。


# %% [markdown]
# #### 数値例
#
# **数値例**：$a=0.3,\ b=0.05,\ \sigma=0.08$ ではフェラー条件は $2ab=0.030\ge\sigma^2=0.0064$ で成立し、自由度 $d=4ab/\sigma^2=9.38\ (\ge 2)$ です。一方 $a=0.2,\ b=0.05,\ \sigma=0.25$ では $2ab=0.020<\sigma^2=0.0625$、$d=0.64\ (<2)$ となり原点が到達可能になります。
#
# **数値例**：$\tau=5$ 年・$r=0.03$ のとき $\gamma=\sqrt{0.3^2+2\times 0.08^2}=0.3206$、$B(\tau)=2.557$、$A(\tau)=0.8872$ となり、割引債は $P=A\,e^{-B r}=0.8217$ です。
# %% [markdown]
# ## スクラッチ実装
#
# `bondlab.models.CIR` に頼らず、CIR の中核を3通り自作する。いずれも
# `SDE: dr = a(b - r)dt + σ√r dW`（$a$=平均回帰速度, $b$=長期水準）に対応する。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `cir_zcb_scratch(a, b, sigma, r0, tau)` | パラメータ, 満期年数 | ZCB 価格 | 解析解 $A e^{-Br}$ を素の式で計算 |
# | `cir_exact_ncx2(a, b, sigma, r0, T, n_steps, n_paths, rng)` | パラメータ, 時間格子, 乱数生成器 | (times, paths) | `scipy.stats.ncx2` による厳密サンプリング |
# | `cir_exact_rng(a, b, sigma, r0, T, n_steps, n_paths, rng)` | 同上 | (times, paths) | `rng.noncentral_chisquare` による厳密サンプリング |
# | `cir_euler_ft(a, b, sigma, r0, T, n_steps, n_paths, rng)` | 同上 | (times, paths) | Euler full-truncation 離散化 |
# | `mc_zcb(paths, times, tau)` | パスと格子, 満期 | (推定値, 標準誤差) | $\mathbb{E}[e^{-\int_0^\tau r\,ds}]$ を台形則で MC 推定 |
#
# 厳密サンプリングを2実装（`scipy` 版と `numpy.Generator` 版）用意するのは、
# 両者が同じ非中心カイ二乗分布を指していることを相互確認するため。

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
for _f in ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Noto Sans JP", "TakaoPGothic", "IPAPGothic"]:
    if any(_f == _n.name for _n in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
from scipy.stats import ncx2, gamma

import bondlab
from bondlab.models import CIR

print("bondlab version:", bondlab.__version__)


def cir_zcb_scratch(a, b, sigma, r0, tau):
    """CIR のゼロクーポン債価格 P = A·exp(-B·r0) を解析式で計算する。"""
    tau = np.asarray(tau, dtype=float)
    g = np.sqrt(a ** 2 + 2 * sigma ** 2)
    eg = np.exp(g * tau) - 1.0
    denom = 2 * g + (a + g) * eg
    A = (2 * g * np.exp((a + g) * tau / 2.0) / denom) ** (2 * a * b / sigma ** 2)
    B = 2 * eg / denom
    return A * np.exp(-B * r0)


def _cir_exact_params(a, b, sigma, dt):
    """厳密サンプリングの遷移分布パラメータ (c, 自由度 d) を返す。"""
    c = sigma ** 2 * (1.0 - np.exp(-a * dt)) / (4.0 * a)
    d = 4.0 * a * b / sigma ** 2
    return c, d


def cir_exact_ncx2(a, b, sigma, r0, T, n_steps, n_paths, rng):
    """scipy.stats.ncx2 を使った厳密サンプリング。

    各ステップで r_{t+dt} = c·χ'^2(d, λ) を引く（λ は前ステップの r に依存）。
    """
    dt = T / n_steps
    c, d = _cir_exact_params(a, b, sigma, dt)
    decay = np.exp(-a * dt)
    r = np.full(n_paths, float(r0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = r
    for i in range(n_steps):
        lam = r * decay / c                      # 非心度 λ
        r = c * ncx2.rvs(df=d, nc=lam, size=n_paths, random_state=rng)
        out[:, i + 1] = r
    return np.linspace(0.0, T, n_steps + 1), out


def cir_exact_rng(a, b, sigma, r0, T, n_steps, n_paths, rng):
    """numpy Generator の noncentral_chisquare を使った厳密サンプリング。"""
    dt = T / n_steps
    c, d = _cir_exact_params(a, b, sigma, dt)
    decay = np.exp(-a * dt)
    r = np.full(n_paths, float(r0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = r
    for i in range(n_steps):
        lam = r * decay / c
        r = c * rng.noncentral_chisquare(d, lam, n_paths)
        out[:, i + 1] = r
    return np.linspace(0.0, T, n_steps + 1), out


def cir_euler_ft(a, b, sigma, r0, T, n_steps, n_paths, rng):
    """Euler-Maruyama 離散化（full-truncation スキーム）。

    負値を許すと √r が計算不能になるため、ドリフト・拡散の両方で r を
    max(r, 0) に切り詰め（full truncation）、記録も切り詰めた値で行う。
    """
    dt = T / n_steps
    sqdt = np.sqrt(dt)
    r = np.full(n_paths, float(r0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = r
    for i in range(n_steps):
        rp = np.maximum(r, 0.0)
        dW = rng.standard_normal(n_paths) * sqdt
        r = r + a * (b - rp) * dt + sigma * np.sqrt(rp) * dW
        out[:, i + 1] = np.maximum(r, 0.0)
    return np.linspace(0.0, T, n_steps + 1), out


def mc_zcb(paths, times, tau):
    """割引債価格 E[exp(-∫_0^τ r ds)] を台形則 + モンテカルロで推定する。

    返り値は (推定値, 標準誤差)。
    """
    idx = int(np.round(tau / (times[1] - times[0])))
    dt = times[1] - times[0]
    seg = paths[:, : idx + 1]
    integral = (seg[:, 0] * 0.5 + seg[:, 1:-1].sum(axis=1) + seg[:, -1] * 0.5) * dt
    disc = np.exp(-integral)
    return disc.mean(), disc.std(ddof=1) / np.sqrt(len(disc))


# フェラー条件を満たす基準パラメータ。
a, b, sigma, r0 = 0.30, 0.05, 0.08, 0.03
model = CIR(a, b, sigma, r0)
print("フェラー条件 2ab ≥ σ²:", model.feller(), f"(2ab={2*a*b:.4f}, σ²={sigma**2:.4f})")

# 自作解析解が bondlab と一致するか。
taus = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
p_scratch = cir_zcb_scratch(a, b, sigma, r0, taus)
p_bondlab = model.zcb(taus)
print("\n満期    自作解析     bondlab      差")
for t, ps, pb in zip(taus, p_scratch, p_bondlab):
    print(f"{t:5.1f} {ps:12.8f} {pb:12.8f} {abs(ps-pb):.2e}")
assert np.allclose(p_scratch, p_bondlab, atol=1e-14)
print("\n自作解析解 == bondlab.models.CIR.zcb（機械精度）")

# %% [markdown]
# ### 3実装の相互一致（解析 ZCB ＝ 厳密 MC ＝ 離散化 MC）
#
# 割引債価格 $P(\tau) = \mathbb{E}\!\left[e^{-\int_0^\tau r_s\,ds}\right]$ を、
# 厳密サンプリング2種と離散化1種のモンテカルロで推定し、解析解と標準誤差の
# 範囲で一致するかを確認する。フェラー条件を満たす基準パラメータでは、離散化も
# 負値をほぼ生まないため4値が揃うはずである。

# %%
T = 5.0
n_steps = 250        # dt = 0.02 年
n_paths = 40000
tau_mc = 5.0

rng = np.random.default_rng(20260707)
t1, paths_ncx2 = cir_exact_ncx2(a, b, sigma, r0, T, n_steps, n_paths, rng)
t2, paths_rng = cir_exact_rng(a, b, sigma, r0, T, n_steps, n_paths, rng)
t3, paths_bl = model.simulate_exact(T, n_steps, n_paths, seed=12345)
t4, paths_eul = cir_euler_ft(a, b, sigma, r0, T, n_steps, n_paths, rng)

p_analytic = float(cir_zcb_scratch(a, b, sigma, r0, tau_mc))
est_ncx2, se_ncx2 = mc_zcb(paths_ncx2, t1, tau_mc)
est_rng, se_rng = mc_zcb(paths_rng, t2, tau_mc)
est_bl, se_bl = mc_zcb(paths_bl, t3, tau_mc)
est_eul, se_eul = mc_zcb(paths_eul, t4, tau_mc)

print(f"満期 {tau_mc:.0f} 年の割引債価格")
print(f"解析解                          : {p_analytic:.6f}")
print(f"厳密 MC (scipy ncx2)            : {est_ncx2:.6f} ± {se_ncx2:.6f}"
      f"  ({(est_ncx2-p_analytic)/se_ncx2:+.2f}σ)")
print(f"厳密 MC (rng noncentral_chisq)  : {est_rng:.6f} ± {se_rng:.6f}"
      f"  ({(est_rng-p_analytic)/se_rng:+.2f}σ)")
print(f"厳密 MC (bondlab simulate_exact): {est_bl:.6f} ± {se_bl:.6f}"
      f"  ({(est_bl-p_analytic)/se_bl:+.2f}σ)")
print(f"離散化 MC (Euler full-trunc)    : {est_eul:.6f} ± {se_eul:.6f}"
      f"  ({(est_eul-p_analytic)/se_eul:+.2f}σ)")

# すべて解析解から 4σ 以内（統計的一致）を要求。
for est, se in [(est_ncx2, se_ncx2), (est_rng, se_rng),
                (est_bl, se_bl), (est_eul, se_eul)]:
    assert abs(est - p_analytic) < 4 * se
print("\n4実装すべて解析解と標準誤差の範囲で一致")

# %% [markdown]
# ゼロカーブ全体でも一致を見る。厳密サンプリング（scipy 版）で複数満期の
# 割引債を一度に推定し、解析カーブと重ねる。

# %%
tau_grid = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
analytic_curve = cir_zcb_scratch(a, b, sigma, r0, tau_grid)
mc_curve = np.array([mc_zcb(paths_ncx2, t1, tt) for tt in tau_grid])

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(tau_grid, analytic_curve, "o-", label="解析解", zorder=3)
ax.errorbar(tau_grid, mc_curve[:, 0], yerr=3 * mc_curve[:, 1],
            fmt="s", capsize=4, label="厳密 MC ±3σ")
ax.set_xlabel("満期 τ（年）")
ax.set_ylabel("割引債価格 P(τ)")
ax.set_title("CIR 割引債カーブ：解析解 vs 厳密サンプリング MC")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## QuantLib検証
#
# CIR の割引債価格を QuantLib の `CoxIngersollRoss` と機械精度で突合する。
# **引数順に注意**する。QuantLib は $dr = k(\theta - r)dt + \sigma\sqrt{r}\,dW$ の
# 規約で、コンストラクタは `(r0, theta=長期水準, k=回帰速度, sigma)` の順に取る。
# bondlab の `(a=回帰速度, b=長期水準)` に対しては `(r0, b, a, sigma)` の順で
# 渡す。割引債は `discountBond(t, T, r)` で、$t=0$ から満期 $\tau$ を評価する。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

qc = ql.CoxIngersollRoss(r0, b, a, sigma)   # (r0, theta=b, k=a, sigma)

print("\n満期    bondlab       QuantLib      差")
max_err = 0.0
for tau in taus:
    p_bl = float(model.zcb(tau))
    p_ql = qc.discountBond(0.0, tau, r0)    # discountBond(t=0, T=τ, r=r0)
    err = abs(p_bl - p_ql)
    max_err = max(max_err, err)
    print(f"{tau:5.1f} {p_bl:12.8f} {p_ql:12.8f} {err:.2e}")

print(f"\n最大誤差 = {max_err:.2e}")
assert max_err < 1e-12
print("bondlab.models.CIR == QuantLib CoxIngersollRoss（機械精度一致）")

# %% [markdown]
# ## 実データ適用
#
# 合成データ（seed 固定）で、フェラー条件を破ると離散化がどう壊れるかを観察する。
# フェラー条件を満たすケースと破るケースの2パラメータを用意し、離散化と厳密
# サンプリングの差を可視化する。

# %%
# 条件成立ケース（基準）と、条件違反ケース（σ を大きく）。
params_ok = dict(a=0.30, b=0.05, sigma=0.08, r0=0.03)   # 2ab=0.030 ≥ σ²=0.0064
params_break = dict(a=0.20, b=0.05, sigma=0.25, r0=0.03)  # 2ab=0.020 < σ²=0.0625

for name, p in [("条件成立", params_ok), ("条件違反", params_break)]:
    m = CIR(p["a"], p["b"], p["sigma"], p["r0"])
    d = 4 * p["a"] * p["b"] / p["sigma"] ** 2
    print(f"{name}: 2ab={2*p['a']*p['b']:.4f}, σ²={p['sigma']**2:.4f}, "
          f"feller={m.feller()}, 自由度 d={d:.3f}")

# %% [markdown]
# ### 離散化の負値発生を数える
#
# 同じ乱数系列（seed 固定）で、条件違反パラメータの離散化パスに何回 $0$ 切り詰め
# が発生したか（＝ full-truncation が無ければ負になっていた回数）を数える。

# %%
def euler_count_truncations(a, b, sigma, r0, T, n_steps, n_paths, seed):
    """Euler full-truncation で、切り詰め前に負となったステップ数を数える。"""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqdt = np.sqrt(dt)
    r = np.full(n_paths, float(r0))
    n_neg = 0
    total = 0
    for _ in range(n_steps):
        rp = np.maximum(r, 0.0)
        dW = rng.standard_normal(n_paths) * sqdt
        r = r + a * (b - rp) * dt + sigma * np.sqrt(rp) * dW
        n_neg += int((r < 0).sum())
        total += n_paths
        r = np.maximum(r, 0.0)
    return n_neg, total


T2, ns2, npth2 = 5.0, 250, 40000
for name, p in [("条件成立", params_ok), ("条件違反", params_break)]:
    n_neg, total = euler_count_truncations(
        p["a"], p["b"], p["sigma"], p["r0"], T2, ns2, npth2, seed=999)
    print(f"{name}: 負値の発生 {n_neg:>8d} / {total} ステップ "
          f"({100*n_neg/total:.3f}%)")

# %% [markdown]
# 条件違反パラメータでは、離散化が頻繁に負値を踏む（full-truncation が無ければ
# $\sqrt{r}$ が NaN になる）。次に、条件違反下での満期分布 $r_T$ を厳密サンプリング
# と離散化で重ね、離散化が原点付近を歪めるさまを見る。

# %%
pb = params_break
mb = CIR(pb["a"], pb["b"], pb["sigma"], pb["r0"])

_, ex_paths = cir_exact_ncx2(pb["a"], pb["b"], pb["sigma"], pb["r0"],
                             T2, ns2, npth2, np.random.default_rng(7))
_, eu_paths = cir_euler_ft(pb["a"], pb["b"], pb["sigma"], pb["r0"],
                           T2, ns2, npth2, np.random.default_rng(7))
rT_exact = ex_paths[:, -1]
rT_euler = eu_paths[:, -1]

# 定常ガンマ分布（理論の極限分布）を参考に重ねる。
shape = 2 * pb["a"] * pb["b"] / pb["sigma"] ** 2
scale = pb["sigma"] ** 2 / (2 * pb["a"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
bins = np.linspace(0, max(rT_exact.max(), rT_euler.max()), 80)
axes[0].hist(rT_exact, bins=bins, density=True, alpha=0.6, label="厳密サンプリング")
axes[0].hist(rT_euler, bins=bins, density=True, alpha=0.6, label="離散化 (Euler FT)")
xx = np.linspace(1e-6, bins[-1], 400)
axes[0].plot(xx, gamma.pdf(xx, a=shape, scale=scale), "k--",
             label=f"定常ガンマ (shape={shape:.2f})")
axes[0].set_title(f"条件違反下の r_T 分布（T={T2:.0f}）")
axes[0].set_xlabel("r_T")
axes[0].set_ylabel("密度")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 原点近傍を拡大。条件違反では厳密解の密度が原点で立ち上がる。
mask = bins <= 0.03
axes[1].hist(rT_exact, bins=bins, density=True, alpha=0.6, label="厳密サンプリング")
axes[1].hist(rT_euler, bins=bins, density=True, alpha=0.6, label="離散化 (Euler FT)")
axes[1].plot(xx, gamma.pdf(xx, a=shape, scale=scale), "k--", label="定常ガンマ")
axes[1].set_xlim(0, 0.03)
axes[1].set_title("原点近傍の拡大")
axes[1].set_xlabel("r_T")
axes[1].set_ylabel("密度")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"厳密サンプリングの r_T=0 到達率（<1e-8）: "
      f"{100*np.mean(rT_exact < 1e-8):.3f}%")
print(f"離散化の r_T=0 到達率（切り詰めで 0 固定）: "
      f"{100*np.mean(rT_euler < 1e-8):.3f}%")

# %% [markdown]
# 離散化は $0$ に切り詰めた質量を原点ちょうどに積み上げるため、原点近傍の
# 分布形状が厳密サンプリングと乖離する。厳密サンプリングは非中心カイ二乗分布に
# 忠実で、$0$ 付近で連続的に密度が立ち上がる。割引債価格の推定値にもこの差が
# 効くので、条件違反領域では厳密サンプリングを使うのが安全である。

# %%
# 条件違反下での割引債価格：解析解と両サンプリングを比較。
tau_b = 5.0
p_ana_b = float(cir_zcb_scratch(pb["a"], pb["b"], pb["sigma"], pb["r0"], tau_b))
e_ex, s_ex = mc_zcb(ex_paths, np.linspace(0, T2, ns2 + 1), tau_b)
e_eu, s_eu = mc_zcb(eu_paths, np.linspace(0, T2, ns2 + 1), tau_b)
print(f"条件違反・満期 {tau_b:.0f} 年の割引債価格")
print(f"解析解            : {p_ana_b:.6f}")
print(f"厳密 MC           : {e_ex:.6f} ± {s_ex:.6f}  "
      f"({(e_ex-p_ana_b)/s_ex:+.2f}σ)")
print(f"離散化 MC         : {e_eu:.6f} ± {s_eu:.6f}  "
      f"({(e_eu-p_ana_b)/s_eu:+.2f}σ)")
print("離散化はバイアスを持ちうる（原点切り詰めの影響）。厳密解を基準に評価する。")

# %% [markdown]
# ## 演習
#
# 1. **フェラー条件と原点近傍の分布形状**：$a=0.5,\ b=0.04,\ r_0=0.04$ を固定し、
#    $\sigma$ を条件成立側（例 $\sigma=0.10$、$2ab/\sigma^2>1$）と条件違反側
#    （例 $\sigma=0.30$）で変えて、厳密サンプリングで満期 $T=10$ の $r_T$ 分布を
#    描け。原点近傍（$r_T < 0.01$）のヒストグラムを比較し、条件違反側で密度が
#    原点に張り付くことを、定常ガンマ分布の shape パラメータ $2ab/\sigma^2$ の
#    値とともに説明せよ。
# 2. **離散化の負値発生と厳密サンプリングの差**：演習1の条件違反パラメータで、
#    離散化（Euler full-truncation）と厳密サンプリングの $r_T$ 分布を重ね、
#    時間刻み $dt$ を $0.1,\ 0.02,\ 0.004$ 年と細かくしたとき、負値（切り詰め）の
#    発生率と割引債価格の推定バイアスがどう変わるかを表にまとめよ。細かくすれば
#    離散化バイアスは縮むが、厳密サンプリングは $dt$ に依存せず正しい値を返す
#    ことを確認せよ。
#
# 解答例は `solutions/S5/sol_0502.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/05_rate_models.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | [平方根過程](../../glossary/05_rate_models.md#square-root-process) | square-root process | 拡散項が $\sigma\sqrt{r}$ で水準依存し、金利を非負に保つ確率過程 |
# | [フェラー条件](../../glossary/05_rate_models.md#feller-condition) | Feller condition | $2ab\ge\sigma^2$。成立時に $r_t$ が原点に触れず厳密に正 |
# | [非中心カイ二乗分布](../../glossary/05_rate_models.md#noncentral-chi-squared-distribution) | noncentral chi-squared distribution | CIR の遷移分布。自由度 $4ab/\sigma^2$・非心度 $r_te^{-a\Delta}/c$ |
# | [厳密サンプリング](../../glossary/05_rate_models.md#exact-sampling) | exact sampling | 遷移分布から直接乱数を引き、離散化誤差ゼロでパスを生成する方法 |
