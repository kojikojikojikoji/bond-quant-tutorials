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
# # S2-3 Nelson-Siegel / Svensson パラメトリックフィット
#
# ## 学習目標
#
# - Nelson-Siegel（NS）と Svensson（NSS）のゼロレート関数を書き下し、
#   3つの因子（レベル・スロープ・曲率）に対応する因子負荷を説明できる
# - 因子負荷の偏微分と、$\tau \to 0$・$\tau \to \infty$ の極限から、各 $\beta$ が
#   カーブのどの部分を動かすのかを導ける
# - 中央銀行（ECB・FRB・BoJ）がこのモデルをどう使っているかを把握する
# - $\lambda$（時定数）を含む最小二乗フィットが不安定になる理由を、多重共線性の
#   言葉で説明できる
# - $\lambda$ を格子探索し、線形部分を最小二乗で解く2段階推定を自分で実装し、
#   `bondlab.curve.nss` / `fit_nss` と一致することを確認できる
# - 既知パラメータから生成した曲線をフィットで復元し（round trip）、実データの
#   パネルから日次の因子時系列を推定・解釈できる

# %% [markdown]
# ## 理論
#
# ### なぜパラメトリックか
#
# S2-1・S2-2 のブートストラップやスプラインは、観測点を（ほぼ）そのまま通す
# ノンパラメトリックな手法でした。対してパラメトリック法は、カーブ全体を
# 少数のパラメータを持つ1本の滑らかな関数で表します。観測点を厳密には通さない
# 代わりに、次の利点があります。
#
# - パラメータ数が固定（NS は4個、NSS は6個）なので、日々のカーブを同じ次元の
#   ベクトルとして比較・時系列化できる
# - 各パラメータがレベル・スロープ・曲率という経済的な意味を持ち、解釈しやすい
# - 少数パラメータゆえノイズに頑健で、外挿もなだらか
#
# この「解釈できる少数パラメータ」という性質が、後述するように多くの中央銀行に
# 採用される理由です。
#
# ### Nelson-Siegel の関数形
#
# Nelson と Siegel（1987）は、瞬間フォワードレートを「定数＋指数関数＋
# 指数×多項式」で表しました。これを積分してゼロレート（連続複利）にすると、
# 年限 $\tau>0$ に対して次の3項和になります。
#
# $$
# z(\tau) = \beta_0
#   + \beta_1 \underbrace{\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}}_{L_1(\tau)}
#   + \beta_2 \underbrace{\left(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}
#       - e^{-\tau/\lambda}\right)}_{L_2(\tau)}
# $$
#
# ここで $L_0(\tau) \equiv 1$、$L_1$、$L_2$ を**因子負荷（factor loading）**と
# 呼びます。$\beta_0,\beta_1,\beta_2$ は線形係数（因子）、$\lambda>0$ は
# 因子負荷の減衰の速さを決める時定数です。$\lambda$ が小さいほど負荷は短期側で
# 速く減衰し、大きいほど長期まで効きます。
#
# ### 3因子の意味：極限と偏微分
#
# ゼロレートを各因子で偏微分すると、その因子の負荷そのものになります。
#
# $$
# \frac{\partial z}{\partial \beta_0} = L_0(\tau) = 1, \quad
# \frac{\partial z}{\partial \beta_1} = L_1(\tau), \quad
# \frac{\partial z}{\partial \beta_2} = L_2(\tau).
# $$
#
# つまり因子負荷は「その因子を1動かしたときにカーブがどう動くか」を年限ごとに
# 表した感応度です。負荷の極限を取ると、各 $\beta$ の役割が確定します。
# $x = \tau/\lambda$ と置くと $(1-e^{-x})/x \to 1$（$x\to0$）、$\to 0$（$x\to\infty$）
# なので、
#
# | 因子 | 負荷 | $\tau \to 0$ | $\tau \to \infty$ | 形 | 役割 |
# |---|---|---|---|---|---|
# | $\beta_0$ | $L_0 = 1$ | 1 | 1 | 定数 | **レベル（長期水準）** |
# | $\beta_1$ | $L_1$ | 1 | 0 | 単調減少 | **スロープ（短期側の傾き）** |
# | $\beta_2$ | $L_2$ | 0 | 0 | 中期に山（hump） | **曲率（中期のふくらみ）** |
#
# ここから2つの端点が読めます。長期端では $L_1,L_2 \to 0$ なので
#
# $$
# \lim_{\tau\to\infty} z(\tau) = \beta_0,
# $$
#
# すなわち $\beta_0$ は**長期ゼロレートの水準**です。短期端（瞬間金利）では
# $L_1\to1,\ L_2\to0$ なので
#
# $$
# \lim_{\tau\to 0} z(\tau) = \beta_0 + \beta_1,
# $$
#
# すなわち $\beta_1$ は**短期金利と長期水準の差の符号反転**（$-$スロープ）です。
# カーブの傾き（長期$-$短期）は $-\beta_1$ になります。$\beta_2$ の負荷は両端で
# ゼロ、中間で山（または谷）を作るので、$\beta_2$ は**中期のふくらみ（曲率）**を
# 動かします。山の位置は $\lambda$ が決めます。
#
# ### Svensson 拡張（NSS）
#
# Svensson（1994）は、2つ目の曲率項を足して山を2つ許し、より複雑な形
# （例えば逆イールドと中期の凹凸が同居するカーブ）を表せるようにしました。
#
# $$
# z(\tau) = \beta_0 + \beta_1 L_1(\tau;\lambda_1)
#   + \beta_2 L_2(\tau;\lambda_1)
#   + \beta_3 \underbrace{\left(\frac{1 - e^{-\tau/\lambda_2}}{\tau/\lambda_2}
#       - e^{-\tau/\lambda_2}\right)}_{L_2(\tau;\lambda_2)}
# $$
#
# 追加された $\beta_3$ は2つ目の曲率因子、$\lambda_2$ はその山の位置を決める
# 2つ目の時定数です。`bondlab.curve.nss` はこの6パラメータ版を実装しています。
#
# ### 中央銀行での採用
#
# このモデルは学術的な曲線当てはめにとどまらず、主要中央銀行が公表カーブの
# 推定に使っています。
#
# - **ECB（欧州中央銀行）**：ユーロ圏各国国債のゼロクーポン利回り曲線を
#   Svensson モデルで日次推定・公表しています（AAA 格付け国債とユーロ圏全体の
#   2系列）。
# - **FRB（米連邦準備制度）**：Gürkaynak, Sack, Wright（2007）による米国債の
#   ゼロクーポン曲線（いわゆる GSW データセット）は Svensson モデルで推定され、
#   Fed のスタッフ研究として日次更新・公開されています。
# - **BoJ（日本銀行）ほか**：BIS のサーベイ（BIS Papers No.25）によれば、
#   ベルギー・フィンランド・フランス・ドイツ・イタリア・スペイン・スイス等は
#   NS/Svensson を採用する一方、日本・英国・カナダ・米国（当局公表値）は
#   平滑化スプラインを主に使う、と整理されています。したがって NSS は
#   「多くの当局の標準」であると同時に、スプライン系との比較ベンチマークとしても
#   広く参照されます。
#
# 要点は、少数の解釈可能なパラメータで各国・各日を横並び比較できることが、
# 政策・市場分析の実務で重宝される、という点です。
#
# ### 最小二乗フィットの落とし穴：$\lambda$ の多重共線性
#
# パラメータを利回りに当てはめる素直な方法は、残差二乗和を最小化する
# **非線形最小二乗（nonlinear least squares）**です。
#
# $$
# \min_{\beta_0,\dots,\beta_3,\ \lambda_1,\lambda_2}
#   \sum_i \bigl(z(\tau_i) - y_i\bigr)^2.
# $$
#
# ところがこれを6変数まとめて最適化すると、しばしば不安定になります。原因は
# $\lambda_1$ と $\lambda_2$ の**多重共線性（multicollinearity）**です。
#
# - 2つの時定数が近い（$\lambda_1 \approx \lambda_2$）と、2つの曲率負荷
#   $L_2(\tau;\lambda_1)$ と $L_2(\tau;\lambda_2)$ がほぼ同じ形になります。すると
#   計画行列の2列がほぼ平行になり、$X^\top X$ が特異に近づきます。結果、
#   $\beta_2,\beta_3$ が絶対値の大きい逆符号のペアに発散し、**曲線の当てはまりは
#   良いのに係数が意味不明**という状態になります。
# - $\lambda$ は関数の中に非線形に入るため、目的関数を $\lambda$ の関数として
#   見ると、浅い谷や平坦な領域を多数持ちます。勾配法は初期値に強く依存し、
#   別の局所解へ落ちます。つまり **$\lambda$ の推定は本質的に不安定**です。
#
# ### 2段階推定で回避する
#
# この不安定性は、パラメータを2種類に分けると避けられます。$\lambda$ を止めると、
# モデルは $\beta$ について**線形**になるからです。
#
# 1. **格子探索**：$\lambda_1,\lambda_2$ を有限個の候補（格子）に固定する。
# 2. **線形部分の OLS**：各 $(\lambda_1,\lambda_2)$ について、因子負荷を並べた
#    計画行列 $X$ を作り、$\hat\beta = (X^\top X)^{-1}X^\top y$ を閉形式で解く。
# 3. 全格子の中で残差二乗和が最小の組を選ぶ。
#
# これは「$\beta$ を消去して $\lambda$ だけの1次元（NS）／2次元（NSS）問題に
# する」概形最小二乗（profile / concentrated least squares）です。不安定な非線形
# 最適化を、頑健な格子探索＋安定な線形解に置き換えるのが要点です。
# `bondlab.curve.fit_nss` もこの2段階法を採っています（内部では線形部分を
# `least_squares` で解いていますが、線形問題なので OLS の閉形式解と一致します）。

# %% [markdown]
# ## スクラッチ実装
#
# 因子負荷・NSS ゼロレート関数・2段階推定（$\lambda$ 格子＋線形 OLS）を自作し、
# `bondlab.curve.nss` / `fit_nss` と一致することを確認します。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `ns_loadings(tau, lam)` | 年限, 時定数 | `(L0, L1, L2)` | 3つの因子負荷（レベル・スロープ・曲率）を返す |
# | `nss_zero(tau, b0,b1,b2,b3, lam1, lam2)` | 年限, 4係数, 2時定数 | ゼロレート | NSS のゼロレート関数（`bondlab.curve.nss` の再実装） |
# | `design_matrix(tenors, lam1, lam2)` | 年限列, 2時定数 | 計画行列 $X$ ($n\times4$) | 各列が因子負荷。線形回帰の入力 |
# | `fit_beta_ols(tenors, yields, lam1, lam2)` | 年限, 利回り, 2時定数 | `(beta, 残差二乗和)` | $\lambda$ 固定下で $\beta$ を OLS で解く（第2段階） |
# | `fit_nss_2stage(tenors, yields, lam_grid)` | 年限, 利回り, 格子 | dict(beta0..beta3, lam1, lam2) | 格子探索＋OLS の2段階推定（`fit_nss` の再実装） |

# %%
import numpy as np

import bondlab
from bondlab import curve as blcurve

np.random.seed(0)
print("bondlab version:", bondlab.__version__)


def ns_loadings(tau, lam):
    """NS の3因子負荷 (L0, L1, L2) を返す。

    L0=1（レベル）、L1=(1-e^-x)/x（スロープ, x=tau/lam）、
    L2=L1-e^-x（曲率）。tau はスカラ/配列いずれも可。
    """
    tau = np.asarray(tau, dtype=float)
    x = tau / lam
    L0 = np.ones_like(tau)
    L1 = (1.0 - np.exp(-x)) / x
    L2 = L1 - np.exp(-x)
    return L0, L1, L2


def nss_zero(tau, b0, b1, b2, b3, lam1, lam2):
    """NSS ゼロレート。lam1 側にスロープ+曲率、lam2 側に第2曲率。"""
    _, L1, L2 = ns_loadings(tau, lam1)
    _, _, L2b = ns_loadings(tau, lam2)
    return b0 + b1 * L1 + b2 * L2 + b3 * L2b


# %% [markdown]
# 自作の `nss_zero` が `bondlab.curve.nss` と一致することを、まず1点で確認します。

# %%
tenors = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
p = dict(b0=0.045, b1=-0.020, b2=0.030, b3=-0.015, lam1=1.5, lam2=5.0)

z_scratch = nss_zero(tenors, **p)
z_bondlab = blcurve.nss(tenors, p["b0"], p["b1"], p["b2"], p["b3"], p["lam1"], p["lam2"])
print("最大差:", np.max(np.abs(z_scratch - z_bondlab)))
assert np.allclose(z_scratch, z_bondlab, atol=1e-14)
print("nss_zero は bondlab.curve.nss と一致")

# %% [markdown]
# 次に2段階推定を実装します。$\lambda$ を固定すると、モデルは
# $y = X\beta$（$X$ は因子負荷を並べた計画行列）という線形回帰になり、
# $\beta$ は正規方程式 $X^\top X \beta = X^\top y$ の閉形式解で求まります。

# %%
def design_matrix(tenors, lam1, lam2):
    """各列が因子負荷 [L0, L1(lam1), L2(lam1), L2(lam2)] の計画行列。"""
    _, L1, L2 = ns_loadings(tenors, lam1)
    _, _, L2b = ns_loadings(tenors, lam2)
    L0 = np.ones_like(np.asarray(tenors, dtype=float))
    return np.column_stack([L0, L1, L2, L2b])


def fit_beta_ols(tenors, yields, lam1, lam2):
    """lambda 固定下で beta を最小二乗（OLS）で解き、(beta, SSR) を返す。"""
    X = design_matrix(tenors, lam1, lam2)
    beta, *_ = np.linalg.lstsq(X, np.asarray(yields, dtype=float), rcond=None)
    resid = X @ beta - yields
    return beta, float(resid @ resid)


def fit_nss_2stage(tenors, yields, lam_grid=None):
    """格子探索＋線形 OLS の2段階推定。bondlab.curve.fit_nss と同じ規約。

    lam_grid の全 (lam1, lam2) 組（lam2 > lam1）について beta を OLS で解き、
    残差二乗和が最小の組を選ぶ。
    """
    tenors = np.asarray(tenors, dtype=float)
    yields = np.asarray(yields, dtype=float)
    if lam_grid is None:
        lam_grid = np.linspace(0.5, 10.0, 20)

    best = None
    for lam1 in lam_grid:
        for lam2 in lam_grid:
            if lam2 <= lam1:
                continue
            beta, ssr = fit_beta_ols(tenors, yields, lam1, lam2)
            if best is None or ssr < best[0]:
                best = (ssr, beta, lam1, lam2)

    _, beta, lam1, lam2 = best
    return dict(beta0=beta[0], beta1=beta[1], beta2=beta[2], beta3=beta[3],
                lam1=lam1, lam2=lam2)


# %% [markdown]
# 既知パラメータ `p` から生成した利回りに2段階推定を当て、`bondlab.curve.fit_nss`
# と係数レベルで一致することを確認します。既定格子 `linspace(0.5,10,20)` は
# 真の $\lambda=(1.5,5.0)$ を格子点に含むため、係数まで厳密に復元できます。

# %%
y_true = nss_zero(tenors, **p)

fit_scratch = fit_nss_2stage(tenors, y_true)
fit_bondlab = blcurve.fit_nss(tenors, y_true)

print(f"{'param':7s} {'true':>10s} {'scratch':>12s} {'bondlab':>12s}")
name_map = [("beta0", "b0"), ("beta1", "b1"), ("beta2", "b2"), ("beta3", "b3"),
            ("lam1", "lam1"), ("lam2", "lam2")]
for k, tk in name_map:
    print(f"{k:7s} {p[tk]:>10.5f} {fit_scratch[k]:>12.6f} {fit_bondlab[k]:>12.6f}")
    assert abs(fit_scratch[k] - fit_bondlab[k]) < 1e-6

print("\nfit_nss_2stage は bondlab.curve.fit_nss と一致")

# %% [markdown]
# ### 因子負荷の可視化
#
# 3つの因子負荷を年限に対して描くと、理論表のとおりレベルは水平、スロープは
# 短期から単調減衰、曲率は中期に山を作る様子が見えます。

# %%
import matplotlib.pyplot as plt

taus = np.linspace(0.05, 30.0, 300)
lam = 2.0
L0, L1, L2 = ns_loadings(taus, lam)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(taus, L0, label=r"$L_0=1$（レベル $\beta_0$）")
ax.plot(taus, L1, label=r"$L_1$（スロープ $\beta_1$）")
ax.plot(taus, L2, label=r"$L_2$（曲率 $\beta_2$）")
ax.axhline(0.0, color="gray", lw=0.6)
ax.set_xlabel("年限 τ（年）")
ax.set_ylabel("因子負荷")
ax.set_title(f"Nelson-Siegel の因子負荷（λ={lam}）")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# レベル負荷はすべての年限で1（全体を平行移動）、スロープ負荷は短期で最大・
# 長期で0（短期側だけを持ち上げ／押し下げ）、曲率負荷は両端で0・中期で最大
# （中期のふくらみ）です。山の位置は $\lambda$ で動きます。

# %% [markdown]
# ## QuantLib検証
#
# QuantLib には Nelson-Siegel / Svensson のフィッティング API が標準では
# 用意されていない（`FittedBondDiscountCurve` の当てはめ手法として NSS は
# 含まれない）ため、ここでは**解析 round-trip で検証**します。すなわち
# 「既知パラメータ → 曲線生成 → フィットで係数・曲線を復元」が成り立つことを、
# 係数レベルと曲線レベルの両方で確認します。
#
# ### round trip 1：係数の完全復元（格子点に真値がある場合）
#
# 真の $\lambda$ が格子点に一致するときは、係数まで機械精度で戻ります。

# %%
fit_rt = fit_nss_2stage(tenors, y_true)
z_rt = nss_zero(tenors, fit_rt["beta0"], fit_rt["beta1"], fit_rt["beta2"],
                fit_rt["beta3"], fit_rt["lam1"], fit_rt["lam2"])

print("係数の復元誤差:")
for k, tk in name_map:
    print(f"  {k:6s}: {abs(fit_rt[k] - p[tk]):.2e}")
print("曲線の最大復元誤差:", np.max(np.abs(z_rt - y_true)))
assert np.max(np.abs(z_rt - y_true)) < 1e-8

# %% [markdown]
# ### round trip 2：曲線の復元（格子点に真値が無い場合）
#
# 真の $\lambda$ が格子から外れていると、選ばれる $\lambda$ は最も近い格子点に
# 丸められ、係数は完全一致しません。しかし**曲線（当てはめ値）は高精度で復元**
# されます。これは前述の多重共線性の裏返しで、「係数は動くが曲線は動かない」
# 方向が存在するためです。フィットの目的は曲線の再現なので、これで十分機能します。

# %%
p2 = dict(b0=0.042, b1=-0.018, b2=0.025, b3=-0.010, lam1=1.3, lam2=4.4)  # 格子外
y_true2 = nss_zero(tenors, **p2)
fit2 = fit_nss_2stage(tenors, y_true2)  # 既定格子（真値を含まない）
z_fit2 = nss_zero(tenors, fit2["beta0"], fit2["beta1"], fit2["beta2"],
                  fit2["beta3"], fit2["lam1"], fit2["lam2"])

print("選ばれた λ:", round(fit2["lam1"], 3), round(fit2["lam2"], 3),
      " (真値 1.3, 4.4)")
print("曲線の最大復元誤差:", np.max(np.abs(z_fit2 - y_true2)))
assert np.max(np.abs(z_fit2 - y_true2)) < 5e-4  # 係数はずれても曲線は復元

# %% [markdown]
# ## 実データ適用
#
# `data/samples/synthetic_ust_par_panel.csv`（列 `date, tenor, par_yield`）は、
# 米国債パー利回りの日次パネル（60営業日 × 9年限、合成データ）です。各営業日の
# 断面に NSS を当て、レベル・スロープ・曲率の因子を日次で推定して時系列化します。
# ネットワークは使わず、同梱 CSV のみを読みます。
#
# > 補足：本来のカーブ構築では、パー利回りはいったんゼロレートへ剥ぎ取って
# > （S2-1）から当てるのが厳密です。ここでは NSS の当てはめと因子の時系列化に
# > 焦点を当てるため、パー利回りを直接ターゲットにします（相対的な因子変動の
# > 解釈には十分です）。

# %%
import pandas as pd

panel = pd.read_csv("data/samples/synthetic_ust_par_panel.csv")
wide = panel.pivot(index="date", columns="tenor", values="par_yield").sort_index()
panel_tenors = wide.columns.values.astype(float)
print("パネル形状:", wide.shape, "（営業日 × 年限）")
print("年限:", panel_tenors)
print(wide.iloc[:3].round(5))

# %% [markdown]
# パネル全体を日次で回します。60日 × 20×20 格子は無駄に重いので、格子は
# `linspace(0.8, 8, 8)` の粗いものを使い、全体を数秒に収めます。各日の推定から、
# 解釈しやすい3つの量を作ります。
#
# - **レベル** $= \beta_0$（長期水準）
# - **スロープ** $= -\beta_1$（長期$-$短期の傾き。正なら順イールド）
# - **曲率** $= \beta_2$（中期のふくらみ）

# %%
coarse_grid = np.linspace(0.8, 8.0, 8)

records = []
for date, row in wide.iterrows():
    f = fit_nss_2stage(panel_tenors, row.values, lam_grid=coarse_grid)
    records.append({
        "date": date,
        "レベル(β0)": f["beta0"],
        "スロープ(-β1)": -f["beta1"],
        "曲率(β2)": f["beta2"],
        "lam1": f["lam1"],
        "lam2": f["lam2"],
    })

betas = pd.DataFrame(records).set_index("date")
betas.index = pd.to_datetime(betas.index)
print(betas.head().round(5))
print("...")
print(betas.tail().round(5))

# %% [markdown]
# フィット品質を1点で確認します。ある日の当てはめ曲線を観測パー利回りに
# 重ねると、9点をよく通ることが見えます。

# %%
sample_date = wide.index[0]
frow = fit_nss_2stage(panel_tenors, wide.loc[sample_date].values, lam_grid=coarse_grid)
grid_tau = np.linspace(0.25, 30.0, 200)
z_curve = blcurve.nss(grid_tau, frow["beta0"], frow["beta1"], frow["beta2"],
                      frow["beta3"], frow["lam1"], frow["lam2"])

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(panel_tenors, wide.loc[sample_date].values * 100, color="k",
           zorder=5, label="観測パー利回り")
ax.plot(grid_tau, z_curve * 100, label="NSS フィット")
ax.set_xlabel("年限（年）")
ax.set_ylabel("利回り (%)")
ax.set_title(f"NSS フィット（{sample_date}）")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# 因子の日次時系列をプロットします。3つの因子が別々のリズムで動くことが
# 見えます。

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(betas.index, betas["レベル(β0)"] * 100, color="C0")
axes[0].set_ylabel("レベル (%)")
axes[0].set_title("NSS 因子の日次時系列")
axes[1].plot(betas.index, betas["スロープ(-β1)"] * 100, color="C1")
axes[1].set_ylabel("スロープ (%)")
axes[2].plot(betas.index, betas["曲率(β2)"] * 100, color="C2")
axes[2].set_ylabel("曲率 (%)")
axes[2].set_xlabel("日付")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# レベル・スロープ・曲率の推定量が、観測利回りから直接作る代理量と整合するかを
# 確認します。レベルは30年、スロープは30年$-$0.5年、曲率は中期の突出
# （2×5年 $-$ 0.5年 $-$ 30年）で近似できます。相関が高ければ、因子が意図した
# カーブの動きを捉えているといえます。

# %%
proxy = pd.DataFrame({
    "レベル代理": wide[30.0].values,
    "スロープ代理": (wide[30.0] - wide[0.5]).values,
    "曲率代理": (2 * wide[5.0] - wide[0.5] - wide[30.0]).values,
}, index=betas.index)

corr = pd.DataFrame({
    "レベル": [np.corrcoef(betas["レベル(β0)"], proxy["レベル代理"])[0, 1]],
    "スロープ": [np.corrcoef(betas["スロープ(-β1)"], proxy["スロープ代理"])[0, 1]],
    "曲率": [np.corrcoef(betas["曲率(β2)"], proxy["曲率代理"])[0, 1]],
}, index=["因子と代理量の相関"])
print(corr.round(4).to_string())

# %% [markdown]
# レベルとスロープは代理量とほぼ完全に連動し（相関 ≈ 1）、因子が素直に
# カーブの水準・傾きを表していることが確認できます。曲率は代理量の取り方に
# 依存するぶん相関はやや緩みますが、符号と動きは対応します。

# %% [markdown]
# ## 演習
#
# 1. **格子の粗さと復元精度**：既知パラメータ（本編の `p`）から曲線を生成し、
#    `fit_nss_2stage` の $\lambda$ 格子を「粗い→細かい」と変えて（例：点数
#    5, 8, 12, 20, 40）フィットする。各格子について「係数の最大復元誤差」と
#    「曲線の最大復元誤差」を表にまとめ、格子を細かくすると何が改善し、何が
#    頭打ちになるかを述べよ。ヒント：曲線誤差は早く小さくなるが、係数誤差は
#    真値が格子点に載るかどうかで階段状に動く。
# 2. **日次因子の解釈**：実データパネルからレベル・スロープ・曲率の日次時系列を
#    推定してプロットし、期間中に最もスティープ化（スロープ拡大）した局面と
#    フラット化した局面を特定せよ。あわせて、その動きが主に短期側・長期側の
#    どちらの利回り変化によるものかを、観測パネルの端点（0.5年・30年）の推移と
#    照合して説明せよ。
#
# 解答例は `solutions/S2/sol_0203.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/02_curves.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | Nelson-Siegel | Nelson-Siegel | レベル・スロープ・曲率の3因子でゼロ曲線を表す4パラメータの関数形 |
# | Svensson | Svensson | NS に第2曲率項を足し山を2つ許した6パラメータ拡張（NSS） |
# | 因子負荷 | factor loading | 各因子を1動かしたときのカーブの年限別感応度。ゼロレートの偏微分 |
# | 非線形最小二乗 | nonlinear least squares | パラメータが非線形に入る残差二乗和の最小化。局所解・初期値依存に注意 |
# | 多重共線性 | multicollinearity | 計画行列の列がほぼ平行で係数推定が不安定になる状態。λ が近いと発生 |
