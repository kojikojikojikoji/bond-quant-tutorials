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
# # S5-1 Vasicekモデル
#
# ## 学習目標
#
# - Vasicek の短期金利 SDE を、平均回帰の3つのパラメータ（回帰速度・長期水準・ボラティリティ）で読める
# - ゼロクーポン債（ZCB）価格の解析解を、アフィンモデルの一般論からゼロベースで導出できる
# - 短期金利が正規分布に従うことと、その帰結である負金利の許容を説明できる
# - 負金利が日本市場では欠点にならなかった理由を、モデル選択の観点から述べられる
# - リスクの市場価格（S4-5）を実モデルに初めて適用し、実測度パラメータからリスク中立パラメータへ変換できる
# - ZCB 解析解・イールドカーブ生成・モンテカルロ（MC）価格を自分で実装し、`bondlab` と QuantLib に突き合わせられる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# 銀行の金利デスクでは、コーラブル債やスワップション、キャンセラブル・スワップといった「金利の将来経路に依存する商品」を値付けし、そのリスクをヘッジすることが日々の仕事です。こうした商品は満期までの割引債価格 $P(t,T)$ の分布が分からないと評価できません。Vasicek の平均回帰・アフィン構造は、短期金利から割引債・イールドカーブ・オプション価値までを閉じた式でつなぐ最小の枠組みで、クオンツが実装する短期金利モデル（S5-3 の Hull-White を含む）の土台になります。ここで組んだ ZCB 解析解とモンテカルロ、`bondlab`・QuantLib への突き合わせは、そのまま本番の評価ライブラリの検算手順です。
#
# 収益への効き方は二段構えです。第一に、値付けの精度がマーケットメイクのスプレッド収益（`docs/債券ファンドの業務.md` の収益源1）を左右します。クォートが甘ければ不利な約定で在庫リスクだけを抱え、辛ければ約定が取れません。第二に、$r_t$ が正規分布に従い負金利を許すという性質は、日本や欧州のマイナス金利局面でモデル選択を分ける決定打になりました。金利が下限にぶつからない前提を置けたかどうかが、フロア性を持つ商品の値付け・ヘッジの当否に直結します。
#
# リスクの市場価格を通じた実測度パラメータ（$\mathbb{P}$）からリスク中立パラメータ（$\mathbb{Q}$）への変換は、ヒストリカルデータでの当てはめと市場整合の評価をつなぐ蝶番です。モデルバリデーション（S5-5、FRB SR 11-7）はこの点を独立に検証します。正規分布・負金利許容という仮定が対象商品に妥当か、$\mathbb{P}\to\mathbb{Q}$ の変換が恣意的でないか、解析解が数値解・ベンチマーク（QuantLib）と一致するかを、開発担当とは別の担当が反証的に確かめ、承認して初めて本番評価系に載ります。この検証が甘いと、モデル要因の評価誤差がそのまま日々の損益とリスク指標を汚染します。
# %% [markdown]
# ## 理論
#
# ### 1. Vasicek の SDE
#
# Vasicek モデル（Vasicek model）は、瞬間短期金利 $r_t$ が単一の確率微分方程式に従うと仮定します。
# リスク中立測度 $\mathbb{Q}$ の下で
#
# $$
# dr_t = a\,(b - r_t)\,dt + \sigma\, dW_t^{\mathbb{Q}} .
# $$
#
# 各パラメータの役割は次の通りです。
#
# | 記号 | 名称 | 役割 |
# |---|---|---|
# | $a>0$ | 平均回帰速度（mean-reversion speed） | $r_t$ が長期水準へ引き戻される強さ |
# | $b$ | 長期水準（long-run level） | 引き戻し先の中心。$r_t$ の期待到達点 |
# | $\sigma>0$ | ボラティリティ（volatility） | ランダムな揺さぶりの大きさ |
#
# ドリフト項 $a(b-r_t)$ が**平均回帰（mean reversion）**を表します。$r_t>b$ なら負のドリフトで下向き、
# $r_t<b$ なら正のドリフトで上向きに、常に $b$ へ引き戻す力が働きます。$a$ が大きいほど戻りが速く、
# $r_t$ は $b$ の近くに留まります。株価の幾何ブラウン運動（ドリフトが水準に比例して発散的）と違い、
# 金利は「ある水準の周りをうろつく」性質を持つため、この定式化が自然です。
#
# ### 2. 短期金利の分布（正規）
#
# SDE は線形なので解析的に解けます。伊藤の補題を $e^{at} r_t$ に適用すると
#
# $$
# d\!\left(e^{at} r_t\right) = a b\, e^{at}\,dt + \sigma e^{at}\, dW_t^{\mathbb{Q}}
# $$
#
# となり、$0$ から $t$ まで積分して
#
# $$
# r_t = r_0 e^{-at} + b\left(1 - e^{-at}\right) + \sigma \int_0^t e^{-a(t-s)}\, dW_s^{\mathbb{Q}} .
# $$
#
# 右辺の確率積分は、被積分関数が確定的なので正規分布に従います。したがって $r_t$ は正規分布です。
#
# $$
# r_t \sim \mathcal{N}\!\left(\; r_0 e^{-at} + b(1 - e^{-at}),\;\; \frac{\sigma^2}{2a}\left(1 - e^{-2at}\right)\right).
# $$
#
# 期待値は $t\to\infty$ で $b$ に、分散は $\dfrac{\sigma^2}{2a}$ に収束します。つまり長時間後の $r_t$ は
# 定常分布 $\mathcal{N}\!\left(b,\, \dfrac{\sigma^2}{2a}\right)$ に落ち着きます。平均回帰があるおかげで分散が
# 発散せず有限に留まる点が、ブラウン運動との決定的な違いです。
#
# ### 3. ZCB 価格の解析解（アフィンモデルの導出）
#
# 満期 $T$ の割引債価格は、リスク中立期待値
#
# $$
# P(t,T) = \mathbb{E}^{\mathbb{Q}}\!\left[\left.\exp\!\left(-\int_t^T r_s\, ds\right)\right| r_t \right]
# $$
#
# で与えられます。これを直接積分するのは難しいので、**アフィンモデル（affine model）**の枠組みで解きます。
# ファインマン–カッツの定理により、$P(t,T)=P(t,r_t)$ は次の偏微分方程式（PDE）を満たします。
#
# $$
# \frac{\partial P}{\partial t} + a(b-r)\frac{\partial P}{\partial r}
# + \tfrac12 \sigma^2 \frac{\partial^2 P}{\partial r^2} - rP = 0,
# \qquad P(T,T)=1 .
# $$
#
# アフィンモデルの要点は、**解が金利について指数アフィン形になる**と仮定することです。残存期間を
# $\tau = T-t$ として
#
# $$
# P(t,T) = \exp\!\big(A(\tau) - B(\tau)\, r\big), \qquad A(0)=0,\; B(0)=0 .
# $$
#
# これを PDE に代入します。$\partial P/\partial t = (-A'(\tau) + B'(\tau) r)P$、
# $\partial P/\partial r = -B(\tau)P$、$\partial^2 P/\partial r^2 = B(\tau)^2 P$ なので、$P$ で割ると
#
# $$
# \big(-A' + B' r\big) - a(b-r)B + \tfrac12 \sigma^2 B^2 - r = 0 .
# $$
#
# これが**全ての $r$ について**成り立つには、$r$ の1次の項と定数項がそれぞれ独立にゼロでなければなりません。
# ここでアフィン形を仮定した効果が効き、PDE が2本の常微分方程式（ODE）に分離します。
#
# $$
# \text{（$r$ の係数）}\quad B'(\tau) = 1 - a B(\tau), \qquad
# \text{（定数項）}\quad A'(\tau) = -a b\, B(\tau) + \tfrac12 \sigma^2 B(\tau)^2 .
# $$
#
# 第1式は1階線形 ODE で、$B(0)=0$ の下に
#
# $$
# B(\tau) = \frac{1 - e^{-a\tau}}{a} .
# $$
#
# これを第2式に代入し $0$ から $\tau$ まで積分すると（$A(0)=0$）
#
# $$
# A(\tau) = \big(B(\tau) - \tau\big)\frac{a^2 b - \tfrac12 \sigma^2}{a^2}
# - \frac{\sigma^2 B(\tau)^2}{4a} .
# $$
#
# よって Vasicek の割引債価格は閉じた形で書けます。
#
# $$
# \boxed{\,P(t,T) = \exp\!\big(A(\tau) - B(\tau)\, r_t\big)\,}
# $$
#
# 数値積分も期待値計算も要らず、パラメータと $r_t$ から一撃で価格が出るのがアフィンモデルの威力です。
# CIR（S5-2）も同じ指数アフィン形で解け、この導出はそのまま一般化します。
#
# ### 4. イールドカーブ
#
# 連続複利のゼロレート（zero rate）は割引債価格から
#
# $$
# y(t,T) = -\frac{\ln P(t,T)}{\tau} = \frac{B(\tau)\, r_t - A(\tau)}{\tau}
# $$
#
# で定まります。$\tau$ を動かせば1本のイールドカーブが得られます。$\tau\to 0$ で $y\to r_t$（現在の短期金利）、
# $\tau\to\infty$ で $y$ は $a,b,\sigma$ だけで決まる長期水準へ収束します。$r_t$ と $b$ の大小関係で、
# 順イールド（右上がり）にも逆イールド（右下がり）にもなります（詳細は後段の実データ適用で数値確認します）。
#
# ### 5. 負金利の許容と、日本市場での位置づけ
#
# $r_t$ は正規分布なので、正の確率で**負の値**を取ります。これは長らく Vasicek の理論的欠点とされ、
# 金利の非負性を保証する CIR（平方根拡散）や対数正規型モデルが好まれた一因でした。
#
# しかし現実の日本市場では、この「欠点」がむしろ利点になりました。日本銀行は 2016 年にマイナス金利政策を
# 導入し、無担保コールレートや短中期の国債利回りが実際にマイナス圏へ沈みました。金利の非負性を**構造的に
# 課してしまう**CIR や対数正規モデルは、この局面を原理的に表現できません。一方 Vasicek は負金利を自然に
# 許容するため、マイナス金利下の日本国債カーブへの当てはめでは、むしろ素直に使えるモデルでした。
# 「モデルの仮定が現実に合うか」は市場と時期に依存する、という実務的教訓です。
#
# ### 6. リスクの市場価格とリスク中立化
#
# ここまでの SDE は測度 $\mathbb{Q}$ の下で書きました。しかし $a,b,\sigma$ を**実データ（実測度 $\mathbb{P}$）**から
# 推定すると、得られるのは $\mathbb{P}$ でのダイナミクスです。S4-5 で導入した**リスクの市場価格
# （market price of risk）**$\lambda$ が、この2つの測度を橋渡しします。Vasicek では $\lambda$ を定数と仮定し、
# 実測度で
#
# $$
# dr_t = a(b - r_t)\,dt + \sigma\, dW_t^{\mathbb{P}}, \qquad
# dW_t^{\mathbb{Q}} = dW_t^{\mathbb{P}} - \lambda\, dt
# $$
#
# とします（$\lambda>0$ は金利リスクを引き受ける投資家が要求するプレミアムに対応）。これを代入すると、
# リスク中立測度でのドリフトが付け替わります。
#
# $$
# dr_t = \big[a(b - r_t) + \lambda\sigma\big]dt + \sigma\, dW_t^{\mathbb{Q}}
#      = a\big(b^{\mathbb{Q}} - r_t\big)dt + \sigma\, dW_t^{\mathbb{Q}},
# \qquad b^{\mathbb{Q}} = b + \frac{\lambda \sigma}{a} .
# $$
#
# 拡散項 $\sigma$ と回帰速度 $a$ は測度を変えても不変で、変わるのは長期水準だけです
# （$b \to b^{\mathbb{Q}} = b + \lambda\sigma/a$）。**価格づけには常にリスク中立パラメータ $b^{\mathbb{Q}}$ を
# 使う**のがポイントです。実測度の $b$ をそのまま ZCB 公式へ入れると、リスクプレミアム分だけ価格を
# 誤ります。S4-5 で抽象的に扱った $\theta=(\mu-r)/\sigma$ 型の測度変換を、金利モデルという実物へ初めて
# 適用する場面です。

# %% [markdown]
# ## スクラッチ実装
#
# Vasicek の ZCB 解析解・イールドカーブ・MC 価格をすべて自作します。QuantLib は後段の検証役です。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `vasicek_B(a, tau)` | 回帰速度, 残存年数 | $B(\tau)$ | ODE 解 $B=(1-e^{-a\tau})/a$ |
# | `vasicek_A(a, b, sigma, tau)` | パラメータ3種, 残存年数 | $A(\tau)$ | ODE 解（指数の中身） |
# | `vasicek_zcb(a, b, sigma, r, tau)` | パラメータ, 金利, 残存年数 | ZCB 価格 | $\exp(A - B r)$ |
# | `vasicek_zero_rate(a, b, sigma, r, tau)` | 同上 | ゼロレート | $-\ln P/\tau$ |
# | `simulate_short_rate(a, b, sigma, r0, T, n_steps, n_paths, seed)` | パラメータ, 満期, ステップ/パス数, seed | (times, paths) | OU 厳密サンプリングで短期金利パス生成 |
# | `vasicek_mc_zcb(a, b, sigma, r0, tau, n_steps, n_paths, seed)` | 同上 | dict（price, stderr, ...） | $\exp(-\int r\,ds)$ の MC 平均で ZCB 価格 |

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.optimize import least_squares

import bondlab
from bondlab.models import Vasicek

print("bondlab version:", bondlab.__version__)


def vasicek_B(a, tau):
    """ODE 解 B(tau) = (1 - exp(-a*tau)) / a。"""
    tau = np.asarray(tau, dtype=float)
    return (1.0 - np.exp(-a * tau)) / a


def vasicek_A(a, b, sigma, tau):
    """ODE 解 A(tau)（P = exp(A - B r) の指数の A 部分）。"""
    tau = np.asarray(tau, dtype=float)
    B = vasicek_B(a, tau)
    return (B - tau) * (a ** 2 * b - 0.5 * sigma ** 2) / a ** 2 - sigma ** 2 * B ** 2 / (4 * a)


def vasicek_zcb(a, b, sigma, r, tau):
    """アフィン解による ZCB 価格 P = exp(A(tau) - B(tau) r)。"""
    return np.exp(vasicek_A(a, b, sigma, tau) - vasicek_B(a, tau) * r)


def vasicek_zero_rate(a, b, sigma, r, tau):
    """連続複利ゼロレート y = -ln P / tau。"""
    tau = np.asarray(tau, dtype=float)
    return -np.log(vasicek_zcb(a, b, sigma, r, tau)) / tau


def simulate_short_rate(a, b, sigma, r0, T, n_steps, n_paths, seed=None):
    """OU 過程の厳密な遷移分布で短期金利パスを生成する。

    r_{t+dt} | r_t ~ Normal(r_t*e^{-a dt} + b(1-e^{-a dt}), σ²/(2a)(1-e^{-2a dt}))。
    Euler 近似と違い、任意の dt で分布が厳密なので離散化誤差が入らない。
    返り値は times shape (n_steps+1,)、paths shape (n_paths, n_steps+1)。
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    mean_rev = np.exp(-a * dt)
    var = sigma ** 2 / (2 * a) * (1 - mean_rev ** 2)
    r = np.full(n_paths, float(r0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = r
    for i in range(n_steps):
        mean = r * mean_rev + b * (1 - mean_rev)
        r = mean + np.sqrt(var) * rng.standard_normal(n_paths)
        out[:, i + 1] = r
    return np.linspace(0.0, T, n_steps + 1), out


def vasicek_mc_zcb(a, b, sigma, r0, tau, n_steps, n_paths, seed=None):
    """∫r ds の割引をシミュレーションして ZCB 価格を MC 推定する。

    各パスで積算金利 ∫_0^tau r_s ds を台形則で近似し、割引 exp(-∫r) の
    標本平均を価格、標本標準偏差/√n を標準誤差とする。
    返り値 dict: price, stderr, ci95, discounts。
    """
    times, paths = simulate_short_rate(a, b, sigma, r0, tau, n_steps, n_paths, seed)
    dt = tau / n_steps
    # 台形則: ∫r ds ≈ dt * (0.5 r_0 + r_1 + ... + r_{n-1} + 0.5 r_n)
    integral = dt * (paths[:, 1:-1].sum(axis=1) + 0.5 * (paths[:, 0] + paths[:, -1]))
    discounts = np.exp(-integral)
    price = discounts.mean()
    stderr = discounts.std(ddof=1) / np.sqrt(n_paths)
    return dict(price=float(price), stderr=float(stderr),
                ci95=1.96 * float(stderr), discounts=discounts)


# %% [markdown]
# ### 解析解 vs `bondlab` の一致
#
# 自作の ZCB 解析解が `bondlab.models.Vasicek` と一致することを、複数年限で確認します。

# %%
a, b, sigma, r0 = 0.30, 0.05, 0.02, 0.03
v = Vasicek(a, b, sigma, r0)

taus = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
print(f"{'年限τ':>6} {'自作 ZCB':>14} {'bondlab ZCB':>14} {'差':>12}")
for tau in taus:
    mine = vasicek_zcb(a, b, sigma, r0, tau)
    lib = v.zcb(tau)
    print(f"{tau:6.1f} {mine:14.10f} {lib:14.10f} {abs(mine - lib):12.2e}")

assert np.allclose(vasicek_zcb(a, b, sigma, r0, taus), v.zcb(taus), atol=1e-12)
assert np.allclose(vasicek_zero_rate(a, b, sigma, r0, taus), v.zero_rate(taus), atol=1e-12)
print("\n→ 自作解析解と bondlab が機械精度で一致しました")

# %% [markdown]
# ### 短期金利パスと正規分布の確認
#
# `simulate_short_rate` でパスを生成し、期末 $r_T$ の標本平均・標本分散が理論値
# （正規分布の平均 $r_0 e^{-aT}+b(1-e^{-aT})$、分散 $\frac{\sigma^2}{2a}(1-e^{-2aT})$）に合うかを見ます。

# %%
T = 10.0
times, paths = simulate_short_rate(a, b, sigma, r0, T, n_steps=2000, n_paths=200_000, seed=1)

mean_th = r0 * np.exp(-a * T) + b * (1 - np.exp(-a * T))
var_th = sigma ** 2 / (2 * a) * (1 - np.exp(-2 * a * T))
rT = paths[:, -1]
print(f"r_T の標本平均 = {rT.mean():.6f}   理論平均 = {mean_th:.6f}")
print(f"r_T の標本分散 = {rT.var(ddof=1):.6e}   理論分散 = {var_th:.6e}")
prob_neg = (rT < 0).mean()
print(f"r_T < 0 となった割合 = {prob_neg:.4f}（正規分布ゆえ負金利が正の確率で発生）")

assert abs(rT.mean() - mean_th) < 5e-4
assert abs(rT.var(ddof=1) - var_th) / var_th < 0.02

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(times, paths[:40].T, color="steelblue", alpha=0.3, lw=0.7)
axes[0].axhline(b, color="black", ls="--", lw=1.2, label=f"長期水準 b={b}")
mean_path = r0 * np.exp(-a * times) + b * (1 - np.exp(-a * times))
axes[0].plot(times, mean_path, color="crimson", lw=2, label="理論平均")
axes[0].set_title("短期金利パス（平均回帰）")
axes[0].set_xlabel("時間 t（年）")
axes[0].set_ylabel("短期金利 r_t")
axes[0].legend(loc="upper right", fontsize=9)

axes[1].hist(rT, bins=80, density=True, color="steelblue", alpha=0.6)
xs = np.linspace(rT.min(), rT.max(), 300)
pdf = np.exp(-(xs - mean_th) ** 2 / (2 * var_th)) / np.sqrt(2 * np.pi * var_th)
axes[1].plot(xs, pdf, color="crimson", lw=2, label="理論正規分布")
axes[1].axvline(0.0, color="black", ls=":", lw=1.2, label="r=0")
axes[1].set_title(f"r_T の分布（T={T:.0f}年）")
axes[1].set_xlabel("r_T")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 解析解 vs MC の一致
#
# $\exp\!\big(-\int_0^\tau r_s\,ds\big)$ の MC 平均で ZCB 価格を推定し、解析解と標準誤差の範囲で
# 一致することを確認します。

# %%
tau = 5.0
analytic = vasicek_zcb(a, b, sigma, r0, tau)
mc = vasicek_mc_zcb(a, b, sigma, r0, tau, n_steps=500, n_paths=200_000, seed=7)

print(f"解析解 ZCB(τ={tau}) = {analytic:.6f}")
print(f"MC 価格            = {mc['price']:.6f} ± {mc['ci95']:.6f}（95%）")
print(f"差                 = {mc['price'] - analytic:+.6f}  （標準誤差 {mc['stderr']:.6f}）")
assert abs(mc["price"] - analytic) < 4 * mc["stderr"]
print("→ MC 価格が解析解と標準誤差の範囲で一致しました")

# %% [markdown]
# ## QuantLib検証
#
# QuantLib の `ql.Vasicek(r0, a, b, sigma)`（`dr=a(b-r)dt+σ dW`、引数順は $r_0, a, b, \sigma$）の
# `discountBond` と、自作解析解を突き合わせます。両者は同一の閉形式なので機械精度で一致します。

# %%
import QuantLib as ql

print("QuantLib version:", ql.__version__)

qv = ql.Vasicek(r0, a, b, sigma)
print(f"\n{'年限τ':>6} {'自作 ZCB':>16} {'QuantLib':>16} {'差':>12}")
max_err = 0.0
for tau_i in taus:
    mine = float(vasicek_zcb(a, b, sigma, r0, tau_i))
    q = qv.discountBond(0.0, float(tau_i), r0)
    max_err = max(max_err, abs(mine - q))
    print(f"{tau_i:6.1f} {mine:16.12f} {q:16.12f} {abs(mine - q):12.2e}")

assert max_err < 1e-12
print(f"\n→ 最大誤差 {max_err:.2e}。自作・bondlab・QuantLib の三者が機械精度で一致")

# %% [markdown]
# ## 実データ適用
#
# 実データは合成データ（seed 固定）で代用します。日本の低金利期を想定した「真の」パラメータで
# ゼロレートを生成し、観測ノイズを載せた合成カーブへ Vasicek を粗くフィットします。あわせて、
# パラメータを変えるとカーブ形状（順イールド／逆イールド）がどう動くかを可視化します。
#
# ### 日本の低金利期を想定した合成カーブへのフィット
#
# 「真の」パラメータは、長期水準をごく低く（$b=0.4\%$）、現在の短期金利を**マイナス**（$r_0=-0.1\%$）に
# 置きます。マイナス金利政策下の日本を模した設定で、正規分布ゆえに負金利をそのまま扱えます。

# %%
a_true, b_true, sigma_true, r0_true = 0.25, 0.004, 0.010, -0.001
tenors = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)

y_true = vasicek_zero_rate(a_true, b_true, sigma_true, r0_true, tenors)
rng = np.random.default_rng(20260707)
y_obs = y_true + rng.normal(0.0, 2e-4, size=tenors.size)   # ±2bp の観測ノイズ

# フィット: r0 は最短年限の観測値で近似し、(a, b, sigma) を最小二乗で推定する。
r0_fit = y_obs[0]


def residuals(params):
    a_, b_, s_ = params
    return vasicek_zero_rate(a_, b_, s_, r0_fit, tenors) - y_obs


sol = least_squares(residuals, x0=[0.1, 0.01, 0.01],
                    bounds=([1e-3, -0.05, 1e-4], [5.0, 0.10, 0.10]))
a_fit, b_fit, sigma_fit = sol.x

print("        真値      推定値")
print(f"a     {a_true:8.4f}  {a_fit:8.4f}")
print(f"b     {b_true:8.4f}  {b_fit:8.4f}")
print(f"sigma {sigma_true:8.4f}  {sigma_fit:8.4f}")
print(f"r0    {r0_true:8.4f}  {r0_fit:8.4f}（最短年限で近似）")
rmse = np.sqrt(np.mean(residuals(sol.x) ** 2))
print(f"\nフィット RMSE = {rmse*1e4:.3f} bp")

# %%
fig, ax = plt.subplots(figsize=(9, 5))
tt = np.linspace(0.25, 30, 200)
ax.plot(tenors, y_obs * 100, "o", color="crimson", label="合成観測ゼロレート")
ax.plot(tt, vasicek_zero_rate(a_fit, b_fit, sigma_fit, r0_fit, tt) * 100,
        color="steelblue", lw=2, label="Vasicek フィット")
ax.axhline(0.0, color="black", ls=":", lw=1)
ax.set_title("日本の低金利期を想定した合成カーブへの Vasicek フィット")
ax.set_xlabel("年限 τ（年）")
ax.set_ylabel("ゼロレート（%）")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### パラメータでカーブ形状が変わる
#
# カーブの向きは、現在の短期金利 $r_0$ と長期水準（リスク中立 $b^{\mathbb{Q}}$）の大小で決まります。
# $r_0 < b$ なら金利が上がる期待で**順イールド**、$r_0 > b$ なら下がる期待で**逆イールド**、
# $r_0 \approx b$ ならほぼ平坦になります。

# %%
a_c, sigma_c, b_c = 0.5, 0.015, 0.03
scenarios = [
    ("順イールド (r0 < b)", 0.005),
    ("ほぼ平坦 (r0 ≈ b)", 0.030),
    ("逆イールド (r0 > b)", 0.055),
]
tt = np.linspace(0.1, 30, 200)

fig, ax = plt.subplots(figsize=(9, 5))
for label, r0_s in scenarios:
    y = vasicek_zero_rate(a_c, b_c, sigma_c, r0_s, tt)
    ax.plot(tt, y * 100, lw=2, label=f"{label}, r0={r0_s:.3f}")
ax.axhline(b_c * 100, color="black", ls="--", lw=1, label=f"長期水準 b={b_c}")
ax.set_title(f"r0 によるカーブ形状の変化（a={a_c}, b={b_c}, σ={sigma_c}）")
ax.set_xlabel("年限 τ（年）")
ax.set_ylabel("ゼロレート（%）")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### リスクの市場価格を通したリスク中立化
#
# 実測度パラメータ $b=3\%$ を、リスクの市場価格 $\lambda$ でリスク中立長期水準
# $b^{\mathbb{Q}} = b + \lambda\sigma/a$ へ変換し、価格づけに使うカーブへの影響を見ます。
# 実測度の $b$ をそのまま使うと、$\lambda>0$（正のプレミアム）の分だけカーブを低く誤ります。

# %%
a_p, b_p, sigma_p, r0_p = 0.30, 0.03, 0.02, 0.02
for lam in [0.0, 0.10, 0.25]:
    b_q = b_p + lam * sigma_p / a_p
    y10 = vasicek_zero_rate(a_p, b_q, sigma_p, r0_p, 10.0)
    print(f"λ={lam:4.2f}: リスク中立 b^Q = {b_q:.5f},  10年ゼロレート = {y10*100:.4f}%")
print("→ 価格づけには実測度 b ではなくリスク中立 b^Q を使う（S4-5 の測度変換を金利モデルへ適用）")

# %% [markdown]
# ## 演習
#
# 1. **カーブ形状のレポート**
#    $a\in\{0.1, 0.5, 1.5\}$、$\sigma\in\{0.005, 0.02\}$ を組み合わせ、$b=3\%$、$r_0=1\%$ に固定して
#    ゼロレートカーブ（年限 $0.1$〜$30$ 年）を描け。$a$ を上げると短期金利が長期水準へ速く回帰するため
#    カーブが早く平坦化すること、$\sigma$ を上げると凸性効果でカーブが押し下げられ長期が沈むことを、
#    図とともに 3〜4 行で説明せよ。
#
# 2. **MC 標準誤差の $1/\sqrt{n}$ 収束**
#    $\tau=5$ 年の ZCB を、パス数 $n\in\{2000, 8000, 32000, 128000, 512000\}$ で MC 価格推定し、
#    各 $n$ の標準誤差を両対数でプロットせよ。標準誤差が傾き $-1/2$ の直線に乗る（$\propto 1/\sqrt{n}$）ことを
#    確認し、解析解との差が標準誤差の範囲に収まることも示せ。
#
# 解答例は `solutions/S5/sol_0501.py` に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/05_rate_models.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | 平均回帰 | mean reversion | 変数を長期水準へ引き戻す力。金利モデルの基本性質 |
# | アフィンモデル | affine model | ZCB 価格が金利の指数アフィン形になる短期金利モデル |
# | ゼロクーポン債（ZCB） | zero-coupon bond | 満期に額面のみを払う債券。割引係数そのもの |
# | リスクの市場価格 | market price of risk | 実測度とリスク中立測度をつなぐ量。長期水準を付け替える |
