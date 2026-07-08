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
# # S3-5 VaRとExpected Shortfall
#
# ## 学習目標
#
# - VaR（Value at Risk）を、ヒストリカル・パラメトリック・モンテカルロの3方式で定義し、自分で実装できる
# - ES（Expected Shortfall）の定義を述べ、なぜ VaR より望ましい性質を持つのかを、劣加法性の観点から説明できる
# - VaR が劣加法性を破る2資産の反例を実際に構成し、同じ設定で ES は破れないことを数値で確認できる
# - Kupiec 検定（POF 検定）で VaR モデルの被覆率を統計的にバックテストできる
# - 合成イールドカーブのパネルから仮想債券ポートフォリオの日次損益を作り、99% VaR を3方式で計算して乖離の理由を説明できる
# - バーゼルの市場リスク規制（FRTB）が VaR から ES へ舵を切った方向性を一言で説明できる
#
# 本ノートは `bondlab.risk` 層（VaR / ES / Kupiec 検定）に対応する。まず定義から3方式を
# 手で実装し、`bondlab.risk` と一致することを確認したうえで、合成カーブの実データに当てる。


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# VaRとExpected Shortfall（ES）は、ポートフォリオが一定期間に被りうる損失を確率で語るための集約指標で、リスク管理の最上流に位置します。使い手はリスク管理クオンツとモデルバリデーションで、全デスクのポジションを日次で1つの損失分布に束ね、「99%の日はこの額以内、それを超える悪い日の平均損失はこの額」という形で経営とリミット管理に落とします。マクロやレラティブバリューのファンドでも、レバレッジをどこまでかけてよいかの上限はVaR／ESで縛られます。VaRリミットは各デスクの取れるリスク量を配分する装置であり、収益機会とリスクの綱引きをここで管理します。
#
# 収益とリスク管理への繋がりは、リスク量の配分がそのまま資本効率を決める点にあります。ESはVaRと違って裾の厚み（超過損失がどれだけ深いか）まで測り、かつ劣加法性を満たすため、ポートフォリオを分散したときにリスクが正しく減る形で集計できます。VaRは分散効果を過小評価したり、極端な場合には合算するとリスクが増えるという劣加法性の破れを起こすことがあり、この反例を実際に構成できることは、なぜ規制と実務がESへ舵を切ったかを腹落ちさせます。バーゼルの市場リスク規制（FRTB）はまさにVaRからESへ移行しており、ESの計測はそのまま所要自己資本の計算に直結します。過大なリスク推計は資本を無駄に積ませ、過小な推計は突然の大損失で資本を毀損します。
#
# 具体的な場面として、VaRモデルは当てているだけでは信用できず、Kupiec検定（POF検定）で「実際の損失超過の頻度が想定した被覆率と統計的に整合するか」をバックテストして初めて承認されます。これはモデルバリデーションの中核業務で、超過が多すぎればモデルはリスクを過小評価しており、規制上のペナルティ（資本乗数の引き上げ）につながります。ヒストリカル・パラメトリック・モンテカルロの3方式を自分で実装し、正規仮定が裾を過小評価する理由まで含めて乖離を説明できないと、どの数字を信じてリミットを引くかを判断できず、リスク管理そのものが機能しません。
# %% [markdown]
# ## 理論
#
# ### 損益系列とリスク指標の約束
#
# 以下では、ポートフォリオの1期間の損益（profit and loss, PnL）を確率変数 $X$ とし、
# **正が利益・負が損失** とします。損失そのものは $L = -X$ で表します。
#
# 信頼水準（confidence level）$\alpha$（例：$\alpha = 0.99$）に対して、
# バリュー・アット・リスク（Value at Risk, VaR）は「損失がそれ以下に収まる確率が
# $\alpha$ となる損失水準」であり、次で定義します。
#
# $$
# \mathrm{VaR}_\alpha(X) = -\,\inf\{\,x : P(X \le x) \ge 1-\alpha\,\} = -\,q_{1-\alpha}(X)
# $$
#
# ここで $q_{1-\alpha}(X)$ は損益分布の下側 $(1-\alpha)$ 分位点です。先頭のマイナスにより、
# VaR は **正の損失** で表されます（$\alpha=0.99$ なら「100日に1日起こる規模の損失」）。
#
# エクスペクテッド・ショートフォール（Expected Shortfall, ES）は、VaR を超える損失の
# 条件付き期待値です。CVaR（conditional VaR）とも呼びます。
#
# $$
# \mathrm{ES}_\alpha(X) = -\,\mathbb{E}\!\left[\,X \mid X \le q_{1-\alpha}(X)\,\right]
# = \frac{1}{1-\alpha}\int_0^{1-\alpha} \mathrm{VaR}_u(X)\,du
# $$
#
# 定義から常に $\mathrm{ES}_\alpha \ge \mathrm{VaR}_\alpha$ です。ES は「悪い方の裾の平均」なので、
# VaR が捉えない裾の厚み（超過損失がどれだけ深いか）まで測ります。
#
# ### 3つの計測方式
#
# 同じ VaR / ES を、分布の与え方で3通りに計算します。
#
# | 方式 | 分布の作り方 | 長所 | 短所 |
# |---|---|---|---|
# | ヒストリカル法 | 過去の PnL の経験分布をそのまま使う | 分布形を仮定しない。裾も過去通り | 過去に無い損失を出せない。標本数に依存 |
# | パラメトリック法 | $X \sim \mathcal{N}(\mu,\sigma^2)$ を当てはめ、分位点を解析式で出す | 計算が速く滑らか | 正規仮定。裾が薄く VaR を過小評価しがち |
# | モンテカルロ法 | 当てはめた分布から乱数を大量生成し、経験分位点を取る | 任意分布・多資産に拡張しやすい | 乱数依存。標本数を増やす必要 |
#
# 正規分布 $X\sim\mathcal{N}(\mu,\sigma^2)$ を仮定すると、$z_{1-\alpha}=\Phi^{-1}(1-\alpha)$（負値）を用いて
#
# $$
# \mathrm{VaR}_\alpha = -(\mu + \sigma z_{1-\alpha}), \qquad
# \mathrm{ES}_\alpha = -\Big(\mu - \sigma\,\frac{\phi(z_{1-\alpha})}{1-\alpha}\Big)
# $$
#
# となります（$\phi$ は標準正規密度）。パラメトリック法はこの閉形式、モンテカルロ法は
# 同じ $\mathcal{N}(\mu,\sigma^2)$ からの標本にヒストリカル法を適用したものです。したがって
# **正規仮定のもとでは、標本数を増やすとモンテカルロ VaR はパラメトリック VaR に収束** します。
# この一致は後の検証パートで数値的に確かめます。
#
# ### 劣加法性と、VaR がそれを破ること
#
# コヒーレントなリスク尺度（coherent risk measure）が満たすべき性質のひとつが
# **劣加法性（subadditivity）** です。任意のポジション $A, B$ に対し
#
# $$
# \rho(A+B) \le \rho(A) + \rho(B)
# $$
#
# が成り立つことを言います。「分散させればリスクは増えない」という、分散投資の常識に対応する
# 性質です。ES はこれを常に満たしますが、**VaR は一般には満たしません**。VaR は分位点という
# 一点しか見ないため、裾の質量の配置しだいで分散がかえって不利に映ることがあります。
#
# 典型的な反例は、独立でまれに大きく損する2資産です。各資産が確率 $p$ で大きな損失 $L$ を出し、
# それ以外は小さな利益 $c$ を出すとします。$p < 1-\alpha$ なら、単独の $(1-\alpha)$ 分位点は
# 損失を捉えず VaR は小さいまま（あるいは負）です。ところが2資産を合算すると
# 「少なくとも一方がデフォルトする確率」は $1-(1-p)^2 > 1-\alpha$ まで上がり、分位点が損失領域に
# 落ちて VaR が跳ね上がります。結果として $\mathrm{VaR}(A+B) > \mathrm{VaR}(A)+\mathrm{VaR}(B)$ となり、
# 劣加法性が破れます。この設定を演習1で実際に組み、ES では破れないことも確認します。
#
# ### バックテスト：Kupiec の POF 検定
#
# VaR モデルが妥当かは、事後に **例外（exception）** ＝実現損失が VaR 予測を超えた回数で検証します。
# 信頼水準 $\alpha$ なら例外は確率 $p=1-\alpha$ で独立に起こるはずで、$n$ 期間なら期待例外数は $pn$ です。
# キューピック検定（Kupiec test, proportion of failures / POF 検定）は、観測例外数 $x$ が二項
# $\mathrm{Bin}(n,p)$ と整合するかを尤度比で調べます。
#
# $$
# LR_{POF} = -2\ln\!\frac{(1-p)^{\,n-x}\,p^{\,x}}{(1-\hat p)^{\,n-x}\,\hat p^{\,x}}, \qquad \hat p = \frac{x}{n}
# $$
#
# 帰無仮説（被覆率が正しい）のもとで $LR_{POF}\sim\chi^2_1$ に従うので、$p$ 値が小さければ
# 「例外が多すぎ／少なすぎ」としてモデルを棄却します。例外が少なすぎる棄却も起こる点に注意します
# （過度に保守的なモデルも POF 検定は嫌います）。
#
# ### 規制上の位置づけ（FRTB の方向性）
#
# バーゼルの市場リスク規制の抜本改定（FRTB, Fundamental Review of the Trading Book）は、
# 内部モデル方式の主指標を **99% VaR から 97.5% ES へ移行** しました。狙いは、VaR が測れない
# 裾リスクを ES で捉えることと、劣加法性を持つ尺度に統一することです。実務では VaR も
# バックテスト（例外計数）の枠組みで併用が続いています。

# %% [markdown]
# ## スクラッチ実装
#
# 定義に忠実に3方式の VaR / ES と Kupiec 検定を実装します。すべて「PnL は正が利益、
# VaR / ES は正の損失で返す」慣行に合わせ、あとで `bondlab.risk` と数値一致を確認します。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `hist_var(pnl, alpha)` | PnL 配列, 信頼水準 | float | 経験分位点から VaR（正の損失） |
# | `hist_es(pnl, alpha)` | PnL 配列, 信頼水準 | float | VaR 超過部分の平均損失 |
# | `param_var(pnl, alpha)` | PnL 配列, 信頼水準 | float | 正規当てはめの解析的 VaR |
# | `param_es(pnl, alpha)` | PnL 配列, 信頼水準 | float | 正規当てはめの解析的 ES |
# | `mc_var(pnl, alpha, n, seed)` | PnL 配列, 信頼水準, 標本数, 乱数種 | float | 正規当てはめ→乱数生成→経験 VaR |
# | `mc_es(pnl, alpha, n, seed)` | PnL 配列, 信頼水準, 標本数, 乱数種 | float | 同上の経験 ES |
# | `kupiec(exceptions, n, alpha)` | 例外数, 期間数, 信頼水準 | dict | POF 検定統計量と p 値 |

# %%
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import bondlab
from bondlab import risk as blrisk

SEED = 20260707
rng = np.random.default_rng(SEED)
print("bondlab version:", bondlab.__version__)


def hist_var(pnl, alpha=0.99):
    """ヒストリカル VaR。損益の経験分布の下側 (1-alpha) 分位点を正の損失で返す。"""
    pnl = np.asarray(pnl, dtype=float)
    return -np.quantile(pnl, 1.0 - alpha)


def hist_es(pnl, alpha=0.99):
    """ヒストリカル ES。VaR を超えて損した部分（裾）の平均損失を正で返す。"""
    pnl = np.asarray(pnl, dtype=float)
    var = hist_var(pnl, alpha)
    tail = pnl[pnl <= -var]  # 損失が VaR 以上（=PnL が -VaR 以下）の標本
    if tail.size == 0:
        return var
    return -tail.mean()


def param_var(pnl, alpha=0.99):
    """正規分布を当てはめたパラメトリック VaR。VaR = -(mu + sigma z)。"""
    pnl = np.asarray(pnl, dtype=float)
    mu, sigma = pnl.mean(), pnl.std(ddof=1)
    z = stats.norm.ppf(1.0 - alpha)  # 下側分位点（負値）
    return -(mu + sigma * z)


def param_es(pnl, alpha=0.99):
    """正規分布を当てはめたパラメトリック ES。ES = -(mu - sigma phi(z)/(1-alpha))。"""
    pnl = np.asarray(pnl, dtype=float)
    mu, sigma = pnl.mean(), pnl.std(ddof=1)
    z = stats.norm.ppf(1.0 - alpha)
    return -(mu - sigma * stats.norm.pdf(z) / (1.0 - alpha))


def mc_var(pnl, alpha=0.99, n=200_000, seed=0):
    """モンテカルロ VaR。当てはめた正規分布から n 標本を引き、経験 VaR を取る。"""
    pnl = np.asarray(pnl, dtype=float)
    mu, sigma = pnl.mean(), pnl.std(ddof=1)
    sims = np.random.default_rng(seed).normal(mu, sigma, size=n)
    return hist_var(sims, alpha)


def mc_es(pnl, alpha=0.99, n=200_000, seed=0):
    """モンテカルロ ES。当てはめた正規分布からの標本に経験 ES を適用する。"""
    pnl = np.asarray(pnl, dtype=float)
    mu, sigma = pnl.mean(), pnl.std(ddof=1)
    sims = np.random.default_rng(seed).normal(mu, sigma, size=n)
    return hist_es(sims, alpha)


def kupiec(exceptions, n, alpha=0.99):
    """Kupiec の POF 検定。例外数が Bin(n, 1-alpha) と整合するかを尤度比で判定する。"""
    p = 1.0 - alpha
    x = int(exceptions)
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        lr = -2.0 * n * np.log(p)
    else:
        pi = x / n
        lr = -2.0 * (
            (n - x) * np.log(1.0 - p) + x * np.log(p)
            - (n - x) * np.log(1.0 - pi) - x * np.log(pi)
        )
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return dict(lr=float(lr), p_value=float(p_value), expected=p * n, observed=x)


# %% [markdown]
# 動作を素の標本で確かめます。標準正規なら、99% VaR は理論値
# $-z_{0.01}=\Phi^{-1}(0.99)\approx 2.326$、99% ES は $\phi(z_{0.01})/0.01\approx 2.665$ に近づくはずです。

# %%
z01 = stats.norm.ppf(0.99)
print(f"理論値: 99%VaR = {z01:.4f},  99%ES = {stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01:.4f}")

demo = rng.standard_normal(500_000)
print(f"hist_var  = {hist_var(demo):.4f}")
print(f"param_var = {param_var(demo):.4f}")
print(f"mc_var    = {mc_var(demo, seed=1):.4f}")
print(f"hist_es   = {hist_es(demo):.4f}")
print(f"param_es  = {param_es(demo):.4f}")

# %% [markdown]
# ## QuantLib検証
#
# VaR / ES とそのバックテストは市場データの統計処理であり、金融商品の評価器である
# **QuantLib の守備範囲外** です。そこで本ノートの検証は次の2本立てで行います。
#
# 1. **自作 == `bondlab.risk`**：スクラッチ実装が、S3-5 に対応するライブラリ層
#    `bondlab.risk`（`historical_var` / `parametric_var` / `historical_es` / `parametric_es` / `kupiec_pof`）
#    と数値一致することを `assert` で守る。
# 2. **パラメトリック ⇔ モンテカルロの収束**：正規仮定のもとで、モンテカルロ VaR / ES が
#    標本数の増加とともにパラメトリックの閉形式へ収束することを確認する。
#
# ### 検証1：自作とライブラリの一致

# %%
sample = rng.normal(0.0, 1.0, 5000)
for alpha in (0.95, 0.99):
    assert np.isclose(hist_var(sample, alpha), blrisk.historical_var(sample, alpha))
    assert np.isclose(hist_es(sample, alpha), blrisk.historical_es(sample, alpha))
    assert np.isclose(param_var(sample, alpha), blrisk.parametric_var(sample, alpha))
    assert np.isclose(param_es(sample, alpha), blrisk.parametric_es(sample, alpha))

for x in (0, 5, 25, 100):
    a = kupiec(x, 1000, 0.99)
    b = blrisk.kupiec_pof(x, 1000, 0.99)
    assert np.isclose(a["lr"], b["lr"]) and np.isclose(a["p_value"], b["p_value"])

print("自作関数は bondlab.risk と一致しました（VaR / ES / Kupiec すべて）")

# %% [markdown]
# ### 検証2：パラメトリックとモンテカルロの収束
#
# 平均 $\mu$・標準偏差 $\sigma$ の正規 PnL を固定し、モンテカルロの標本数 $n$ を増やすと、
# MC-VaR がパラメトリック VaR（閉形式）へ収束していく様子を表とグラフで見ます。

# %%
mu_true, sigma_true = 0.02, 1.3
base = np.array([mu_true, sigma_true])  # パラメトリックは (mu, sigma) さえ分かればよい
pv = param_var(rng.normal(mu_true, sigma_true, 2_000_000), 0.99)  # 事実上の真値
pe = param_es(rng.normal(mu_true, sigma_true, 2_000_000), 0.99)

rows = []
for n in (1_000, 10_000, 100_000, 1_000_000):
    mvar = mc_var(rng.normal(mu_true, sigma_true, 5000), 0.99, n=n, seed=n)
    mes = mc_es(rng.normal(mu_true, sigma_true, 5000), 0.99, n=n, seed=n)
    rows.append((n, mvar, abs(mvar - pv), mes, abs(mes - pe)))

conv = pd.DataFrame(rows, columns=["標本数n", "MC-VaR", "VaR誤差", "MC-ES", "ES誤差"])
print(conv.to_string(index=False,
      formatters={"MC-VaR": "{:.4f}".format, "VaR誤差": "{:.5f}".format,
                  "MC-ES": "{:.4f}".format, "ES誤差": "{:.5f}".format}))
print(f"\nパラメトリック 99%VaR = {pv:.4f},  99%ES = {pe:.4f}")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(conv["標本数n"], conv["VaR誤差"], "o-", label="MC-VaR 誤差")
ax.loglog(conv["標本数n"], conv["ES誤差"], "s-", label="MC-ES 誤差")
ref = conv["VaR誤差"].iloc[0] * np.sqrt(conv["標本数n"].iloc[0] / conv["標本数n"])
ax.loglog(conv["標本数n"], ref, "k--", alpha=0.6, label=r"傾き $\propto 1/\sqrt{n}$ の目安")
ax.set_xlabel("モンテカルロ標本数 n")
ax.set_ylabel("パラメトリック値との絶対誤差")
ax.set_title("MC-VaR / MC-ES はパラメトリック値へ収束する")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
plt.show()
print("標本数を増やすと誤差はおよそ 1/√n で縮む")

# %% [markdown]
# ## 実データ適用
#
# 合成米国債パーカーブのパネル（`data/samples/synthetic_ust_par_panel.csv`, 60営業日 × 9テナー）から、
# 仮想の債券ポートフォリオの日次損益系列を作り、99% VaR を3方式で計算します。ネットワークには
# 一切アクセスしません。
#
# ### ポートフォリオと日次損益の作り方
#
# 2年・5年・10年・30年の固定利付債（半年払い）を等ウェイトで保有するとします。各営業日に、
# その日のパーカーブを債券満期テナーへ線形補間して利回りとし、`FixedRateBond.clean_price` で
# 再評価します。日次損益は「翌日の合計時価 − 当日の合計時価」で定義します（額面 100 建て）。

# %%
import datetime as _dt
from bondlab.bond import FixedRateBond

panel = pd.read_csv("data/samples/synthetic_ust_par_panel.csv", parse_dates=["date"])
wide = panel.pivot(index="date", columns="tenor", values="par_yield").sort_index()
tenor_grid = wide.columns.to_numpy(dtype=float)
dates = [d.date() for d in wide.index]
print(f"パネル: {wide.shape[0]}営業日 × {wide.shape[1]}テナー "
      f"({dates[0]} 〜 {dates[-1]})")

# 満期テナー別の保有債券（発行はパネル開始より前、償還は十分先に置く）。
holdings = [
    ("2年", 2.0, FixedRateBond(_dt.date(2024, 1, 2), _dt.date(2028, 1, 2), 0.030, 2)),
    ("5年", 5.0, FixedRateBond(_dt.date(2024, 1, 2), _dt.date(2031, 1, 2), 0.035, 2)),
    ("10年", 10.0, FixedRateBond(_dt.date(2024, 1, 2), _dt.date(2036, 1, 2), 0.040, 2)),
    ("30年", 30.0, FixedRateBond(_dt.date(2024, 1, 2), _dt.date(2056, 1, 2), 0.045, 2)),
]
weights = np.full(len(holdings), 1.0 / len(holdings))  # 等ウェイト

# 各日・各債券の clean price を評価して合計時価を作る。
values = np.zeros(len(dates))
for i, d in enumerate(dates):
    row = wide.iloc[i].to_numpy(dtype=float)
    pv = 0.0
    for j, (_lbl, ten, bond) in enumerate(holdings):
        y = float(np.interp(ten, tenor_grid, row))
        pv += weights[j] * bond.clean_price(y, d)
    values[i] = pv

pnl = np.diff(values)  # 日次損益（正が利益）
print(f"日次損益系列: {pnl.size}日分,  平均 {pnl.mean():+.4f},  標準偏差 {pnl.std(ddof=1):.4f}")
print(f"最小(最大損失日) {pnl.min():+.4f},  最大 {pnl.max():+.4f}")

# %%
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(range(pnl.size), pnl, color=np.where(pnl < 0, "#c0392b", "#2e86c1"))
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("営業日インデックス")
ax.set_ylabel("日次損益（額面100建て）")
ax.set_title("仮想債券ポートフォリオの日次損益")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 3方式の 99% VaR と乖離の理由

# %%
alpha = 0.99
res = pd.DataFrame({
    "方式": ["ヒストリカル", "パラメトリック", "モンテカルロ"],
    "99%VaR": [hist_var(pnl, alpha), param_var(pnl, alpha), mc_var(pnl, alpha, seed=SEED)],
    "99%ES": [hist_es(pnl, alpha), param_es(pnl, alpha), mc_es(pnl, alpha, seed=SEED)],
})
print(res.to_string(index=False,
      formatters={"99%VaR": "{:.4f}".format, "99%ES": "{:.4f}".format}))

skew = stats.skew(pnl)
kurt = stats.kurtosis(pnl, fisher=True)  # 超過尖度
print(f"\n損益の歪度 = {skew:+.3f},  超過尖度 = {kurt:+.3f},  標本数 = {pnl.size}")

# %% [markdown]
# **乖離の読み方**。3方式が一致しない主因は次の通りです。
#
# - **パラメトリック法** は正規分布を当てはめるため、VaR は $\mu,\sigma$ だけで決まり滑らかです。
#   損益に歪みや裾の厚み（尖度）があると、正規の裾はそれを取りこぼし、VaR を過小／過大評価します。
# - **ヒストリカル法** は経験分位点をそのまま使うので、標本が $n\approx 59$ 日と少ない今回は
#   99% 点が「ほぼ最悪の1日」に張り付き、離散的で不安定です（1日入れ替わると値が飛ぶ）。
# - **モンテカルロ法** は当てはめた正規から大量に引くため、値はパラメトリック法に近く滑らかです。
#   両者の差は乱数誤差のオーダーに収まり、ヒストリカル法との差が「正規で近似したことの代償」を表します。
#
# 実務では標本数・分布の当てはめ・裾の扱いのどれを重視するかで方式を選びます。
#
# ### Kupiec 検定によるバックテスト
#
# ヒストリカル 99% VaR をしきい値として、実現損益がそれを超えた日（例外）を数え、
# Kupiec の POF 検定にかけます。

# %%
var99 = hist_var(pnl, 0.99)
exceptions = int(np.sum(pnl < -var99))
kp = kupiec(exceptions, pnl.size, 0.99)
print(f"99%VaR しきい値 = {var99:.4f}")
print(f"期間数 n = {pnl.size},  例外数 = {exceptions},  期待例外数 = {kp['expected']:.2f}")
print(f"Kupiec LR = {kp['lr']:.4f},  p値 = {kp['p_value']:.4f}")
verdict = "棄却しない（被覆率と整合）" if kp["p_value"] > 0.05 else "棄却（被覆率が不整合）"
print(f"有意水準5%での判定: {verdict}")
# ライブラリと一致することも確認する。
assert np.isclose(kp["lr"], blrisk.kupiec_pof(exceptions, pnl.size, 0.99)["lr"])
print("→ bondlab.risk.kupiec_pof と一致")

# %% [markdown]
# 標本が短い（約60日）ため検出力は弱く、例外1〜2件では被覆率のズレを統計的に言い切れないのが普通です。
# 規制上のバックテストが少なくとも1年（約250日）を要求するのは、この検出力の問題が理由です。

# %% [markdown]
# ## 演習
#
# 1. **VaR が劣加法性を破る反例の構成**。独立な2資産 $A, B$ を、それぞれ確率 $p=0.03$ で
#    損失 $L=100$、確率 $1-p$ で利益 $c=1$ を出すよう定義する（PnL は正が利益）。
#    信頼水準 $\alpha=0.95$ で、単独の VaR と合算ポートフォリオ $A+B$ の VaR を計算し、
#    $\mathrm{VaR}(A+B) > \mathrm{VaR}(A)+\mathrm{VaR}(B)$ となって劣加法性が破れることを示せ。
#    同じ設定で ES を計算し、$\mathrm{ES}(A+B) \le \mathrm{ES}(A)+\mathrm{ES}(B)$ が保たれることを確認せよ。
# 2. **損益系列のバックテスト**。本編で作った日次損益 `pnl` について、パラメトリック
#    99% VaR をしきい値に取り直して例外数を数え、Kupiec 検定にかけよ。ヒストリカル VaR を
#    使った場合と例外数・p 値がどう変わるかを比較し、方式の違いがバックテスト結論に与える
#    影響を一言で述べよ。
#
# 解答例は `solutions/S3/sol_0305.py` に置く。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/03_risk.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | バリュー・アット・リスク | VaR | 信頼水準 $\alpha$ で、損失がそれ以下に収まる損失水準。損益分布の下側分位点を正の損失で表す |
# | エクスペクテッド・ショートフォール | Expected Shortfall | VaR を超える損失の条件付き期待値。裾の平均損失で、常に VaR 以上 |
# | 劣加法性 | subadditivity | $\rho(A+B)\le\rho(A)+\rho(B)$。ES は満たし VaR は一般に満たさない |
# | キューピック検定 | Kupiec test | VaR 例外数が想定被覆率と整合するかを尤度比で調べる POF 検定 |
# | バックテスティング | backtesting | 実現損益が VaR 予測を超えた回数を数え、モデルの妥当性を事後検証すること |
