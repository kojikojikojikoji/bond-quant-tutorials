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
# # S9-1 カーブ対比 rich/cheap 分析
#
# ## 学習目標
#
# - フィット済みイールドカーブからの利回り乖離（残差）として、銘柄の割高
#   （rich）・割安（cheap）を定義し、符号の意味を説明できる
# - 全銘柄を NSS カーブでフィットし、残差の Z スコア時系列を計算する分析基盤を
#   スクラッチで実装できる
# - 残差の平均回帰半減期を AR(1) 係数から換算する式を導出し、統計的有意性を
#   確認できる
# - on-the-run/off-the-run プレミアムと流動性を踏まえ、「安く見える銘柄」が
#   本当に割安かを切り分けて考察できる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# rich/cheap 分析は、相対価値（RV）ヘッジファンドが国債デスクで最初に回すエンジンです。フィット済みカーブからの残差 $r_i$ とその Z スコアは「どの銘柄がフェアカーブに対して安く（cheap）、どれが高く（rich）値付けされているか」を全銘柄で横並びにするので、ファンドは $z \gg 0$ の cheap 銘柄を買い、$z \ll 0$ の rich 銘柄を売って、両者の乖離が平均へ戻る（収束する）方に賭けます。ここで方向性リスク（金利が上がるか下がるか）は取りません。ロングとショートを DV01 でほぼ相殺し、残差そのものの縮小だけを取りにいくのが RV の骨格です。稼ぎの源泉は、カーブに対する一時的な需給の歪み（新発債の発行圧力、指数入替、特定年限への買い需要）が剥落するときの数 bp の収束であり、これは docs/債券ファンドの業務.md でいう「証券間の価格のゆがみが収束する方向に賭ける」相対価値そのものです。
#
# 1トレードあたりの収束幅は数 bp と薄いので、レバレッジが収益の前提になります。買った cheap 銘柄はレポ市場で担保に出して資金を調達し、その資金でさらにポジションを積むことで、自己資本の何倍もの名目を回します。したがって実現損益は「残差の収束（キャピタルゲイン）＋保有期間のキャリー（銘柄利回り − レポ調達コスト）」の合算で決まります。cheap 銘柄はカーブより利回りが高いぶんキャリーもプラスに乗りやすく、収束を待つ間も時間で稼げる——この二重取りが rich/cheap トレードの旨味です。逆に、狙った銘柄が品薄でスペシャルレポになると調達コスト構造が崩れるため、レポ市場の状態は残差と同じくらい重要な入力になります。
#
# 平均回帰半減期の推定は、机上の統計ではなくポジションサイズと保有期間を決める実務パラメータです。AR(1) 係数 $\phi$ から換算した半減期が短ければ資金回転を上げて多くの機会を取りにいけますが、半減期が長い（$\phi$ が 1 に近い）銘柄に大きく張ると、収束を待つ間に金利ショックやマージンコールでポジションを畳まされ、収束前に損切りする「キャリーの負け方」に陥ります。半減期が推定できない、あるいは統計的に有意な平均回帰が確認できない乖離は、単なる恒常的な流動性プレミアム（構造的に cheap なだけ）である可能性が高く、張っても戻らないため見送ります。
#
# on-the-run/off-the-run の切り分けはこの見送り判断の中核です。直近発行の on-the-run 銘柄は流動性が高いぶん恒常的に rich に、少し古い off-the-run は cheap に値付けされます。この恒常的な水準差を平均回帰の機会と誤認して off-the-run を買い続けると、戻らない乖離に資本を寝かせることになります。銘柄ごとの平均 $\bar{r}_i$ を差し引いた Z スコアで評価するのは、この構造的な安さと一時的な安さを分離し、「本当に戻る歪み」だけに資本を配分するためです。
# %% [markdown]
# ## 理論
#
# ### フィット残差としての rich/cheap
#
# ある評価日に、ユニバース $N$ 銘柄の満期 $\tau_i$ と利回り $y_i$ が観測される。
# 平滑なパラメトリックカーブ（ここでは Nelson-Siegel-Svensson, NSS）を
# 最小二乗でフィットし、モデル利回り $\hat{y}(\tau)$ を得る。銘柄 $i$ の
# **フィット残差**（fitted residual）を
#
# $$ r_i = y_i - \hat{y}(\tau_i) $$
#
# と定義する。カーブは「その日のフェアな期間構造」を表すと解釈するので、
# 残差はカーブから見た個別銘柄の割高・割安を測る。符号の約束は次のとおり。
#
# | 残差 $r_i$ | 利回り | 価格 | 呼称 |
# |---|---|---|---|
# | $r_i > 0$ | カーブより高い | カーブより安い | cheap（割安） |
# | $r_i < 0$ | カーブより低い | カーブより高い | rich（割高） |
#
# 利回りと価格は逆向きなので、利回りが高い（残差が正）ほど価格は安く cheap に
# なる。残差はベーシスポイント（bp, $=10^{-4}$）で表すのが実務の慣習である。
#
# NSS はスプラインの一種ではないが、少数パラメータで滑らかな期間構造を張る点で
# 平滑化スプラインと同じ役割を果たす。残差の定義は「平滑カーブからの乖離」で
# あって、平滑化の手段（NSS でも三次スプラインでも）には依存しない。
#
# ### 残差の Z スコア
#
# 残差の絶対水準 $r_i$ はカーブ形状の日々の変化を含み、銘柄間の比較には向かない。
# そこで各銘柄の残差時系列 $\{r_{i,t}\}_t$ を、その平均と標準偏差で基準化した
# **Z スコア**（z-score）を使う。
#
# $$ z_{i,t} = \frac{r_{i,t} - \bar{r}_i}{s_i}, \qquad
#    \bar{r}_i = \frac{1}{T}\sum_t r_{i,t}, \quad
#    s_i = \sqrt{\frac{1}{T-1}\sum_t (r_{i,t}-\bar{r}_i)^2} $$
#
# $z_{i,t}$ は「その銘柄の平常時の乖離幅に対して、今日は何シグマ安い/高いか」を
# 表す。$z \gg 0$ は普段より際立って cheap、$z \ll 0$ は際立って rich である。
# 銘柄ごとに恒常的な乖離（平均 $\bar{r}_i$）を差し引くので、構造的に安い銘柄
# （後述の流動性プレミアム等）と、一時的に安くなった銘柄を分離しやすい。
#
# ### 平均回帰性と半減期の導出
#
# rich/cheap トレードが成立するのは、残差が平均へ戻る（平均回帰する,
# mean-reverting）ときに限る。残差時系列を1次自己回帰 AR(1) でモデル化する。
#
# $$ r_t - \mu = \phi\,(r_{t-1} - \mu) + \varepsilon_t,
#    \qquad \varepsilon_t \sim \text{i.i.d.}(0, \sigma^2) $$
#
# $|\phi|<1$ なら定常で平均 $\mu$ へ回帰する。$h$ 期先の条件付き期待値は、
# 漸化式を反復して
#
# $$ \mathbb{E}[r_{t+h}-\mu \mid r_t] = \phi^{h}\,(r_t - \mu) $$
#
# となる。乖離が初期値の半分まで縮む期間を**平均回帰半減期**（mean-reversion
# half-life）$h_{1/2}$ と呼ぶ。定義より $\phi^{h_{1/2}} = \tfrac{1}{2}$ を解いて
#
# $$ h_{1/2} = \frac{\ln(1/2)}{\ln \phi} = -\frac{\ln 2}{\ln \phi} $$
#
# を得る。$\phi \to 1$ で $h_{1/2}\to\infty$（回帰しない=ランダムウォークに近い）、
# $\phi$ が小さいほど半減期は短い。連続時間の Ornstein-Uhlenbeck 過程
# $dr = \kappa(\mu-r)dt+\sigma dW$ とは $\phi=e^{-\kappa\Delta t}$ で対応し、
# $h_{1/2}=\ln 2/\kappa$ と一致する。
#
# ### 平均回帰性の検定
#
# $\phi$ は $r_t$ を $r_{t-1}$ に回帰する OLS 傾きで推定できる。平均回帰の有無は
# 「$\phi<1$ か」を問う単位根の話だが、実務では回帰係数 $\phi$ の
# 標準誤差から $t$ 値を作り、$\phi$ が 1 から有意に離れているか（あるいは
# 反平均回帰でない=$\phi$ が 0 と区別できる水準で 1 未満か）を確認する。
# 本 notebook では $\hat\phi$ の推定値・標準誤差・$H_0:\phi=0$ の $t$ 値を出し、
# 併せて $\hat\phi<1$ を目視で確認して半減期の解釈可能性を判断する
# （厳密な単位根検定=Augmented Dickey-Fuller は S9 の後半で扱う）。
#
# ### on-the-run / off-the-run プレミアムと流動性の交絡
#
# 直近に発行された最も出来高の多い銘柄を **on-the-run**、それ以前の同年限を
# **off-the-run** と呼ぶ。on-the-run は流動性が高く、投資家が流動性に対して
# 割増を払うため利回りが低く（価格が高く）なりやすい。これが
# **on-the-run プレミアム**である。結果として on-the-run は残差が負（rich）に、
# off-the-run は残差が正（cheap）に寄る傾向を持つ。
#
# ここに交絡（confounding）がある。残差が正の銘柄は、
#
# 1. 本当に割安（フェアバリューに対して価格が安く、いずれ収束して儲かる）
# 2. 流動性が低いことへの対価（安いのではなく、売買コスト・保有リスクの補償）
#
# のどちらでも起こりうる。2 の場合、残差は縮まず（半減期が長い）、収束を狙った
# ロングは流動性プレミアムを負担し続けるだけで報われない。したがって
# 「cheap かどうか」の判定には、残差の大きさに加えて **半減期の短さ**（平均回帰
# の速さ）と、流動性指標（出来高・ビッド/アスク・発行後経過・on/off の別）を
# 併せて見る必要がある。本 notebook のデータは満期・クーポン・残差のみで流動性
# 列を持たないため、実データ適用ではこの限界を明示して考察する。

# %% [markdown]
# ## スクラッチ実装
#
# 分析基盤を4つの自作関数に分ける。日次でカーブをフィットして残差パネルを作り、
# それを Z スコアに基準化し、各銘柄の残差の AR(1) 半減期を推定する。
#
# ### 使用する自作関数
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `fit_date_residuals(mat, y, lam_grid)` | 満期配列, 利回り配列, λ格子 | 残差配列(bp) | 1営業日ぶんのNSSフィット残差 |
# | `build_residual_panel(panel, lam_grid)` | 縦持ちパネル, λ格子 | DataFrame(日付×銘柄) | 全日付をループして残差パネルを構築 |
# | `residual_zscores(resid_wide)` | 残差パネル | DataFrame(日付×銘柄) | 銘柄ごとに残差時系列をZ基準化 |
# | `ar1_halflife(series)` | 1銘柄の残差時系列 | dict(phi, halflife, se, tstat) | 自作OLSでAR(1)を推定し半減期へ換算 |

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from bondlab.curve import fit_nss, nss

np.random.seed(0)

# λ格子は粗く取る。既定の 20x20 格子は日次ループでは重いため、8 点に絞って
# フィット速度を確保する（残差の相対順位はこの粗さでも安定する）。
LAM_GRID = np.linspace(0.8, 8.0, 8)
BP = 1e4  # 小数利回り → ベーシスポイント


def fit_date_residuals(mat, y, lam_grid=LAM_GRID):
    """1営業日の (満期, 利回り) に NSS をフィットし、残差(bp)を返す。

    残差 = 実測利回り - モデル利回り。正なら利回りが高く価格が安い=cheap。
    """
    mat = np.asarray(mat, dtype=float)
    y = np.asarray(y, dtype=float)
    fit = fit_nss(mat, y, lam_grid=lam_grid)
    model_y = nss(mat, **fit)
    return (y - model_y) * BP


def build_residual_panel(panel, lam_grid=LAM_GRID):
    """縦持ちパネル(date, bond_id, maturity_years, yield)から
    残差パネル DataFrame(index=date, columns=bond_id, 値=残差bp) を作る。
    """
    rows = {}
    for date, grp in panel.groupby("date"):
        grp = grp.sort_values("maturity_years")
        resid = fit_date_residuals(grp["maturity_years"].values, grp["yield"].values, lam_grid)
        rows[date] = pd.Series(resid, index=grp["bond_id"].values)
    wide = pd.DataFrame(rows).T
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def residual_zscores(resid_wide):
    """残差パネルを銘柄ごとに時系列平均・標準偏差で基準化した Z スコアにする。"""
    mean = resid_wide.mean(axis=0)
    std = resid_wide.std(axis=0, ddof=1)
    return (resid_wide - mean) / std


def ar1_halflife(series):
    """1銘柄の残差時系列に AR(1) を自作 OLS でフィットし、半減期へ換算する。

    r_t = c + phi * r_{t-1} + eps を最小二乗で解く。返り値は
    phi, halflife(=-ln2/ln phi), phi の標準誤差 se, H0:phi=0 の t 値。
    phi<=0 など半減期が定義できないときは halflife=NaN を返す。
    """
    r = np.asarray(series, dtype=float)
    r = r[~np.isnan(r)]
    y = r[1:]
    x = r[:-1]
    n = y.size
    X = np.column_stack([np.ones(n), x])  # 定数項 + ラグ
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_phi = np.sqrt(cov[1, 1])
    phi = beta[1]
    tstat = phi / se_phi
    halflife = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.nan
    return dict(phi=phi, halflife=halflife, se=se_phi, tstat=tstat, n=n)


# %% [markdown]
# 読み込みと残差パネルの構築。60 営業日 × 40 銘柄のパネルを日次でフィットする。

# %%
universe = pd.read_csv("data/samples/synthetic_jgb_universe.csv")
panel = pd.read_csv("data/samples/synthetic_jgb_yield_panel.csv")

resid_panel = build_residual_panel(panel)
z_panel = residual_zscores(resid_panel)

print("残差パネル形状 (日付 x 銘柄):", resid_panel.shape)
print("期間:", resid_panel.index.min().date(), "〜", resid_panel.index.max().date())
print("\n最終営業日の残差(bp) 先頭5銘柄:")
display(resid_panel.iloc[-1].head().round(2))

# %% [markdown]
# ある1日を取り出し、フィット済みカーブと実測利回りを重ねる。カーブから上に
# 外れた点が cheap、下が rich である。

# %%
last_date = panel["date"].max()
snap = panel[panel["date"] == last_date].sort_values("maturity_years")
fit = fit_nss(snap["maturity_years"].values, snap["yield"].values, lam_grid=LAM_GRID)
tau_grid = np.linspace(snap["maturity_years"].min(), snap["maturity_years"].max(), 200)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(tau_grid, nss(tau_grid, **fit) * 100, color="steelblue", label="NSS フィット")
ax.scatter(snap["maturity_years"], snap["yield"] * 100, s=18, color="black", label="実測利回り")
ax.set_xlabel("残存年数")
ax.set_ylabel("利回り (%)")
ax.set_title(f"{last_date} の JGB カーブと実測利回り")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## QuantLib検証
#
# rich/cheap 分析の核はイールドカーブのパラメトリックフィット（NSS）と、その
# 残差の時系列統計である。QuantLib は債券・デリバティブの評価基盤だが、
# **NSS のようなクロスセクション・カーブフィットや残差の平均回帰推定は守備範囲
# 外**で、対応する部品を持たない。したがってここでは QuantLib と突合するのでは
# なく、次の2つを **合成データの既知の真値に対して回収できるか** で検証する。
# 検証の位置づけは「実装が正しく作られていることを、ground truth を仕込んだ
# 合成データで確かめる」ことであり、実データの利益を保証するものではない。
#
# ### 検証1：既知の rich/cheap 残差の回収
#
# ユニバース CSV の `rich_cheap_bp` 列は、合成データ生成時に各銘柄へ埋め込んだ
# **真の割高・割安**（bp）である。カーブフィットの日次残差を銘柄ごとに時系列
# 平均した $\bar{r}_i$ が、この埋め込み値を回収できるかを確認する。

# %%
mean_resid = resid_panel.mean(axis=0).rename("mean_resid_bp")
check = universe.merge(mean_resid, left_on="bond_id", right_index=True)
corr = np.corrcoef(check["mean_resid_bp"], check["rich_cheap_bp"])[0, 1]
mae = np.mean(np.abs(check["mean_resid_bp"] - check["rich_cheap_bp"]))
print(f"平均残差 vs 埋め込み rich_cheap_bp  相関: {corr:.3f}")
print(f"平均絶対誤差 (MAE): {mae:.2f} bp")
assert corr > 0.9, "既知残差の回収に失敗（相関が低い）"
print("検証1 合格: カーブ残差が埋め込んだ割高割安を回収できている")

# %% [markdown]
# 散布図でも、平均残差と真の値がほぼ 45 度線上に乗ることを確認する。

# %%
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(check["rich_cheap_bp"], check["mean_resid_bp"], s=20, color="steelblue")
lim = [check[["rich_cheap_bp", "mean_resid_bp"]].min().min() - 1,
       check[["rich_cheap_bp", "mean_resid_bp"]].max().max() + 1]
ax.plot(lim, lim, color="gray", lw=1, ls="--", label="45度線")
ax.set_xlabel("埋め込み rich_cheap (bp)")
ax.set_ylabel("推定 平均残差 (bp)")
ax.set_title("既知残差の回収")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 検証2：AR(1) 半減期の回収
#
# 半減期の換算式 $h_{1/2}=-\ln 2/\ln\phi$ と自作 OLS が正しいかを、既知の
# $\phi$ から生成した AR(1) 系列で確かめる。真の半減期に対し、推定半減期が近い
# 値を返すことを確認する。

# %%
rng = np.random.default_rng(1)
phi_true = 0.85
n_sim = 2000
sim = np.zeros(n_sim)
for t in range(1, n_sim):
    sim[t] = phi_true * sim[t - 1] + rng.normal(0, 1)

est = ar1_halflife(sim)
hl_true = -np.log(2) / np.log(phi_true)
print(f"真の phi = {phi_true},  推定 phi = {est['phi']:.3f}  (t値 {est['tstat']:.1f})")
print(f"真の半減期 = {hl_true:.2f} 期,  推定半減期 = {est['halflife']:.2f} 期")
assert abs(est["phi"] - phi_true) < 0.05, "AR(1) 係数の回収に失敗"
print("検証2 合格: 半減期の導出式と自作 OLS が既知値を回収できている")

# %% [markdown]
# 自作 OLS の妥当性を、`statsmodels` の OLS と突合して二重確認する（同じ係数・
# 標準誤差を返すはず）。`statsmodels` が無い環境では読み飛ばしても本文は動く。

# %%
try:
    import statsmodels.api as sm

    Xsm = sm.add_constant(sim[:-1])
    res = sm.OLS(sim[1:], Xsm).fit()
    print(f"statsmodels phi = {res.params[1]:.6f},  自作 phi = {est['phi']:.6f}")
    print(f"statsmodels se  = {res.bse[1]:.6f},  自作 se  = {est['se']:.6f}")
    assert abs(res.params[1] - est["phi"]) < 1e-8
    print("突合 合格: 自作 OLS と statsmodels が一致")
except ImportError:
    print("statsmodels 未導入のため突合はスキップ（本文は自作 OLS で完結）")

# %% [markdown]
# ## 実データ適用
#
# 合成 JGB ユニバースの直近営業日で、最も cheap（残差が正で大きい）銘柄を抽出
# する。ここでは絶対残差ではなく Z スコアを使う。恒常的に安い銘柄ではなく、
# 「その銘柄基準で今日際立って安い」銘柄を拾うためである。

# %%
z_last = z_panel.iloc[-1]
resid_last = resid_panel.iloc[-1]

ranking = pd.DataFrame({
    "z_score": z_last,
    "resid_bp": resid_last,
}).merge(universe.set_index("bond_id")[["maturity_years", "coupon"]],
         left_index=True, right_index=True)

cheapest5 = ranking.sort_values("z_score", ascending=False).head(5)
print("直近営業日の cheapest 上位5銘柄 (Z スコア降順):")
display(cheapest5.round(3))

# %% [markdown]
# 各 cheapest 銘柄の半減期を推定し、「収束を狙う取引妙味」を評価する。半減期が
# 短く Z スコアの絶対値が大きいほど、短期での収束益が見込める。

# %%
hl_rows = []
for bond_id in cheapest5.index:
    est = ar1_halflife(resid_panel[bond_id])
    hl_rows.append({
        "bond_id": bond_id,
        "z_score": z_last[bond_id],
        "resid_bp": resid_last[bond_id],
        "phi": est["phi"],
        "halflife_days": est["halflife"],
        "tstat": est["tstat"],
    })
hl_table = pd.DataFrame(hl_rows).set_index("bond_id")
display(hl_table.round(3))

# %% [markdown]
# 残差の時系列を描き、cheapest 上位銘柄が平均へ戻る動きを持つかを目視する。

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
for bond_id in cheapest5.index:
    ax.plot(resid_panel.index, resid_panel[bond_id], label=bond_id, lw=1.2)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("日付")
ax.set_ylabel("フィット残差 (bp)")
ax.set_title("cheapest 上位5銘柄の残差時系列")
ax.legend(ncol=5, fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 「本当に安いのか」— 流動性を含めた考察
#
# 抽出された cheapest 銘柄が本当に割安かは、残差の正しさ以上に**なぜ安いか**に
# よる。ポイントを整理する。
#
# - **半減期が短い銘柄**（`halflife_days` が小さく、`z_score` の絶対値が大きい）
#   は、一時的にカーブから外れただけで、収束に賭けるロングが報われやすい。
#   `phi` が有意に 1 未満であることが前提になる。
# - **半減期が長い/推定不能（NaN）な銘柄**は、残差が縮まず高止まりしている。
#   これは「安い」のではなく、流動性が低いことへの恒常的な対価
#   （off-the-run プレミアム）である可能性が高い。収束を狙っても流動性
#   プレミアムを負担し続けるだけになりうる。
# - 本データは満期・クーポン・利回りしか持たず、**出来高・ビッド/アスク・
#   発行後経過・on-the-run 区分といった流動性指標を欠く**。したがって
#   「Z スコアが高い=買い」と即断できない。実務では、流動性の代理変数で
#   残差を回帰し、残差から流動性成分を除いた「純粋な割安」で判定する。
#
# 合成データでは埋め込み値 `rich_cheap_bp` が真の割安なので、抽出銘柄の残差が
# それと整合するかを確認できる。実市場ではこの真値が観測できない点が、
# rich/cheap 分析の本質的な難しさである。

# %%
liq_check = cheapest5.join(universe.set_index("bond_id")["rich_cheap_bp"])
print("cheapest 上位5と埋め込み真値の整合:")
display(liq_check[["z_score", "resid_bp", "rich_cheap_bp"]].round(2))

# %% [markdown]
# ## 演習
#
# 1. **cheapest 銘柄抽出と流動性込み考察**：直近営業日ではなく、任意の過去
#    営業日を1つ選び、その日の Z スコアで cheapest 上位5銘柄を抽出せよ。抽出
#    された銘柄のうち、残差が「一時的な割安」か「構造的な（流動性起因の）割安」
#    かを、後述の半減期と残差の水準から論じよ。流動性指標が無いこのデータで
#    断定できない点を明記すること。
# 2. **半減期の推定と取引妙味の評価**：全銘柄について残差の AR(1) 半減期を推定
#    し、半減期の分布をヒストグラムにせよ。半減期が短い銘柄群と長い銘柄群を
#    比較し、収束トレードの対象としてどちらが妥当か、$\phi$ の有意性（$t$ 値）
#    も踏まえて述べよ。
#
# 解答例は `solutions/S9/sol_0901.py` にある。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/09_trading.md`。ここでは初出語の一行要約のみ示す。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | rich/cheap | rich/cheap | フィットカーブに対する割高（rich）・割安（cheap） |
# | フィット残差 | fitted residual | 実測利回りとモデル利回りの差。rich/cheap の尺度 |
# | 平均回帰半減期 | mean-reversion half-life | 乖離が半分に縮む期間。$-\ln 2/\ln\phi$ |
# | Z スコア | z-score | 残差を時系列平均・標準偏差で基準化した無次元量 |
# | on-the-run | on-the-run | 直近発行で最も流動性の高い銘柄。rich に寄りやすい |
