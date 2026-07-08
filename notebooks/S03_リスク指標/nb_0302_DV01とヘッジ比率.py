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
# # S3-2 DV01とヘッジ比率
#
# ## 学習目標
#
# - DV01（dollar value of a basis point）・BPV（basis point value）・PV01（present value of a basis point）の定義を数式から導出し、使い分けられる
# - デュレーションヘッジが平行シフトしか消せないことを理解し、非平行シフトで残るPnLの発生源を説明できる
# - ポートフォリオDV01の集計と、複数銘柄でのヘッジ比率計算機を自分で実装できる
# - バンプ幅（1bp/10bp）と差分方式（片側/中心差分）の選択が、数値DV01の誤差にどう効くかを定量化できる
# - 解析DV01・バンプ再評価DV01・QuantLib の modified duration 由来DV01の三者が一致することを確認できる
# - 10年債ロングを2年+30年でDV01ニュートラルにヘッジし、非平行シフト日に残るPnLを水準・傾き・曲率へ要因分解できる


# %% [markdown]
# ## 実務での位置づけ（ファンドはどう稼ぐか）
#
# DV01は「利回りが1bp動いたときの損益額」を円やドルの金額に揃えた指標で、実務のヘッジはほぼこの単位で回ります。使い手の中心はレラティブバリュー（RV）のリサーチとトレーダー、そしてマーケットメイクのデスクです。RVは割安な銘柄を買い割高な銘柄を売って収束に賭けますが、この差益（キャリー）を取り出すには両者の金利感応度を打ち消して「価格のゆがみ」だけを残す必要があります。異なるクーポン・異なる年限の銘柄でも、DV01という共通の金額単位に揃えれば足し引きしてニュートラルに組めます。リスク管理クオンツはこのDV01をポートフォリオ単位で集計し、金利リスクの総量とデスク別リミットを管理します。
#
# 収益との繋がりは、ヘッジの品質がそのままアルファの純度になる点にあります。10年債ロングを2年+30年でDV01ニュートラルに組めば、カーブが平行に動く日はほぼ無風になり、狙ったRVスプレッドの収束だけがPnLに出ます。逆に言えば、デュレーションヘッジは平行シフトしか消せないため、スティープ化やバタフライのような非平行シフトの日には残存PnLが必ず出ます。この残差を水準・傾き・曲率へ要因分解できないと、儲かった／損した理由が「読んだ相場観が当たった」のか「消し切れなかったカーブリスクにたまたま賭けていた」のか区別できません。要因分解はキーレートヘッジ（S3-3）や損益要因分析の入口です。
#
# 具体的な場面として、マーケットメイカーは顧客のRFQに応じて在庫を抱えるたびにDV01が増減するため、日中これを見ながら先物やスワップで即座にヘッジして、稼ぎたいビッド・オファースプレッドだけを残し金利方向のリスクを落とします。DV01を解析式・バンプ再評価・QuantLib由来の三者で一致させて検算できないこと、あるいはバンプ幅や差分方式の誤差を把握できないことは、ヘッジ数量を誤ることに直結します。ヘッジ比率を1桁でも取り違えれば、ニュートラルのつもりのポジションが実は大きな金利ベットになっていた、という事故につながります。
# %% [markdown]
# ## 理論
#
# ### DV01・BPV・PV01 の定義
#
# 債券価格 $P$ を利回り $y$ の関数とみなします。DV01 は利回りが $1\,\mathrm{bp}=10^{-4}$
# 上がったときの価格の下落幅（円/ドル建て）として定義します。
#
# $$
# \mathrm{DV01} \;=\; -\frac{\partial P}{\partial y}\times 10^{-4}.
# $$
#
# 符号は「利回り上昇に対して価格が下がる」向きを正で測るための約束です。DV01 は
# 額面あたりの金額であり、パーセントでもデュレーション（年）でもありません。この
# 金額次元こそがヘッジで扱いやすい理由です。異なる銘柄・異なるクーポンでも、DV01
# は「1bp あたり何ドル動くか」という共通の単位に揃うため、そのまま足し引きできます。
#
# BPV は basis point value の略で、実務では DV01 と同義に使います。本 notebook でも
# DV01 と BPV は同じ量として扱います。一方 PV01 は present value of a basis point で、
# 「各利払日に $1\,\mathrm{bp}$ のクーポンが乗った年金（アニュイティ）の現在価値」を指し、
# 割引カーブから直接求めます。スワップのフィックス脚の感応度など、キャッシュフロー
# 側を $1\,\mathrm{bp}$ 動かす発想の量です。両者は近い値になりますが、DV01 が「利回りを
# 動かして再評価した価格差」なのに対し、PV01 は「$1\,\mathrm{bp}$ のクーポン年金の現在
# 価値」であり、出所が異なります。本 notebook では債券のヘッジを主題にするため、
# 利回り基準の DV01/BPV を軸に進めます。
#
# | 量 | 定義 | 主な用途 |
# |---|---|---|
# | DV01 | $-\partial P/\partial y\times10^{-4}$（利回りを1bp動かした価格差） | 債券・ポートフォリオの金利感応度、ヘッジ |
# | BPV | DV01 と同義 | 同上（呼び名が違うだけ） |
# | PV01 | 1bp のクーポン年金の現在価値（$\sum \tau_i\,\mathrm{DF}_i\times10^{-4}$ 系） | スワップのフィックス脚、割引ベースの感応度 |
#
# ### 修正デュレーションからの導出
#
# street convention の債券価格は、次クーポンまでの残り期間割合を $w$、$j$ 番目クーポン
# の割引指数を $n_j=w+j$、年 $f$ 回払いとして
#
# $$
# P(y)=\sum_j c_j\left(1+\frac{y}{f}\right)^{-n_j}.
# $$
#
# 利回りで微分すると
#
# $$
# \frac{\partial P}{\partial y}
# = -\frac{1}{f}\sum_j n_j\, c_j\left(1+\frac{y}{f}\right)^{-n_j-1}
# = -\frac{1}{1+y/f}\sum_j \frac{n_j}{f}\, c_j\left(1+\frac{y}{f}\right)^{-n_j}.
# $$
#
# ここで $t_j=n_j/f$ は年数です。マコーレー・デュレーション
# $D_{\mathrm{mac}}=\frac{1}{P}\sum_j t_j\,c_j(1+y/f)^{-n_j}$ と修正デュレーション
# $D_{\mathrm{mod}}=D_{\mathrm{mac}}/(1+y/f)$ を使うと
#
# $$
# \frac{\partial P}{\partial y}=-D_{\mathrm{mod}}\,P
# \quad\Longrightarrow\quad
# \boxed{\;\mathrm{DV01}=D_{\mathrm{mod}}\,P\times10^{-4}\;}
# $$
#
# が得られます。これは `bondlab.analytics.duration_convexity` が `dv01` として返す式
# そのものです。DV01 は修正デュレーションと価格（dirty price）の積に $10^{-4}$ を掛けた
# 金額であり、デュレーションが「年」、価格が「額面あたり金額」なので、DV01 は金額に
# なります。
#
# ### デュレーションヘッジの限界（非平行シフト）
#
# 対象ポートフォリオの DV01 を $\mathrm{DV01}_P$、ヘッジ商品の DV01 を $\mathrm{DV01}_H$
# とすると、ヘッジ数量
#
# $$
# h=-\frac{\mathrm{DV01}_P}{\mathrm{DV01}_H}
# $$
#
# だけ持てば、全銘柄の利回りが同じ $\Delta y$ だけ動く**平行シフト**に対して
# 一次の損益が打ち消えます。実際、平行シフトのPnLは各ポジションで
# $-\mathrm{DV01}_i\,(\Delta y/10^{-4})$ の和なので、$\sum_i \mathrm{DV01}_i=0$ ならゼロです。
#
# 問題は、現実のカーブが平行には動かない点です。年限 $T$ ごとに利回り変化を
# $\Delta y(T)$ と書き、基準年限 $T_0$ のまわりで
#
# $$
# \Delta y(T)=\underbrace{a}_{\text{水準}}+\underbrace{b\,(T-T_0)}_{\text{傾き}}
# +\underbrace{c\,(T-T_0)^2}_{\text{曲率}}+\cdots
# $$
#
# と分解します。ポートフォリオの一次PnLは
#
# $$
# \Delta P\approx -\sum_i \frac{\mathrm{DV01}_i}{10^{-4}}\,\Delta y(T_i)
# = -\frac{1}{10^{-4}}\Big[a\!\sum_i\!\mathrm{DV01}_i
# + b\!\sum_i\!\mathrm{DV01}_i(T_i-T_0)
# + c\!\sum_i\!\mathrm{DV01}_i(T_i-T_0)^2+\cdots\Big].
# $$
#
# DV01 中立（$\sum_i\mathrm{DV01}_i=0$）は水準項 $a$ しか消しません。傾き項は
# $\sum_i\mathrm{DV01}_i(T_i-T_0)$、曲率項は $\sum_i\mathrm{DV01}_i(T_i-T_0)^2$ が残ります。
# **単一商品での DV01 ヘッジは平行シフト専用**であり、傾き・曲率で損益が漏れます。
# これがデュレーションヘッジの本質的な限界です。
#
# ### 先物・スワップでのヘッジと、2商品ヘッジ
#
# ヘッジ商品としては、国債先物（受渡適格銘柄=CTDのDV01を転換係数で割った量が
# 先物1枚のDV01）や金利スワップ（レシーブ/ペイのDV01）を使います。いずれも
# $h=-\mathrm{DV01}_P/\mathrm{DV01}_H$ の形は同じで、先物なら枚数、スワップなら想定元本に
# 読み替えます。
#
# 傾きまで消したいときは、ヘッジ商品を2本にします。10年債（対象）を2年+30年で
# ヘッジする場合、$T_0$ を対象年限に取り、次の2本の連立を解きます。
#
# $$
# \begin{cases}
# \mathrm{DV01}_{P}+n_2\,\mathrm{DV01}_2+n_{30}\,\mathrm{DV01}_{30}=0 & \text{（水準中立）}\\[2pt]
# n_2\,\mathrm{DV01}_2\,(T_2-T_0)+n_{30}\,\mathrm{DV01}_{30}\,(T_{30}-T_0)=0 & \text{（傾き中立）}
# \end{cases}
# $$
#
# $T_0=T_{10}$ に取ると、対象10年債の傾き寄与 $(T_{10}-T_0)=0$ なので第2式の右辺が
# 消え、扱いが簡単になります。水準と傾きの2本を消すと、残るのは曲率
# $\sum_i\mathrm{DV01}_i(T_i-T_0)^2$ の項です。これがヘッジ後PnLの主因になります。
#
# ### ヘッジ後に残るPnLの要因分解
#
# 上の一次近似に加え、各ポジションのコンベクシティ $C_i$ による二次項
# $+\tfrac12 C_i P_i(\Delta y_i)^2$ も残ります。したがってヘッジ後の残存PnLは
#
# $$
# \Delta P_{\text{hedged}}
# \approx \underbrace{-\tfrac{c}{10^{-4}}\sum_i \mathrm{DV01}_i(T_i-T_0)^2}_{\text{曲率（非平行シフト）}}
# \;+\;\underbrace{\sum_i \tfrac12 C_i P_i(\Delta y_i)^2}_{\text{コンベクシティ（二次）}}
# \;+\;(\text{ベーシスリスク等})
# $$
#
# の3種に分けられます。第1項はカーブ形状の非平行成分、第2項は利回り変化の二乗に
# 効く曲率（コンベクシティ）、第3項は対象とヘッジ商品の利回りが完全に連動しない
# **ベーシスリスク**です。本 notebook では第1項を数値で分離して見せます。

# %% [markdown]
# ## スクラッチ実装
#
# `bondlab.analytics.duration_convexity` の `dv01`（解析DV01）を部品として使い、
# その上に「バンプ再評価DV01」「ポートフォリオDV01集計」「2商品ヘッジ比率」の
# 3つを自作します。すべて額面100あたりで統一します。
#
# ### 自作関数の仕様
#
# | 関数 | 引数 | 返り値 | 役割 |
# |---|---|---|---|
# | `dv01_bump(bond, ytm, settle, bump, method)` | 債券, 利回り, 決済日, バンプ幅, 差分方式 | DV01（額面100あたり） | 利回りをバンプして再評価した数値DV01。`central`=中心差分, `forward`=片側差分 |
# | `portfolio_dv01(positions, settle)` | ポジション列, 決済日 | (合計DV01, 明細DataFrame) | 各ポジションの解析DV01を額面比で集計する |
# | `solve_two_bond_hedge(target, hedge_a, hedge_b, settle, t_ref)` | 対象, ヘッジ2本, 決済日, 基準年限 | (面額A, 面額B) | 水準・傾き中立の2本連立を解きヘッジ面額を返す |
# | `pnl_reprice(bond, y0, y1, settle)` | 債券, 前利回り, 後利回り, 決済日 | PnL（額面100あたり） | 決済日固定で利回りだけ動かした dirty price 差 |

# %%
import datetime as dt

import numpy as np
import pandas as pd

import bondlab
from bondlab import bond as blbond
from bondlab.analytics import duration_convexity

np.random.seed(0)
print("bondlab version:", bondlab.__version__)


def dv01_bump(bond, ytm, settle, bump=1e-4, method="central"):
    """利回りをバンプして再評価した数値DV01（額面100あたり）を返す。

    DV01 = -dP/dy * 1e-4。中心差分は O(h^2)、片側差分は O(h) の打ち切り誤差。
    """
    p0 = bond.dirty_price(ytm, settle)
    p_up = bond.dirty_price(ytm + bump, settle)
    if method == "central":
        p_dn = bond.dirty_price(ytm - bump, settle)
        deriv = (p_up - p_dn) / (2.0 * bump)
    elif method == "forward":
        deriv = (p_up - p0) / bump
    else:
        raise ValueError(f"未知の差分方式: {method!r}")
    return -deriv * 1e-4


def portfolio_dv01(positions, settle):
    """ポジション列の解析DV01を額面比で集計する。

    positions: (ラベル, 債券, 利回り, 面額) のリスト。面額は額面通貨建て。
    DV01 は額面100あたりなので、面額/100 を掛けて合算する。
    """
    rows = []
    total = 0.0
    for label, bond, ytm, face in positions:
        dc = duration_convexity(bond, ytm, settle)
        pos_dv01 = (face / 100.0) * dc["dv01"]
        total += pos_dv01
        rows.append({
            "銘柄": label,
            "面額": face,
            "利回り%": round(ytm * 100, 4),
            "修正デュレ": round(dc["modified"], 4),
            "DV01/100": round(dc["dv01"], 6),
            "ポジションDV01": round(pos_dv01, 4),
        })
    return total, pd.DataFrame(rows)


def solve_two_bond_hedge(target, hedge_a, hedge_b, settle, t_ref):
    """水準・傾き中立の2本連立を解き、ヘッジ2本の面額 (n_a*100, n_b*100) を返す。

    target/hedge_*: (年限T, 債券, 利回り, 面額) のタプル。基準年限 t_ref のまわりで
        n_a*DV01_a + n_b*DV01_b = -DV01_target                     （水準中立）
        n_a*DV01_a*(T_a-t_ref) + n_b*DV01_b*(T_b-t_ref) = -DV01_target*(T_t-t_ref)
    を解く。n は「面額/100」。返り値は面額（n*100）。
    """
    def dv01_of(item):
        T, bond, ytm, face = item
        return T, (face / 100.0) * duration_convexity(bond, ytm, settle)["dv01"]

    Tt, dt_dv01 = dv01_of(target)
    Ta, da = dv01_of(hedge_a)  # 面額100基準の単位DV01（面額=100を渡す）
    Tb, db = dv01_of(hedge_b)

    A = np.array([[da, db],
                  [da * (Ta - t_ref), db * (Tb - t_ref)]])
    rhs = np.array([-dt_dv01,
                    -dt_dv01 * (Tt - t_ref)])
    n_a, n_b = np.linalg.solve(A, rhs)
    return n_a * 100.0, n_b * 100.0


def pnl_reprice(bond, y0, y1, settle):
    """決済日を固定し、利回りだけ y0→y1 に動かした dirty price 差（額面100あたり）。"""
    return bond.dirty_price(y1, settle) - bond.dirty_price(y0, settle)


# %% [markdown]
# ### バンプ幅・差分方式の影響を検証
#
# 額面100・クーポン4.35%・満期10年の債券で、解析DV01を真値として、バンプ再評価DV01の
# 誤差がバンプ幅（1bp/10bp）と差分方式（中心/片側）でどう変わるかを見ます。中心差分は
# 打ち切り誤差が $O(h^2)$、片側差分は $O(h)$ なので、片側×10bp が最も誤差が大きく、
# 中心×1bp が最も小さいはずです。

# %%
issue = dt.date(2026, 1, 2)
mat10 = dt.date(2036, 1, 2)
demo = blbond.FixedRateBond(issue, mat10, coupon=0.0435, frequency=2)
y_demo = 0.0435
settle = issue

dv01_exact = duration_convexity(demo, y_demo, settle)["dv01"]

records = []
for bump in (1e-4, 10e-4):
    for method in ("central", "forward"):
        approx = dv01_bump(demo, y_demo, settle, bump=bump, method=method)
        records.append({
            "バンプ幅": f"{int(round(bump * 1e4))}bp",
            "差分方式": "中心" if method == "central" else "片側",
            "数値DV01": round(approx, 8),
            "解析DV01": round(dv01_exact, 8),
            "誤差(絶対)": abs(approx - dv01_exact),
            "相対誤差": abs(approx - dv01_exact) / dv01_exact,
        })
bump_tbl = pd.DataFrame(records)
print(bump_tbl.to_string(index=False,
      formatters={"誤差(絶対)": "{:.2e}".format, "相対誤差": "{:.2e}".format}))

# %% [markdown]
# 中心差分は片側差分より数桁精度が高く、片側差分ではバンプ幅を10倍にすると誤差もほぼ
# 10倍になります（$O(h)$）。中心差分はバンプ幅を10倍にすると誤差がほぼ100倍になり
# ますが、絶対値そのものが小さいため実務では中心差分×小さめのバンプが安全です。

# %%
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(8, 5))
for method, marker, color in [("central", "o", "#1f77b4"), ("forward", "s", "#d62728")]:
    hs = np.array([0.5e-4, 1e-4, 2e-4, 5e-4, 10e-4, 20e-4])
    errs = [abs(dv01_bump(demo, y_demo, settle, bump=h, method=method) - dv01_exact) for h in hs]
    ax.loglog(hs * 1e4, errs, marker + "-", color=color,
              label="中心差分" if method == "central" else "片側差分")
ax.set_xlabel("バンプ幅 (bp)")
ax.set_ylabel("解析DV01との絶対誤差")
ax.set_title("差分方式・バンプ幅と数値DV01の誤差")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# %% [markdown]
# 両対数プロットで、片側差分の傾きは約1（$O(h)$）、中心差分の傾きは約2（$O(h^2)$）に
# なります。傾きが精度の次数をそのまま表します。

# %% [markdown]
# ## QuantLib検証
#
# 3つのDV01が一致することを確認します。(1) 解析DV01（`duration_convexity`）、
# (2) バンプ再評価DV01（中心差分・1bp）、(3) QuantLib の `BondFunctions.duration`
# で求めた modified duration から $\mathrm{DV01}=D_{\mathrm{mod}}\,P\times10^{-4}$ で作った値。
#
# QuantLib と機械精度で合わせるため、決済日を発行日（=クーポン日）に置き、day count を
# ActualActual(ISMA) にします。こうすると割引指数が整数の周期に一致し、street convention
# と同じ式になります。

# %%
import QuantLib as ql

dc_exact = duration_convexity(demo, y_demo, settle)
dv01_analytic = dc_exact["dv01"]
dv01_numeric = dv01_bump(demo, y_demo, settle, bump=1e-4, method="central")

ql_issue = ql.Date(2, 1, 2026)
ql_mat = ql.Date(2, 1, 2036)
ql_sched = ql.Schedule(
    ql_issue, ql_mat, ql.Period(ql.Semiannual), ql.NullCalendar(),
    ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False,
)
ql_daycount = ql.ActualActual(ql.ActualActual.ISMA, ql_sched)
ql_bond = ql.FixedRateBond(0, 100.0, ql_sched, [0.0435], ql_daycount)
ql.Settings.instance().evaluationDate = ql_issue

ql_rate = ql.InterestRate(y_demo, ql_daycount, ql.Compounded, ql.Semiannual)
ql_moddur = ql.BondFunctions.duration(ql_bond, ql_rate, ql.Duration.Modified)
ql_dirty = ql.BondFunctions.cleanPrice(ql_bond, ql_rate) + ql_bond.accruedAmount()
dv01_quantlib = ql_moddur * ql_dirty * 1e-4

cmp_tbl = pd.DataFrame({
    "手法": ["解析（duration_convexity）", "バンプ再評価（中心1bp）", "QuantLib（BondFunctions）"],
    "modified": [round(dc_exact["modified"], 8), np.nan, round(ql_moddur, 8)],
    "DV01": [round(dv01_analytic, 10), round(dv01_numeric, 10), round(dv01_quantlib, 10)],
})
print(cmp_tbl.to_string(index=False))

# バンプ再評価は1bpの中心差分なので、打ち切り誤差の分だけ緩めに判定する。
assert abs(dv01_numeric - dv01_analytic) < 1e-6
assert abs(dv01_quantlib - dv01_analytic) < 1e-6
print("\n解析・バンプ・QuantLib の3手法でDV01が一致します（< 1e-6）")

# %% [markdown]
# 3手法が一致することで、`duration_convexity` の解析式・自作のバンプ再評価・QuantLib の
# デュレーションが同じ量を測っていると確認できました。以降はこのDV01を部品として
# ヘッジを組みます。

# %% [markdown]
# ## 実データ適用
#
# 合成の実在風パーカーブ・パネル（`data/samples/synthetic_ust_par_panel.csv`、鍵不要・
# 再配布可）を使い、仮想ポートフォリオを組みます。手順は次の通りです。
#
# 1. パネルから、カーブ変化が最も非平行だった日（2s10s30sバタフライ変化が最大の日）を選ぶ
# 2. その前日の水準で、2年・10年・30年の実在風銘柄（パー発行）を作る
# 3. 10年債ロングを、2年+30年で「水準・傾き中立」にヘッジする
# 4. 当日の非平行シフトで決済日を固定して再評価し、残存PnLを水準・傾き・曲率へ分解する
#
# ネットワークアクセスは行いません。すべてローカルの合成データで完結します。

# %%
panel = pd.read_csv("data/samples/synthetic_ust_par_panel.csv")
wide = panel.pivot(index="date", columns="tenor", values="par_yield").sort_index()

# 2s10s30s バタフライ変化が最大の日（=最も非平行な日）を選ぶ。
chg = wide.diff().dropna()
butterfly = (chg[2.0] - 2.0 * chg[10.0] + chg[30.0]).abs()
day1 = butterfly.idxmax()
dates = list(wide.index)
day0 = dates[dates.index(day1) - 1]

shift_bp = (wide.loc[day1] - wide.loc[day0]) * 1e4
print(f"基準日(前日): {day0}   シフト日: {day1}")
print("年限別の利回り変化 (bp):")
print(shift_bp.round(3).to_string())
print(f"\n2s10s30s バタフライ変化: {butterfly.loc[day1] * 1e4:.2f} bp（非平行の度合い）")

# %% [markdown]
# 2年・10年・30年で符号や大きさが揃わない（例：短中期は小動き、超長期が大きく上昇）
# ため、平行シフトではありません。この非平行日でヘッジの効き方を検証します。

# %%
# 前日の水準でパー発行の実在風銘柄を作る（クーポン=当該年限のパー利回り→価格≒100）。
issue_d = dt.date(*map(int, day0.split("-")))


def make_par_bond(tenor_years):
    """発行日 issue_d・満期 tenor_years 年・クーポン=前日パー利回りの半年債を作る。"""
    y0 = float(wide.loc[day0, tenor_years])
    mat = dt.date(issue_d.year + tenor_years, issue_d.month, issue_d.day)
    bond = blbond.FixedRateBond(issue_d, mat, coupon=y0, frequency=2)
    return bond, y0


bond2, y2_0 = make_par_bond(2)
bond10, y10_0 = make_par_bond(10)
bond30, y30_0 = make_par_bond(30)
settle_d = issue_d  # 発行日決済（経過利子ゼロ、価格≒100）

# 各銘柄の当日利回り（前日 + 年限別シフト）。
y2_1 = y2_0 + float(shift_bp[2.0]) * 1e-4
y10_1 = y10_0 + float(shift_bp[10.0]) * 1e-4
y30_1 = y30_0 + float(shift_bp[30.0]) * 1e-4

# %% [markdown]
# ### ポートフォリオDV01の集計とヘッジ構築
#
# 10年債を額面 1,000万でロングします。まずヘッジ前のDV01を集計し、次に2年+30年で
# 水準・傾き中立になるヘッジ面額を解きます。

# %%
FACE10 = 10_000_000.0

# ヘッジ前（10年ロングのみ）。
pre_total, pre_tbl = portfolio_dv01([("10年ロング", bond10, y10_0, FACE10)], settle_d)
print("=== ヘッジ前ポートフォリオ ===")
print(pre_tbl.to_string(index=False))
print(f"合計DV01: {pre_total:,.2f}\n")

# 2年+30年で水準・傾き中立のヘッジ面額を解く（基準年限=10）。
face2, face30 = solve_two_bond_hedge(
    target=(10.0, bond10, y10_0, FACE10),
    hedge_a=(2.0, bond2, y2_0, 100.0),
    hedge_b=(30.0, bond30, y30_0, 100.0),
    settle=settle_d,
    t_ref=10.0,
)
print(f"ヘッジ面額: 2年 = {face2:,.0f} / 30年 = {face30:,.0f}")

positions = [
    ("10年ロング", bond10, y10_0, FACE10),
    ("2年ヘッジ", bond2, y2_0, face2),
    ("30年ヘッジ", bond30, y30_0, face30),
]
post_total, post_tbl = portfolio_dv01(positions, settle_d)
print("\n=== ヘッジ後ポートフォリオ ===")
print(post_tbl.to_string(index=False))
print(f"合計DV01: {post_total:,.6f}（≒0 なら水準中立）")

assert abs(post_total) < 1e-6
print("ヘッジ後の合計DV01は実質ゼロ（水準中立を確認）")

# %% [markdown]
# ### 非平行シフト日のPnLと要因分解
#
# 決済日を発行日に固定し、利回りだけ当日水準へ動かして各銘柄を再評価します。ヘッジ前
# （10年ロングのみ）とヘッジ後（2年+30年込み）のPnLを比べ、ヘッジ後の残存PnLを
# 水準・傾き・曲率へ分解します。

# %%
# 各銘柄の実PnL（額面100あたり×面額比）。
pnl10 = FACE10 / 100.0 * pnl_reprice(bond10, y10_0, y10_1, settle_d)
pnl2 = face2 / 100.0 * pnl_reprice(bond2, y2_0, y2_1, settle_d)
pnl30 = face30 / 100.0 * pnl_reprice(bond30, y30_0, y30_1, settle_d)

pnl_unhedged = pnl10
pnl_hedged = pnl10 + pnl2 + pnl30
print(f"ヘッジ前PnL（10年のみ）: {pnl_unhedged:,.2f}")
print(f"ヘッジ後PnL（2+10+30） : {pnl_hedged:,.2f}")
print(f"ヘッジで消えたPnL      : {pnl_unhedged - pnl_hedged:,.2f}")

# %% [markdown]
# 続いて、ヘッジ後の残存PnLを理論の分解式で説明します。年限 $\{2,10,30\}$ の利回り変化を
# 基準 $T_0=10$ のまわりで $\Delta y(T)=a+b(T-10)+c(T-10)^2$ に当てはめます（3点3係数で
# 一意）。各項がヘッジ後PnLに与える寄与を計算します。

# %%
Tn = np.array([2.0, 10.0, 30.0])
dy = np.array([shift_bp[2.0], shift_bp[10.0], shift_bp[30.0]]) * 1e-4  # 実数の利回り変化
# a + b(T-10) + c(T-10)^2 = dy を解く（Vandermonde、3x3）。
V = np.column_stack([np.ones_like(Tn), (Tn - 10.0), (Tn - 10.0) ** 2])
a, b, c = np.linalg.solve(V, dy)
print(f"水準 a = {a * 1e4:.3f} bp / 傾き b = {b * 1e4:.4f} bp/年 / 曲率 c = {c * 1e4:.5f} bp/年^2")

# ポジションDV01（額面比、符号付き）。
dv01_pos = {
    2.0: face2 / 100.0 * duration_convexity(bond2, y2_0, settle_d)["dv01"],
    10.0: FACE10 / 100.0 * duration_convexity(bond10, y10_0, settle_d)["dv01"],
    30.0: face30 / 100.0 * duration_convexity(bond30, y30_0, settle_d)["dv01"],
}
# 各項の一次PnL寄与 = -(1/1e-4) * Σ DV01_i * (項)。
level_pnl = -sum(dv01_pos[T] * a for T in Tn) / 1e-4
slope_pnl = -sum(dv01_pos[T] * b * (T - 10.0) for T in Tn) / 1e-4
curv_pnl = -sum(dv01_pos[T] * c * (T - 10.0) ** 2 for T in Tn) / 1e-4
linear_sum = level_pnl + slope_pnl + curv_pnl

decomp = pd.DataFrame({
    "要因": ["水準（平行）", "傾き（スロープ）", "曲率（カーブ）", "一次合計", "実測ヘッジ後PnL", "残差（二次以上）"],
    "PnL": [level_pnl, slope_pnl, curv_pnl, linear_sum, pnl_hedged, pnl_hedged - linear_sum],
})
print("\n=== ヘッジ後PnLの要因分解 ===")
print(decomp.to_string(index=False, formatters={"PnL": "{:,.2f}".format}))

# %% [markdown]
# 水準項と傾き項はほぼゼロ（2本のヘッジで消したため）で、残存PnLの大半は曲率項が
# 占めます。一次合計と実測ヘッジ後PnLの差はコンベクシティ（二次）とベーシスの寄与で、
# 利回り変化が小さいため僅かです。DV01ニュートラルにしても非平行シフトの曲率成分は
# 消せない、という理論の結論を数値で確認できました。

# %%
fig, ax = plt.subplots(figsize=(9, 5))
labels = ["ヘッジ前\n(10年のみ)", "ヘッジ後\n(2+10+30)", "水準", "傾き", "曲率"]
vals = [pnl_unhedged, pnl_hedged, level_pnl, slope_pnl, curv_pnl]
colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
ax.bar(labels, vals, color=colors)
ax.axhline(0.0, color="k", lw=0.8)
ax.set_ylabel("PnL")
ax.set_title(f"非平行シフト日（{day1}）のヘッジ効果と残存PnLの分解")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 演習
#
# 1. **DV01ニュートラルヘッジと残存PnLの要因分解**：本編と同じパネルから、
#    2s10s30sバタフライ変化が **2番目に大きい** 日を選べ。その日について、
#    5年債ロング（額面 1,000万）を2年+30年で水準・傾き中立にヘッジし、非平行シフトの
#    残存PnLを水準・傾き・曲率へ分解せよ。10年ヘッジのときと比べ、基準年限 $T_0$ を
#    対象年限（5年）に取ると第2式の右辺がどうなるかを述べよ。
# 2. **バンプ幅・差分方式の定量化**：満期30年・クーポン4.458%の債券について、
#    バンプ幅 $\{1,10\}\,\mathrm{bp}$ × 差分方式 $\{$中心, 片側$\}$ の4通りで数値DV01を求め、
#    解析DV01との相対誤差を表にせよ。さらにバンプ幅を $0.5$〜$50\,\mathrm{bp}$ で振り、
#    両対数プロットの傾きから中心差分が $O(h^2)$、片側差分が $O(h)$ であることを確認せよ。
#
# 解答例は `solutions/S3/sol_0302.py` に置きます。

# %% [markdown]
# ## 用語集
#
# 定義の正は `glossary/03_risk.md`。ここでは初出語の一行要約のみ示します。
#
# | 用語 | 英語 | 一行定義 |
# |---|---|---|
# | DV01 | dollar value of a basis point | 利回りを1bp動かしたときの価格変化額。$D_{\mathrm{mod}}\,P\times10^{-4}$ |
# | BPV | basis point value | DV01 と同義。1bpあたりの価格感応度を金額で表したもの |
# | ヘッジ比率 | hedge ratio | 対象と反対のDV01を持つためのヘッジ数量。$-\mathrm{DV01}_P/\mathrm{DV01}_H$ |
# | ベーシスリスク | basis risk | 対象とヘッジ商品の利回りが完全連動せず残るリスク |
# | 中心差分 | central difference | 前後を対称にバンプする数値微分。打ち切り誤差が $O(h^2)$ |
