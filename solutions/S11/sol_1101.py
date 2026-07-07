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
# # S11-1 演習 解答例
#
# 1. `bondlab` の関数を1つ選んでテストを書く
# 2. 層間の依存関係図を描き、循環が無いことを確認する

# %%
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import ast
import re
from pathlib import Path

import numpy as np

import bondlab
from bondlab.curve import bootstrap_par, DiscountCurve

# %% [markdown]
# ## 演習1：`DiscountCurve.forward_rate` のテストを書く
#
# 選んだ対象は `bondlab.curve.DiscountCurve.forward_rate`。連続複利フォワードは
# 定義から $f(t_1,t_2) = \ln(DF(t_1)/DF(t_2))/(t_2-t_1)$ で、次の性質を持つ。
#
# - **(a) 単体（閉形式一致）**：フラットな連続複利カーブでは、任意の区間の
#   フォワードはゼロレートに等しく、水準そのものになる。
# - **(b) 回帰（別経路の往復一致）**：`forward_rate` の値は割引係数から直接
#   計算した $\ln(DF_1/DF_2)/(t_2-t_1)$ と一致しなければならない。
# - **(c) 整合（0起点フォワード＝ゼロレート）**：始点を $t_1=0$ に取ると
#   $DF(0)=1$ なのでフォワードはゼロレートに一致する
#   （$f(0,t)=-\ln DF(t)/t = z(t)$）。補間方式によらず厳密に成り立つ。
#
# `pytest` の `tests/` に置く形の関数として書き、この解答内でも実行して緑を確認する。

# %%
def _flat_cc_curve(z=0.02, tmax=30.0):
    """連続複利ゼロレート一定（フラット）の割引カーブ。DF(t)=exp(-z t)。"""
    times = np.arange(1.0, tmax + 1.0)
    dfs = np.exp(-z * times)
    return DiscountCurve(times, dfs, interp="log_linear"), z


def test_forward_flat_equals_zero_rate():
    """(a) フラットカーブでは任意区間のフォワード = ゼロレート = 水準。"""
    curve, z = _flat_cc_curve(0.025)
    for t1, t2 in [(1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0)]:
        f = curve.forward_rate(t1, t2)
        assert abs(f - z) < 1e-10
        assert abs(f - curve.zero_rate(t2)) < 1e-10


def test_forward_matches_discount_ratio():
    """(b) 別経路：ln(DF(t1)/DF(t2))/(t2-t1) と一致（回帰の下地）。"""
    # わざと傾きのあるカーブ（フラットでない）で確認する。
    times = np.arange(1.0, 11.0)
    zeros = 0.01 + 0.002 * times           # 右肩上がり
    curve = DiscountCurve(times, np.exp(-zeros * times), interp="log_linear")
    for t1, t2 in [(1.0, 3.0), (2.0, 7.0), (4.0, 9.0)]:
        d1, d2 = curve.discount(t1), curve.discount(t2)
        expected = np.log(d1 / d2) / (t2 - t1)
        assert abs(curve.forward_rate(t1, t2) - expected) < 1e-12


def test_forward_from_zero_equals_zero_rate():
    """(c) 始点0のフォワードはゼロレートに厳密一致（DF(0)=1 より）。"""
    times = np.arange(1.0, 11.0)
    zeros = 0.01 + 0.002 * times
    curve = DiscountCurve(times, np.exp(-zeros * times), interp="log_linear")
    for t in [1.0, 2.5, 5.0, 9.0]:
        assert abs(curve.forward_rate(0.0, t) - curve.zero_rate(t)) < 1e-12


# この解答内でも実行して緑を確認する（pytest では自動収集される）。
for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
        fn()
        print(f"PASS {name}")

# %% [markdown]
# `tests/test_curve.py` の末尾にこれらを追記すれば、`python -m pytest -q` が
# 収集して実行する。3観点（閉形式一致・別経路の往復・極限）をカバーしており、
# 将来 `forward_rate` の実装を変えても、これらが回帰テストとして守りになる。

# %% [markdown]
# ## 演習2：依存関係図を構築し、循環が無いことを確認する
#
# `bondlab/*/__init__.py` を AST で解析し、`from bondlab.X import ...` と
# `import bondlab.X` を抽出して層間の有向グラフを作る。`A -> B` は「A が B に
# 依存」。トポロジカルソート（Kahn 法）が全ノードを消化できれば閉路は無い。

# %%
LAYERS = ["data", "rates", "daycount", "bond", "curve", "analytics", "risk",
          "sim", "models", "pricing", "credit", "mbs", "bt"]

pkg_root = Path(bondlab.__file__).resolve().parent


def layer_dependencies(layer):
    """1つの層の __init__.py を AST 解析し、依存する他の bondlab 層名の集合を返す。"""
    src = (pkg_root / layer / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    deps = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # from bondlab.X import ...
            m = re.match(r"bondlab\.(\w+)", node.module)
            if m and m.group(1) in LAYERS and m.group(1) != layer:
                deps.add(m.group(1))
            # from bondlab import X（X が層名）
            if node.module == "bondlab":
                for alias in node.names:
                    if alias.name in LAYERS and alias.name != layer:
                        deps.add(alias.name)
        # import bondlab.X
        elif isinstance(node, ast.Import):
            for alias in node.names:
                m = re.match(r"bondlab\.(\w+)", alias.name)
                if m and m.group(1) in LAYERS and m.group(1) != layer:
                    deps.add(m.group(1))
    return deps


graph = {layer: layer_dependencies(layer) for layer in LAYERS}

print("層間の依存エッジ（A -> B は「A が B に依存」）:")
for layer in LAYERS:
    if graph[layer]:
        print(f"  {layer:>10} -> {', '.join(sorted(graph[layer]))}")
print("  （上記以外の層は他の bondlab 層に依存しない）")


# %% [markdown]
# 抽出したエントリを Kahn のアルゴリズムでトポロジカルソートする。入次数0の
# ノードから順に取り除き、全ノードを消化できれば閉路は無い。

# %%
def topological_order(graph):
    """Kahn 法。閉路があれば ValueError。返り値は依存の浅い順の層リスト。"""
    # in_degree[n] = n に依存しているノード数ではなく、n が依存するノード数を消化する形にする。
    remaining = {n: set(deps) for n, deps in graph.items()}
    order = []
    while remaining:
        # 依存先が全て解決済み（remaining 内に依存が無い）ノードを取り出す。
        ready = [n for n, deps in remaining.items() if not deps]
        if not ready:
            raise ValueError(f"閉路を検出: {sorted(remaining)}")
        for n in sorted(ready):
            order.append(n)
            del remaining[n]
        for deps in remaining.values():
            deps.difference_update(order)
    return order


order = topological_order(graph)
print("トポロジカル順序（依存の浅い層から）:")
print("  " + " < ".join(order))
print("\n閉路なし（DAG）を確認：全13層をトポロジカルソートで消化できた")

# 理論節の宣言（bond->daycount, analytics->curve, models->curve）と一致するか。
expected_edges = {("bond", "daycount"), ("analytics", "curve"), ("models", "curve")}
actual_edges = {(a, b) for a, deps in graph.items() for b in deps}
assert actual_edges == expected_edges, f"宣言と実装の依存が不一致: {actual_edges}"
print("実装の依存エッジが理論節の宣言（3本）と完全一致")

# %% [markdown]
# ### もし上位層への依存を足したら
#
# 例えば `curve` 層に `from bondlab.analytics import duration_convexity` を
# 加えると、`analytics -> curve`（既存）と `curve -> analytics`（新規）が同時に
# 立ち、2ノードの閉路 `curve <-> analytics` が生じる。このときトポロジカルソートは
# 入次数0のノードを見つけられず `ValueError` を送出する。下位層（`curve`）が上位層
# （`analytics`）を知る設計は、変更の波及を「自分より上だけ見れば読める」という
# 不変条件を壊すため、依存は常に下向き一方向に保つ。

# %%
# 反例のデモ：curve -> analytics を人為的に足すと閉路になることを確認する。
broken = {n: set(d) for n, d in graph.items()}
broken["curve"].add("analytics")
try:
    topological_order(broken)
    print("（想定外：閉路を検出できなかった）")
except ValueError as e:
    print(f"想定どおり閉路を検出: {e}")
