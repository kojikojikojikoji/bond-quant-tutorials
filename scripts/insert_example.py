"""数式を説明している markdown セルの直後に「数値例」markdown を挿入するヘルパ。

コードセルの出力は触らない（再実行不要）。.ipynb と jupytext の .py の両方へ入れる。
anchor（対象 markdown セル内の一意な部分文字列）で挿入位置を決める。冪等。
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def insert_after(stem: str, anchor: str, md: str) -> str:
    ip = Path(stem + ".ipynb")
    py = Path(stem + ".py")
    key = md.strip()[:24]
    # --- ipynb ---
    nb = nbformat.read(ip, as_version=4)
    if any(key in c.source for c in nb.cells if c.cell_type == "markdown"):
        return "skip(dup)"
    idx = next((i for i, c in enumerate(nb.cells)
                if c.cell_type == "markdown" and anchor in c.source), None)
    if idx is None:
        return f"anchor NOT found in ipynb: {anchor[:30]!r}"
    nb.cells.insert(idx + 1, nbformat.v4.new_markdown_cell(md))
    nbformat.write(nb, ip)
    # --- py ---
    t = py.read_text(encoding="utf-8")
    pos = t.find(anchor)
    if pos < 0:
        return "ipynb OK / py anchor missing"
    nxt = t.find("\n# %%", pos)
    if nxt < 0:
        nxt = len(t)
    block = "\n\n# %% [markdown]\n" + "\n".join(("# " + ln) if ln else "#" for ln in md.splitlines())
    py.write_text(t[:nxt] + block + t[nxt:], encoding="utf-8")
    return "ok"


if __name__ == "__main__":
    # python insert_example.py <stem> <anchor> <md_file>
    stem, anchor, mdfile = sys.argv[1], sys.argv[2], sys.argv[3]
    print(insert_after(stem, anchor, Path(mdfile).read_text(encoding="utf-8")))
