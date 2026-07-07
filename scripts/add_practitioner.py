"""各 notebook の「学習目標」直後に実務観点セクションを挿入するヘルパ。

markdown セルを1つ差し込むだけで、コードセルの実行済み出力は保持する
（再実行しない）。.ipynb と jupytext の .py の両方へ入れる。冪等。
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

HEADING = "## 実務での位置づけ（ファンドはどう稼ぐか）"


def insert_ipynb(path: Path, section_md: str) -> bool:
    nb = nbformat.read(path, as_version=4)
    if any(HEADING in c.source for c in nb.cells if c.cell_type == "markdown"):
        return False
    idx = next((i for i, c in enumerate(nb.cells)
                if c.cell_type == "markdown" and "## 学習目標" in c.source), None)
    if idx is None:
        raise RuntimeError(f"'## 学習目標' が見つからない: {path}")
    nb.cells.insert(idx + 1, nbformat.v4.new_markdown_cell(section_md))
    nbformat.write(nb, path)
    return True


def insert_py(path: Path, section_md: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if HEADING in text:
        return False
    pos = text.find("## 学習目標")
    if pos < 0:
        raise RuntimeError(f"'## 学習目標' が見つからない: {path}")
    # 学習目標セルの次の「# %%」マーカー位置に、新しい markdown セルを差し込む
    nxt = text.find("\n# %%", pos)
    if nxt < 0:
        raise RuntimeError(f"次のセル境界が見つからない: {path}")
    body = "\n".join(("# " + ln) if ln else "#" for ln in section_md.splitlines())
    block = f"\n\n# %% [markdown]\n{body}"
    new_text = text[:nxt] + block + text[nxt:]
    path.write_text(new_text, encoding="utf-8")
    return True


def add(nb_stem: str, section_md: str) -> tuple[bool, bool]:
    """nb_stem は拡張子なしのパス（.ipynb と .py の両方を編集）。"""
    ipynb = Path(nb_stem + ".ipynb")
    py = Path(nb_stem + ".py")
    a = insert_ipynb(ipynb, section_md)
    b = insert_py(py, section_md)
    return a, b


if __name__ == "__main__":
    # 動作確認用: python add_practitioner.py <nb_stem> <section_md_file>
    stem = sys.argv[1]
    md = Path(sys.argv[2]).read_text(encoding="utf-8")
    print(add(stem, md))
