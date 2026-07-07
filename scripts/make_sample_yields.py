"""S0-2 の教材用サンプル（合成の10年国債利回り）を生成する。

実データは再配布しないため、鍵の無い読者でも notebook が動くよう、現実的な
水準・変動を持つ合成系列を決定論的に生成して data/samples に置く。
本物の市場データではない点を README に明記する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "samples"


def synth_series(seed: int, start_level: float, drift: float, vol: float) -> pd.Series:
    rng = np.random.default_rng(seed)
    # 平日の営業日インデックス（2000-01 以降）。
    idx = pd.bdate_range("2000-01-03", "2025-12-31")
    n = len(idx)
    # 平均回帰つきのランダムウォークで利回りらしい系列を作る。
    x = np.empty(n)
    x[0] = start_level
    kappa = 0.001  # 緩やかな平均回帰
    theta = start_level + drift
    for i in range(1, n):
        x[i] = x[i - 1] + kappa * (theta - x[i - 1]) + vol * rng.standard_normal()
    # わずかに欠損を差し込み、欠損処理を演習で扱えるようにする。
    x = np.round(x, 2)
    s = pd.Series(x, index=idx, name="value")
    miss = rng.choice(n, size=n // 500, replace=False)
    s.iloc[miss] = np.nan
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {
        "synthetic_us10y": dict(seed=1, start_level=6.5, drift=-4.0, vol=0.04),
        "synthetic_jp10y": dict(seed=2, start_level=1.7, drift=-1.4, vol=0.02),
    }
    for name, kw in specs.items():
        s = synth_series(**kw)
        out = OUT / f"{name}.csv"
        s.rename("value").to_frame().reset_index(names="date").to_csv(out, index=False)
        print("wrote", out, "rows", len(s))


if __name__ == "__main__":
    main()
