"""S2 の教材用サンプル（合成のパー利回りカーブ）を生成する。

実データ（FRED CMT）は再配布しないため、鍵の無い読者でも動くよう、現実的な
形状のパー利回りカーブを決定論的に生成して data/samples に置く。本物では
ない点を README に明記する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "samples"
TENORS = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])


def base_curve() -> np.ndarray:
    # なだらかな順イールド（Nelson-Siegel 風の形状）。
    beta0, beta1, beta2, lam = 0.045, -0.020, 0.015, 2.5
    x = TENORS / lam
    t1 = (1 - np.exp(-x)) / x
    t2 = t1 - np.exp(-x)
    return beta0 + beta1 * t1 + beta2 * t2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = base_curve()
    pd.DataFrame({"tenor": TENORS, "par_yield": np.round(snap, 5)}).to_csv(
        OUT / "synthetic_ust_par_curve.csv", index=False
    )

    # 60営業日ぶんの合成パネル（水準と傾きをゆっくり動かす）。
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2026-01-02", periods=60)
    rows = []
    level = 0.0
    slope = 0.0
    for d in dates:
        level += 0.02 * (0.0 - level) + 0.01 * rng.standard_normal()
        slope += 0.02 * (0.0 - slope) + 0.005 * rng.standard_normal()
        curve = base_curve() + level * 0.01 + slope * 0.01 * (TENORS - 10) / 20
        for tn, y in zip(TENORS, curve):
            rows.append({"date": d.date().isoformat(), "tenor": tn, "par_yield": round(float(y), 5)})
    pd.DataFrame(rows).to_csv(OUT / "synthetic_ust_par_panel.csv", index=False)
    print("wrote synthetic_ust_par_curve.csv and synthetic_ust_par_panel.csv")


if __name__ == "__main__":
    main()
