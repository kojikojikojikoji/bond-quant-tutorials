"""S9 の教材用サンプル（合成の銘柄別 JGB ユニバースと日次利回りパネル）を生成する。

実データ（JSDA 売買参考統計値）は再配布しないため、カーブ＋銘柄固有の rich/cheap
残差＋日次のカーブ変動を持つ合成データを決定論的に生成する。本物ではない。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "samples"


def base_zero(tenor, level=0.0):
    b0, b1, b2, lam = 0.018 + level, -0.015, 0.012, 3.0
    x = tenor / lam
    t1 = (1 - np.exp(-x)) / x
    t2 = t1 - np.exp(-x)
    return b0 + b1 * t1 + b2 * t2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    # 40銘柄: 残存 0.5〜30年、クーポンは発行時カーブ近辺、銘柄固有の rich/cheap 残差
    mats = np.sort(rng.uniform(0.5, 30.0, 40))
    coupons = np.round(np.clip(base_zero(mats) + rng.uniform(-0.003, 0.003, 40), 0.001, None), 3)
    rc = rng.normal(0, 0.0004, 40)  # 銘柄固有の rich/cheap（±4bp 程度）
    issues = pd.DataFrame({
        "bond_id": [f"JGB{i:03d}" for i in range(40)],
        "maturity_years": np.round(mats, 3),
        "coupon": coupons,
        "rich_cheap_bp": np.round(rc * 1e4, 2),
    })
    issues.to_csv(OUT / "synthetic_jgb_universe.csv", index=False)

    # 60営業日の銘柄別利回りパネル（カーブ水準がゆっくり動く＋固有残差は persistent）
    dates = pd.bdate_range("2026-01-02", periods=60)
    level = 0.0
    rows = []
    for d in dates:
        level += 0.02 * (0.0 - level) + 0.0008 * rng.standard_normal()
        for _, r in issues.iterrows():
            fair = base_zero(r["maturity_years"], level)
            # 固有残差は AR(1) 的に少し揺れる
            y = fair + r["rich_cheap_bp"] / 1e4 + rng.normal(0, 0.00005)
            rows.append({"date": d.date().isoformat(), "bond_id": r["bond_id"],
                         "maturity_years": r["maturity_years"], "yield": round(float(y), 6)})
    pd.DataFrame(rows).to_csv(OUT / "synthetic_jgb_yield_panel.csv", index=False)
    print("wrote synthetic_jgb_universe.csv (40 bonds) and synthetic_jgb_yield_panel.csv")


if __name__ == "__main__":
    main()
