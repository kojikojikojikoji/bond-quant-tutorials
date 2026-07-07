"""RFQ プライサーの FastAPI ラッパー（S11-2 notebook のプライサーを HTTP 化）。

notebook `notebooks/S11_capstone/nb_1102_rfq_pricer.py` で作ったクォートエンジンを、
電子取引プラットフォームから叩ける `/quote` エンドポイントとして公開する。ミッドは
NSS カーブフィット値＋rich/cheap 残差、スプレッドはサイズ・在庫・ボラの関数で、
Avellaneda-Stoikov のリザベーション価格で在庫スキューを与える（notebook と同一ロジック）。

起動方法（このファイル自体はサーバを起動しない）:

    pip install fastapi uvicorn
    # リポジトリルートで（PYTHONPATH にルートを通す）
    PYTHONPATH=$PWD uvicorn app.rfq_pricer:app --reload

    curl -X POST localhost:8000/quote \\
      -H 'Content-Type: application/json' \\
      -d '{"bond_id": "JGB010", "size": 10, "side": "client_buy", "inventory": 4}'

FastAPI/pydantic が未導入でも、純粋関数（quote_core など）は import できるように、
Web 層は例外を握って任意化している。
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from bondlab.curve import fit_nss, nss

# ---- notebook と同一の定数・純粋関数 ---------------------------------------

LAM_GRID = np.linspace(0.8, 8.0, 8)
BP = 1e4
FREQ = 2


@dataclass
class RFQParams:
    gamma: float = 0.15      # リスク回避度
    kappa: float = 1.3       # 注文到達の強度（参考パラメータ）
    horizon: float = 5.0     # 在庫解消の想定日数 (T-t)
    base_ticks: float = 0.02  # 最低半スプレッド


def price_from_yield(coupon: float, years: float, ytm: float, freq: int = FREQ,
                     face: float = 100.0) -> float:
    """半年利払い標準クーポン債の YTM→クリーン価格（利払日決済を仮定）。"""
    n = max(int(round(years * freq)), 1)
    c = coupon / freq * face
    y = ytm / freq
    disc = (1.0 + y) ** (-np.arange(1, n + 1))
    return float(c * disc.sum() + face * disc[-1])


def modified_duration(coupon: float, years: float, ytm: float, freq: int = FREQ,
                      face: float = 100.0, h: float = 1e-4) -> float:
    """修正デュレーション（価格ボラを利回りボラから作るために使う）。"""
    p_up = price_from_yield(coupon, years, ytm + h, freq, face)
    p_dn = price_from_yield(coupon, years, ytm - h, freq, face)
    return -((p_up - p_dn) / (2.0 * h)) / price_from_yield(coupon, years, ytm, freq, face)


def quote_core(mid: float, sigma_p: float, inventory: float, size: float,
               params: RFQParams | None = None) -> dict:
    """在庫スキュー＋サイズ依存スプレッドでの bid/ask 計算（notebook と同一）。

    reservation = mid - q·γ·σ_P²·(T-t)
    half_spread = base + 0.5·γ·σ_P²·(T-t)·Q
    """
    p = params or RFQParams()
    reservation = mid - inventory * p.gamma * sigma_p ** 2 * p.horizon
    half = p.base_ticks + 0.5 * p.gamma * sigma_p ** 2 * p.horizon * size
    return dict(mid=mid, reservation=reservation,
                bid=reservation - half, ask=reservation + half, half_spread=half)


# ---- 評価日ブックの構築（CSV から前計算） ---------------------------------

@functools.lru_cache(maxsize=1)
def build_book(data_dir: str = "data/samples") -> pd.DataFrame:
    """ユニバース／利回りパネルから、直近営業日の (mid, sigma_p) ブックを作る。

    リポジトリルートを作業ディレクトリ（PYTHONPATH）に想定した相対パスで読む。
    結果は lru_cache で1回だけ計算する。
    """
    universe = pd.read_csv(f"{data_dir}/synthetic_jgb_universe.csv")
    panel = pd.read_csv(f"{data_dir}/synthetic_jgb_yield_panel.csv")

    # 残差パネル（rich/cheap の時系列平均）と日次利回りボラ
    rows = {}
    for date, grp in panel.groupby("date"):
        grp = grp.sort_values("maturity_years")
        fit = fit_nss(grp["maturity_years"].values, grp["yield"].values, lam_grid=LAM_GRID)
        resid = (grp["yield"].values - nss(grp["maturity_years"].values, **fit)) * BP
        rows[date] = pd.Series(resid, index=grp["bond_id"].values)
    resid_panel = pd.DataFrame(rows).T
    resid_mean_bp = resid_panel.mean(axis=0)
    wide = panel.pivot(index="date", columns="bond_id", values="yield").sort_index()
    daily_vol = wide.diff().std(axis=0, ddof=1)

    val_date = panel["date"].max()
    snap = panel[panel["date"] == val_date].set_index("bond_id")
    snap_yields = snap["yield"]

    univ = universe.set_index("bond_id")
    mat = univ["maturity_years"].reindex(snap_yields.index).values
    fit = fit_nss(mat, snap_yields.values, lam_grid=LAM_GRID)
    model_y = pd.Series(nss(mat, **fit), index=snap_yields.index)
    mid_yield = model_y + resid_mean_bp.reindex(snap_yields.index) / BP

    recs = {}
    for b in snap_yields.index:
        cpn = univ.loc[b, "coupon"]
        yrs = univ.loc[b, "maturity_years"]
        my = mid_yield[b]
        m = price_from_yield(cpn, yrs, my)
        dur = modified_duration(cpn, yrs, my)
        recs[b] = dict(mid=m, sigma_p=dur * m * daily_vol.get(b, np.nan))
    return pd.DataFrame(recs).T


def quote_bond(bond_id: str, size: float = 1.0, side: str = "two_way",
               inventory: float = 0.0, params: RFQParams | None = None) -> dict:
    """ブックを引いて1銘柄の bid/ask を返す。side で顧客約定価格も付す。"""
    book = build_book()
    if bond_id not in book.index:
        raise KeyError(f"unknown bond_id: {bond_id}")
    row = book.loc[bond_id]
    out = quote_core(row["mid"], row["sigma_p"], inventory, size, params)
    fill, inv_delta = None, 0.0
    if side == "client_buy":
        fill, inv_delta = out["ask"], -size
    elif side == "client_sell":
        fill, inv_delta = out["bid"], +size
    out.update(bond_id=bond_id, size=size, side=side,
               fill_price=fill, fill_inventory_delta=inv_delta)
    return out


# ---- Web 層（fastapi 未導入でもモジュールは import 可能にする） ------------

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class QuoteRequest(BaseModel):
        bond_id: str = Field(..., description="銘柄ID（例 JGB010）")
        size: float = Field(1.0, gt=0, description="RFQ サイズ")
        side: str = Field("two_way", description="two_way / client_buy / client_sell")
        inventory: float = Field(0.0, description="現在在庫（ロングが正）")

    app = FastAPI(title="RFQ Pricer", version="0.1.0",
                  description="S11-2 のミニ電子取引エンジン（bid/ask クォート）")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "bonds": int(build_book().shape[0])}

    @app.post("/quote")
    def quote(req: QuoteRequest) -> dict:
        try:
            return quote_bond(req.bond_id, req.size, req.side, req.inventory)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/params")
    def params() -> dict:
        return asdict(RFQParams())

except ImportError:  # fastapi 未導入環境では純粋関数のみ提供
    app = None
