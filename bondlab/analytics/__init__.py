"""リスク指標。

デュレーション・コンベクシティ・DV01（S3-1/3-2）、カーブ変動の主成分分析
（S3-4）を提供する。KRD（S3-3）はカーブとポートフォリオに依存するため、
カーブをバンプするヘルパ bump_curve を用意し、notebook 側で組み立てる。
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from bondlab.curve import DiscountCurve


def duration_convexity(bond, ytm: float, settlement: _dt.date) -> dict:
    """固定利付債のデュレーション・コンベクシティ・DV01 を解析的に求める。

    street convention の割引指数 $n=w+j$、年数 $t=n/f$ を使う。

        P = Σ c_j (1+y/f)^{-n_j}
        D_mac = (1/P) Σ t_j c_j (1+y/f)^{-n_j}
        D_mod = D_mac / (1+y/f)
        DV01  = D_mod · P · 1e-4
        C     = (1/P) Σ n_j (n_j+1) c_j (1+y/f)^{-n_j-2} / f^2

    Returns
    -------
    dict(macaulay, modified, convexity, dv01, dirty_price)
    """
    f = bond.frequency
    n, cf = bond.period_cashflows(settlement)
    if n.size == 0:
        return dict(macaulay=0.0, modified=0.0, convexity=0.0, dv01=0.0, dirty_price=0.0)
    disc = (1.0 + ytm / f) ** (-n)
    pv = cf * disc
    price = pv.sum()
    t = n / f
    macaulay = (t * pv).sum() / price
    modified = macaulay / (1.0 + ytm / f)
    dv01 = modified * price * 1e-4
    d2 = (cf * n * (n + 1) * (1.0 + ytm / f) ** (-n - 2)).sum() / (f ** 2)
    convexity = d2 / price
    return dict(
        macaulay=float(macaulay),
        modified=float(modified),
        convexity=float(convexity),
        dv01=float(dv01),
        dirty_price=float(price),
    )


def effective_duration(price_fn, y0: float, bump: float = 1e-4) -> float:
    """価格関数 price_fn(y) の実効デュレーションを中心差分で求める。

    コーラブル債など解析式が無い商品にも使える汎用版（S3-1/S6-4 で利用）。
    """
    p_up = price_fn(y0 + bump)
    p_dn = price_fn(y0 - bump)
    p0 = price_fn(y0)
    return -(p_up - p_dn) / (2.0 * bump * p0)


def bump_curve(curve: DiscountCurve, tenor: float, size: float, width: float | None = None) -> DiscountCurve:
    """カーブの特定テナー近傍のゼロレートをバンプした新カーブを返す。

    KRD（S3-3）でテナー別感応度を測るために使う。width を与えると、その幅で
    テナーへ線形に減衰する三角バンプ、None ならノード1点のみをバンプする。
    """
    times = curve.times.copy()
    zeros = np.array([curve.zero_rate(t) if t > 0 else 0.0 for t in times])
    if width is None:
        w = np.where(np.isclose(times, tenor), 1.0, 0.0)
    else:
        w = np.clip(1.0 - np.abs(times - tenor) / width, 0.0, 1.0)
    bumped_zero = zeros + size * w
    mask = times > 0
    dfs = np.exp(-bumped_zero[mask] * times[mask])
    return DiscountCurve(times[mask], dfs, interp=curve.interp)


def pca(changes: np.ndarray) -> dict:
    """カーブ変動行列（行=日, 列=テナー）の主成分分析。

    共分散行列を固有値分解し、寄与率の降順に並べて返す。

    Returns
    -------
    dict(eigenvalues, eigenvectors, explained_ratio)
        eigenvectors は列が各主成分（負荷ベクトル）。
    """
    x = np.asarray(changes, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    explained = vals / vals.sum()
    return dict(eigenvalues=vals, eigenvectors=vecs, explained_ratio=explained)
