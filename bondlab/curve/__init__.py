"""イールドカーブ構築。

割引カーブ（時点→割引係数）を保持する DiscountCurve と、パー利回りからの
ブートストラップ、Nelson-Siegel-Svensson フィットを提供する。S2 で理論を
扱い、以降のデリバティブ評価・リスク層が土台に使う。

補間は割引係数のログ線形（既定）と、ゼロレートの線形に対応する。QuantLib の
PiecewiseLogLinearDiscount と突合して検証する。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class DiscountCurve:
    """時点（年）と割引係数からなる割引カーブ。

    Parameters
    ----------
    times : sequence of float
         node の年数（昇順、正）。
    dfs : sequence of float
        各 node の割引係数（減少列を想定）。
    interp : str
        "log_linear"（割引係数の対数を線形補間）または
        "linear_zero"（連続複利ゼロレートを線形補間）。
    """

    def __init__(self, times: Sequence[float], dfs: Sequence[float], interp: str = "log_linear"):
        t = np.asarray(times, dtype=float)
        d = np.asarray(dfs, dtype=float)
        if t.ndim != 1 or t.shape != d.shape:
            raise ValueError("times と dfs は同じ長さの1次元配列である必要がある")
        if np.any(np.diff(t) <= 0):
            raise ValueError("times は狭義単調増加である必要がある")
        # t=0, DF=1 を暗黙のノードとして先頭に加える。
        if t[0] > 0:
            t = np.concatenate([[0.0], t])
            d = np.concatenate([[1.0], d])
        self.times = t
        self.dfs = d
        self.interp = interp
        self._logdf = np.log(d)
        self._zero = np.where(t > 0, -self._logdf / np.where(t > 0, t, 1.0), 0.0)

    def discount(self, t):
        """時点 t の割引係数。ノード間は補間、端は最近傍の傾きで外挿する。

        スカラー入力にはスカラー（float）、配列入力には配列を返す。
        """
        scalar = np.ndim(t) == 0
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        if self.interp == "log_linear":
            log_df = np.interp(tt, self.times, self._logdf)
            # 端点の外挿（np.interp は端でクリップするため、傾きで延長する）
            log_df = self._extrapolate(tt, self.times, self._logdf, log_df)
            df = np.exp(log_df)
        elif self.interp == "linear_zero":
            z = np.interp(tt, self.times[1:], self._zero[1:])
            z = self._extrapolate(tt, self.times[1:], self._zero[1:], z)
            df = np.exp(-z * tt)
        else:
            raise ValueError(f"未知の補間: {self.interp!r}")
        return float(df[0]) if scalar else df

    @staticmethod
    def _extrapolate(t, xs, ys, interp_vals):
        """np.interp のクリップを、端の傾きによる線形外挿へ置き換える。"""
        t = np.atleast_1d(t).astype(float)
        out = np.atleast_1d(interp_vals).astype(float)
        left = t < xs[0]
        right = t > xs[-1]
        if np.any(left):
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            out[left] = ys[0] + slope * (t[left] - xs[0])
        if np.any(right):
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            out[right] = ys[-1] + slope * (t[right] - xs[-1])
        return out if out.shape != () else float(out)

    def zero_rate(self, t):
        """連続複利ゼロレート $z(t) = -\\ln DF(t)/t$。t>0 で定義する。

        スカラー入力にはスカラー、配列入力には配列を返す。
        """
        scalar = np.ndim(t) == 0
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        z = np.where(tt > 0, -np.log(self.discount(tt)) / np.where(tt > 0, tt, 1.0), np.nan)
        return float(z[0]) if scalar else z

    def forward_rate(self, t1, t2):
        """区間 [t1, t2] の連続複利フォワードレート。

        スカラー入力にはスカラー、配列入力には配列を返す。
        """
        scalar = np.ndim(t1) == 0 and np.ndim(t2) == 0
        d1 = self.discount(t1)
        d2 = self.discount(t2)
        f = np.log(np.asarray(d1) / np.asarray(d2)) / (np.asarray(t2, float) - np.asarray(t1, float))
        return float(f) if scalar else f


def bootstrap_par(
    tenors: Sequence[float],
    par_rates: Sequence[float],
    frequency: int = 1,
    interp: str = "log_linear",
) -> DiscountCurve:
    """パー利回りから割引係数を逐次に剥ぎ取る（ブートストラップ）。

    各テナー $t_i$ のパー債（クーポン率 = パー利回り、額面1）が価格1になる
    条件から $DF(t_i)$ を解く。年 $f$ 回払いを仮定し、テナーは払込日に一致
    するものとする（等間隔グリッド）。

        1 = c_i/f * Σ_{k≤i} DF(t_k) + DF(t_i)
        DF(t_i) = (1 - c_i/f * Σ_{k<i} DF(t_k)) / (1 + c_i/f)

    Parameters
    ----------
    tenors : sequence of float
        テナー（年）。等間隔（1/frequency 刻み）を想定。
    par_rates : sequence of float
        各テナーのパー利回り（小数）。
    frequency : int
        年間クーポン回数。
    """
    tenors = np.asarray(tenors, dtype=float)
    par = np.asarray(par_rates, dtype=float)
    if tenors.shape != par.shape:
        raise ValueError("tenors と par_rates は同じ長さ")
    # この単純なブートストラップは「テナー = 等間隔のクーポン払込日」を仮定する。
    # 非等間隔グリッド（例 0.5,1,2,3,5,7,10,20,30）を渡すと、間の払込日を無視して
    # ゼロカーブが歪む。年次グリッドへ補間してから渡すこと。誤用を早く気づけるよう
    # 警告する（例外にはしない。呼び出し側が意図的に近似する場合もあるため）。
    if tenors.size >= 3:
        steps = np.diff(tenors)
        if not np.allclose(steps, steps[0], rtol=0.05, atol=1e-9):
            import warnings
            warnings.warn(
                "bootstrap_par: テナーが等間隔でない。間のクーポン払込日を無視して"
                "カーブが歪む可能性がある。1/frequency 刻みのグリッドへ補間してから"
                "渡すことを推奨。",
                stacklevel=2,
            )
    dfs = np.empty_like(tenors)
    running = 0.0  # Σ DF over previous coupon dates
    for i, (t, c) in enumerate(zip(tenors, par)):
        cpn = c / frequency
        dfs[i] = (1.0 - cpn * running) / (1.0 + cpn)
        running += dfs[i]
    return DiscountCurve(tenors, dfs, interp=interp)


def _ns_terms(tau, lam1, lam2):
    """NSS の3つの因子負荷 (t1, t2, t3) を返す。tau=0 の極限も正しく扱う。

    tau→0 で (1-e^{-x})/x → 1 なので t1→1, t2→0, t3→0。
    """
    tau = np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x1 = tau / lam1
        x2 = tau / lam2
        e1 = np.exp(-x1)
        e2 = np.exp(-x2)
        loading1 = np.where(tau > 0, (1 - e1) / np.where(x1 == 0, 1.0, x1), 1.0)
        t1 = loading1
        t2 = loading1 - e1
        loading2 = np.where(tau > 0, (1 - e2) / np.where(x2 == 0, 1.0, x2), 1.0)
        t3 = loading2 - e2
    return t1, t2, t3


def nss(tau, beta0, beta1, beta2, beta3, lam1, lam2):
    """Nelson-Siegel-Svensson のゼロレート関数。

    tau は年数（0以上）。beta0=長期水準、beta1=スロープ、beta2/beta3=曲率。
    tau=0 では極限 $z(0)=\\beta_0+\\beta_1$ を返す。
    """
    t1, t2, t3 = _ns_terms(tau, lam1, lam2)
    return beta0 + beta1 * t1 + beta2 * t2 + beta3 * t3


def fit_nss(tenors, yields, lam_grid=None):
    """NSS をゼロ/パー利回りにフィットして係数を返す。

    lambda を格子探索し、各 lambda で線形部分（beta）を最小二乗で解く2段階法。
    返り値は dict(beta0, beta1, beta2, beta3, lam1, lam2)。
    """
    tenors = np.asarray(tenors, dtype=float)
    yields = np.asarray(yields, dtype=float)
    if lam_grid is None:
        lam_grid = np.linspace(0.5, 10.0, 20)

    lam_grid = np.atleast_1d(np.asarray(lam_grid, dtype=float))
    if lam_grid.size < 2:
        raise ValueError("lam_grid には lam1<lam2 を作れる2点以上が必要")
    best = None
    for lam1 in lam_grid:
        for lam2 in lam_grid:
            if lam2 <= lam1:
                continue
            # 与えた lambda では NSS は beta について線形なので、線形最小二乗の
            # 閉形式（lstsq）で beta を一発で解ける。非線形ソルバより速く安定。
            t1, t2, t3 = _ns_terms(tenors, lam1, lam2)
            design = np.column_stack([np.ones_like(tenors), t1, t2, t3])
            beta, *_ = np.linalg.lstsq(design, yields, rcond=None)
            cost = np.sum((design @ beta - yields) ** 2)
            if best is None or cost < best[0]:
                best = (cost, beta, lam1, lam2)

    _, beta, lam1, lam2 = best
    return dict(beta0=beta[0], beta1=beta[1], beta2=beta[2], beta3=beta[3], lam1=lam1, lam2=lam2)
