"""クレジット。

強度（ハザード）モデルによる生存確率とリスキー割引、CDS の評価とハザード
カーブのブートストラップ、Merton の構造モデルを提供する。S7 で理論を扱い、
CVA（S7-5）は金利モデル（S5）と組み合わせて構築する。
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq


class HazardCurve:
    """区分定数ハザードレート λ による生存確率カーブ。

    times は各区間の右端（年、昇順）、hazards は対応する区間のハザード。
    生存確率 S(t) = exp(-∫_0^t λ(u) du)。
    """

    def __init__(self, times, hazards):
        self.times = np.asarray(times, dtype=float)
        self.hazards = np.asarray(hazards, dtype=float)
        if self.times.shape != self.hazards.shape:
            raise ValueError("times と hazards は同じ長さ")

    def _cum_hazard(self, t):
        """累積ハザード ∫_0^t λ。最終ノードより先は最後のハザードで外挿する。"""
        t = np.asarray(t, dtype=float)
        edges = np.concatenate([[0.0], self.times])
        cumh = np.concatenate([[0.0], np.cumsum(np.diff(edges) * self.hazards)])

        def one(tt):
            if tt <= 0:
                return 0.0
            i = int(np.searchsorted(self.times, tt))
            if i >= len(self.times):
                # 最終ノードより先: 最後のハザードで延長
                return cumh[-1] + (tt - self.times[-1]) * self.hazards[-1]
            return cumh[i] + (tt - edges[i]) * self.hazards[i]

        if t.ndim == 0:
            return one(float(t))
        return np.array([one(float(x)) for x in t])

    def survival(self, t):
        """生存確率 S(t)。"""
        return np.exp(-self._cum_hazard(t))

    def default_prob(self, t):
        """時点 t までの累積デフォルト確率 1 - S(t)。"""
        return 1.0 - self.survival(t)


def cds_legs(hazard: HazardCurve, disc_curve, maturity, recovery=0.4, freq=4, n_int=None):
    """CDS のプレミアムレッグ（アニュイティ）とプロテクションレッグを返す。

    プロテクションレッグは (1-R)∫ DF(t)(-dS(t)) を細かいグリッドで積分する。
    プレミアムレッグは、生存時の保険料 Σ τ DF S に加え、デフォルト時に直前
    クーポンからの経過保険料を払う分（accrual on default）を各クーポン区間で
    近似して足す。これを入れると QuantLib の CDS フェアスプレッドとよく一致する。

    Returns
    -------
    (risky_annuity, protection_pv)
        risky_annuity は「スプレッド1あたりのプレミアムレッグ PV」。
    """
    n_int = n_int or int(round(maturity * 200))
    tau = 1.0 / freq
    pay_times = np.arange(1, int(round(maturity * freq)) + 1) / freq
    surv = hazard.survival(pay_times)
    # 生存時の保険料
    risky_annuity = float(np.sum(tau * disc_curve.discount(pay_times) * surv))
    # accrual on default: 各クーポン区間でデフォルトすると平均で τ/2 の経過分を払う
    prev = np.concatenate([[0.0], pay_times[:-1]])
    s_prev = hazard.survival(prev)
    dS_period = s_prev - surv  # 区間内デフォルト確率
    mid = 0.5 * (prev + pay_times)
    risky_annuity += float(np.sum(0.5 * tau * disc_curve.discount(mid) * dS_period))
    # プロテクションレッグ（細かいグリッドで積分）
    grid = np.linspace(0.0, maturity, n_int + 1)
    S = hazard.survival(grid)
    DF = disc_curve.discount(0.5 * (grid[1:] + grid[:-1]))
    dS = -(S[1:] - S[:-1])
    protection = float((1 - recovery) * np.sum(DF * dS))
    return risky_annuity, protection


def cds_par_spread(hazard: HazardCurve, disc_curve, maturity, recovery=0.4, freq=4):
    """CDS のパースプレッド = プロテクションPV / リスキーアニュイティ。"""
    ann, prot = cds_legs(hazard, disc_curve, maturity, recovery, freq)
    return prot / ann


def bootstrap_hazard(disc_curve, tenors, spreads, recovery=0.4, freq=4):
    """CDS パースプレッド列から区分定数ハザードカーブを剥ぎ取る。"""
    tenors = np.asarray(tenors, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    hazards = []
    for i, (T, s) in enumerate(zip(tenors, spreads)):
        def obj(lam, i=i, T=T, s=s):
            h = HazardCurve(tenors[: i + 1], np.array(hazards + [lam]))
            return cds_par_spread(h, disc_curve, T, recovery, freq) - s

        lam = brentq(obj, 1e-8, 5.0, xtol=1e-12, maxiter=200)
        hazards.append(lam)
    return HazardCurve(tenors, np.array(hazards))


# ---- Merton 構造モデル ------------------------------------------------------

def merton_equity(asset, asset_vol, debt, expiry, r):
    """企業価値を原資産とするコールオプションとしての株式価値。"""
    V, sV, D, T = asset, asset_vol, debt, expiry
    d1 = (np.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
    d2 = d1 - sV * np.sqrt(T)
    return V * stats.norm.cdf(d1) - D * np.exp(-r * T) * stats.norm.cdf(d2)


def distance_to_default(asset, asset_vol, debt, expiry, r):
    """距離 to default（d2 に相当）。"""
    V, sV, D, T = asset, asset_vol, debt, expiry
    return (np.log(V / D) + (r - 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))


def merton_pd(asset, asset_vol, debt, expiry, r):
    """リスク中立デフォルト確率 N(-d2)。"""
    return float(stats.norm.cdf(-distance_to_default(asset, asset_vol, debt, expiry, r)))


def solve_asset(equity, equity_vol, debt, expiry, r, tol=1e-10):
    """観測できる株式価値・株式ボラから企業価値と資産ボラを逆算する。

    2式（株式=コール、株式ボラ=資産ボラ×Δ×V/E）を反復で解く。
    """
    V, sV = equity + debt * np.exp(-r * expiry), equity_vol
    for _ in range(200):
        d1 = (np.log(V / debt) + (r + 0.5 * sV ** 2) * expiry) / (sV * np.sqrt(expiry))
        E_model = merton_equity(V, sV, debt, expiry, r)
        delta = stats.norm.cdf(d1)
        sV_new = equity_vol * equity / (delta * V)
        V_new = V + (equity - E_model)
        if abs(V_new - V) < tol and abs(sV_new - sV) < tol:
            V, sV = V_new, sV_new
            break
        V, sV = V_new, sV_new
    return dict(asset=float(V), asset_vol=float(sV))
