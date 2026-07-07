"""金利デリバティブ評価。

金利スワップ、キャップ/フロア（Black'76・Bachelier）、スワップション、そして
SABR スマイル（Hagan 近似）を提供する。S6 で理論を扱い、割引・フォワードは
S2 のカーブ、ボラは市場から与える。QuantLib と突合して検証する。
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq


# ---- スワップ ---------------------------------------------------------------

def swap_annuity(curve, times, start: float = 0.0):
    """固定レッグのアニュイティ Σ τ_i DF(t_i)。times は支払時点（年）の配列。

    start はスワップ開始時点（年）。既定 0 はスポットスタート。フォワード
    スタートのスワップ（例 5y5y）では start=5 を渡し、最初の発生期間を正しく
    1 期間にする（既定のままだと最初の τ が支払時点そのものになり誤る）。
    """
    times = np.asarray(times, dtype=float)
    tau = np.diff(np.concatenate([[start], times]))
    return float(np.sum(tau * curve.discount(times)))


def par_swap_rate(disc_curve, proj_curve, times, start: float = 0.0):
    """フォワード（推計）カーブと割引カーブからパースワップレートを求める。

    固定レッグ = 変動レッグ となるレート。変動レッグは単利フォワード
    $L_i = (P^{proj}(t_{i-1})/P^{proj}(t_i) - 1)/\\tau_i$ で評価する（連続複利
    フォワードではなくスワップ慣行の単利）。シングルカーブでは変動レッグが
    $1 - P(t_n)$ にテレスコープし、パーレートは $(1-P(t_n))/\\text{annuity}$ になる。
    マルチカーブでは割引と推計を分ける。
    """
    times = np.asarray(times, dtype=float)
    t_all = np.concatenate([[start], times])
    tau = np.diff(t_all)
    disc = disc_curve.discount(times)
    ann = float(np.sum(tau * disc))
    p_prev = proj_curve.discount(t_all[:-1])
    p_curr = proj_curve.discount(times)
    simple_fwd = (p_prev / p_curr - 1.0) / tau
    float_pv = float(np.sum(tau * disc * simple_fwd))
    return float_pv / ann


# ---- Black'76 / Bachelier --------------------------------------------------

def black76(forward, strike, expiry, vol, df=1.0, option="call"):
    """Black'76（対数正規）でのコール/プット価格。df は割引係数。"""
    F, K, T, s = forward, strike, expiry, vol
    if T <= 0 or s <= 0:
        payoff = max(F - K, 0.0) if option == "call" else max(K - F, 0.0)
        return df * payoff
    d1 = (np.log(F / K) + 0.5 * s ** 2 * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    if option == "call":
        return df * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    return df * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))


def bachelier(forward, strike, expiry, vol, df=1.0, option="call"):
    """Bachelier（正規）モデルでのコール/プット価格。低金利・負金利で使う。"""
    F, K, T, s = forward, strike, expiry, vol
    if T <= 0 or s <= 0:
        payoff = max(F - K, 0.0) if option == "call" else max(K - F, 0.0)
        return df * payoff
    d = (F - K) / (s * np.sqrt(T))
    sign = 1.0 if option == "call" else -1.0
    return df * (sign * (F - K) * stats.norm.cdf(sign * d) + s * np.sqrt(T) * stats.norm.pdf(d))


def implied_vol_black(price, forward, strike, expiry, df=1.0, option="call"):
    """Black'76 のインプライドボラティリティを数値解法で求める。"""
    intrinsic = df * (max(forward - strike, 0.0) if option == "call" else max(strike - forward, 0.0))
    if price <= intrinsic + 1e-14:
        return 0.0

    def f(s):
        return black76(forward, strike, expiry, s, df, option) - price

    return brentq(f, 1e-8, 5.0, xtol=1e-10, maxiter=200)


def caplet(forward, strike, expiry, vol, tau, df, model="black"):
    """1本のキャップレット価格。tau は対象期間、df はその期末までの割引係数。"""
    fn = black76 if model == "black" else bachelier
    return tau * fn(forward, strike, expiry, vol, df, option="call")


# ---- スワップション（Black） ------------------------------------------------

def swaption_black(forward_swap, strike, expiry, vol, annuity, option="payer", model="black"):
    """アニュイティ測度下の Black/Bachelier スワップション価格。

    payer = スワップレートのコール、receiver = プット。価格 = アニュイティ × Black。
    """
    opt = "call" if option == "payer" else "put"
    fn = black76 if model == "black" else bachelier
    return annuity * fn(forward_swap, strike, expiry, vol, df=1.0, option=opt)


# ---- SABR（Hagan 近似） -----------------------------------------------------

def sabr_vol(forward, strike, expiry, alpha, beta, rho, nu):
    """Hagan 近似式による SABR のインプライド（対数正規）ボラティリティ。

    forward=フォワード、alpha=水準ボラ、beta=CEV指数、rho=相関、nu=ボラのボラ。
    """
    F, K, T = forward, strike, expiry
    if abs(F - K) < 1e-12:  # ATM
        term1 = alpha / (F ** (1 - beta))
        term2 = (((1 - beta) ** 2 / 24) * alpha ** 2 / F ** (2 - 2 * beta)
                 + 0.25 * rho * beta * nu * alpha / F ** (1 - beta)
                 + (2 - 3 * rho ** 2) / 24 * nu ** 2)
        return term1 * (1 + term2 * T)
    logFK = np.log(F / K)
    fkbeta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * fkbeta * logFK
    x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    denom = fkbeta * (1 + (1 - beta) ** 2 / 24 * logFK ** 2 + (1 - beta) ** 4 / 1920 * logFK ** 4)
    factor = alpha / denom * (z / x)
    term = (((1 - beta) ** 2 / 24) * alpha ** 2 / fkbeta ** 2
            + 0.25 * rho * beta * nu * alpha / fkbeta
            + (2 - 3 * rho ** 2) / 24 * nu ** 2)
    return factor * (1 + term * T)
