"""証券化商品（MBS）。

期限前償還（プリペイメント）モデル（PSA・条件付き計量モデル）と、MBS の
キャッシュフロー生成を提供する。S8 で理論を扱い、MC-OAS は S5 の金利モデルと
組み合わせて構築する。S8 はオプション回（米系運用志望向け）。
"""
from __future__ import annotations

import numpy as np


def psa_cpr(age_months, psa=100.0):
    """PSA モデルの年率 CPR（条件付き期限前償還率）。

    PSA 100 は、経過月 age に対し 0.2%×age で上昇し30ヶ月で6%に達して以降一定。
    psa はその倍率（PSA 200 なら2倍）。
    """
    age = np.asarray(age_months, dtype=float)
    base = np.minimum(0.06, 0.002 * age)
    return base * (psa / 100.0)


def cpr_to_smm(cpr):
    """年率 CPR を月次 SMM（single monthly mortality）へ変換する。"""
    cpr = np.asarray(cpr, dtype=float)
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def smm_to_cpr(smm):
    smm = np.asarray(smm, dtype=float)
    return 1.0 - (1.0 - smm) ** 12.0


def mbs_cashflows(balance, wac, wam, smm):
    """パススルー MBS の月次キャッシュフローを生成する。

    Parameters
    ----------
    balance : float
        初期プール残高。
    wac : float
        加重平均クーポン（年率）。月次金利は wac/12。
    wam : int
        加重平均残存月数。
    smm : array-like
        各月の SMM（長さ wam）。定数でもベクトルでもよい。

    Returns
    -------
    dict(month, interest, scheduled_principal, prepayment, total_principal,
         cashflow, balance) の配列
    """
    r = wac / 12.0
    smm = np.broadcast_to(np.asarray(smm, dtype=float), (wam,))
    bal = balance
    rows = {k: [] for k in ("month", "interest", "scheduled_principal",
                            "prepayment", "total_principal", "cashflow", "balance")}
    for m in range(1, wam + 1):
        n_rem = wam - m + 1
        interest = bal * r
        # 元利均等の予定元本
        if abs(r) < 1e-12:
            pmt = bal / n_rem + interest
        else:
            pmt = bal * r / (1.0 - (1.0 + r) ** (-n_rem)) if bal > 0 else 0.0
        sched_prin = min(pmt - interest, bal)
        prepay = (bal - sched_prin) * smm[m - 1]
        total_prin = sched_prin + prepay
        cf = interest + total_prin
        bal = bal - total_prin
        for k, v in zip(rows, (m, interest, sched_prin, prepay, total_prin, cf, bal)):
            rows[k].append(v)
    return {k: np.array(v) for k, v in rows.items()}


def weighted_average_life(cashflows) -> float:
    """加重平均年限 WAL = Σ (月/12) × 元本 / Σ 元本。"""
    m = cashflows["month"]
    prin = cashflows["total_principal"]
    return float(np.sum((m / 12.0) * prin) / np.sum(prin))
