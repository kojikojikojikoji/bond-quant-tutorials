"""日次カーブモニタリングダッシュボード（Streamlit 版）。

このアプリは notebook `notebooks/S11_総合演習/nb_1103_dashboard.py` の
`## スクラッチ実装` と**同一ロジック**の計算関数を用い、同じ6パネル＋アラート帯
のダッシュボードを描画します。計算（compute_dashboard）と描画（render_dashboard）
は notebook と概念的に共通で、ここでは Streamlit の UI（基準日と分位点の選択）を
かぶせただけです。

実行方法（このファイルは実行せず、構文チェックのみ想定）:

    cd <repo root>
    PYTHONPATH=$PWD MPLBACKEND=Agg streamlit run app/dashboard.py

サイドバーで「基準日（as-of）」と「アラート閾値の分位点 q」を選ぶと、その日
までの履歴だけ（先読みなし）で全パネルとアラートが再計算されます。
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bondlab.curve import bootstrap_par, fit_nss, nss, DiscountCurve
from bondlab.analytics import bump_curve

BP = 1e4
BENCH_TENORS = (0.5, 2.0, 5.0, 10.0, 30.0)
FLY_LEGS = (2.0, 5.0, 10.0)
KEY_TENORS = (2.0, 5.0, 10.0, 30.0)
HOLD_HORIZON = 0.25
DATA_PATH = Path("data") / "samples" / "synthetic_ust_par_panel.csv"


# --------------------------------------------------------------------------
# 共通計算関数（notebook S11-3 と同一ロジック）
# --------------------------------------------------------------------------
def load_par_panel(path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    expected = {"date", "tenor", "par_yield"}
    if set(raw.columns) != expected:
        raise ValueError(f"想定外の列: {list(raw.columns)}")
    raw["date"] = pd.to_datetime(raw["date"])
    return raw.pivot(index="date", columns="tenor", values="par_yield").sort_index().sort_index(axis=1)


def _curve_from_row(tenors, par_rates, freq=1) -> DiscountCurve:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bootstrap_par(np.asarray(tenors, float), np.asarray(par_rates, float), frequency=freq)


def build_zero_panel(panel, freq=1) -> pd.DataFrame:
    tenors = np.asarray(panel.columns, dtype=float)
    rows = {d: _curve_from_row(tenors, row.to_numpy(float), freq).zero_rate(tenors)
            for d, row in panel.iterrows()}
    out = pd.DataFrame(rows).T
    out.columns = tenors
    out.index.name = "date"
    return out


def price_bond_on_curve(curve, coupon, maturity, freq=1):
    n = int(round(maturity * freq))
    times = np.arange(1, n + 1) / freq
    cfs = np.full(n, coupon / freq)
    cfs[-1] += 1.0
    return float(np.sum(cfs * curve.discount(times)))


def key_rate_durations(curve, coupon, maturity, keys=KEY_TENORS, freq=1, bump=1e-4, width=None):
    keys = np.asarray(keys, dtype=float)
    if width is None:
        width = float(np.median(np.diff(keys)))
    p0 = price_bond_on_curve(curve, coupon, maturity, freq)
    krd = {}
    for k in keys:
        p_up = price_bond_on_curve(bump_curve(curve, k, bump, width=width), coupon, maturity, freq)
        p_dn = price_bond_on_curve(bump_curve(curve, k, -bump, width=width), coupon, maturity, freq)
        krd[k] = -(p_up - p_dn) / (2.0 * bump * p0)
    return pd.Series(krd, name="KRD")


def richcheap_residuals(panel, bench=BENCH_TENORS, lam_grid=None):
    if lam_grid is None:
        lam_grid = np.linspace(0.8, 8.0, 8)
    tenors = np.asarray(panel.columns, dtype=float)
    bidx = [list(tenors).index(b) for b in bench]
    rows = {}
    for d, row in panel.iterrows():
        y = row.to_numpy(float)
        fit = fit_nss(tenors[bidx], y[bidx], lam_grid=lam_grid)
        rows[d] = (y - nss(tenors, **fit)) * BP
    out = pd.DataFrame(rows).T
    out.columns = tenors
    out.index.name = "date"
    return out


def zscore_panel(df, upto=None):
    hist = df.loc[:upto] if upto is not None else df
    mu = hist.mean(axis=0)
    sd = hist.std(axis=0, ddof=1).replace(0, np.nan)
    return (df - mu) / sd


def carry_roll(curve, tenors, horizon=HOLD_HORIZON):
    tenors = np.asarray(tenors, dtype=float)
    out = {}
    for t in tenors:
        if t <= horizon:
            continue
        out[t] = (curve.zero_rate(float(t)) - curve.forward_rate(float(horizon), float(t))) * BP
    return pd.Series(out, name="carry_roll_bp")


def butterfly_series(panel, legs=FLY_LEGS):
    w_lo, belly, w_hi = legs
    return ((2.0 * panel[belly] - panel[w_lo] - panel[w_hi]) * BP).rename("butterfly_bp")


def alert_threshold(series, q=0.95):
    s = pd.Series(series).dropna().abs()
    return float(s.quantile(q)) if not s.empty else np.nan


def evaluate_alerts(state, q=0.95):
    alerts = []
    rc_z_hist, rc_today = state["richcheap_z_hist"], state["richcheap_z_today"]
    for t in rc_today.index:
        thr, val = alert_threshold(rc_z_hist[t], q), rc_today[t]
        if np.isfinite(thr) and np.isfinite(val) and abs(val) > thr:
            side = "cheap（割安）" if val > 0 else "rich（割高）"
            alerts.append({"panel": "rich/cheap", "key": f"{t:g}Y",
                           "value": f"z={val:+.2f}", "threshold": f"±{thr:.2f}", "note": side})
    thr = alert_threshold(state["fly_z_hist"], q)
    if np.isfinite(thr) and abs(state["fly_z_today"]) > thr:
        side = "ベリー割安" if state["fly_z_today"] > 0 else "ベリー割高"
        alerts.append({"panel": "butterfly 2-5-10", "key": "5Y belly",
                       "value": f"z={state['fly_z_today']:+.2f}", "threshold": f"±{thr:.2f}", "note": side})
    dod_hist, dod_today = state["dod_hist"], state["dod_today"]
    for t in dod_today.index:
        thr, val = alert_threshold(dod_hist[t], q), dod_today[t]
        if np.isfinite(thr) and np.isfinite(val) and abs(val) > thr:
            alerts.append({"panel": "前日比", "key": f"{t:g}Y",
                           "value": f"{val:+.1f}bp", "threshold": f"±{thr:.1f}bp", "note": "大きめの日次変化"})
    return alerts


def compute_dashboard(panel, asof=None, freq=1, bench=BENCH_TENORS, legs=FLY_LEGS,
                      keys=KEY_TENORS, horizon=HOLD_HORIZON, q=0.95):
    if asof is None:
        asof = panel.index[-1]
    tenors = np.asarray(panel.columns, dtype=float)
    hist_panel = panel.loc[:asof]
    hist_zero = build_zero_panel(hist_panel, freq)
    asof_row = panel.loc[asof]
    curve = _curve_from_row(tenors, asof_row.to_numpy(float), freq)

    rc = richcheap_residuals(hist_panel, bench)
    rc_z = zscore_panel(rc, upto=asof)
    fly = butterfly_series(hist_panel, legs)
    fly_z = (fly - fly.mean()) / fly.std(ddof=1)
    dod = hist_zero.diff() * BP

    coupon = float(asof_row[float(legs[2])])
    maturity = float(legs[2])
    krd = key_rate_durations(curve, coupon, maturity, keys, freq)
    cr = carry_roll(curve, tenors, horizon)

    return {
        "asof": asof, "tenors": tenors, "panel": hist_panel, "zero_panel": hist_zero,
        "curve": curve, "richcheap_today": rc.loc[asof], "richcheap_z_today": rc_z.loc[asof],
        "richcheap_z_hist": rc_z, "fly_series": fly, "fly_z_today": float(fly_z.loc[asof]),
        "fly_z_hist": fly_z, "fly_z_series": fly_z, "dod_today": dod.loc[asof], "dod_hist": dod,
        "krd": krd, "carry_roll": cr, "hold_maturity": maturity, "q": q, "alerts": None,
    }


def render_dashboard(state):
    asof, tenors = state["asof"], state["tenors"]
    panel, zero_panel, q = state["panel"], state["zero_panel"], state["q"]
    alerts = state["alerts"] if state["alerts"] is not None else evaluate_alerts(state, q)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28, height_ratios=[1.0, 1.0, 0.6])
    fig.suptitle(f"日次カーブモニタリング — 今朝の画面（as-of {asof.date()}）",
                 fontsize=16, fontweight="bold")

    axA = fig.add_subplot(gs[0, 0])
    idx = len(panel.index) - 1
    for back, style in [(0, dict(color="black", lw=2.0, label="当日")),
                        (5, dict(color="tab:blue", lw=1.2, ls="--", label="5営業日前")),
                        (20, dict(color="tab:gray", lw=1.0, ls=":", label="20営業日前"))]:
        j = idx - back
        if j >= 0:
            axA.plot(tenors, panel.iloc[j].to_numpy(float) * 100, **style)
    axA.set_title("カーブ推移（パー利回り, %）")
    axA.set_xlabel("テナー (年)"); axA.set_ylabel("利回り (%)")
    axA.legend(fontsize=8); axA.grid(alpha=0.3)

    axB = fig.add_subplot(gs[0, 1])
    for t in [x for x in (2.0, 5.0, 10.0, 30.0) if x in tenors]:
        axB.plot(zero_panel.index, zero_panel[t] * 100, lw=1.1, label=f"{t:g}Y")
    axB.axvline(asof, color="red", lw=0.8, alpha=0.6)
    axB.set_title("主要テナーのゼロレート推移（%）")
    axB.set_xlabel("営業日"); axB.set_ylabel("ゼロレート (%)")
    axB.legend(fontsize=8, ncol=2); axB.grid(alpha=0.3); axB.tick_params(axis="x", rotation=30)

    axC = fig.add_subplot(gs[0, 2])
    rc = state["richcheap_today"]
    colors = ["tab:red" if v > 0 else "tab:blue" for v in rc.values]
    axC.bar([f"{t:g}" for t in rc.index], rc.values, color=colors)
    axC.axhline(0, color="black", lw=0.7)
    axC.set_title("rich/cheap 残差（bp, 赤=cheap 青=rich）")
    axC.set_xlabel("テナー (年)"); axC.set_ylabel("残差 (bp)"); axC.grid(alpha=0.3, axis="y")

    axD = fig.add_subplot(gs[1, 0])
    krd = state["krd"]
    axD.bar([f"{k:g}Y" for k in krd.index], krd.values, color="tab:purple")
    axD.set_title(f"KRD（代表:{state['hold_maturity']:g}年債, 合計={krd.sum():.2f}）")
    axD.set_xlabel("キーテナー"); axD.set_ylabel("KRD (年)"); axD.grid(alpha=0.3, axis="y")

    axE = fig.add_subplot(gs[1, 1])
    cr = state["carry_roll"]
    axE.bar([f"{t:g}" for t in cr.index], cr.values, color="tab:green")
    axE.axhline(0, color="black", lw=0.7)
    axE.set_title("キャリー＋ロール（3か月保有, bp）")
    axE.set_xlabel("テナー (年)"); axE.set_ylabel("bp"); axE.grid(alpha=0.3, axis="y")

    axF = fig.add_subplot(gs[1, 2])
    fz = state["fly_z_series"]
    thr = alert_threshold(fz, q)
    axF.plot(fz.index, fz.values, color="tab:orange", lw=1.2)
    axF.axhline(thr, color="red", ls="--", lw=0.9, label=f"±閾値({q:.0%}分位)")
    axF.axhline(-thr, color="red", ls="--", lw=0.9)
    axF.axhline(0, color="black", lw=0.6)
    axF.scatter([asof], [state["fly_z_today"]], color="red", zorder=5, s=40)
    axF.set_title("バタフライ 2-5-10 シグナル（zスコア）")
    axF.set_xlabel("営業日"); axF.set_ylabel("z"); axF.legend(fontsize=8)
    axF.grid(alpha=0.3); axF.tick_params(axis="x", rotation=30)

    axG = fig.add_subplot(gs[2, :])
    axG.axis("off")
    if alerts:
        head = f"アラート {len(alerts)} 件（閾値={q:.0%}分位）"
        lines = [f"[{a['panel']}] {a['key']}: {a['value']}（閾値 {a['threshold']}）… {a['note']}"
                 for a in alerts]
        txt = head + "\n" + "\n".join(lines)
        box = dict(boxstyle="round", facecolor="#ffecec", edgecolor="tab:red")
    else:
        txt = "アラートなし（全指標が過去分位点しきい値の内側）"
        box = dict(boxstyle="round", facecolor="#eefaee", edgecolor="tab:green")
    axG.text(0.01, 0.95, txt, va="top", ha="left", fontsize=10, family="monospace",
             transform=axG.transAxes, bbox=box)
    return fig


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main():
    import streamlit as st

    st.set_page_config(page_title="日次カーブモニタリング", layout="wide")
    st.title("日次カーブモニタリングダッシュボード")

    panel = load_par_panel(DATA_PATH)
    dates = list(panel.index)

    st.sidebar.header("設定")
    asof = st.sidebar.select_slider(
        "基準日（as-of）",
        options=dates,
        value=dates[-1],
        format_func=lambda d: pd.Timestamp(d).date().isoformat(),
    )
    q = st.sidebar.slider("アラート閾値の分位点 q", 0.80, 0.99, 0.95, 0.01)

    state = compute_dashboard(panel, asof=asof, freq=1, q=q)
    state["alerts"] = evaluate_alerts(state, q=q)

    col1, col2, col3 = st.columns(3)
    col1.metric("基準日", pd.Timestamp(asof).date().isoformat())
    col2.metric("KRD 合計（実効Dur）", f"{state['krd'].sum():.2f} 年")
    col3.metric("アラート件数", len(state["alerts"]))

    st.pyplot(render_dashboard(state))

    if state["alerts"]:
        st.subheader("アラート明細")
        st.dataframe(pd.DataFrame(state["alerts"]))
    else:
        st.success("アラートなし（全指標が過去分位点しきい値の内側）")


if __name__ == "__main__":
    main()
