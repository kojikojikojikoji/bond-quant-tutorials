# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # S2-2 演習 解答例

# %% [markdown]
# ## 準備：4補間器と共通ノード
#
# 本編の補間器を最小限だけ再掲する。実装は本編と同一で、演習で使う分に絞る。

# %%
import numpy as np
import pandas as pd

from bondlab.curve import bootstrap_par

np.random.seed(0)


class CurveInterpolator:
    def __init__(self, times, dfs):
        self.t = np.asarray(times, dtype=float)
        self.df = np.asarray(dfs, dtype=float)
        self.rt = -np.log(self.df)

    def neglogdf(self, t):
        raise NotImplementedError

    def inst_forward(self, t):
        raise NotImplementedError

    @staticmethod
    def _vec(fn, t):
        arr = np.asarray(t, dtype=float)
        if arr.ndim == 0:
            return float(fn(float(arr)))
        return np.array([fn(float(x)) for x in arr])

    def zero(self, t):
        arr = np.asarray(t, dtype=float)
        return self._vec(self.neglogdf, arr) / arr

    def forward_curve(self, ts):
        return self._vec(self.inst_forward, ts)


class LinearZero(CurveInterpolator):
    def __init__(self, times, dfs):
        super().__init__(times, dfs)
        mask = self.t > 0
        self.tz = self.t[mask]
        self.z = self.rt[mask] / self.tz

    def _seg(self, t):
        i = int(np.searchsorted(self.tz, t, side="right"))
        i = min(max(i, 1), len(self.tz) - 1)
        s = (self.z[i] - self.z[i - 1]) / (self.tz[i] - self.tz[i - 1])
        return i, s

    def neglogdf(self, t):
        return np.interp(t, self.tz, self.z) * t

    def inst_forward(self, t):
        i, s = self._seg(t)
        z = self.z[i - 1] + s * (t - self.tz[i - 1])
        return z + t * s


class LogLinearDF(CurveInterpolator):
    def neglogdf(self, t):
        return -np.interp(t, self.t, -self.rt)

    def inst_forward(self, t):
        i = int(np.searchsorted(self.t, t, side="right"))
        i = min(max(i, 1), len(self.t) - 1)
        return (self.rt[i] - self.rt[i - 1]) / (self.t[i] - self.t[i - 1])


def _thomas_natural(x, y):
    n = len(x)
    h = np.diff(x)
    M = np.zeros(n)
    if n < 3:
        return M
    lower = np.zeros(n - 2)
    diag = np.zeros(n - 2)
    upper = np.zeros(n - 2)
    rhs = np.zeros(n - 2)
    for k in range(1, n - 1):
        j = k - 1
        lower[j] = h[k - 1]
        diag[j] = 2.0 * (h[k - 1] + h[k])
        upper[j] = h[k]
        rhs[j] = 6.0 * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1])
    for j in range(1, n - 2):
        w = lower[j] / diag[j - 1]
        diag[j] -= w * upper[j - 1]
        rhs[j] -= w * rhs[j - 1]
    m = np.zeros(n - 2)
    m[-1] = rhs[-1] / diag[-1]
    for j in range(n - 4, -1, -1):
        m[j] = (rhs[j] - upper[j] * m[j + 1]) / diag[j]
    M[1:-1] = m
    return M


class NaturalCubicDF(CurveInterpolator):
    def __init__(self, times, dfs):
        super().__init__(times, dfs)
        self.x = self.t
        self.y = -self.rt
        self.M = _thomas_natural(self.x, self.y)

    def _eval(self, t, deriv=False):
        i = int(np.searchsorted(self.x, t, side="right"))
        i = min(max(i, 1), len(self.x) - 1)
        h = self.x[i] - self.x[i - 1]
        A = (self.x[i] - t) / h
        B = (t - self.x[i - 1]) / h
        yi_1, yi = self.y[i - 1], self.y[i]
        Mi_1, Mi = self.M[i - 1], self.M[i]
        if not deriv:
            return (A * yi_1 + B * yi
                    + ((A ** 3 - A) * Mi_1 + (B ** 3 - B) * Mi) * h ** 2 / 6.0)
        return ((yi - yi_1) / h
                - (3 * A ** 2 - 1) / 6.0 * h * Mi_1
                + (3 * B ** 2 - 1) / 6.0 * h * Mi)

    def neglogdf(self, t):
        return -self._eval(t, deriv=False)

    def inst_forward(self, t):
        return -self._eval(t, deriv=True)


class MonotoneConvex(CurveInterpolator):
    def __init__(self, times, dfs):
        super().__init__(times, dfs)
        T, RT = self.t, self.rt
        self.fd = np.diff(RT) / np.diff(T)
        n = len(self.fd)
        f = np.zeros(n + 1)
        for i in range(1, n):
            w_l = (T[i] - T[i - 1]) / (T[i + 1] - T[i - 1])
            w_r = (T[i + 1] - T[i]) / (T[i + 1] - T[i - 1])
            f[i] = w_l * self.fd[i] + w_r * self.fd[i - 1]
        f[0] = self.fd[0] - 0.5 * (f[1] - self.fd[0])
        f[n] = self.fd[n - 1] - 0.5 * (f[n - 1] - self.fd[n - 1])
        f[0] = min(max(f[0], 0.0), 2.0 * self.fd[0])
        for i in range(1, n):
            f[i] = min(max(f[i], 0.0), 2.0 * min(self.fd[i - 1], self.fd[i]))
        f[n] = min(max(f[n], 0.0), 2.0 * self.fd[n - 1])
        self.fnode = f
        self.n = n

    def _locate(self, t):
        i = int(np.searchsorted(self.t, t, side="right"))
        return min(max(i, 1), self.n)

    def _g_params(self, i):
        return self.fnode[i - 1] - self.fd[i - 1], self.fnode[i] - self.fd[i - 1]

    @staticmethod
    def _G(x, g0, g1):
        if g0 == 0.0 and g1 == 0.0:
            return 0.0
        if ((g0 < 0 and -0.5 * g0 <= g1 <= -2.0 * g0)
                or (g0 > 0 and -0.5 * g0 >= g1 >= -2.0 * g0)
                or (g0 * g1 < 0 and 0.5 * abs(g0) <= abs(g1) <= 2.0 * abs(g0))):
            return g0 * (1 - 4 * x + 3 * x ** 2) + g1 * (-2 * x + 3 * x ** 2)
        if (g0 < 0 and g1 > -2.0 * g0) or (g0 > 0 and g1 < -2.0 * g0):
            eta = (g1 + 2.0 * g0) / (g1 - g0)
            return g0 if x <= eta else g0 + (g1 - g0) * ((x - eta) / (1 - eta)) ** 2
        if (g0 > 0 and 0 > g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
            eta = 3.0 * g1 / (g1 - g0)
            return g1 + (g0 - g1) * ((eta - x) / eta) ** 2 if x < eta else g1
        eta = g1 / (g1 + g0)
        A = -g0 * g1 / (g0 + g1)
        if x <= eta:
            return A + (g0 - A) * ((eta - x) / eta) ** 2
        return A + (g1 - A) * ((x - eta) / (1 - eta)) ** 2

    @staticmethod
    def _int_G(x, g0, g1):
        if g0 == 0.0 and g1 == 0.0:
            return 0.0
        if ((g0 < 0 and -0.5 * g0 <= g1 <= -2.0 * g0)
                or (g0 > 0 and -0.5 * g0 >= g1 >= -2.0 * g0)
                or (g0 * g1 < 0 and 0.5 * abs(g0) <= abs(g1) <= 2.0 * abs(g0))):
            return g0 * (x - 2 * x ** 2 + x ** 3) + g1 * (-x ** 2 + x ** 3)
        if (g0 < 0 and g1 > -2.0 * g0) or (g0 > 0 and g1 < -2.0 * g0):
            eta = (g1 + 2.0 * g0) / (g1 - g0)
            if x <= eta:
                return g0 * x
            return g0 * x + (g1 - g0) * (x - eta) ** 3 / (3 * (1 - eta) ** 2)
        if (g0 > 0 and 0 > g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
            eta = 3.0 * g1 / (g1 - g0)
            if x <= eta:
                return g1 * x + (g0 - g1) / (eta ** 2) * (eta ** 3 - (eta - x) ** 3) / 3.0
            return g1 * eta + (g0 - g1) * eta / 3.0 + g1 * (x - eta)
        eta = g1 / (g1 + g0)
        A = -g0 * g1 / (g0 + g1)
        if x <= eta:
            return A * x + (g0 - A) / (eta ** 2) * (eta ** 3 - (eta - x) ** 3) / 3.0
        Ieta = A * eta + (g0 - A) * eta / 3.0
        return Ieta + A * (x - eta) + (g1 - A) * (x - eta) ** 3 / (3 * (1 - eta) ** 2)

    def inst_forward(self, t):
        i = self._locate(t)
        dt = self.t[i] - self.t[i - 1]
        x = (t - self.t[i - 1]) / dt
        g0, g1 = self._g_params(i)
        return self.fd[i - 1] + self._G(x, g0, g1)

    def neglogdf(self, t):
        if t <= 0:
            return 0.0
        i = self._locate(t)
        dt = self.t[i] - self.t[i - 1]
        x = (t - self.t[i - 1]) / dt
        g0, g1 = self._g_params(i)
        return self.rt[i - 1] + self.fd[i - 1] * (t - self.t[i - 1]) + dt * self._int_G(x, g0, g1)


def build_interps():
    par_df = pd.read_csv("data/samples/synthetic_ust_par_curve.csv")
    grid = np.arange(1, 31, dtype=float)
    par = np.interp(grid, par_df["tenor"].to_numpy(), par_df["par_yield"].to_numpy())
    curve = bootstrap_par(grid, par, frequency=1)
    t, d = curve.times, curve.dfs
    return {
        "ゼロ線形": LinearZero(t, d),
        "log-DF 線形": LogLinearDF(t, d),
        "自然三次スプライン": NaturalCubicDF(t, d),
        "単調凸 (Hagan-West)": MonotoneConvex(t, d),
    }, curve


interps, curve = build_interps()

# %% [markdown]
# ## 演習1：フォワードのギザギザ度と推奨
#
# 密なグリッド上の瞬間フォワードの2階差分 $\Delta^2 f$ の L2 ノルムを求める。
# 値が小さいほどフォワードが滑らかである。

# %%
rough_grid = np.linspace(1.0, 30.0, 600)


def forward_roughness(obj, ts):
    return float(np.linalg.norm(np.diff(obj.forward_curve(ts), n=2)))


rows = []
for label, obj in interps.items():
    f = obj.forward_curve(rough_grid)
    rows.append({
        "補間法": label,
        "2階差分ノルム(×1e-4)": round(forward_roughness(obj, rough_grid) * 1e4, 4),
        "最小フォワード%": round(f.min() * 100, 4),
        "最大フォワード%": round(f.max() * 100, 4),
    })

ex1 = pd.DataFrame(rows)
print(ex1.to_string(index=False))

# %% [markdown]
# **解釈と推奨**：
#
# - **log-DF 線形**と**ゼロ線形**はフォワードが階段状・鋸歯状に折れるため、2階差分
#   ノルムが大きい（ギザギザ）。ただし局所的で、log-DF 線形は正値を保証する。
# - **自然三次スプライン**は 2階差分ノルムが小さく最も滑らかに見えるが、フォワードが
#   入力にない振れ（オーバーシュート）を起こしやすく、非局所である。最小フォワードが
#   極端に振れる場合は負値の危険もある。
# - **単調凸**は、滑らかさ（小さめの2階差分ノルム）と非振動・正値保存・局所性を
#   両立する。
#
# **推奨**：フォワードを一次入力に使う用途（金利オプション評価・キャリー分析）では
# **単調凸**を推す。理由は、滑らかさだけを最大化する三次スプラインが非局所な振動で
# 誤ったフォワードを生むのに対し、単調凸は入力を厳密に再現しつつ正値・非振動・局所性を
# 保つためである。実装の単純さ・堅牢さ最優先なら **log-DF 線形**（`bondlab` の既定）が
# 無難な第二候補になる。

# %% [markdown]
# ## 演習2：局所性の実験（1点を動かした波及範囲）
#
# 10 年ノードのゼロレートを $+20$bp 動かし、各補間でフォワードカーブがどこまで
# 変わるかを測る。波及した年限の範囲（変化が $0.5$bp を超える区間）を比べる。

# %%
def bumped_dfs(curve, bump_tenor=10.0, bump_bp=20.0):
    """指定テナーのゼロレートを bump_bp だけ動かした割引係数列を返す。"""
    t = curve.times.copy()
    z = np.where(t > 0, -np.log(curve.dfs) / np.where(t > 0, t, 1.0), 0.0)
    idx = int(np.argmin(np.abs(t - bump_tenor)))
    z = z.copy()
    z[idx] += bump_bp / 1e4
    d = curve.dfs.copy()
    d[idx] = np.exp(-z[idx] * t[idx])
    return t, d


t_base, d_base = curve.times, curve.dfs
t_bump, d_bump = bumped_dfs(curve, bump_tenor=10.0, bump_bp=20.0)

cls_map = {
    "ゼロ線形": LinearZero,
    "log-DF 線形": LogLinearDF,
    "自然三次スプライン": NaturalCubicDF,
    "単調凸 (Hagan-West)": MonotoneConvex,
}

probe = np.linspace(1.0, 30.0, 600)
rows = []
for label, cls in cls_map.items():
    base = cls(t_base, d_base)
    bumped = cls(t_bump, d_bump)
    diff_bp = np.abs(bumped.forward_curve(probe) - base.forward_curve(probe)) * 1e4
    affected = probe[diff_bp > 0.5]
    if affected.size:
        span = f"{affected.min():.1f}〜{affected.max():.1f} 年"
        width = affected.max() - affected.min()
    else:
        span, width = "なし", 0.0
    rows.append({
        "補間法": label,
        "波及した年限": span,
        "波及幅(年)": round(width, 1),
        "最大変化(bp)": round(diff_bp.max(), 3),
    })

ex2 = pd.DataFrame(rows)
print(ex2.to_string(index=False))

# %% [markdown]
# **解釈**：
#
# - **log-DF 線形**・**ゼロ線形**・**単調凸**は、10 年ノードの変更が隣接する2区間
#   （おおむね 7〜20 年）だけに波及し、それより外側は不変である。**局所的**な補間で
#   あることが数値で確認できる。
# - **自然三次スプライン**は、10 年の変更がカーブ全体（短期側・超長期側まで）に波及
#   する。三次スプラインの係数が大域的な連立方程式で決まるためで、**非局所**である
#   ことを示す。
#
# 局所性は実務で重要である。ある年限の気配だけが動いたとき、遠く離れた年限の
# フォワードまで動く補間は、リスク要因の帰属（どの年限のリスクか）を曖昧にする。
# この点でも log-DF 線形や単調凸のような局所的な補間が好まれる。

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for label, cls in cls_map.items():
    base = cls(t_base, d_base)
    bumped = cls(t_bump, d_bump)
    d_bp = (bumped.forward_curve(probe) - base.forward_curve(probe)) * 1e4
    ax = axes[0] if label in ("自然三次スプライン",) else axes[1]
    ax.plot(probe, d_bp, label=label, lw=1.3)
axes[0].set_title("非局所：自然三次スプライン")
axes[1].set_title("局所：線形系・単調凸")
for ax in axes:
    ax.axvline(10.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("年限")
    ax.set_ylabel("フォワード変化 (bp)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
